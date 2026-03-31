#!/usr/bin/env bash
# =============================================================================
# Torch Profiler Trace Capture for ATOM/vLLM Inference (MI355X)
#
# Captures GPU kernel timelines via PyTorch Profiler for kernel-level
# performance analysis on AMD MI355X GPUs.
#
# Flow:
#   1. Cleanup residual processes (port, GPU, shared memory)
#   2. Start ATOM server with --torch-profiler-dir --mark-trace
#   3. Warmup with full benchmark load (reach steady state)
#   4. Start profile, run benchmark (same config as warmup), stop profile
#   5. Wait for trace flush (.json.gz appears AND .json disappears)
#   6. Stop server, copy traces, parse decode wall times by batch size
#
# Key design decisions (learned from repeated debugging):
#   - Port cleanup uses SIGKILL (-9) + sleep 2 (SIGTERM not enough)
#   - GPU cleanup kills all /dev/kfd users + clears /dev/shm/aiter_*
#   - Trace flush waits for .json.gz to appear AND .json to vanish
#     (.json.gz is created before compression finishes; .json gone = done)
#   - Profiling prompts default to warmup prompts (not concurrency) to
#     capture prefill-decode interleaving that occurs in real workloads
#   - Python parse script uses heredoc (not python3 -c) to avoid shell
#     eating newlines and causing syntax errors
#   - After stop_profile, commands use ; not && (kill returns non-zero)
#
# Usage:
#   bash scripts/collect_atom_trace.sh \
#     --model /shared/data/amd_int/models/DeepSeek-R1-0528-MTP-MoE-MXFP4-Attn-PTPC-FP8 \
#     --scenario chat --concurrency 64 --tp 4 \
#     --result-dir ./results_mi355x_mxfp4_mtp0_ep1_tp4
#
#   # Smaller trace (faster, less memory):
#   bash scripts/collect_atom_trace.sh \
#     --model ... --concurrency 64 --profile-prompts 128 \
#     --result-dir ./results_mi355x_mxfp4_mtp0_ep1_tp4
#
# Prerequisites:
#   - AMD Instinct MI355X GPUs with ROCm
#   - ATOM installed (python3 -m atom.entrypoints.openai_server)
#   - Run inside the ATOM container (model paths must be accessible)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ======================== Defaults ============================================
MODEL=""
SCENARIO="chat"
CONCURRENCY=64
TP=4
RESULT_DIR=""
TRACE_DIR="/tmp/trace"
SERVER_PORT=8000
GPU_MEM_UTIL=0.90
KV_CACHE_DTYPE="fp8"
MAX_NUM_SEQS=512
MAX_MODEL_LEN=""
PROFILE_NUM_PROMPTS=""      # defaults to WARMUP_NUM_PROMPTS if not set
ROCTRACER_MAX_EVENTS=""     # if set, auto-generate libkineto.conf (default 1M, use 10M+ for long traces)
ROCTX_MARKERS=false         # inject roctx markers into engine for decode/prefill step timing
FLUSH_TIMEOUT=300
LAYER=40

# ======================== Argument Parsing ====================================
usage() {
    cat <<EOF
Usage: bash $(basename "$0") [options]

Torch Profiler trace capture for ATOM/vLLM kernel-level profiling on MI355X.

Required:
  --model PATH              Model path
  --result-dir DIR          Output directory for trace files

Options:
  --scenario SCENARIO       chat (1K/1K), reasoning (1K/8K), summarize (8K/1K) [default: chat]
  --concurrency N           Request concurrency [default: 64]
  --tp N                    Tensor parallelism [default: 4]
  --port PORT               Server port [default: 8000]
  --gpu-mem-util FLOAT      GPU memory utilization [default: 0.90]
  --max-model-len N         Override max model length
  --profile-prompts N       Prompts during profiling [default: conc*2, same as warmup]
                            Use smaller values (e.g. 128) for faster runs with less memory.
                            Must be >= concurrency to maintain steady-state batch size.
  --roctracer-max-events N  Set ROCTRACER_MAX_EVENTS (default 1M; use 10000000 for longer traces)
  --roctx-markers           Inject roctx markers (decode_step/prefill_step) into engine
  --flush-timeout N         Max seconds to wait for trace flush [default: 300]
  --layer N                 Layer index for parse_trace.py [default: 40]
  -h, --help                Show this help

Trace size estimates (chat scenario, concurrency 64):
  --profile-prompts 64   ~  64 prompts: ~500MB compressed, ~5GB RAM to parse (no prefill interleave)
  --profile-prompts 128  ~ 128 prompts: ~800MB compressed, ~8GB RAM (some interleave)
  --profile-prompts 256  ~ 256 prompts: ~1.5GB compressed, ~15GB RAM (good interleave)
  --profile-prompts 640  ~ 640 prompts: ~3GB compressed, ~30GB RAM (full SA match)

Examples:
  # Full SA-matching trace (chat, concurrency 64, TP=4)
  bash $(basename "$0") \\
    --model /shared/data/amd_int/models/DeepSeek-R1-0528-MTP-MoE-MXFP4-Attn-PTPC-FP8 \\
    --scenario chat --concurrency 64 --tp 4 \\
    --result-dir ./results_mi355x_mxfp4_mtp0_ep1_tp4

  # Smaller trace for quick iteration
  bash $(basename "$0") \\
    --model /shared/data/amd_int/models/DeepSeek-R1-0528-MTP-MoE-MXFP4-Attn-PTPC-FP8 \\
    --scenario chat --concurrency 64 --tp 4 --profile-prompts 128 \\
    --result-dir ./results_mi355x_mxfp4_mtp0_ep1_tp4
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)            MODEL="$2"; shift 2 ;;
        --scenario)         SCENARIO="$2"; shift 2 ;;
        --concurrency)      CONCURRENCY="$2"; shift 2 ;;
        --tp)               TP="$2"; shift 2 ;;
        --result-dir)       RESULT_DIR="$2"; shift 2 ;;
        --port)             SERVER_PORT="$2"; shift 2 ;;
        --gpu-mem-util)     GPU_MEM_UTIL="$2"; shift 2 ;;
        --max-model-len)    MAX_MODEL_LEN="$2"; shift 2 ;;
        --profile-prompts)  PROFILE_NUM_PROMPTS="$2"; shift 2 ;;
        --roctracer-max-events) ROCTRACER_MAX_EVENTS="$2"; shift 2 ;;
        --roctx-markers)    ROCTX_MARKERS=true; shift ;;
        --flush-timeout)    FLUSH_TIMEOUT="$2"; shift 2 ;;
        --layer)            LAYER="$2"; shift 2 ;;
        -h|--help)          usage ;;
        *)                  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ======================== Validation ==========================================
if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required"; usage
fi
if [[ -z "$RESULT_DIR" ]]; then
    echo "ERROR: --result-dir is required"; usage
fi

case "$SCENARIO" in
    chat)      ISL=1024; OSL=1024 ;;
    reasoning) ISL=1024; OSL=8192 ;;
    summarize) ISL=8192; OSL=1024 ;;
    *) echo "ERROR: --scenario must be chat, reasoning, or summarize"; exit 1 ;;
esac

# Compute defaults
WARMUP_NUM_PROMPTS=$(( CONCURRENCY * 2 ))

# Profile prompts default to warmup prompts to capture prefill-decode interleaving
[[ -z "$PROFILE_NUM_PROMPTS" ]] && PROFILE_NUM_PROMPTS=$WARMUP_NUM_PROMPTS

if [[ -z "$MAX_MODEL_LEN" ]]; then
    MAX_MODEL_LEN=$(( ISL + OSL + 200 ))
fi

# ======================== Utilities ===========================================
TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(TS)] $*"; }

# ======================== Cleanup =============================================
cleanup_residual() {
    log "Cleaning up residual processes..."

    # Kill anything on the server port (SIGKILL, not SIGTERM)
    fuser -k -9 "${SERVER_PORT}/tcp" 2>/dev/null || true
    sleep 2

    # Kill residual GPU processes
    local gpu_pids
    gpu_pids=$(fuser /dev/kfd 2>/dev/null) || true
    if [[ -n "$gpu_pids" ]]; then
        log "  Killing residual GPU processes: $gpu_pids"
        kill -9 $gpu_pids 2>/dev/null || true
        sleep 2
    fi

    # Clean shared memory (aiter blocks from previous runs)
    rm -f /dev/shm/aiter_*

    # Clean stale roctx .pth hook from previous crashed runs
    local stale_pth
    stale_pth=$(python3 -c "import site; print(site.getsitepackages()[0])")/_roctx_hook.pth
    if [[ -f "$stale_pth" ]]; then
        rm -f "$stale_pth"
        log "  Removed stale roctx hook: $stale_pth"
    fi

    # Clean trace dir
    rm -rf "$TRACE_DIR"

    # Verify port is free
    if fuser "${SERVER_PORT}/tcp" 2>/dev/null | grep -q .; then
        log "  WARNING: Port $SERVER_PORT still in use after cleanup!"
        fuser -v "${SERVER_PORT}/tcp" 2>/dev/null || true
    fi

    log "  Cleanup done."
}

# ======================== Wait for Trace Flush ================================
# Waits for the run trace .json.gz to appear AND the uncompressed .json to
# vanish. The profiler writes .tmp -> .json -> .json.gz; if we only check
# for .json.gz we may catch it mid-compression (corrupt file).
wait_for_trace_flush() {
    log "Waiting for trace flush (timeout ${FLUSH_TIMEOUT}s)..."
    local wait_elapsed=0

    while true; do
        local gz_file json_file
        gz_file=$(ls "$TRACE_DIR"/rank_0/*.json.gz 2>/dev/null | grep -v capture_graph | head -1) || true
        json_file=$(ls "$TRACE_DIR"/rank_0/*.json 2>/dev/null | grep -v capture_graph | head -1) || true

        # Done when .json.gz exists AND .json (uncompressed) is gone
        if [[ -n "$gz_file" ]] && [[ -z "$json_file" ]]; then
            local size
            size=$(du -h "$gz_file" | cut -f1)
            log "  Trace flush complete: $(basename "$gz_file") ($size)"
            return 0
        fi

        sleep 5
        wait_elapsed=$((wait_elapsed + 5))

        if [[ $wait_elapsed -ge $FLUSH_TIMEOUT ]]; then
            log "  WARNING: Trace flush timeout after ${FLUSH_TIMEOUT}s"
            log "  Files in trace dir:"
            ls -lh "$TRACE_DIR"/rank_0/ 2>/dev/null || true
            return 1
        fi

        # Progress indicator every 30s
        if [[ $((wait_elapsed % 30)) -eq 0 ]]; then
            log "  Still waiting... (${wait_elapsed}s elapsed)"
            ls "$TRACE_DIR"/rank_0/ 2>/dev/null | head -5 || true
        fi
    done
}

# ======================== Parse Decode Events =================================
# Extracts decode wall times from trace, groups by batch size, and prints
# a summary table. This is the key metric for comparing with B200 nsys data.
parse_decode_walltime() {
    local trace_file="$1"
    local output_csv="$2"

    python3 - "$trace_file" "$output_csv" << 'PYEOF'
import gzip, json, re, sys, csv
from collections import defaultdict

trace_file = sys.argv[1]
output_csv = sys.argv[2] if len(sys.argv) > 2 else ""

opener = gzip.open if trace_file.endswith(".gz") else open
print(f"Loading trace: {trace_file}")
with opener(trace_file, "rt", encoding="utf-8") as f:
    data = json.load(f)

events = data.get("traceEvents", [])
print(f"Total trace events: {len(events)}")

decodes = sorted(
    [e for e in events
     if e.get("name", "").startswith("decode[")
     and e.get("ph") == "X"
     and e.get("cat") == "gpu_user_annotation"],
    key=lambda x: x["ts"]
)

prefills = [e for e in events
    if e.get("name", "").startswith("prefill")
    and e.get("ph") == "X"
    and e.get("cat") == "gpu_user_annotation"]

# roctx markers from roctx_patch.py: decode_step_N_bs=M / prefill_step_N_bs=M
roctx_pat = re.compile(r"(decode|prefill)_step_\d+_bs=(\d+)")

# Try X events (profiler typically converts push/pop to complete events)
roctx_all = [e for e in events if roctx_pat.match(e.get("name", ""))]
roctx_x = [e for e in roctx_all if e.get("ph") == "X"]

# B/E fallback: stack-based matching per thread
if not roctx_x:
    roctx_b = [e for e in roctx_all if e.get("ph") == "B"]
    if roctx_b:
        roctx_tids = {e.get("tid") for e in roctx_b}
        be_events = sorted(
            [e for e in events if e.get("tid") in roctx_tids and e.get("ph") in ("B", "E")],
            key=lambda x: x["ts"]
        )
        stacks = defaultdict(list)
        for e in be_events:
            tid = e.get("tid")
            if e["ph"] == "B":
                stacks[tid].append(e)
            elif stacks[tid]:
                b = stacks[tid].pop()
                if roctx_pat.match(b.get("name", "")):
                    roctx_x.append({**b, "dur": e["ts"] - b["ts"], "ph": "X"})

roctx_decodes = sorted(
    [e for e in roctx_x if e.get("name", "").startswith("decode")],
    key=lambda x: x["ts"]
)
roctx_prefills_list = [e for e in roctx_x if e.get("name", "").startswith("prefill")]

print(f"Decode events: {len(decodes)}")
print(f"Prefill events: {len(prefills)}")
print(f"roctx decode events: {len(roctx_decodes)}")
print(f"roctx prefill events: {len(roctx_prefills_list)}")

if not decodes and not roctx_decodes:
    print("No decode events found!")
    sys.exit(1)

groups = defaultdict(list)
for d in decodes:
    bs = re.search(r"bs=(\d+)", d["name"])
    if bs:
        groups[int(bs.group(1))].append(d["dur"] / 1000)

csv_rows = []
if groups:
    header = f"{'bs':<6} {'count':>8} {'avg(ms)':>10} {'min(ms)':>10} {'max(ms)':>10} {'p50(ms)':>10} {'p99(ms)':>10}"
    print(f"\n  ATOM --mark-trace decode timing")
    print(f"{header}")
    print("-" * len(header))
    for bs in sorted(groups):
        vals = sorted(groups[bs])
        n = len(vals)
        avg = sum(vals) / n
        p50 = vals[n // 2]
        p99 = vals[min(int(n * 0.99), n - 1)]
        print(f"{bs:<6d} {n:>8d} {avg:>10.2f} {min(vals):>10.2f} {max(vals):>10.2f} {p50:>10.2f} {p99:>10.2f}")
        csv_rows.append({
            "bs": bs, "count": n, "avg_ms": round(avg, 2),
            "min_ms": round(min(vals), 2), "max_ms": round(max(vals), 2),
            "p50_ms": round(p50, 2), "p99_ms": round(p99, 2)
        })
    steady_bs = max(groups, key=lambda b: len(groups[b]))
    steady_vals = groups[steady_bs]
    steady_avg = sum(steady_vals) / len(steady_vals)
    steady_p50 = sorted(steady_vals)[len(steady_vals) // 2]
    print(f"\n{'='*60}")
    print(f"Steady-state decode (bs={steady_bs}):")
    print(f"  avg={steady_avg:.2f}ms  p50={steady_p50:.2f}ms  ({len(steady_vals)} iterations)")
    print(f"  Compare with B200 nsys decode wall time for gap analysis")
    print(f"{'='*60}")

# roctx execute_model analysis
roctx_groups = defaultdict(list)
roctx_csv_rows = []
if roctx_decodes:
    for d in roctx_decodes:
        m = re.search(r"bs=(\d+)", d["name"])
        if m:
            roctx_groups[int(m.group(1))].append(d["dur"] / 1000)

    print(f"\n{'='*60}")
    print(f"  roctx execute_model() decode timing")
    print(f"{'='*60}")
    rh = f"{'bs':<6} {'count':>8} {'avg(ms)':>10} {'min(ms)':>10} {'max(ms)':>10} {'p50(ms)':>10} {'p99(ms)':>10}"
    print(rh)
    print("-" * len(rh))
    for bs in sorted(roctx_groups):
        vals = sorted(roctx_groups[bs])
        n = len(vals)
        avg = sum(vals) / n
        p50 = vals[n // 2]
        p99 = vals[min(int(n * 0.99), n - 1)]
        print(f"{bs:<6d} {n:>8d} {avg:>10.2f} {min(vals):>10.2f} {max(vals):>10.2f} {p50:>10.2f} {p99:>10.2f}")
        roctx_csv_rows.append({
            "bs": bs, "count": n, "avg_ms": round(avg, 2),
            "min_ms": round(min(vals), 2), "max_ms": round(max(vals), 2),
            "p50_ms": round(p50, 2), "p99_ms": round(p99, 2)
        })

    # Comparison: ATOM native decode[ vs roctx execute_model
    common_bs = set(groups.keys()) & set(roctx_groups.keys())
    if common_bs:
        print(f"\n{'='*60}")
        print(f"  ATOM --mark-trace vs roctx execute_model comparison")
        print(f"{'='*60}")
        print(f"{'bs':<6} {'mark-trace':>12} {'roctx':>12} {'diff(ms)':>10} {'note':>20}")
        print("-" * 64)
        for bs in sorted(common_bs):
            atom_avg = sum(groups[bs]) / len(groups[bs])
            roctx_avg = sum(roctx_groups[bs]) / len(roctx_groups[bs])
            diff = atom_avg - roctx_avg
            note = "sched overhead" if diff > 0.5 else ("roctx wider?" if diff < -0.5 else "~same")
            print(f"{bs:<6d} {atom_avg:>12.2f} {roctx_avg:>12.2f} {diff:>+10.2f} {note:>20}")

# Save CSV if requested
if output_csv:
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bs", "count", "avg_ms", "min_ms", "max_ms", "p50_ms", "p99_ms"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved: {output_csv}")
    if roctx_csv_rows:
        roctx_csv_path = output_csv.replace(".csv", "_roctx.csv")
        with open(roctx_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["bs", "count", "avg_ms", "min_ms", "max_ms", "p50_ms", "p99_ms"])
            writer.writeheader()
            writer.writerows(roctx_csv_rows)
        print(f"roctx CSV saved: {roctx_csv_path}")
PYEOF
}

# ======================== Main ================================================
SERVER_PID=""

ROCTX_PTH=""
trap_cleanup() {
    [[ -n "$ROCTX_PTH" ]] && rm -f "$ROCTX_PTH" 2>/dev/null
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Trap: killing server PID=$SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap trap_cleanup EXIT INT TERM

TAG="trace_${SCENARIO}_c${CONCURRENCY}_tp${TP}_p${PROFILE_NUM_PROMPTS}"

log "============================================================"
log "  ATOM Torch Profiler Trace Capture"
log "============================================================"
log "  Model:         $MODEL"
log "  Scenario:      $SCENARIO (ISL=$ISL, OSL=$OSL)"
log "  Concurrency:   $CONCURRENCY"
log "  TP:            $TP"
log "  Warmup:        $WARMUP_NUM_PROMPTS prompts"
log "  Profile:       $PROFILE_NUM_PROMPTS prompts"
log "  Max Model Len: $MAX_MODEL_LEN"
log "  GPU Mem Util:  $GPU_MEM_UTIL"
log "  Result Dir:    $RESULT_DIR"
log "  Trace Dir:     $TRACE_DIR"
log "  ROCTracer Max: ${ROCTRACER_MAX_EVENTS:-default (1M)}"
log "  Tag:           $TAG"
log "============================================================"

# Log file (don't use exec+tee — it buffers Python's tqdm/asyncio and hangs)
mkdir -p "$RESULT_DIR"
SCRIPT_LOG="$RESULT_DIR/${TAG}.log"
log "Log file: $SCRIPT_LOG"

# Step 1: Cleanup
cleanup_residual

# Apply Kineto/ROCTracer config if specified
if [[ -n "$ROCTRACER_MAX_EVENTS" ]]; then
    KINETO_CONF="$RESULT_DIR/libkineto.conf"
    echo "ROCTRACER_MAX_EVENTS=$ROCTRACER_MAX_EVENTS" > "$KINETO_CONF"
    export KINETO_CONFIG="$KINETO_CONF"
    log "Kineto config: ROCTRACER_MAX_EVENTS=$ROCTRACER_MAX_EVENTS ($KINETO_CONF)"
fi

# Step 2: Start ATOM server with profiling enabled
log "Starting ATOM server (TP=$TP, profiler enabled)..."

# Install roctx hook into all Python processes (main + spawned workers)
if [[ "$ROCTX_MARKERS" == "true" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROCTX_SITE_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])")
    ROCTX_PTH="$ROCTX_SITE_DIR/_roctx_hook.pth"
    echo "import sys; sys.path.insert(0, '$SCRIPT_DIR'); import roctx_patch" > "$ROCTX_PTH"
    export ROCTX_PATCH_ENABLED=1
    log "roctx hook installed: $ROCTX_PTH (activates in all Python processes)"
fi

python3 -m atom.entrypoints.openai_server \
    --model "$MODEL" \
    --server-port "$SERVER_PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv_cache_dtype "$KV_CACHE_DTYPE" \
    --torch-profiler-dir "$TRACE_DIR" \
    --mark-trace > "$RESULT_DIR/server_${TAG}.log" 2>&1 &
SERVER_PID=$!
log "Server PID: $SERVER_PID"

# Wait for server ready (with dead-process detection)
log "Waiting for server to be ready..."
server_wait=0
while ! curl -s "http://0.0.0.0:${SERVER_PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "ERROR: Server died during startup. Last 30 lines:"
        tail -30 "$RESULT_DIR/server_${TAG}.log"
        exit 1
    fi
    sleep 5
    server_wait=$((server_wait + 5))
    if [[ $server_wait -ge 600 ]]; then
        log "ERROR: Server not ready after 600s"
        tail -30 "$RESULT_DIR/server_${TAG}.log"
        exit 1
    fi
done
log "Server ready (took ~${server_wait}s)."

# Auto-detect served model name
SERVED_MODEL=$(curl -s "http://0.0.0.0:${SERVER_PORT}/v1/models" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
log "Served model: $SERVED_MODEL"

# Step 3: Warmup (match SA benchmark config: conc*10 prompts)
log "Running warmup benchmark ($WARMUP_NUM_PROMPTS prompts, concurrency $CONCURRENCY)..."
python3 -u -m atom.benchmarks.benchmark_serving \
    --model "$SERVED_MODEL" \
    --backend vllm \
    --base-url "http://0.0.0.0:${SERVER_PORT}" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio 0.8 \
    --num-prompts "$WARMUP_NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --request-rate inf \
    --ignore-eos >> "$SCRIPT_LOG" 2>&1 || {
    log "WARNING: Warmup benchmark failed"
}
log "Warmup benchmark done."

# Step 4: Profile (same config to capture prefill-decode interleaving)
log "Starting profiler..."
curl -s -X POST "http://0.0.0.0:${SERVER_PORT}/start_profile" || {
    log "ERROR: Failed to start profiler"; exit 1
}

TRACE_RESULT_FILE="result_profiled_${TAG}.json"
log "Running profiled benchmark ($PROFILE_NUM_PROMPTS prompts, concurrency $CONCURRENCY)..."
log "  Result file: $RESULT_DIR/$TRACE_RESULT_FILE"
python3 -u -m atom.benchmarks.benchmark_serving \
    --model "$SERVED_MODEL" \
    --backend vllm \
    --base-url "http://0.0.0.0:${SERVER_PORT}" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio 0.8 \
    --num-prompts "$PROFILE_NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --request-rate inf \
    --ignore-eos \
    --save-result \
    --save-detailed \
    --result-dir "$RESULT_DIR" \
    --result-filename "$TRACE_RESULT_FILE" >> "$SCRIPT_LOG" 2>&1 || {
    log "WARNING: Profiled benchmark failed"
}
log "Profiled benchmark done."

# Extract per-request debug CSV from detailed JSON, then strip large arrays
DETAILED_JSON="$RESULT_DIR/$TRACE_RESULT_FILE"
if [[ -f "$DETAILED_JSON" ]]; then
    PERREQ_CSV="$RESULT_DIR/per_request_${TAG}.csv"
    if python3 - "$DETAILED_JSON" "$PERREQ_CSV" <<'PYEOF'
import json, csv, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
ttfts = d.get('ttfts', [])
itls = d.get('itls', [])
olens = d.get('output_lens', [])
with open(sys.argv[2], 'w', newline='') as out:
    w = csv.writer(out)
    w.writerow(['req_id','output_len','ttft_ms','tpot_ms','decode_time_ms','e2e_latency_ms'])
    for i in range(len(ttfts)):
        ttft = ttfts[i] * 1000
        raw = itls[i] if i < len(itls) else []
        itl_list = raw if isinstance(raw, list) else [raw]
        olen = olens[i] if i < len(olens) else 0
        decode = sum(t * 1000 for t in itl_list)
        tpot = (decode / len(itl_list)) if itl_list else 0
        e2e = ttft + decode
        w.writerow([i, olen, f'{ttft:.2f}', f'{tpot:.2f}', f'{decode:.2f}', f'{e2e:.2f}'])
PYEOF
    then log "Saved per-request CSV: $PERREQ_CSV"
    else log "WARNING: per-request CSV extraction failed"
    fi
    # Keep original as _detailed, write stripped version as main JSON
    DETAILED_BACKUP="${DETAILED_JSON%.json}_detailed.json"
    mv "$DETAILED_JSON" "$DETAILED_BACKUP"
    if python3 - "$DETAILED_BACKUP" "$DETAILED_JSON" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
for k in ['input_lens','output_lens','ttfts','itls','generated_texts','errors']:
    d.pop(k, None)
with open(sys.argv[2], 'w') as f:
    json.dump(d, f)
PYEOF
    then log "Kept detailed: $(basename $DETAILED_BACKUP), stripped: $TRACE_RESULT_FILE"
    else log "WARNING: JSON strip failed"
    fi
fi

log "Stopping profiler..."
curl -s -X POST "http://0.0.0.0:${SERVER_PORT}/stop_profile" || true

# Step 5: Wait for trace flush (must complete before killing server)
wait_for_trace_flush

# Step 6: Stop server
log "Stopping server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
sleep 3

# Step 7: Copy traces and parse
log "Trace files in $TRACE_DIR/rank_0/:"
ls -lh "$TRACE_DIR"/rank_0/ 2>/dev/null || true

TRACE_FILE=$(ls "$TRACE_DIR"/rank_0/*.json.gz 2>/dev/null | grep -v capture_graph | head -1) || true

if [[ -n "$TRACE_FILE" ]]; then
    log "Copying traces to $RESULT_DIR/"
    cp -v "$TRACE_DIR"/rank_0/*.json.gz "$RESULT_DIR/"

    log "============================================================"
    log "  DECODE WALL TIME ANALYSIS"
    log "============================================================"
    DECODE_CSV="$RESULT_DIR/decode_walltime_${TAG}.csv"
    parse_decode_walltime "$TRACE_FILE" "$DECODE_CSV"

    # Also run parse_trace.py if available (for kernel breakdown)
    PARSE_TRACE="/app/ATOM/tools/parse_trace.py"
    if [[ -f "$PARSE_TRACE" ]]; then
        log "Running parse_trace.py --layer $LAYER..."
        python3 "$PARSE_TRACE" "$TRACE_FILE" --layer "$LAYER" 2>&1 || \
            log "WARNING: parse_trace.py failed (norm module detection may need patching)"
    fi
else
    log "ERROR: No trace file found in $TRACE_DIR/rank_0/"
    ls -la "$TRACE_DIR"/rank_0/ 2>/dev/null || true
    exit 1
fi

log "============================================================"
log "  TRACE CAPTURE COMPLETE"
log "============================================================"
log "  Traces:    $RESULT_DIR/"
log "  Decode CSV: $DECODE_CSV"
log "  Log:       $SCRIPT_LOG"
log "============================================================"
