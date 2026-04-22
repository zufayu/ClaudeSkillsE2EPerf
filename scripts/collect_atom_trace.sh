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
PROFILE_NUM_PROMPTS=""      # defaults to CONC*2 (matches ATOM CI regression flow + SA InferenceX)
                            # Old default CONC*10 produced huge traces (~3GB compressed for c=64).
                            # CONC*2 still yields >100 decode steps for stable operator-time averaging.
ROCTRACER_MAX_EVENTS=""     # if set, auto-generate libkineto.conf (default 1M, use 10M+ for long traces)
FLUSH_TIMEOUT=300
LAYER=40
EXPERT_PARALLEL="false"
MODEL_NAME="dsr1"
QUANT="mxfp4"
ENV=""

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
  --ep                      Enable expert parallelism (--enable-expert-parallel)
  --port PORT               Server port [default: 8000]
  --gpu-mem-util FLOAT      GPU memory utilization [default: 0.90]
  --max-model-len N         Override max model length
  --profile-prompts N       Prompts during profiling [default: conc*2, same as warmup]
                            Use smaller values (e.g. 128) for faster runs with less memory.
                            Must be >= concurrency to maintain steady-state batch size.
  --roctracer-max-events N  Set ROCTRACER_MAX_EVENTS (default 1M; use 10000000 for longer traces)
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
        --flush-timeout)    FLUSH_TIMEOUT="$2"; shift 2 ;;
        --layer)            LAYER="$2"; shift 2 ;;
        --ep)               EXPERT_PARALLEL="true"; shift 1 ;;
        --model-name)       MODEL_NAME="$2"; shift 2 ;;
        --quant)            QUANT="$2"; shift 2 ;;
        --env)              ENV="$2"; shift 2 ;;
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

# Profile prompts default to benchmark standard: CONC * 10
[[ -z "$PROFILE_NUM_PROMPTS" ]] && PROFILE_NUM_PROMPTS=$(( CONCURRENCY * 2 ))

# If MAX_MODEL_LEN not set, let ATOM use its default (no --max-model-len passed)

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
# Two complementary signals (in OR — first to fire wins):
#  (1) server log "Profiler stopped." count incremented
#      — most reliable; same pattern as ATOM CI's atom_test.sh
#  (2) .json.gz appears AND uncompressed .json is gone
#      — fallback; profiler writes .tmp -> .json -> .json.gz
#
# Caller must record PROFILER_STOPPED_COUNT_BEFORE before invoking.
wait_for_trace_flush() {
    local server_log="$1"            # path to server log file (optional, omit to skip log signal)
    local before_count="${2:-0}"     # baseline "Profiler stopped." count (optional)
    log "Waiting for trace flush (timeout ${FLUSH_TIMEOUT}s)..."
    local wait_elapsed=0

    while true; do
        # Signal 1: server log "Profiler stopped." count incremented
        if [[ -n "$server_log" && -f "$server_log" ]]; then
            local now_count
            now_count=$(grep -c "Profiler stopped\." "$server_log" 2>/dev/null || echo 0)
            if [[ "$now_count" -gt "$before_count" ]]; then
                log "  Trace flush signal: server log '$now_count > $before_count Profiler stopped.' (after ${wait_elapsed}s)"
                # Brief settle so .json.gz finishes appearing
                sleep 2
                local gz_file
                gz_file=$(ls "$TRACE_DIR"/rank_0/*.json.gz 2>/dev/null | grep -v capture_graph | head -1) || true
                [[ -n "$gz_file" ]] && log "  Trace file: $(basename "$gz_file") ($(du -h "$gz_file" | cut -f1))"
                return 0
            fi
        fi

        # Signal 2: file rename (.json -> .json.gz) — fallback if log path unavailable
        local gz_file json_file
        gz_file=$(ls "$TRACE_DIR"/rank_0/*.json.gz 2>/dev/null | grep -v capture_graph | head -1) || true
        json_file=$(ls "$TRACE_DIR"/rank_0/*.json 2>/dev/null | grep -v capture_graph | head -1) || true
        if [[ -n "$gz_file" && -z "$json_file" ]]; then
            log "  Trace flush complete (file rename): $(basename "$gz_file") ($(du -h "$gz_file" | cut -f1))"
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

print(f"Decode events: {len(decodes)}")
print(f"Prefill events: {len(prefills)}")

if not decodes:
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

# Save CSV if requested
if output_csv:
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bs", "count", "avg_ms", "min_ms", "max_ms", "p50_ms", "p99_ms"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved: {output_csv}")
PYEOF
}

# ======================== Main ================================================
SERVER_PID=""

trap_cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Trap: killing server PID=$SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap trap_cleanup EXIT INT TERM

EP_SIZE=1
[[ "$EXPERT_PARALLEL" == "true" ]] && EP_SIZE=$TP
TAG="trace_torch_mi355x_atom_${MODEL_NAME}_${QUANT}_${ENV}_${SCENARIO}_ep${EP_SIZE}_tp${TP}_c${CONCURRENCY}_full"

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
log "  ROCTracer Max: ${ROCTRACER_MAX_EVENTS:-default}"
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
EP_ARGS=()
if [[ "$EXPERT_PARALLEL" == "true" ]]; then
    EP_ARGS+=(--enable-expert-parallel)
fi
log "Starting ATOM server (TP=$TP, EP=$EXPERT_PARALLEL, profiler enabled)..."

MAX_MODEL_LEN_ARGS=()
if [[ -n "$MAX_MODEL_LEN" ]]; then
    MAX_MODEL_LEN_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

PYTORCH_PROFILER_WITH_STACK=0 \
python3 -m atom.entrypoints.openai_server \
    --model "$MODEL" \
    --server-port "$SERVER_PORT" \
    --tensor-parallel-size "$TP" \
    "${MAX_MODEL_LEN_ARGS[@]}" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv_cache_dtype "$KV_CACHE_DTYPE" \
    --torch-profiler-dir "$TRACE_DIR" \
    --mark-trace \
    "${EP_ARGS[@]}" \
    > "$RESULT_DIR/server_${TAG}.log" 2>&1 &
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
# Snapshot server log "Profiler stopped." count BEFORE start, so flush wait
# can detect new completions reliably (per ATOM CI atom_test.sh).
SERVER_LOG_PATH="$RESULT_DIR/server_${TAG}.log"
PROFILER_STOPPED_BEFORE=$(grep -c "Profiler stopped\." "$SERVER_LOG_PATH" 2>/dev/null || echo 0)
log "Profiler stopped count before run: $PROFILER_STOPPED_BEFORE"
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
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --result-dir "$RESULT_DIR" \
    --result-filename "$TRACE_RESULT_FILE" \
    --metadata \
        "max_model_len=$MAX_MODEL_LEN" \
        "gpu_memory_utilization=$GPU_MEM_UTIL" \
        "kv_cache_dtype=$KV_CACHE_DTYPE" \
        "tensor_parallel_size=$TP" \
        "expert_parallel=$EXPERT_PARALLEL" \
        "max_num_seqs=$MAX_NUM_SEQS" \
        "profile_num_prompts=$PROFILE_NUM_PROMPTS" \
        "warmup_num_prompts=$WARMUP_NUM_PROMPTS" >> "$SCRIPT_LOG" 2>&1 || {
    log "WARNING: Profiled benchmark failed"
}
log "Profiled benchmark done."

log "Stopping profiler..."
curl -s -X POST "http://0.0.0.0:${SERVER_PORT}/stop_profile" || true

# Step 5: Wait for trace flush (must complete before killing server).
# Pass server log + baseline count: log signal fires first, file rename is fallback.
wait_for_trace_flush "$SERVER_LOG_PATH" "$PROFILER_STOPPED_BEFORE"

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

    # Rename trace to include config info (TAG)
    TRACE_BASENAME=$(basename "$TRACE_FILE")
    TRACE_FILE="$RESULT_DIR/$TRACE_BASENAME"
    RENAMED_TRACE="$RESULT_DIR/${TAG}.json.gz"
    if [[ "$TRACE_FILE" != "$RENAMED_TRACE" ]]; then
        mv -v "$TRACE_FILE" "$RENAMED_TRACE"
        TRACE_FILE="$RENAMED_TRACE"
        # Also rename capture_graph if present
        for cg in "$RESULT_DIR"/capture_graph_*.json.gz; do
            [[ -f "$cg" ]] && mv -v "$cg" "$RESULT_DIR/capture_graph_${TAG}.json.gz" && break
        done
    fi

    log "============================================================"
    log "  DECODE WALL TIME ANALYSIS"
    log "============================================================"
    DECODE_CSV="$RESULT_DIR/decode_walltime_${TAG}.csv"
    parse_decode_walltime "$TRACE_FILE" "$DECODE_CSV"

    # Auto-trim: create a trimmed trace with only the steady-state decode window
    # This makes analysis 10-50x faster (550MB → 50-100MB)
    TRIM_FILE="${TRACE_FILE%.json.gz}_trim_decode.json.gz"
    log "Creating trimmed trace (middle 60% of decode window)..."
    python3 - "$TRACE_FILE" "$TRIM_FILE" << 'TRIM_PYEOF'
import gzip, json, sys

trace_path, trim_out = sys.argv[1], sys.argv[2]
print(f"Loading {trace_path} for trimming...")
with gzip.open(trace_path, 'rt') as f:
    data = json.load(f)

events = data.get('traceEvents', [])
print(f"Total events: {len(events)}")

# Find decode timestamps
decode_ts = [e['ts'] for e in events if e.get('ph') == 'X' and 'user_annotation' in e.get('cat', '') and (e.get('name', '').startswith('decode[') or e.get('name', '').startswith('decode '))]
decode_ts.sort()

if not decode_ts:
    print("No decode events found, skipping trim")
    sys.exit(0)

n = len(decode_ts)
start_idx = n * 2 // 10
end_idx = n * 8 // 10
trim_start = decode_ts[start_idx] - 1000
trim_end = decode_ts[end_idx] + 50000

print(f"Decode events: {n}, trim window: {trim_start/1e6:.1f}s - {trim_end/1e6:.1f}s ({end_idx-start_idx} events)")

trimmed = [e for e in events if trim_start <= e.get('ts', 0) <= trim_end]
del events, data

print(f"Trimmed events: {len(trimmed)}")
with gzip.open(trim_out, 'wt') as f:
    json.dump({"traceEvents": trimmed}, f)

import os
print(f"Trimmed trace: {trim_out} ({os.path.getsize(trim_out)/1e6:.0f}MB)")
TRIM_PYEOF
    if [[ -f "$TRIM_FILE" ]]; then
        log "Trimmed trace: $(du -h "$TRIM_FILE" | cut -f1)"
    fi

    # Run kernel breakdown analysis (selects steady-state bs for decode)
    RUN_PARSE="$SCRIPT_DIR/run_parse_trace.py"
    if [[ -f "$RUN_PARSE" ]]; then
        CAPTURE_ARG=""
        CAPTURE_FILE="$RESULT_DIR/capture_graph_${TAG}.json.gz"
        if [[ -f "$CAPTURE_FILE" ]]; then
            CAPTURE_ARG="--capture-trace $(basename "$CAPTURE_FILE")"
            log "Using explicit capture trace: $(basename "$CAPTURE_FILE")"
        fi
        log "Running run_parse_trace.py --layer $LAYER (target-bs=auto)..."
        (cd "$RESULT_DIR" && ATOM_TOOLS=/app/ATOM/tools python3 "$RUN_PARSE" "$(basename "$TRACE_FILE")" --layer "$LAYER" $CAPTURE_ARG 2>&1) || \
            log "WARNING: run_parse_trace.py failed"
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
log "  Trimmed:   ${TRIM_FILE:-N/A}"
log "  Decode CSV: $DECODE_CSV"
log "  Log:       $SCRIPT_LOG"
log "============================================================"

# Step 8: Generate summary.md
SUMMARY_FILE="$RESULT_DIR/summary.md"
cat > "$SUMMARY_FILE" << 'HEADER'
# DeepSeek R1 Profiling Results (ATOM/vLLM)
## MI355X 8×GPU

| Config | Scenario | CONC | Total Tput | Output Tput | TPOT (ms) | TTFT (ms) |
|--------|----------|------|------------|-------------|-----------|-----------|
HEADER

for f in "$RESULT_DIR"/result_profiled_*.json; do
    [[ -f "$f" ]] || continue
    python3 -c "
import json, sys, os
try:
    with open('$f') as fh:
        data = json.load(fh)
    fname = os.path.basename('$f').replace('.json', '')
    out_tps = data.get('output_throughput', 0)
    in_tps = data.get('input_throughput', 0)
    total_tps = data.get('total_token_throughput', in_tps + out_tps)
    ttft_p50 = data.get('ttft_p50', data.get('median_ttft_ms', 0))
    tpot_p50 = data.get('tpot_p50', data.get('median_tpot_ms', 0))
    print(f'| profiling | $SCENARIO | $CONCURRENCY | {total_tps:.1f} | {out_tps:.1f} | {tpot_p50:.1f} | {ttft_p50:.1f} |')
except Exception as e:
    print(f'| ERROR | $SCENARIO | $CONCURRENCY | - | - | - | {e} |', file=sys.stderr)
" >> "$SUMMARY_FILE" 2>/dev/null || true
done

log "Summary written to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
