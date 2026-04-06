#!/usr/bin/env bash
# =============================================================================
# Nsight Systems Trace Capture for SGLang Inference (B200)
#
# Captures GPU kernel timelines with per-layer NVTX markers for kernel-level
# performance analysis. Uses SGLang's --enable-layerwise-nvtx-marker to annotate
# each transformer layer, equivalent to TRT-LLM's TLLM_LLMAPI_ENABLE_NVTX.
#
# Approach (same as TRT-LLM collect_nsys_trace.sh serve mode):
#   1. Start SGLang server under nsys (background, always-on capture)
#   2. Wait for server ready
#   3. Warmup: CONC*2 prompts
#   4. Benchmark: CONC*10 prompts (this is the steady-state we analyze)
#   5. Kill server → nsys exits and writes .nsys-rep
#   6. Post-process: if trace >1GB, trim to benchmark window; export SQLite + jsonlines
#
# The trace captures everything (startup + warmup + benchmark). For large
# traces (>1GB), we trim exports to the benchmark window using recorded
# timestamps. Exports include SQLite (for kernel analysis) and jsonlines
# (Perfetto-compatible visualization). --cuda-graph-trace node expands
# CUDA Graph internals.
#
# Usage:
#   bash scripts/collect_sglang_nsys_trace.sh \
#     --model /path/to/DeepSeek-R1-0528-NVFP4-v2 \
#     --tp 4 --ep 1 --scenario chat --concurrency 64 \
#     --result-dir ./results/sglang_nsys_profiling
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/benchmark_lib.sh"

# ======================== Defaults ============================================

MODEL=""
TP=4
EP=4
PORT=8888
SCENARIO="chat"
CONCURRENCY=64
RESULT_DIR=""
SKIP_EXPORT=false
CONTAINER_IMAGE=""
MODEL_NAME="dsr1"
QUANT="fp4"
ENV=""

# SA InferenceX server parameters
MEM_FRACTION_STATIC=0.85
CHUNKED_PREFILL_SIZE=16384
CUDA_GRAPH_MAX_BS=256
MAX_RUNNING_REQUESTS=256
STREAM_INTERVAL=10

# ======================== CLI Parsing =========================================

usage() {
    cat <<EOF
Usage: $0 --model MODEL_PATH --result-dir DIR [options]

Nsight Systems trace capture for SGLang on B200 with per-layer NVTX markers.

Required:
  --model PATH          Path to DeepSeek-R1 FP4 model
  --result-dir DIR      Output directory for traces and results

Options:
  --tp N                Tensor parallel size (default: 4)
  --ep N                Expert parallel size (default: 4)
  --scenario NAME       chat|reasoning|summarize (default: chat)
  --concurrency N       Max concurrency (default: 64)
  --port N              Server port (default: 8888)
  --skip-export         Skip post-processing (just capture .nsys-rep)
  --container-image IMG Container image name for metadata
  --model-name NAME     Model short name for trace tag (default: dsr1)
  --quant QUANT         Quantization for trace tag (default: fp4)
  --env ENV             Container env for trace tag (e.g. sglang059)
  -h, --help            Show this help

Profiling methodology (same as TRT-LLM serve mode):
  1. Start SGLang server under nsys (always-on capture)
  2. Wait for server ready
  3. Warmup: CONC*2 prompts
  4. Benchmark: CONC*10 prompts (steady state)
  5. Kill server -> nsys writes .nsys-rep
  6. Post-process: trim if >1GB, export SQLite + jsonlines (.json.gz)
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)           MODEL="$2"; shift 2 ;;
        --tp)              TP="$2"; shift 2 ;;
        --ep)              EP="$2"; shift 2 ;;
        --scenario)        SCENARIO="$2"; shift 2 ;;
        --concurrency)     CONCURRENCY="$2"; shift 2 ;;
        --result-dir)      RESULT_DIR="$2"; shift 2 ;;
        --port)            PORT="$2"; shift 2 ;;
        --skip-export)     SKIP_EXPORT=true; shift ;;
        --container-image) CONTAINER_IMAGE="$2"; shift 2 ;;
        --model-name)      MODEL_NAME="$2"; shift 2 ;;
        --quant)           QUANT="$2"; shift 2 ;;
        --env)             ENV="$2"; shift 2 ;;
        -h|--help)         usage ;;
        *)                 echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$MODEL" ]] && { echo "ERROR: --model is required"; usage; }
[[ -z "$RESULT_DIR" ]] && { echo "ERROR: --result-dir is required"; usage; }

# ======================== Scenario -> ISL/OSL ==================================

case "$SCENARIO" in
    chat)       ISL=1024; OSL=1024 ;;
    reasoning)  ISL=1024; OSL=8192 ;;
    summarize)  ISL=8192; OSL=1024 ;;
    *)          echo "ERROR: Unknown scenario '$SCENARIO'"; exit 1 ;;
esac

WARMUP_NUM_PROMPTS=$((CONCURRENCY * 2))
BENCH_NUM_PROMPTS=$((CONCURRENCY * 10))
GPU_COUNT=$((TP > EP ? TP : EP))

SCHEDULER_RECV_INTERVAL=10
[[ $CONCURRENCY -ge 16 ]] && SCHEDULER_RECV_INTERVAL=30

# ======================== Logging =============================================

log() { echo "[$(date '+%H:%M:%S')] $*"; }

TAG="trace_nsys_b200_sglang_${MODEL_NAME}_${QUANT}_${ENV}_${SCENARIO}_ep${EP}_tp${TP}_c${CONCURRENCY}"

log "============================================================"
log "  SGLang Nsight Systems Trace Capture"
log "============================================================"
log "  Model:       $MODEL"
log "  Scenario:    $SCENARIO (ISL=$ISL, OSL=$OSL)"
log "  TP=$TP  EP=$EP  GPU_COUNT=$GPU_COUNT"
log "  Concurrency: $CONCURRENCY"
log "  Warmup:      $WARMUP_NUM_PROMPTS prompts"
log "  Benchmark:   $BENCH_NUM_PROMPTS prompts"
log "  Result Dir:  $RESULT_DIR"
log "  Tag:         $TAG"
log "============================================================"

mkdir -p "$RESULT_DIR"

# ======================== Version Capture =====================================

sglang_version=$(python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "unknown")
log "SGLang version: $sglang_version"

nsys_version=$(nsys --version 2>/dev/null | head -1 || echo "unknown")
log "nsys version: $nsys_version"

# ======================== GPU Visibility ======================================

if [[ $GPU_COUNT -lt 8 ]]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
    export CUDA_VISIBLE_DEVICES
    log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# ======================== Cleanup =============================================

cleanup() {
    log "Cleaning up..."
    fuser -k -9 "${PORT}/tcp" 2>/dev/null || true
    sleep 2
    pkill -f "sglang.launch_server" 2>/dev/null || true
    sleep 2
    log "Cleanup done."
}

# ======================== Trap ================================================

SERVER_PID=""
trap_cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Trap: killing server PID=$SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    pkill -f "sglang.launch_server" 2>/dev/null || true
}
trap trap_cleanup EXIT INT TERM

# ======================== Step 1: Cleanup =====================================

cleanup

# ======================== Step 2: Start Server under nsys =====================

SERVER_LOG="$RESULT_DIR/server_${TAG}.log"
log "Starting SGLang server under nsys (always-on capture)..."
log "  --enable-layerwise-nvtx-marker for per-layer NVTX annotations"
log "  --cuda-graph-trace node to expand CUDA Graph kernels"

PYTHONNOUSERSITE=1 \
nsys profile -o "$RESULT_DIR/${TAG}" -f true -t 'cuda,nvtx' --cuda-graph-trace node --sample=none --cpuctxsw=none --trace-fork-before-exec=true \
    python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size="$TP" \
    --data-parallel-size=1 \
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --ep-size "$EP" \
    --quantization modelopt_fp4 \
    --enable-flashinfer-allreduce-fusion \
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL" \
    --enable-symm-mem \
    --disable-radix-cache \
    --attention-backend trtllm_mla \
    --moe-runner-backend flashinfer_trtllm \
    --stream-interval "$STREAM_INTERVAL" \
    --enable-layerwise-nvtx-marker \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
NSYS_START_EPOCH=$(date +%s)
log "Server PID=$SERVER_PID (nsys-wrapped)"

# Wait for server ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID" --max-wait 1200

# ======================== Step 3: Warmup ======================================

WARMUP_START_EPOCH=$(date +%s)
log "Running warmup ($WARMUP_NUM_PROMPTS prompts, concurrency $CONCURRENCY)..."

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio 0.8 \
    --num-prompts "$WARMUP_NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --num-warmups 0 \
    --result-filename "result_warmup_${TAG}" \
    --result-dir "$RESULT_DIR"

log "Warmup done."

# ======================== Step 4: Benchmark ===================================

BENCH_START_EPOCH=$(date +%s)
log "Running benchmark ($BENCH_NUM_PROMPTS prompts, concurrency $CONCURRENCY)..."

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio 0.8 \
    --num-prompts "$BENCH_NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --num-warmups 0 \
    --result-filename "result_profiled_${TAG}" \
    --result-dir "$RESULT_DIR" \
    --metadata "sglang_version=$sglang_version" "container_image=$CONTAINER_IMAGE" "tp=$TP" "ep=$EP" "scenario=$SCENARIO" "profiler=nsys"

BENCH_END_EPOCH=$(date +%s)
log "Benchmark done."
log "Timestamps: warmup_start=${WARMUP_START_EPOCH}, bench_start=${BENCH_START_EPOCH}, bench_end=${BENCH_END_EPOCH}"

# ======================== Step 5: Kill server -> nsys writes trace =============

log "Stopping server (triggers nsys trace flush)..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
pkill -f "sglang.launch_server" 2>/dev/null || true
# Give nsys time to finalize the trace
sleep 10

# ======================== Step 6: Post-Processing =============================

TRACE_FILE="$RESULT_DIR/${TAG}.nsys-rep"

if [[ ! -f "$TRACE_FILE" ]]; then
    log "ERROR: No .nsys-rep trace file found at $TRACE_FILE"
    ls -la "$RESULT_DIR/" 2>/dev/null
    exit 1
fi

TRACE_SIZE_MB=$(du -m "$TRACE_FILE" | cut -f1)
log "Trace file: $TRACE_FILE (${TRACE_SIZE_MB} MB)"

if [[ "$SKIP_EXPORT" == "false" ]]; then
    log "=== POST-PROCESSING ==="

    TRIM_NEEDED=false
    if [[ $TRACE_SIZE_MB -gt 1024 ]]; then
        TRIM_NEEDED=true
        # Calculate trim window: benchmark phase relative to nsys start, with 5s buffer
        TRIM_START=$(( BENCH_START_EPOCH - NSYS_START_EPOCH - 5 ))
        TRIM_END=$(( BENCH_END_EPOCH - NSYS_START_EPOCH + 5 ))
        [[ $TRIM_START -lt 0 ]] && TRIM_START=0
        TRIM_TAG="trim_${TRIM_START}-${TRIM_END}"
        SQLITE_OUT="$RESULT_DIR/${TAG}_${TRIM_TAG}.sqlite"
        JSON_OUT="$RESULT_DIR/${TAG}_${TRIM_TAG}.json.gz"
        log "Trace >1GB (${TRACE_SIZE_MB}MB), trimming to benchmark window: ${TRIM_START}s-${TRIM_END}s"

        # Export trimmed SQLite
        log "Exporting trimmed SQLite: $(basename "$SQLITE_OUT")"
        nsys export --type sqlite --times="${TRIM_START}s/${TRIM_END}s" -f true -o "$SQLITE_OUT" "$TRACE_FILE" 2>&1 || log "WARN: Trimmed SQLite export failed"

        # Export trimmed jsonlines compressed (Perfetto-compatible)
        log "Exporting trimmed jsonlines: $(basename "$JSON_OUT")"
        nsys export --type jsonlines --times="${TRIM_START}s/${TRIM_END}s" -f true -o "$JSON_OUT" "$TRACE_FILE" 2>&1 || log "WARN: Trimmed jsonlines export failed"

        SQLITE_FILE="$SQLITE_OUT"
    else
        log "Trace ≤1GB (${TRACE_SIZE_MB}MB), exporting full trace"

        # Export full SQLite
        SQLITE_FILE="$RESULT_DIR/${TAG}.sqlite"
        log "Exporting to SQLite..."
        nsys export --type sqlite -f true -o "$SQLITE_FILE" "$TRACE_FILE" 2>&1 || log "WARN: SQLite export failed"

        # Export full jsonlines compressed
        JSON_OUT="$RESULT_DIR/${TAG}.json.gz"
        log "Exporting jsonlines: $(basename "$JSON_OUT")"
        nsys export --type jsonlines -f true -o "$JSON_OUT" "$TRACE_FILE" 2>&1 || log "WARN: Jsonlines export failed"
    fi

    # Print top kernels from whichever SQLite we have
    if [[ -f "$SQLITE_FILE" ]]; then
        log "SQLite: $(ls -lh "$SQLITE_FILE" | awk '{print $5}')"

        log "=== Top 10 Kernels by Total Duration ==="
        python3 -c "
import sqlite3
conn = sqlite3.connect('$SQLITE_FILE')
cur = conn.cursor()
rows = cur.execute('''SELECT shortName, COUNT(*) as count, SUM(end-start) as total_ns, AVG(end-start) as avg_ns FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY shortName ORDER BY total_ns DESC LIMIT 10''').fetchall()
print(f'{\"Kernel\":<60} {\"Count\":>8} {\"Total(ms)\":>12} {\"Avg(us)\":>12}')
print('-'*94)
for name, cnt, total_ns, avg_ns in rows:
    print(f'{name[:60]:<60} {cnt:>8} {total_ns/1e6:>12.2f} {avg_ns/1e3:>12.2f}')
conn.close()
" 2>/dev/null || log "WARN: Could not query kernel summary"

        log "=== NVTX Layer Markers ==="
        python3 -c "
import sqlite3
conn = sqlite3.connect('$SQLITE_FILE')
rows = conn.execute('''SELECT s.value, COUNT(*) as count, AVG(n.end-n.start)/1e6 as avg_ms, SUM(n.end-n.start)/1e6 as total_ms FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id WHERE s.value LIKE '%layer%' OR s.value LIKE '%Layer%' GROUP BY s.value ORDER BY total_ms DESC LIMIT 20''').fetchall()
if rows:
    print(f'{\"Marker\":<60} {\"Count\":>8} {\"Avg(ms)\":>10} {\"Total(ms)\":>12}')
    print('-'*92)
    for name, cnt, avg_ms, total_ms in rows:
        print(f'{name[:60]:<60} {cnt:>8} {avg_ms:>10.3f} {total_ms:>12.1f}')
else:
    print('No layer markers found')
conn.close()
" 2>/dev/null || log "WARN: Could not query NVTX events"
    fi

    # Show file sizes
    log "=== Export files ==="
    ls -lh "$RESULT_DIR/${TAG}"*.sqlite "$RESULT_DIR/${TAG}"*.json* 2>/dev/null || true
else
    log "Skipping post-processing (--skip-export)"
fi

# ======================== Step 7: Summary =====================================

SUMMARY_FILE="$RESULT_DIR/summary.md"
cat > "$SUMMARY_FILE" << 'HEADER'
# DeepSeek R1 Profiling Results (SGLang)
## B200 Nsys Trace

| Config | Scenario | CONC | Total Tput | Output Tput | TPOT (ms) | TTFT (ms) |
|--------|----------|------|------------|-------------|-----------|-----------|
HEADER

if [[ -f "$RESULT_DIR/result_profiled_${TAG}.json" ]]; then
    python3 -c "
import json, sys
with open('$RESULT_DIR/result_profiled_${TAG}.json') as f:
    data = json.load(f)
out_tps = data.get('output_throughput', 0)
in_tps = data.get('input_throughput', 0)
total_tps = data.get('total_token_throughput', in_tps + out_tps)
ttft_p50 = data.get('ttft_p50', data.get('median_ttft_ms', 0))
tpot_p50 = data.get('tpot_p50', data.get('median_tpot_ms', 0))
print(f'| profiling | $SCENARIO | $CONCURRENCY | {total_tps:.1f} | {out_tps:.1f} | {tpot_p50:.1f} | {ttft_p50:.1f} |')
" >> "$SUMMARY_FILE" 2>/dev/null || true
fi

# Append top kernels if SQLite export was done
if [[ -n "${SQLITE_FILE:-}" ]] && [[ -f "$SQLITE_FILE" ]]; then
    python3 -c "
import sqlite3
conn = sqlite3.connect('$SQLITE_FILE')
cur = conn.cursor()

# Top kernels
rows = cur.execute('''SELECT shortName, COUNT(*) as count, SUM(end-start) as total_ns, AVG(end-start) as avg_ns FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY shortName ORDER BY total_ns DESC LIMIT 10''').fetchall()
if rows:
    print()
    print('## Top 10 Kernels by Total Duration')
    print()
    print('\`\`\`')
    print(f'{\"Kernel\":<60} {\"Count\":>8} {\"Total(ms)\":>12} {\"Avg(us)\":>12}')
    print('-'*94)
    for name, cnt, total_ns, avg_ns in rows:
        print(f'{name[:60]:<60} {cnt:>8} {total_ns/1e6:>12.2f} {avg_ns/1e3:>12.2f}')
    print('\`\`\`')

# NVTX markers
rows = conn.execute('''SELECT s.value, COUNT(*) as count, AVG(n.end-n.start)/1e6 as avg_ms, SUM(n.end-n.start)/1e6 as total_ms FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id WHERE s.value LIKE '%layer%' OR s.value LIKE '%Layer%' GROUP BY s.value ORDER BY total_ms DESC LIMIT 20''').fetchall()
if rows:
    print()
    print('## NVTX Layer Markers')
    print()
    print('\`\`\`')
    print(f'{\"Marker\":<60} {\"Count\":>8} {\"Avg(ms)\":>10} {\"Total(ms)\":>12}')
    print('-'*92)
    for name, cnt, avg_ms, total_ms in rows:
        print(f'{name[:60]:<60} {cnt:>8} {avg_ms:>10.3f} {total_ms:>12.1f}')
    print('\`\`\`')
conn.close()
" >> "$SUMMARY_FILE" 2>/dev/null || true
fi

log "Summary written to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"

log "============================================================"
log "  NSYS TRACE CAPTURE COMPLETE"
log "============================================================"
log "  Trace:      $TRACE_FILE (${TRACE_SIZE_MB} MB)"
if [[ "${TRIM_NEEDED:-false}" == "true" ]]; then
    log "  Trimmed:    ${TRIM_START}s-${TRIM_END}s (benchmark window)"
    log "  SQLite:     $(basename "${SQLITE_FILE:-}")"
    log "  Jsonlines:  $(basename "${JSON_OUT:-}")"
fi
log "  Summary:    $SUMMARY_FILE"
log "  Server log: $SERVER_LOG"
log "  Result dir: $RESULT_DIR/"
log "============================================================"

log "Done."
