#!/usr/bin/env bash
# =============================================================================
# Nsight Systems Trace Capture for SGLang Inference (B200)
#
# Captures GPU kernel timelines with per-layer NVTX markers for kernel-level
# performance analysis. Uses SGLang's --enable-layerwise-nvtx-marker to annotate
# each transformer layer, equivalent to TRT-LLM's TLLM_LLMAPI_ENABLE_NVTX.
#
# Approach:
#   1. Start SGLang server under nsys with delay (covers warmup) + duration
#   2. Warmup: CONC*2 prompts during nsys delay period
#   3. Benchmark: CONC*10 prompts during nsys capture window
#   4. nsys auto-stops after duration, server killed, trace collected
#
# Key difference from TRT-LLM nsys:
#   - TRT-LLM uses TLLM_PROFILE_START_STOP with -c cudaProfilerApi
#   - SGLang uses nsys --delay/--duration (no cudaProfilerApi support)
#   - SGLang provides --enable-layerwise-nvtx-marker for per-layer NVTX
#
# Usage:
#   bash scripts/collect_sglang_nsys_trace.sh \
#     --model /path/to/DeepSeek-R1-0528-NVFP4-v2 \
#     --tp 4 --ep 4 --scenario chat --concurrency 64 \
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
NSYS_DELAY=180
NSYS_DURATION=60
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
  --nsys-delay N        Seconds before nsys starts capturing (covers server startup + warmup) (default: 180)
  --nsys-duration N     Seconds of nsys capture (default: 60)
  --skip-export         Skip post-processing (just capture .nsys-rep)
  --container-image IMG Container image name for metadata
  -h, --help            Show this help

Profiling methodology:
  1. Start SGLang server under nsys with --delay and --duration
  2. Wait for server ready, run warmup (during nsys delay period)
  3. nsys capture begins automatically after delay
  4. Run benchmark during capture window
  5. nsys auto-stops, collect .nsys-rep trace
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
        --nsys-delay)      NSYS_DELAY="$2"; shift 2 ;;
        --nsys-duration)   NSYS_DURATION="$2"; shift 2 ;;
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

TAG="trace_nsys_b200_sglang_${MODEL_NAME}_${QUANT}_${ENV}_${SCENARIO}_ep${EP}_tp${TP}_c${CONCURRENCY}_delay${NSYS_DELAY}s-dur${NSYS_DURATION}s"

log "============================================================"
log "  SGLang Nsight Systems Trace Capture"
log "============================================================"
log "  Model:       $MODEL"
log "  Scenario:    $SCENARIO (ISL=$ISL, OSL=$OSL)"
log "  TP=$TP  EP=$EP  GPU_COUNT=$GPU_COUNT"
log "  Concurrency: $CONCURRENCY"
log "  Warmup:      $WARMUP_NUM_PROMPTS prompts"
log "  Benchmark:   $BENCH_NUM_PROMPTS prompts"
log "  nsys delay:  ${NSYS_DELAY}s (covers startup + warmup)"
log "  nsys duration: ${NSYS_DURATION}s (capture window)"
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
log "Starting SGLang server under nsys (delay=${NSYS_DELAY}s, duration=${NSYS_DURATION}s)..."
log "  --enable-layerwise-nvtx-marker for per-layer NVTX annotations"
log "  --cuda-graph-trace node to expand CUDA Graph kernels"

nsys profile -o "$RESULT_DIR/${TAG}" -f true -t 'cuda,nvtx' --delay "$NSYS_DELAY" --duration "$NSYS_DURATION" --cuda-graph-trace node --sample=none --cpuctxsw=none --trace-fork-before-exec=true \
    env PYTHONNOUSERSITE=1 \
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
log "Server PID=$SERVER_PID (nsys-wrapped)"

# Wait for server ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID" --max-wait 1200

# ======================== Step 3: Warmup ======================================

log "Running warmup ($WARMUP_NUM_PROMPTS prompts, concurrency $CONCURRENCY)..."
log "  This runs during nsys delay period (no tracing yet)"

WARMUP_START=$(date +%s)

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

WARMUP_END=$(date +%s)
WARMUP_ELAPSED=$((WARMUP_END - WARMUP_START))
log "Warmup done in ${WARMUP_ELAPSED}s."

# Check if we need to wait for nsys delay to expire
ELAPSED_SINCE_START=$((WARMUP_END - $(date -d "$(ps -p $SERVER_PID -o lstart= 2>/dev/null || echo 'now')" +%s 2>/dev/null || echo $WARMUP_END)))
log "NOTE: nsys delay=${NSYS_DELAY}s. If warmup finishes early, benchmark starts before nsys capture."
log "  This is OK - nsys will capture the tail end of the benchmark (steady state)."

# ======================== Step 4: Benchmark ===================================

log "Running benchmark ($BENCH_NUM_PROMPTS prompts, concurrency $CONCURRENCY)..."
log "  nsys capture will overlap with this benchmark phase"

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
    --metadata "sglang_version=$sglang_version" "container_image=$CONTAINER_IMAGE" "tp=$TP" "ep=$EP" "scenario=$SCENARIO" "profiler=nsys" "nsys_delay=$NSYS_DELAY" "nsys_duration=$NSYS_DURATION"

log "Benchmark done."

# ======================== Step 5: Wait for nsys to finish =====================

log "Waiting for nsys to finish (duration=${NSYS_DURATION}s from start of capture)..."
# nsys will auto-stop after duration seconds of capture
# The server process will continue running; we wait for the .nsys-rep to appear

WAIT_START=$(date +%s)
NSYS_TIMEOUT=$((NSYS_DELAY + NSYS_DURATION + 60))
while true; do
    if [[ -f "$RESULT_DIR/${TAG}.nsys-rep" ]]; then
        log "nsys trace file created."
        break
    fi
    WAITED=$(( $(date +%s) - WAIT_START ))
    if [[ $WAITED -ge $NSYS_TIMEOUT ]]; then
        log "WARNING: nsys trace not found after ${NSYS_TIMEOUT}s"
        break
    fi
    if [[ $((WAITED % 30)) -eq 0 ]] && [[ $WAITED -gt 0 ]]; then
        log "  Waiting for nsys... (${WAITED}s)"
    fi
    sleep 5
done

# ======================== Step 6: Stop Server =================================

log "Stopping server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
pkill -f "sglang.launch_server" 2>/dev/null || true
sleep 5

# Wait a bit more for nsys to finalize
sleep 5

# ======================== Step 7: Post-Processing =============================

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

    # Export to SQLite
    log "Exporting to SQLite..."
    nsys export --type sqlite -o "$RESULT_DIR/${TAG}.sqlite" "$TRACE_FILE" 2>&1 || log "WARN: SQLite export failed"

    # Export kernel CSV
    log "Exporting kernel trace CSV..."
    nsys stats --report cuda_gpu_trace --format csv -o "$RESULT_DIR/${TAG}_kernels" "$TRACE_FILE" 2>&1 || log "WARN: Kernel CSV export failed"

    # Print top kernels
    if [[ -f "$RESULT_DIR/${TAG}.sqlite" ]]; then
        log "=== Top 10 Kernels by Total Duration ==="
        sqlite3 -header -column "$RESULT_DIR/${TAG}.sqlite" "SELECT shortName, COUNT(*) as count, SUM(end-start) as total_ns, AVG(end-start) as avg_ns, ROUND(CAST(SUM(end-start) AS FLOAT) / 1e6, 2) as total_ms FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY shortName ORDER BY total_ns DESC LIMIT 10;" 2>/dev/null || log "WARN: Could not query kernel summary"

        log "=== NVTX Layer Markers ==="
        sqlite3 -header -column "$RESULT_DIR/${TAG}.sqlite" "SELECT text, COUNT(*) as count, ROUND(AVG(end-start)/1e6, 3) as avg_ms, ROUND(SUM(end-start)/1e6, 1) as total_ms FROM NVTX_EVENTS WHERE text LIKE '%layer%' OR text LIKE '%Layer%' GROUP BY text ORDER BY total_ms DESC LIMIT 20;" 2>/dev/null || log "WARN: Could not query NVTX events"
    fi
else
    log "Skipping post-processing (--skip-export)"
fi

# ======================== Step 8: Summary =====================================

log "============================================================"
log "  NSYS TRACE CAPTURE COMPLETE"
log "============================================================"
log "  Trace:      $TRACE_FILE (${TRACE_SIZE_MB} MB)"
log "  Server log: $SERVER_LOG"
log "  Result dir: $RESULT_DIR/"
log "============================================================"

# Show profiled benchmark result
if [[ -f "$RESULT_DIR/result_profiled_${TAG}.json" ]]; then
    python3 -c "
import json
with open('$RESULT_DIR/result_profiled_${TAG}.json') as f:
    d = json.load(f)
out_tps = d.get('output_throughput', 0)
tpot = d.get('tpot_p50', d.get('median_tpot_ms', 0))
ttft = d.get('ttft_p50', d.get('median_ttft_ms', 0))
print(f'Profiled benchmark: Output Tput={out_tps:.1f}, TPOT p50={tpot:.1f}ms, TTFT p50={ttft:.1f}ms')
" || true
fi

log "Done."
