#!/usr/bin/env bash
# =============================================================================
# Torch Profiler Trace Capture for SGLang Inference (B200)
#
# Mirrors collect_atom_trace.sh methodology for cross-platform comparison:
#   1. Cleanup residual processes
#   2. Start SGLang server with SGLANG_TORCH_PROFILER_DIR
#   3. Warmup with CONC*2 prompts (reach steady state)
#   4. Start profile via HTTP, run full benchmark (CONC*10), stop profile
#   5. Wait for trace flush
#   6. Stop server, copy traces
#
# SGLang profiling uses the same /start_profile /stop_profile HTTP endpoints
# as ATOM/vLLM. The key difference from SA InferenceX's PROFILE=1 mode:
#   - SA reduces num_prompts to CONC (small trace, not steady state)
#   - We keep num_prompts = CONC*10 (matches benchmark, captures steady state)
#
# Usage:
#   bash scripts/collect_sglang_trace.sh \
#     --model /path/to/DeepSeek-R1-0528-NVFP4-v2 \
#     --tp 4 --ep 4 --scenario chat --concurrency 64 \
#     --result-dir ./results/sglang_profiling
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
TRACE_DIR="/tmp/sglang_trace"
PROFILE_NUM_PROMPTS=""
FLUSH_TIMEOUT=300
CONTAINER_IMAGE=""

# SA InferenceX server parameters (from dsr1_fp4_b200.sh)
MEM_FRACTION_STATIC=0.85
CHUNKED_PREFILL_SIZE=16384
CUDA_GRAPH_MAX_BS=256
MAX_RUNNING_REQUESTS=256
STREAM_INTERVAL=10

# ======================== CLI Parsing =========================================

usage() {
    cat <<EOF
Usage: $0 --model MODEL_PATH --result-dir DIR [options]

Torch Profiler trace capture for SGLang on B200, matching collect_atom_trace.sh
methodology for cross-platform comparison.

Required:
  --model PATH          Path to DeepSeek-R1 FP4 model
  --result-dir DIR      Output directory for traces and results

Options:
  --tp N                Tensor parallel size (default: 4)
  --ep N                Expert parallel size (default: 4)
  --scenario NAME       chat|reasoning|summarize (default: chat)
  --concurrency N       Max concurrency (default: 64)
  --port N              Server port (default: 8888)
  --profile-prompts N   Prompts during profiling (default: CONC*10, full load)
  --flush-timeout N     Max seconds to wait for trace flush (default: 300)
  --container-image IMG Container image name for metadata
  -h, --help            Show this help

Profiling methodology (aligned with collect_atom_trace.sh):
  1. Start server with SGLANG_TORCH_PROFILER_DIR
  2. Warmup: CONC*2 prompts (steady state)
  3. HTTP POST /start_profile
  4. Benchmark: CONC*10 prompts (full load, same as SA benchmark)
  5. HTTP POST /stop_profile
  6. Wait for trace flush, collect traces
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
        --profile-prompts) PROFILE_NUM_PROMPTS="$2"; shift 2 ;;
        --flush-timeout)   FLUSH_TIMEOUT="$2"; shift 2 ;;
        --container-image) CONTAINER_IMAGE="$2"; shift 2 ;;
        -h|--help)         usage ;;
        *)                 echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$MODEL" ]] && { echo "ERROR: --model is required"; usage; }
[[ -z "$RESULT_DIR" ]] && { echo "ERROR: --result-dir is required"; usage; }

# ======================== Scenario → ISL/OSL ==================================

case "$SCENARIO" in
    chat)       ISL=1024; OSL=1024 ;;
    reasoning)  ISL=1024; OSL=8192 ;;
    summarize)  ISL=8192; OSL=1024 ;;
    *)          echo "ERROR: Unknown scenario '$SCENARIO'"; exit 1 ;;
esac

WARMUP_NUM_PROMPTS=$((CONCURRENCY * 2))
[[ -z "$PROFILE_NUM_PROMPTS" ]] && PROFILE_NUM_PROMPTS=$((CONCURRENCY * 10))
GPU_COUNT=$((TP > EP ? TP : EP))

# Scheduler recv interval: SA uses 30 for conc>=16, 10 otherwise
SCHEDULER_RECV_INTERVAL=10
[[ $CONCURRENCY -ge 16 ]] && SCHEDULER_RECV_INTERVAL=30

# ======================== Logging =============================================

log() { echo "[$(date '+%H:%M:%S')] $*"; }

TAG="trace_${SCENARIO}_tp${TP}_ep${EP}_c${CONCURRENCY}_p${PROFILE_NUM_PROMPTS}"

log "============================================================"
log "  SGLang Torch Profiler Trace Capture"
log "============================================================"
log "  Model:       $MODEL"
log "  Scenario:    $SCENARIO (ISL=$ISL, OSL=$OSL)"
log "  TP=$TP  EP=$EP  GPU_COUNT=$GPU_COUNT"
log "  Concurrency: $CONCURRENCY"
log "  Warmup:      $WARMUP_NUM_PROMPTS prompts"
log "  Profile:     $PROFILE_NUM_PROMPTS prompts"
log "  Trace Dir:   $TRACE_DIR"
log "  Result Dir:  $RESULT_DIR"
log "  Tag:         $TAG"
log "============================================================"

mkdir -p "$RESULT_DIR"

# ======================== Version Capture =====================================

sglang_version=$(python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "unknown")
log "SGLang version: $sglang_version"

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
    rm -rf "$TRACE_DIR"
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

# ======================== Step 2: Start Server ================================

SERVER_LOG="$RESULT_DIR/server_${TAG}.log"
log "Starting SGLang server with profiler enabled..."

SGLANG_TORCH_PROFILER_DIR="$TRACE_DIR" \
SGLANG_PROFILE_WITH_STACK=False \
PYTHONNOUSERSITE=1 \
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
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
log "Server PID=$SERVER_PID"

# Wait for server ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID" --max-wait 1200

# ======================== Step 3: Warmup ======================================

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

# ======================== Step 4: Profile =====================================

log "Starting profiler via HTTP..."
curl -s -X POST "http://0.0.0.0:${PORT}/start_profile" || {
    log "ERROR: Failed to start profiler"; exit 1
}
log "Profiler started."

log "Running profiled benchmark ($PROFILE_NUM_PROMPTS prompts, concurrency $CONCURRENCY)..."

TRACE_RESULT_FILE="result_profiled_${TAG}"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio 0.8 \
    --num-prompts "$PROFILE_NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --num-warmups 0 \
    --result-filename "$TRACE_RESULT_FILE" \
    --result-dir "$RESULT_DIR" \
    --metadata "sglang_version=$sglang_version" "container_image=$CONTAINER_IMAGE" "tp=$TP" "ep=$EP" "scenario=$SCENARIO" "profile_num_prompts=$PROFILE_NUM_PROMPTS"

log "Profiled benchmark done."

log "Stopping profiler via HTTP..."
curl -s -X POST "http://0.0.0.0:${PORT}/stop_profile" || true
log "Profiler stopped."

# ======================== Step 5: Wait for Trace Flush ========================

log "Waiting for trace flush (timeout ${FLUSH_TIMEOUT}s)..."
wait_elapsed=0

while true; do
    # SGLang writes traces to SGLANG_TORCH_PROFILER_DIR
    # Look for .json.gz files (compressed traces)
    gz_files=$(find "$TRACE_DIR" -name "*.json.gz" -o -name "*.trace.json.gz" 2>/dev/null | head -5) || true
    json_files=$(find "$TRACE_DIR" -name "*.json" ! -name "*.json.gz" 2>/dev/null | head -5) || true

    # Done when .json.gz exists AND no uncompressed .json remains
    if [[ -n "$gz_files" ]] && [[ -z "$json_files" ]]; then
        log "Trace flush complete."
        break
    fi

    # Also check for profiles/ subdirectory (SGLang v0.5.9 puts traces there)
    gz_profiles=$(find "$TRACE_DIR" -path "*/profiles/*" -name "*.json.gz" 2>/dev/null | head -5) || true
    if [[ -n "$gz_profiles" ]]; then
        json_profiles=$(find "$TRACE_DIR" -path "*/profiles/*" -name "*.json" ! -name "*.json.gz" 2>/dev/null | head -5) || true
        if [[ -z "$json_profiles" ]]; then
            log "Trace flush complete (profiles/ dir)."
            break
        fi
    fi

    sleep 5
    wait_elapsed=$((wait_elapsed + 5))

    if [[ $wait_elapsed -ge $FLUSH_TIMEOUT ]]; then
        log "WARNING: Trace flush timeout after ${FLUSH_TIMEOUT}s"
        log "Files in trace dir:"
        find "$TRACE_DIR" -type f -ls 2>/dev/null | head -20
        break
    fi

    if [[ $((wait_elapsed % 30)) -eq 0 ]]; then
        log "  Still waiting... (${wait_elapsed}s elapsed)"
        find "$TRACE_DIR" -type f -name "*.json*" 2>/dev/null | head -10
    fi
done

# ======================== Step 6: Stop Server =================================

log "Stopping server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
pkill -f "sglang.launch_server" 2>/dev/null || true
sleep 3

# ======================== Step 7: Collect Traces ==============================

log "Trace files:"
find "$TRACE_DIR" -type f -name "*.json*" -ls 2>/dev/null | head -20

# Copy all trace files to result dir
TRACE_COUNT=0
while IFS= read -r -d '' trace_file; do
    cp -v "$trace_file" "$RESULT_DIR/"
    TRACE_COUNT=$((TRACE_COUNT + 1))
done < <(find "$TRACE_DIR" -type f \( -name "*.json.gz" -o -name "*.trace.json.gz" \) -print0 2>/dev/null)

if [[ $TRACE_COUNT -eq 0 ]]; then
    log "WARNING: No trace files found"
    # Check for uncompressed traces
    find "$TRACE_DIR" -type f -ls 2>/dev/null | head -20
else
    log "Copied $TRACE_COUNT trace file(s) to $RESULT_DIR/"
fi

# ======================== Step 8: Summary =====================================

log "============================================================"
log "  TRACE CAPTURE COMPLETE"
log "============================================================"
log "  Traces:     $RESULT_DIR/"
log "  Server log: $SERVER_LOG"
log "============================================================"

# Show profiled benchmark result
if [[ -f "$RESULT_DIR/${TRACE_RESULT_FILE}.json" ]]; then
    python3 -c "
import json
with open('$RESULT_DIR/${TRACE_RESULT_FILE}.json') as f:
    d = json.load(f)
out_tps = d.get('output_throughput', 0)
tpot = d.get('tpot_p50', d.get('median_tpot_ms', 0))
ttft = d.get('ttft_p50', d.get('median_ttft_ms', 0))
print(f'Profiled benchmark: Output Tput={out_tps:.1f}, TPOT p50={tpot:.1f}ms, TTFT p50={ttft:.1f}ms')
" || true
fi

log "Done."
