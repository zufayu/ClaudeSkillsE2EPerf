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
# Empty = auto-derive from bench params (OSL × num_prompts / concurrency).
# Old hardcoded 1000 caused torch profiler buffer to swamp CUDA queue,
# triggering SGLang scheduler watchdog (timeout=300s) → SIGQUIT on large MoE
# (DeepSeek-R1 671B FP4). New default uses ~2% of total bench in steady-state
# middle window — enough decode-step samples for kernel breakdown, well
# within profiler buffer + watchdog safety margin.
PROFILE_STEPS=""
PROFILE_START_STEP=""
FLUSH_TIMEOUT=300
CONTAINER_IMAGE=""
MODEL_NAME="dsr1"
QUANT="fp4"
ENV=""
# Platform tag — used in TAG (filename) and passed to extract_cuda_kernels --platform.
# Valid values: b200, b300, h20, mi355x. Default kept as b200 for backward compat with
# pre-refactor workflows that don't pass --platform.
PLATFORM_TAG="b200"

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
  --profile-steps N     Forward steps to profile (default: auto = max(30, min(100, total/50))
                        where total = OSL * num_prompts / concurrency)
  --profile-start-step N Skip first N steps before profiling (default: auto = total/4,
                        enters steady-state region)
  --flush-timeout N     Max seconds to wait for trace flush (default: 300)
  --container-image IMG Container image name for metadata
  --platform NAME       Platform tag for output filename + analysis (default: b200;
                        valid: b200|b300|h20|mi355x). Affects kernel_registry mapping.
  -h, --help            Show this help

Profiling methodology:
  1. Start server with SGLANG_TORCH_PROFILER_DIR
  2. Warmup: CONC*2 prompts (reach steady state)
  3. HTTP POST /start_profile with num_steps + start_step
     (skip initial prefill ramp-up, capture steady-state decode)
  4. Benchmark: CONC*10 prompts (full load)
  5. Profiler auto-stops after num_steps, or /stop_profile as fallback
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
        --profile-steps)   PROFILE_STEPS="$2"; shift 2 ;;
        --profile-start-step) PROFILE_START_STEP="$2"; shift 2 ;;
        --flush-timeout)   FLUSH_TIMEOUT="$2"; shift 2 ;;
        --container-image) CONTAINER_IMAGE="$2"; shift 2 ;;
        --model-name)      MODEL_NAME="$2"; shift 2 ;;
        --quant)           QUANT="$2"; shift 2 ;;
        --env)             ENV="$2"; shift 2 ;;
        --platform)        PLATFORM_TAG="$2"; shift 2 ;;
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

# Auto-derive profile window from bench params (if not user-overridden).
# Each scheduler iteration produces 1 token per in-flight request, so:
#   total_steps ≈ OSL × num_prompts / concurrency
# Steady state lives in the middle ~50%; we sample 2% deep inside it.
ESTIMATED_TOTAL_STEPS=$(( OSL * PROFILE_NUM_PROMPTS / CONCURRENCY ))
PROFILE_PARAMS_AUTO=""
if [[ -z "$PROFILE_START_STEP" ]]; then
    PROFILE_START_STEP=$(( ESTIMATED_TOTAL_STEPS / 4 ))
    PROFILE_PARAMS_AUTO="${PROFILE_PARAMS_AUTO}start_step "
fi
if [[ -z "$PROFILE_STEPS" ]]; then
    PROFILE_STEPS=$(( ESTIMATED_TOTAL_STEPS / 50 ))
    # Floor 100: downstream analysis averages operator time over decode steps;
    # need ≥100 samples for stable per-operator means (user requirement).
    [[ $PROFILE_STEPS -lt 100 ]] && PROFILE_STEPS=100
    # Cap 100: empirically safe for watchdog (300s) and profiler buffer at c=4..256.
    # Larger windows risk SGLang scheduler stall on 671B MoE. Override --profile-steps
    # if you've validated higher works for your config.
    [[ $PROFILE_STEPS -gt 100 ]] && PROFILE_STEPS=100
    PROFILE_PARAMS_AUTO="${PROFILE_PARAMS_AUTO}num_steps"
fi

# Scheduler recv interval: SA uses 30 for conc>=16, 10 otherwise
SCHEDULER_RECV_INTERVAL=10
[[ $CONCURRENCY -ge 16 ]] && SCHEDULER_RECV_INTERVAL=30

# ======================== Logging =============================================

log() { echo "[$(date '+%H:%M:%S')] $*"; }

PROFILE_END_STEP=$((PROFILE_START_STEP + PROFILE_STEPS))
TAG="trace_torch_${PLATFORM_TAG}_sglang_${MODEL_NAME}_${QUANT}_${ENV}_${SCENARIO}_ep${EP}_tp${TP}_c${CONCURRENCY}_step${PROFILE_START_STEP}-${PROFILE_END_STEP}"

log "============================================================"
log "  SGLang Torch Profiler Trace Capture"
log "============================================================"
log "  Model:       $MODEL"
log "  Scenario:    $SCENARIO (ISL=$ISL, OSL=$OSL)"
log "  TP=$TP  EP=$EP  GPU_COUNT=$GPU_COUNT"
log "  Concurrency: $CONCURRENCY"
log "  Warmup:      $WARMUP_NUM_PROMPTS prompts"
log "  Profile:     $PROFILE_NUM_PROMPTS prompts (steps=$PROFILE_STEPS, start_step=$PROFILE_START_STEP)"
if [[ -n "$PROFILE_PARAMS_AUTO" ]]; then
    log "  Profile window auto-derived: $PROFILE_PARAMS_AUTO  (est total steps=$ESTIMATED_TOTAL_STEPS = OSL×N/CONC)"
fi
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
    safe_kill "sglang.launch_server"
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
    safe_kill "sglang.launch_server"
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

log "Starting profiler via HTTP (num_steps=$PROFILE_STEPS, start_step=$PROFILE_START_STEP)..."
log "  start_step=$PROFILE_START_STEP skips initial prefill ramp-up to capture steady-state decode"
PROFILE_RESP=$(curl -s -X POST "http://0.0.0.0:${PORT}/start_profile" -H "Content-Type: application/json" -d "{\"num_steps\": $PROFILE_STEPS, \"start_step\": $PROFILE_START_STEP}") || {
    log "ERROR: Failed to start profiler"; exit 1
}
log "Profiler started. Response: $PROFILE_RESP"

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

log "Stopping profiler via HTTP (fallback, should have auto-stopped after $PROFILE_STEPS steps)..."
curl -s -X POST "http://0.0.0.0:${PORT}/stop_profile" || true
log "Profiler stop sent."

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
safe_kill "sglang.launch_server"
sleep 3

# ======================== Step 7: Collect Traces ==============================

log "Trace files:"
find "$TRACE_DIR" -type f -name "*.json*" -ls 2>/dev/null | head -20

# Copy all trace files to result dir, rename with TAG for traceability
TRACE_COUNT=0
while IFS= read -r -d '' trace_file; do
    orig_name=$(basename "$trace_file")
    # Extract rank info (e.g. TP-0-EP-0) from original filename
    rank_info=$(echo "$orig_name" | grep -oP 'TP-\d+-EP-\d+' || echo "rank${TRACE_COUNT}")
    new_name="${TAG}_${rank_info}.trace.json.gz"
    cp -v "$trace_file" "$RESULT_DIR/$new_name"
    TRACE_COUNT=$((TRACE_COUNT + 1))
done < <(find "$TRACE_DIR" -type f \( -name "*.json.gz" -o -name "*.trace.json.gz" \) -print0 2>/dev/null)

if [[ $TRACE_COUNT -eq 0 ]]; then
    log "WARNING: No trace files found"
    # Check for uncompressed traces
    find "$TRACE_DIR" -type f -ls 2>/dev/null | head -20
else
    log "Copied $TRACE_COUNT trace file(s) to $RESULT_DIR/ (renamed with TAG)"
fi

# ======================== Step 7.5: Serialize Traces (fix multi-stream overlap) ==

SERIALIZE_SCRIPT="$SCRIPT_DIR/serialize_trace.py"
if [[ -f "$SERIALIZE_SCRIPT" ]] && [[ $TRACE_COUNT -gt 0 ]]; then
    log "Serializing traces to fix multi-stream overlap for Perfetto..."
    while IFS= read -r -d '' trace_file; do
        log "  Serializing $(basename "$trace_file")..."
        python3 "$SERIALIZE_SCRIPT" "$trace_file" -o "${trace_file%.json.gz}_serialized.json.gz" 2>&1 || log "WARNING: serialization failed for $(basename "$trace_file")"
    done < <(find "$RESULT_DIR" -maxdepth 1 -name "*.trace.json.gz" -print0 2>/dev/null)
    log "Trace serialization done. Serialized files (*_serialized.json.gz) are for Perfetto viewing."
fi

# ======================== Step 8: Kernel Breakdown ============================

EXTRACT_SCRIPT="$SCRIPT_DIR/extract_cuda_kernels_torch_trace.py"
FIRST_TRACE=$(find "$RESULT_DIR" -maxdepth 1 -name "*.trace.json.gz" ! -name "*_serialized*" -type f 2>/dev/null | head -1)

if [[ -f "$EXTRACT_SCRIPT" ]] && [[ -n "$FIRST_TRACE" ]]; then
    log "Running CUDA Graph kernel breakdown on $(basename "$FIRST_TRACE")..."
    BREAKDOWN_CSV="$RESULT_DIR/kernel_breakdown_${TAG}.csv"
    python3 "$EXTRACT_SCRIPT" "$FIRST_TRACE" --platform "$PLATFORM_TAG" --csv "$BREAKDOWN_CSV" --max-steps 0 --skip-first 5 --show-steps 0 2>&1 | tee "$RESULT_DIR/kernel_breakdown_${TAG}.log" || log "WARNING: kernel breakdown extraction failed"

    log "Running per-layer analysis (layers 10-40)..."
    PER_LAYER_CSV="$RESULT_DIR/per_layer_breakdown_${TAG}.csv"
    python3 "$EXTRACT_SCRIPT" "$FIRST_TRACE" --platform "$PLATFORM_TAG" --max-steps 0 --skip-first 5 --show-steps 0 --per-layer --layer-range 10-40 --per-layer-csv "$PER_LAYER_CSV" 2>&1 | tee "$RESULT_DIR/per_layer_breakdown_${TAG}.log" || log "WARNING: per-layer analysis failed"
else
    if [[ ! -f "$EXTRACT_SCRIPT" ]]; then
        log "WARNING: extract_cuda_kernels_torch_trace.py not found at $EXTRACT_SCRIPT"
    else
        log "WARNING: No trace .json.gz found for kernel breakdown"
    fi
fi

# ======================== Step 9: Summary =====================================

SUMMARY_FILE="$RESULT_DIR/summary.md"
cat > "$SUMMARY_FILE" << 'HEADER'
# DeepSeek R1 Profiling Results (SGLang)
## B200 Torch Profiler Trace

| Config | Scenario | CONC | Total Tput | Output Tput | TPOT (ms) | TTFT (ms) |
|--------|----------|------|------------|-------------|-----------|-----------|
HEADER

if [[ -f "$RESULT_DIR/${TRACE_RESULT_FILE}.json" ]]; then
    python3 -c "
import json, sys
with open('$RESULT_DIR/${TRACE_RESULT_FILE}.json') as f:
    data = json.load(f)
out_tps = data.get('output_throughput', 0)
in_tps = data.get('input_throughput', 0)
total_tps = data.get('total_token_throughput', in_tps + out_tps)
ttft_p50 = data.get('ttft_p50', data.get('median_ttft_ms', 0))
tpot_p50 = data.get('tpot_p50', data.get('median_tpot_ms', 0))
print(f'| profiling | $SCENARIO | $CONCURRENCY | {total_tps:.1f} | {out_tps:.1f} | {tpot_p50:.1f} | {ttft_p50:.1f} |')
" >> "$SUMMARY_FILE" 2>/dev/null || true
fi

# Append kernel breakdown summary if available
if [[ -f "$BREAKDOWN_CSV" ]]; then
    echo "" >> "$SUMMARY_FILE"
    echo "## Kernel Breakdown (per decode step, averaged)" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo '```' >> "$SUMMARY_FILE"
    head -20 "$RESULT_DIR/kernel_breakdown_${TAG}.log" 2>/dev/null | grep -E '^\s*[0-9]+\s*\||^#|^=|Per-Decode|Total per' >> "$SUMMARY_FILE" || true
    echo '```' >> "$SUMMARY_FILE"
fi

log "Summary written to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"

log "============================================================"
log "  TRACE CAPTURE COMPLETE"
log "============================================================"
log "  Traces:     $RESULT_DIR/"
log "  Breakdown:  ${BREAKDOWN_CSV:-N/A}"
log "  Summary:    $SUMMARY_FILE"
log "  Server log: $SERVER_LOG"
log "============================================================"

log "Done."
