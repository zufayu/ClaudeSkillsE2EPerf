#!/usr/bin/env bash
# =============================================================================
# SGLang Benchmark Script for DeepSeek-R1 FP4 on B200
#
# Reproduces SA InferenceX dsr1_fp4_b200.sh configuration.
# Server params taken from SemiAnalysisAI/InferenceX benchmarks/single_node/dsr1_fp4_b200.sh
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
RESULT_DIR="./results_sglang"
CONTAINER_IMAGE=""

# SA InferenceX server parameters (from dsr1_fp4_b200.sh)
MEM_FRACTION_STATIC=0.85
CHUNKED_PREFILL_SIZE=16384
CUDA_GRAPH_MAX_BS=256
MAX_RUNNING_REQUESTS=256
STREAM_INTERVAL=10

# ======================== CLI Parsing =========================================

usage() {
    echo "Usage: $0 --model MODEL_PATH [options]"
    echo ""
    echo "Required:"
    echo "  --model PATH          Path to DeepSeek-R1 FP4 model"
    echo ""
    echo "Options:"
    echo "  --tp N                Tensor parallel size (default: 4)"
    echo "  --ep N                Expert parallel size (default: 4)"
    echo "  --scenario NAME       chat|reasoning|summarize (default: chat)"
    echo "  --concurrency N       Max concurrency (default: 64)"
    echo "  --result-dir DIR      Result output directory (default: ./results_sglang)"
    echo "  --container-image IMG Container image name for metadata"
    echo "  --port N              Server port (default: 8888)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)           MODEL="$2"; shift 2 ;;
        --tp)              TP="$2"; shift 2 ;;
        --ep)              EP="$2"; shift 2 ;;
        --scenario)        SCENARIO="$2"; shift 2 ;;
        --concurrency)     CONCURRENCY="$2"; shift 2 ;;
        --result-dir)      RESULT_DIR="$2"; shift 2 ;;
        --container-image) CONTAINER_IMAGE="$2"; shift 2 ;;
        --port)            PORT="$2"; shift 2 ;;
        -h|--help)         usage ;;
        *)                 echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$MODEL" ]] && { echo "ERROR: --model is required"; usage; }

# ======================== Scenario → ISL/OSL ==================================

case "$SCENARIO" in
    chat)       ISL=1024; OSL=1024 ;;
    reasoning)  ISL=1024; OSL=8192 ;;
    summarize)  ISL=8192; OSL=1024 ;;
    *)          echo "ERROR: Unknown scenario '$SCENARIO'"; exit 1 ;;
esac

NUM_PROMPTS=$((CONCURRENCY * 10))
NUM_WARMUPS=$((CONCURRENCY * 2))
GPU_COUNT=$((TP > EP ? TP : EP))

# ======================== Logging =============================================

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "========================================================"
log " SGLang Benchmark (SA InferenceX config)"
log "========================================================"
log "Model:      $MODEL"
log "TP=$TP  EP=$EP  GPU_COUNT=$GPU_COUNT"
log "Scenario:   $SCENARIO (ISL=$ISL, OSL=$OSL)"
log "Concurrency: $CONCURRENCY  Prompts: $NUM_PROMPTS  Warmups: $NUM_WARMUPS"
log "Result dir: $RESULT_DIR"
log "Container:  ${CONTAINER_IMAGE:-unknown}"
log "========================================================"

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

# ======================== Start Server ========================================

SERVER_LOG="$RESULT_DIR/server_sglang_${SCENARIO}_tp${TP}_ep${EP}_c${CONCURRENCY}.log"

# Scheduler recv interval: SA uses 30 for conc>=16, 10 otherwise
SCHEDULER_RECV_INTERVAL=10
[[ $CONCURRENCY -ge 16 ]] && SCHEDULER_RECV_INTERVAL=30

log "Starting SGLang server (TP=$TP, EP=$EP, port=$PORT)..."

PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" --trust-remote-code --tensor-parallel-size="$TP" --data-parallel-size=1 --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" --max-running-requests "$MAX_RUNNING_REQUESTS" --mem-fraction-static "$MEM_FRACTION_STATIC" --kv-cache-dtype fp8_e4m3 --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --ep-size "$EP" --quantization modelopt_fp4 --enable-flashinfer-allreduce-fusion --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL" --enable-symm-mem --disable-radix-cache --attention-backend trtllm_mla --moe-runner-backend flashinfer_trtllm --stream-interval "$STREAM_INTERVAL" > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
log "Server PID=$SERVER_PID"

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID" --max-wait 1200

# ======================== Run Benchmark =======================================

RESULT_TAG="fp4_throughput_${SCENARIO}_tp${TP}_ep${EP}_c${CONCURRENCY}"

log "Running benchmark: $RESULT_TAG"

start_gpu_monitor --output "$RESULT_DIR/gpu_sglang_${RESULT_TAG}.csv"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio 1.0 \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --num-warmups "$NUM_WARMUPS" \
    --result-filename "result_${RESULT_TAG}" \
    --result-dir "$RESULT_DIR" \
    --metadata "sglang_version=$sglang_version" "container_image=$CONTAINER_IMAGE" "tp=$TP" "ep=$EP" "scenario=$SCENARIO"

stop_gpu_monitor

# ======================== Kill Server =========================================

log "Stopping server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
pkill -f "sglang.launch_server" 2>/dev/null || true
sleep 3

# ======================== Summary Report ======================================

generate_summary() {
    local summary_file="$RESULT_DIR/summary.md"

    cat > "$summary_file" <<EOF
# DeepSeek R1 Benchmark Results (SGLang)
## B200 ${GPU_COUNT}×GPU

| Config | Quant | Scenario | TP | EP | CONC | Total Tput | Output Tput | Interac. | TPOT (ms) | TTFT (ms) |
|--------|-------|----------|----|----|------|------------|-------------|----------|-----------|-----------|
EOF

    for f in "$RESULT_DIR"/result_*.json; do
        [[ -f "$f" ]] || continue
        python3 -c "
import json, sys, os
try:
    with open('$f') as fh:
        data = json.load(fh)
    fname = os.path.basename('$f').replace('result_', '').replace('.json', '')
    parts = fname.split('_')
    quant = parts[0] if len(parts) > 0 else '-'
    config = parts[1] if len(parts) > 1 else '-'
    scenario = parts[2] if len(parts) > 2 else '-'
    tp_val = [p for p in parts if p.startswith('tp') and p[2:].isdigit()]
    tp = tp_val[0].replace('tp','') if tp_val else '-'
    ep_val = [p for p in parts if p.startswith('ep') and p[2:].isdigit()]
    ep = ep_val[0].replace('ep','') if ep_val else '-'
    conc_val = [p for p in parts if p.startswith('c') and p[1:].isdigit()]
    conc = conc_val[0].replace('c','') if conc_val else '-'
    out_tps = data.get('output_throughput', 0)
    total_tps = data.get('total_token_throughput', 0)
    ttft_p50 = data.get('ttft_p50', data.get('median_ttft_ms', 0))
    tpot_p50 = data.get('tpot_p50', data.get('median_tpot_ms', 0))
    interactivity = 1000.0 / tpot_p50 if tpot_p50 > 0 else 0
    print(f'| {config} | {quant} | {scenario} | {tp} | {ep} | {conc} | {total_tps:.1f} | {out_tps:.1f} | {interactivity:.2f} | {tpot_p50:.1f} | {ttft_p50:.0f} |')
except Exception as e:
    print(f'| ERROR | - | $f | - | - | - | - | - | - | - | {e} |', file=sys.stderr)
" >> "$summary_file" 2>/dev/null || true
    done

    log "Summary written to: $summary_file"
}

generate_summary

log "========================================================"
log " BENCHMARK COMPLETE"
log "========================================================"
cat "$RESULT_DIR/summary.md"
