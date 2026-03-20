#!/usr/bin/env bash
# =============================================================================
# DAR (Draft Acceptance Rate) Measurement for MTP on B200
#
# Measures speculative decoding acceptance rate per scenario using
# trtllm-bench throughput. Runs separately from sa_bench_b200.sh because
# trtllm-bench loads the model directly (no server).
#
# Usage:
#   bash scripts/bench_dar_b200.sh \
#     --model-fp8 /home/models/models--DeepSeek-R1-0528 \
#     --result-dir ./results_b200_fp8_mtp3
#
#   # Run only one scenario:
#   bash scripts/bench_dar_b200.sh \
#     --model-fp8 /home/models/models--DeepSeek-R1-0528 \
#     --scenario chat
#
# Output:
#   dar_dataset_chat.jsonl            - generated dataset
#   dar_fp8_latency_chat.json         - trtllm-bench report with DAR stats
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/benchmark_lib.sh"

# ======================== Argument Parsing ====================================
MODEL_FP8=""
CONFIGS="fp8-latency"
RESULT_DIR="./results_b200"
TP=8
EP=1
CONCURRENCY=32
NUM_REQUESTS=200
MTP_LAYERS=3
SCENARIO_FILTER="all"
WARMUP=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-fp8)      MODEL_FP8="$2"; shift 2 ;;
        --configs)        CONFIGS="$2"; shift 2 ;;
        --result-dir)     RESULT_DIR="$2"; shift 2 ;;
        --tp)             TP="$2"; shift 2 ;;
        --ep)             EP="$2"; shift 2 ;;
        --concurrency)    CONCURRENCY="$2"; shift 2 ;;
        --num-requests)   NUM_REQUESTS="$2"; shift 2 ;;
        --mtp-layers)     MTP_LAYERS="$2"; shift 2 ;;
        --scenario)       SCENARIO_FILTER="$2"; shift 2 ;;
        --warmup)         WARMUP="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash bench_dar_b200.sh --model-fp8 <path> [options]"
            echo ""
            echo "Options:"
            echo "  --model-fp8 PATH       FP8 model path (required)"
            echo "  --configs CONFIG       Config tag for output filenames (default: fp8-latency)"
            echo "  --result-dir DIR       Results directory (default: ./results_b200)"
            echo "  --tp N                 Tensor parallelism (default: 8)"
            echo "  --ep N                 Expert parallelism (default: 1)"
            echo "  --concurrency N        Concurrency for trtllm-bench (default: 32)"
            echo "  --num-requests N       Number of requests (default: 200)"
            echo "  --mtp-layers N         MTP speculative layers (default: 3)"
            echo "  --scenario FILTER      chat|reasoning|summarize|all (default: all)"
            echo "  --warmup N             Warmup requests (default: 5)"
            exit 0
            ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_FP8" ]]; then
    echo "ERROR: --model-fp8 is required"
    exit 1
fi

mkdir -p "$RESULT_DIR"

# ======================== Scenarios ===========================================
declare -a ALL_SCENARIOS=(
    "1024:1024:chat"
    "1024:8192:reasoning"
    "8192:1024:summarize"
)

declare -a SCENARIOS=()
for s in "${ALL_SCENARIOS[@]}"; do
    IFS=':' read -r _isl _osl _tag <<< "$s"
    if [[ "$SCENARIO_FILTER" == "all" || "$SCENARIO_FILTER" == "$_tag" ]]; then
        SCENARIOS+=("$s")
    fi
done
if [[ ${#SCENARIOS[@]} -eq 0 ]]; then
    echo "ERROR: --scenario '$SCENARIO_FILTER' matched nothing. Use: chat|reasoning|summarize|all"
    exit 1
fi

# ======================== Utility =============================================
TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(TS)] $*"; }

# ======================== Main ================================================

# Parse config tag for quant/config_name
IFS='-' read -r QUANT CONFIG_NAME <<< "$CONFIGS"

trap 'kill_server 2>/dev/null; exit' INT TERM

log "============================================================"
log "  DAR (Draft Acceptance Rate) Measurement"
log "============================================================"
log "  Model FP8:    $MODEL_FP8"
log "  Config:       $CONFIGS"
log "  Result Dir:   $RESULT_DIR"
log "  TP=$TP  EP=$EP  MTP=$MTP_LAYERS"
log "  Concurrency:  $CONCURRENCY"
log "  Num Requests: $NUM_REQUESTS"
log "  Scenarios:    ${SCENARIOS[*]}"
log "============================================================"
echo ""

for scenario_entry in "${SCENARIOS[@]}"; do
    IFS=':' read -r ISL OSL SCENARIO_TAG <<< "$scenario_entry"

    log "========================================================"
    log " DAR: $SCENARIO_TAG (ISL=$ISL, OSL=$OSL)"
    log "========================================================"

    # --- Step 1: Generate dataset ---
    DATASET_FILE="$RESULT_DIR/dar_dataset_${SCENARIO_TAG}.jsonl"
    log "  Generating dataset: $DATASET_FILE"

    python3 "$SCRIPT_DIR/gen_dataset.py" \
        --tokenizer "$MODEL_FP8" \
        --fixed_input_len "$ISL" \
        --output_tokens "$OSL" \
        --num_requests "$NUM_REQUESTS" \
        --input_mode fixed_len \
        --output "$DATASET_FILE"

    # --- Step 2: Generate MTP config YAML ---
    local_max_model_len=10240
    if [[ $((ISL + OSL)) -le 2048 ]]; then
        local_max_model_len=8192
    fi

    CONFIG_FILE="$RESULT_DIR/dar_config_${SCENARIO_TAG}.yml"
    cat > "$CONFIG_FILE" << EOF
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: $MTP_LAYERS
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.8
    enable_block_reuse: false
moe_config:
    backend: TRTLLM
cuda_graph_config:
    enable_padding: true
    max_batch_size: $CONCURRENCY
print_iter_log: true
EOF

    # --- Piecewise CUDA Graphs (significant perf improvement) ---
    local max_num_tokens
    if [[ $MTP_LAYERS -gt 0 ]]; then
        max_num_tokens=$(( ((MTP_LAYERS + 1) * CONCURRENCY + ISL + 64 + 63) / 64 * 64 ))
    else
        max_num_tokens=$(( (CONCURRENCY + ISL + 64 + 63) / 64 * 64 ))
    fi
    [[ $max_num_tokens -lt 8192 ]] && max_num_tokens=8192

    local capture_tokens=(1 2 4 8 16 32 64 128)
    capture_tokens+=( $(seq 256 256 $max_num_tokens) )
    if [[ $((max_num_tokens % 256)) -ne 0 ]]; then
        capture_tokens+=($max_num_tokens)
    fi
    local capture_list
    capture_list=$(printf "%s, " "${capture_tokens[@]}")

    cat >> "$CONFIG_FILE" << EOF
torch_compile_config:
    capture_num_tokens: [${capture_list%, }]
    enable_piecewise_cuda_graph: true
EOF

    # --- Step 3: Run trtllm-bench ---
    REPORT_FILE="$RESULT_DIR/dar_${QUANT}_${CONFIG_NAME}_${SCENARIO_TAG}.json"
    log "  Running trtllm-bench: concurrency=$CONCURRENCY, requests=$NUM_REQUESTS"

    # Kill any lingering server processes to free GPU memory
    kill_server

    set -x
    trtllm-bench --model "$MODEL_FP8" \
        --model_path "$MODEL_FP8" \
        --backend pytorch \
        --extra_llm_api_options "$CONFIG_FILE" \
        --max_seq_len "$local_max_model_len" \
        --tp "$TP" \
        --ep "$EP" \
        throughput \
        --dataset "$DATASET_FILE" \
        --concurrency "$CONCURRENCY" \
        --num_requests "$NUM_REQUESTS" \
        --report_json "$REPORT_FILE" \
        --warmup "$WARMUP"
    local_rc=$?
    set +x

    # --- Step 4: Log DAR from report ---
    if [[ $local_rc -eq 0 && -f "$REPORT_FILE" ]]; then
        log "  DAR report: $REPORT_FILE"
        python3 -c "
import json, sys
with open('$REPORT_FILE') as f:
    data = json.load(f)
ds = data.get('decoding_stats', {})
dar = ds.get('draft_acceptance_rate_percentiles', {})
acc_len = ds.get('acceptance_length_percentiles', {})
if dar:
    print(f'  DAR:  avg={dar.get(\"avg\", \"N/A\")}, p50={dar.get(\"p50\", \"N/A\")}, p90={dar.get(\"p90\", \"N/A\")}, p99={dar.get(\"p99\", \"N/A\")}')
if acc_len:
    print(f'  AccLen: avg={acc_len.get(\"avg\", \"N/A\")}, p50={acc_len.get(\"p50\", \"N/A\")}')
if not dar and not acc_len:
    print('  WARN: No decoding_stats found in report', file=sys.stderr)
"
    else
        log "  WARN: trtllm-bench failed or report missing for $SCENARIO_TAG"
    fi

    echo ""
done

log "============================================================"
log "  DAR MEASUREMENT COMPLETE"
log "  Results in: $RESULT_DIR/dar_*.json"
log "============================================================"
