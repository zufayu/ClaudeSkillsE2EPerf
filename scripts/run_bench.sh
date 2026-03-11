#!/bin/bash
# End-to-end benchmark for DeepSeek R1 on TRT-LLM
#
# Usage: bash run_bench.sh <model_path> [tp_size] [num_requests] [output_tokens]
# Example: bash run_bench.sh /models/deepseekr1/ 8 50 128

set -e

MODEL_PATH="${1:?Usage: bash run_bench.sh <model_path> [tp_size] [num_requests] [output_tokens]}"
TP_SIZE="${2:-8}"
NUM_REQUESTS="${3:-50}"
OUTPUT_TOKENS="${4:-128}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
KV_CACHE_FRAC="${KV_CACHE_FRAC:-0.85}"
WORKSPACE="${WORKSPACE:-/tmp/trtllm_bench}"
BACKEND="${BACKEND:-pytorch}"
INPUT_MODE="${INPUT_MODE:-synthetic}"
FIXED_INPUT_LEN="${FIXED_INPUT_LEN:-512}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_FILE="${WORKSPACE}/benchmark_dataset.json"
REPORT_FILE="${WORKSPACE}/benchmark_report.json"

mkdir -p "${WORKSPACE}"

echo "=============================================="
echo " DeepSeek R1 - TRT-LLM Benchmark"
echo "=============================================="
echo " Model:          ${MODEL_PATH}"
echo " Backend:        ${BACKEND}"
echo " TP Size:        ${TP_SIZE}"
echo " Requests:       ${NUM_REQUESTS}"
echo " Output Tokens:  ${OUTPUT_TOKENS}"
echo " Max Batch Size: ${MAX_BATCH_SIZE}"
echo " Max Seq Len:    ${MAX_SEQ_LEN}"
echo " Input Mode:     ${INPUT_MODE}"
echo " Workspace:      ${WORKSPACE}"
echo "=============================================="

# Step 1: Generate dataset
echo ""
echo "[Step 1/2] Generating benchmark dataset..."
python3 "${SCRIPT_DIR}/gen_dataset.py" \
    --tokenizer "${MODEL_PATH}" \
    --num_requests "${NUM_REQUESTS}" \
    --output_tokens "${OUTPUT_TOKENS}" \
    --output "${DATASET_FILE}" \
    --input_mode "${INPUT_MODE}" \
    --fixed_input_len "${FIXED_INPUT_LEN}"

# Step 2: Run throughput benchmark
echo ""
echo "[Step 2/2] Running throughput benchmark..."
trtllm-bench \
    -m deepseek-ai/DeepSeek-R1 \
    --model_path "${MODEL_PATH}" \
    -w "${WORKSPACE}" \
    throughput \
    --backend "${BACKEND}" \
    --tp "${TP_SIZE}" \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --kv_cache_free_gpu_mem_fraction "${KV_CACHE_FRAC}" \
    --dataset "${DATASET_FILE}" \
    --num_requests "${NUM_REQUESTS}" \
    --warmup 2 \
    --report_json "${REPORT_FILE}"

echo ""
echo "=============================================="
echo " Benchmark Complete"
echo "=============================================="
echo " Report saved to: ${REPORT_FILE}"

if [ -f "${REPORT_FILE}" ]; then
    echo ""
    echo "--- Report Summary ---"
    python3 -c "
import json
with open('${REPORT_FILE}') as f:
    r = json.load(f)
print(json.dumps(r, indent=2))
" 2>/dev/null || cat "${REPORT_FILE}"
fi
