#!/bin/bash
# Serve DeepSeek R1 with TRT-LLM PyTorch backend (OpenAI-compatible API)
#
# Usage: bash serve.sh <model_path> [tp_size] [port]
# Example: bash serve.sh /models/deepseekr1/ 8 8000

set -e

MODEL_PATH="${1:?Usage: bash serve.sh <model_path> [tp_size] [port]}"
TP_SIZE="${2:-8}"
PORT="${3:-8000}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
KV_CACHE_FRAC="${KV_CACHE_FRAC:-0.85}"

echo "=============================================="
echo " DeepSeek R1 - TRT-LLM Serving"
echo "=============================================="
echo " Model:          ${MODEL_PATH}"
echo " TP Size:        ${TP_SIZE}"
echo " Port:           ${PORT}"
echo " Max Batch Size: ${MAX_BATCH_SIZE}"
echo " Max Seq Len:    ${MAX_SEQ_LEN}"
echo " KV Cache Frac:  ${KV_CACHE_FRAC}"
echo "=============================================="

trtllm-serve serve \
    "${MODEL_PATH}" \
    --backend pytorch \
    --tp_size "${TP_SIZE}" \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --kv_cache_free_gpu_memory_fraction "${KV_CACHE_FRAC}" \
    --trust_remote_code \
    --reasoning_parser deepseek-r1 \
    --host 0.0.0.0 \
    --port "${PORT}"
