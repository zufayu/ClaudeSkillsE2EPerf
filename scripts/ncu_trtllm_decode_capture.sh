#!/usr/bin/env bash
# NCU decode capture: --devices 0 + --profile-from-start off + gdb trigger
# Usage: docker exec zufa_trtllm bash /path/to/ncu_trtllm_decode_capture.sh
set -euo pipefail

MODEL=/models/DeepSeek-R1-0528-NVFP4-v2
ISL=${ISL:-1024}
OSL=${OSL:-1024}
NUM=${NUM:-64}
TP=${TP:-8}
BS=${BS:-64}
DELAY=${DELAY:-101}
DURATION=${DURATION:-1}
LC=${LC:-500}
NCU_OUT=${NCU_OUT:-/tmp/ncu_decode_dev0}

echo "=== NCU Decode Capture ==="
echo "  TP=$TP BS=$BS ISL=$ISL OSL=$OSL NUM=$NUM"
echo "  DELAY=${DELAY}s DURATION=${DURATION}s LC=$LC"

# Generate dataset
python3 -c "
import json
prompt = 'Explain the architecture of modern large language models in detail. ' * ($ISL // 10)
for i in range($NUM):
    print(json.dumps({'task_id': i, 'prompt': prompt, 'output_tokens': $OSL}))
" > /tmp/ds_decode.jsonl
echo "Dataset: $NUM requests"

# Background: gdb trigger after DELAY seconds
(
    sleep $DELAY
    echo "[trigger] Injecting cudaProfilerStart after ${DELAY}s..."
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | tr -d " "); do
        gdb -batch -ex "call (int)cudaProfilerStart()" -p $pid 2>/dev/null &
    done
    wait
    echo "[trigger] cudaProfilerStart done"
    sleep $DURATION
    echo "[trigger] Injecting cudaProfilerStop after ${DURATION}s..."
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | tr -d " "); do
        gdb -batch -ex "call (int)cudaProfilerStop()" -p $pid 2>/dev/null &
    done
    wait
    echo "[trigger] cudaProfilerStop done"
) &
TRIGGER_PID=$!

# Foreground: NCU + trtllm-bench
echo ""
echo "=== Starting NCU + trtllm-bench ==="
ncu --devices 0 --profile-from-start off --target-processes all \
    --launch-count $LC \
    --section PmSampling --section PmSampling_WarpStates \
    -f -o $NCU_OUT \
    trtllm-bench -m deepseek_r1 --model_path $MODEL \
    throughput \
    --dataset /tmp/ds_decode.jsonl \
    --backend pytorch \
    --tp $TP \
    --max_batch_size $BS \
    --num_requests $NUM \
    2>&1 | tail -30

# Wait for trigger to finish
wait $TRIGGER_PID 2>/dev/null || true

echo ""
echo "=== Results ==="
ls -lh ${NCU_OUT}*.ncu-rep 2>/dev/null || echo "NO .ncu-rep"
