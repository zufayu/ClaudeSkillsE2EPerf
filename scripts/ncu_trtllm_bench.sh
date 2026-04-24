#!/usr/bin/env bash
# NCU + trtllm-bench on B300
# Usage: docker exec zufa_trtllm bash /home/zufa/ClaudeSkillsE2EPerf/scripts/ncu_trtllm_bench.sh
set -euo pipefail

MODEL=/models/DeepSeek-R1-0528-NVFP4-v2
ISL=${ISL:-1024}
OSL=${OSL:-64}
NUM=${NUM:-64}
LC=${LC:-50}
LS=${LS:-0}
TP=${TP:-4}
EP=${EP:-4}
BS=${BS:-64}
NCU_SET=${NCU_SET:-pmsampling}
NCU_OUT=${NCU_OUT:-/tmp/ncu_trtllm}

echo "=== Config ==="
echo "  TP=$TP EP=$EP BS=$BS"
echo "  ISL=$ISL OSL=$OSL NUM=$NUM"
echo "  LC=$LC LS=$LS NCU_SET=$NCU_SET"

echo ""
echo "=== Generate dataset ==="
python3 -c "
import json
prompt = 'Explain the architecture of modern large language models in detail. ' * ($ISL // 10)
for i in range($NUM):
    d = {'task_id': i, 'prompt': prompt, 'output_tokens': $OSL}
    print(json.dumps(d))
" > /tmp/dataset.jsonl
echo "Dataset: $NUM requests, ISL~$ISL, OSL=$OSL"

# NCU options
NCU_OPTS="--target-processes all --launch-skip $LS --launch-count $LC -f -o $NCU_OUT"
if [ "$NCU_SET" = "pmsampling" ]; then
    NCU_OPTS="$NCU_OPTS --section PmSampling --section PmSampling_WarpStates"
else
    NCU_OPTS="$NCU_OPTS --set $NCU_SET"
fi

echo ""
echo "=== NCU + trtllm-bench throughput ==="
echo "  ncu $NCU_OPTS trtllm-bench ... --tp $TP --ep $EP --num_requests $NUM"
ncu $NCU_OPTS \
    trtllm-bench -m deepseek_r1 --model_path $MODEL \
    throughput \
    --dataset /tmp/dataset.jsonl \
    --backend pytorch \
    --tp $TP \
    --ep $EP \
    --max_batch_size $BS \
    --num_requests $NUM \
    2>&1 | tail -80

echo ""
echo "=== Results ==="
ls -lh ${NCU_OUT}*.ncu-rep 2>/dev/null || echo "NO .ncu-rep"
# Show captured kernel names (for probe runs)
if [ -f "${NCU_OUT}.ncu-rep" ]; then
    echo "=== Captured kernels ==="
    ncu -i ${NCU_OUT}.ncu-rep --csv --page raw 2>/dev/null | head -1
    ncu -i ${NCU_OUT}.ncu-rep --csv --page raw 2>/dev/null | tail -n +2 | cut -d'"' -f4 | head -20
fi
