#!/usr/bin/env bash
# NCU + trtllm-bench with time-based decode capture
#
# Step 1 (dry run): Run without NCU, measure timing
# Step 2 (profile): Run with NCU --profile-from-start off + auto-trigger
#
# Usage:
#   # Dry run to measure timing:
#   DRY_RUN=1 bash scripts/ncu_trtllm_timed.sh
#   # Profile with delay:
#   NCU_DELAY=120 NCU_DURATION=10 bash scripts/ncu_trtllm_timed.sh
set -euo pipefail

MODEL=/models/DeepSeek-R1-0528-NVFP4-v2
ISL=${ISL:-1024}
OSL=${OSL:-1024}
NUM=${NUM:-64}
TP=${TP:-4}
EP=${EP:-4}
BS=${BS:-64}
DRY_RUN=${DRY_RUN:-0}
NCU_DELAY=${NCU_DELAY:-120}
NCU_DURATION=${NCU_DURATION:-10}
LC=${LC:-2000}
NCU_SET=${NCU_SET:-pmsampling}
NCU_OUT=${NCU_OUT:-/tmp/ncu_trtllm_decode}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Config ==="
echo "  TP=$TP EP=$EP BS=$BS ISL=$ISL OSL=$OSL NUM=$NUM"
echo "  DRY_RUN=$DRY_RUN"
echo "  NCU_DELAY=$NCU_DELAY NCU_DURATION=$NCU_DURATION LC=$LC"

# Generate dataset
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

# Deploy auto-trigger as sitecustomize.py
SITE_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "Site packages: $SITE_DIR"

if [ "$DRY_RUN" = "1" ]; then
    echo ""
    echo "=== DRY RUN (no NCU) ==="
    echo "  Measuring timing to determine NCU_DELAY..."
    START_TIME=$(date +%s)
    trtllm-bench -m deepseek_r1 --model_path $MODEL \
        throughput \
        --dataset /tmp/dataset.jsonl \
        --backend pytorch \
        --tp $TP --ep $EP \
        --max_batch_size $BS \
        --num_requests $NUM \
        2>&1 | tail -30
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ""
    echo "=== Timing ==="
    echo "  Total duration: ${DURATION}s"
    echo "  Suggested NCU_DELAY: $((DURATION * 2 / 3))s (2/3 of total = mid-decode)"
    echo "  Suggested NCU_DURATION: 10s"
else
    echo ""
    echo "=== PROFILE RUN ==="
    echo "  Deploying ncu_auto_trigger.py as sitecustomize.py..."
    cp "$SCRIPT_DIR/ncu_auto_trigger.py" "$SITE_DIR/sitecustomize.py"
    echo "  Installed at $SITE_DIR/sitecustomize.py"

    # NCU options
    NCU_OPTS="--target-processes all --profile-from-start off --launch-count $LC -f -o $NCU_OUT"
    if [ "$NCU_SET" = "pmsampling" ]; then
        NCU_OPTS="$NCU_OPTS --section PmSampling --section PmSampling_WarpStates"
    else
        NCU_OPTS="$NCU_OPTS --set $NCU_SET"
    fi

    echo "  NCU_DELAY=${NCU_DELAY}s NCU_DURATION=${NCU_DURATION}s"
    echo "  ncu $NCU_OPTS trtllm-bench ..."

    NCU_TRIGGER=1 NCU_DELAY=$NCU_DELAY NCU_DURATION=$NCU_DURATION \
    ncu $NCU_OPTS \
        trtllm-bench -m deepseek_r1 --model_path $MODEL \
        throughput \
        --dataset /tmp/dataset.jsonl \
        --backend pytorch \
        --tp $TP --ep $EP \
        --max_batch_size $BS \
        --num_requests $NUM \
        2>&1 | tail -40

    # Cleanup sitecustomize.py
    rm -f "$SITE_DIR/sitecustomize.py"

    echo ""
    echo "=== Results ==="
    ls -lh ${NCU_OUT}*.ncu-rep 2>/dev/null || echo "NO .ncu-rep"

    if [ -f "${NCU_OUT}.ncu-rep" ]; then
        echo "=== Captured kernels ==="
        ncu -i ${NCU_OUT}.ncu-rep --csv --page details 2>/dev/null | tail -n +2 | wc -l
        echo "total kernels profiled"
    fi
fi
