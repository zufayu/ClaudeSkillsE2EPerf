#!/usr/bin/env bash
# NCU decode capture via --profile-from-start off + gdb cudaProfilerStart injection
#
# Flow:
#   1. Start NCU + trtllm-bench in background
#   2. Sleep NCU_DELAY (model loads + runs to decode mid-point)
#   3. gdb attaches to worker PIDs, calls cudaProfilerStart()
#   4. Sleep NCU_DURATION (NCU captures decode kernels)
#   5. gdb calls cudaProfilerStop()
#   6. Wait for NCU to write .ncu-rep
set -euo pipefail

MODEL=/models/DeepSeek-R1-0528-NVFP4-v2
ISL=${ISL:-1024}
OSL=${OSL:-1024}
NUM=${NUM:-64}
TP=${TP:-8}
BS=${BS:-64}
NCU_DELAY=${NCU_DELAY:-101}
NCU_DURATION=${NCU_DURATION:-1}
LC=${LC:-1000}
NCU_SET=${NCU_SET:-pmsampling}
NCU_OUT=${NCU_OUT:-/tmp/ncu_trtllm_decode}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Config ==="
echo "  TP=$TP BS=$BS ISL=$ISL OSL=$OSL NUM=$NUM"
echo "  NCU_DELAY=${NCU_DELAY}s NCU_DURATION=${NCU_DURATION}s LC=$LC"

# Generate dataset
python3 -c "
import json
prompt = 'Explain the architecture of modern large language models in detail. ' * ($ISL // 10)
for i in range($NUM):
    print(json.dumps({'task_id': i, 'prompt': prompt, 'output_tokens': $OSL}))
" > /tmp/dataset.jsonl
echo "Dataset: $NUM requests"

# NCU options
NCU_OPTS="--profile-from-start off --target-processes all --launch-count $LC -f -o $NCU_OUT"
if [ "$NCU_SET" = "pmsampling" ]; then
    NCU_OPTS="$NCU_OPTS --section PmSampling --section PmSampling_WarpStates"
else
    NCU_OPTS="$NCU_OPTS --set $NCU_SET"
fi

# Step 1: Start NCU + trtllm-bench in background
echo ""
echo "=== Step 1: Start NCU + trtllm-bench ==="
ncu $NCU_OPTS \
    trtllm-bench -m deepseek_r1 --model_path $MODEL \
    throughput \
    --dataset /tmp/dataset.jsonl \
    --backend pytorch \
    --tp $TP \
    --max_batch_size $BS \
    --num_requests $NUM \
    > /tmp/ncu_gdb.log 2>&1 &
NCU_PID=$!
echo "NCU PID=$NCU_PID"

# Step 2: Sleep until decode mid-point
echo ""
echo "=== Step 2: Waiting ${NCU_DELAY}s for decode mid-point ==="
sleep $NCU_DELAY

# Step 3: Find worker PIDs and inject cudaProfilerStart
echo ""
echo "=== Step 3: Inject cudaProfilerStart via gdb ==="
# Find worker PIDs: all descendant processes of NCU_PID that are on GPU
# First get all descendants of NCU
ALL_DESC=$(ps -eo pid,ppid --no-headers | python3 -c "
import sys
children = {}
for line in sys.stdin:
    parts = line.split()
    if len(parts) == 2:
        pid, ppid = int(parts[0]), int(parts[1])
        children.setdefault(ppid, []).append(pid)
# BFS from NCU_PID
queue = children.get($NCU_PID, [])
desc = []
while queue:
    pid = queue.pop(0)
    desc.append(pid)
    queue.extend(children.get(pid, []))
print(' '.join(str(p) for p in desc))
" 2>/dev/null || true)
echo "  All NCU descendants: $ALL_DESC"
# Filter to GPU-active ones
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " " | sort -u)
WORKER_PIDS=""
for pid in $ALL_DESC; do
    if echo "$GPU_PIDS" | grep -q "^${pid}$"; then
        WORKER_PIDS="$WORKER_PIDS $pid"
    fi
done
WORKER_PIDS=$(echo $WORKER_PIDS | tr ' ' '\n' | sort -u | tr '\n' ' ')
echo "Worker PIDs: $WORKER_PIDS"

for pid in $WORKER_PIDS; do
    echo "  Injecting cudaProfilerStart into PID=$pid..."
    gdb -batch -ex "call (int)cudaProfilerStart()" -p $pid 2>/dev/null &
done
wait
echo "  cudaProfilerStart injected"

# Step 4: Wait for capture duration
echo ""
echo "=== Step 4: Capturing for ${NCU_DURATION}s ==="
sleep $NCU_DURATION

# Step 5: Inject cudaProfilerStop
echo ""
echo "=== Step 5: Inject cudaProfilerStop via gdb ==="
for pid in $WORKER_PIDS; do
    echo "  Injecting cudaProfilerStop into PID=$pid..."
    gdb -batch -ex "call (int)cudaProfilerStop()" -p $pid 2>/dev/null &
done
wait
echo "  cudaProfilerStop injected"

# Step 6: Wait for NCU to finish
echo ""
echo "=== Step 6: Waiting for NCU to finish ==="
echo "  Waiting up to 300s..."
for i in $(seq 1 60); do
    if ! kill -0 $NCU_PID 2>/dev/null; then
        echo "  NCU exited after $((i*5))s"
        break
    fi
    sleep 5
done

if kill -0 $NCU_PID 2>/dev/null; then
    echo "  NCU still running, sending SIGTERM..."
    kill -TERM $NCU_PID 2>/dev/null || true
    sleep 10
fi

echo ""
echo "=== Results ==="
ls -lh ${NCU_OUT}*.ncu-rep 2>/dev/null || echo "NO .ncu-rep"
echo ""
echo "=== NCU log (last 20 lines) ==="
tail -20 /tmp/ncu_gdb.log 2>/dev/null
