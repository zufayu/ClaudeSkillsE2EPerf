#!/usr/bin/env bash
# =============================================================================
# SemiAnalysis InferenceX-style Benchmark Suite for DeepSeek R1 on 8×H20
#
# Adapted from: https://github.com/SemiAnalysisAI/InferenceX
# Configs ported from H200 scripts, adjusted for H20 (96GB/GPU).
#
# Usage:
#   bash sa_bench_h20.sh --model /path/to/DeepSeek-R1 [--configs all] [--port 8888]
#
# Configs available:
#   trt-throughput    TRT-LLM PyTorch, max throughput (no MTP)
#   trt-latency       TRT-LLM PyTorch, min latency (MTP-3)
#   trt-balanced      TRT-LLM PyTorch, balanced (MTP-1 + attention DP)
#   all               Run all configs sequentially
#
# Prerequisites:
#   - 8× NVIDIA H20 GPUs
#   - TensorRT-LLM installed with PyTorch backend
#   - DeepSeek R1 FP8 model weights downloaded
# =============================================================================

set -euo pipefail

# ======================== Argument Parsing ====================================
MODEL=""
CONFIGS="all"
PORT=8888
RESULT_DIR="./results_sa_bench"
WARMUP_REQUESTS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)        MODEL="$2"; shift 2 ;;
        --configs)      CONFIGS="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --result-dir)   RESULT_DIR="$2"; shift 2 ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required"
    echo "Usage: bash sa_bench_h20.sh --model /path/to/DeepSeek-R1 [--configs all]"
    exit 1
fi

mkdir -p "$RESULT_DIR"

# ======================== Hardware Constants (8×H20) ==========================
TP=8
NUM_GPUS=8
GPU_MEM_GB=96    # H20 = 96GB per GPU

# ======================== ISL/OSL Test Matrix =================================
# SemiAnalysis tests 3 scenarios: chat(1k/1k), reasoning(1k/8k), summarize(8k/1k)
# For H20 we start with chat(1k/1k) and reasoning(1k/2k) to avoid OOM
declare -a SCENARIOS=(
    "1024:1024:chat"
    "1024:2048:reasoning"
)

# ======================== Concurrency Sweep ===================================
# SA sweeps concurrency to build Pareto frontier (throughput vs latency)
declare -a CONC_SWEEP=(1 4 8 16 32 64)

# ======================== Utility Functions ===================================
TS() { date '+%Y-%m-%d %H:%M:%S'; }

log() { echo "[$(TS)] $*"; }

kill_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping server (PID=$SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    # Also kill any leftover trtllm-serve / sglang processes
    pkill -f "trtllm-serve" 2>/dev/null || true
    sleep 3
}

wait_for_server() {
    local port=$1
    local log_file=$2
    local max_wait=600  # 10 minutes max
    local elapsed=0

    log "Waiting for server on port $port (max ${max_wait}s)..."
    while ! curl -s "http://localhost:${port}/health" >/dev/null 2>&1 && \
          ! curl -s "http://localhost:${port}/v1/models" >/dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [[ $elapsed -ge $max_wait ]]; then
            log "ERROR: Server did not start within ${max_wait}s"
            log "Last 30 lines of server log:"
            tail -30 "$log_file"
            return 1
        fi
        # Check if process died
        if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "ERROR: Server process died"
            tail -30 "$log_file"
            return 1
        fi
    done
    log "Server is ready (took ${elapsed}s)"
}

start_gpu_monitor() {
    local csv_file="$1"
    nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.current.sm,utilization.gpu,memory.used \
        --format=csv -l 2 > "$csv_file" 2>/dev/null &
    GPU_MONITOR_PID=$!
}

stop_gpu_monitor() {
    if [[ -n "${GPU_MONITOR_PID:-}" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
}

run_benchmark() {
    local model=$1
    local port=$2
    local isl=$3
    local osl=$4
    local conc=$5
    local result_file=$6
    local num_requests=$((conc * 8))

    # Minimum 20 requests
    if [[ $num_requests -lt 20 ]]; then
        num_requests=20
    fi

    log "  Benchmark: ISL=$isl, OSL=$osl, CONC=$conc, REQS=$num_requests"

    # Use trtllm-bench for throughput measurement
    trtllm-bench \
        --model "$model" \
        throughput \
        --backend pytorch \
        --dataset <(python3 -c "
import json
for i in range($num_requests):
    print(json.dumps({'input_ids': list(range($isl)), 'output_tokens': $osl, 'task_id': i}))
") \
        --tp "$TP" \
        --ep 4 \
        --max_batch_size "$conc" \
        --max_seq_len $(( isl + osl + 128 )) \
        --kv_cache_free_gpu_mem_fraction 0.80 \
        --num_requests "$num_requests" \
        --warmup "$WARMUP_REQUESTS" \
        --report_json "$result_file" \
        2>&1 | tee "${result_file%.json}.log"
}

# ======================== Config Definitions ===================================

run_trt_throughput() {
    local scenario_tag=$1
    local isl=$2
    local osl=$3

    log "========================================================"
    log " CONFIG: trt-throughput | Scenario: $scenario_tag"
    log " TRT-LLM PyTorch, max throughput, no MTP"
    log " (Ported from SA: dsr1_fp8_h200_trt.sh)"
    log "========================================================"

    local config_file="$RESULT_DIR/config_throughput_${scenario_tag}.yml"

    cat > "$config_file" << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: 64
enable_attention_dp: true
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.80
    enable_block_reuse: false
stream_interval: 10
moe_config:
    backend: CUTLASS
attention_dp_config:
    batching_wait_iters: 0
    enable_balance: true
    timeout_iters: 60
EOF

    for conc in "${CONC_SWEEP[@]}"; do
        kill_server

        local max_num_tokens=$(( (conc + isl + 64 + 63) / 64 * 64 ))
        [[ $max_num_tokens -lt 4096 ]] && max_num_tokens=4096
        local max_seq_len=$(( isl + osl + 256 ))
        [[ $max_seq_len -lt 4096 ]] && max_seq_len=4096

        local server_log="$RESULT_DIR/server_throughput_${scenario_tag}_c${conc}.log"
        local result_file="$RESULT_DIR/result_throughput_${scenario_tag}_c${conc}.json"
        local gpu_csv="$RESULT_DIR/gpu_throughput_${scenario_tag}_c${conc}.csv"

        log "  Starting server: CONC=$conc, MAX_NUM_TOKENS=$max_num_tokens"

        start_gpu_monitor "$gpu_csv"

        PYTHONNOUSERSITE=1 mpirun -n 1 --oversubscribe --allow-run-as-root \
            trtllm-serve "$MODEL" --port="$PORT" \
            --trust_remote_code \
            --backend=pytorch \
            --max_seq_len="$max_seq_len" \
            --max_num_tokens="$max_num_tokens" \
            --tp_size=$TP --ep_size=4 \
            --extra_llm_api_options="$config_file" \
            > "$server_log" 2>&1 &
        SERVER_PID=$!

        if wait_for_server "$PORT" "$server_log"; then
            run_benchmark "$MODEL" "$PORT" "$isl" "$osl" "$conc" "$result_file"
        else
            log "  SKIP: Server failed to start for CONC=$conc"
        fi

        stop_gpu_monitor
        kill_server
    done
}

run_trt_latency() {
    local scenario_tag=$1
    local isl=$2
    local osl=$3

    log "========================================================"
    log " CONFIG: trt-latency | Scenario: $scenario_tag"
    log " TRT-LLM PyTorch, min latency, MTP-3 speculative decoding"
    log " (Ported from SA: dsr1_fp8_h200_trt_mtp.sh)"
    log "========================================================"

    local config_file="$RESULT_DIR/config_latency_${scenario_tag}.yml"

    cat > "$config_file" << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: 64
enable_attention_dp: false
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.80
    enable_block_reuse: false
stream_interval: 10
moe_config:
    backend: CUTLASS
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: 3
EOF

    # Min-latency: only low concurrency values
    local latency_concs=(1 4 8)

    for conc in "${latency_concs[@]}"; do
        kill_server

        local max_batch_size=$conc
        local max_num_tokens=$(( (4 * max_batch_size + isl + 64 + 63) / 64 * 64 ))
        [[ $max_num_tokens -lt 4096 ]] && max_num_tokens=4096
        local max_seq_len=$(( isl + osl + 256 ))
        [[ $max_seq_len -lt 4096 ]] && max_seq_len=4096

        local server_log="$RESULT_DIR/server_latency_${scenario_tag}_c${conc}.log"
        local result_file="$RESULT_DIR/result_latency_${scenario_tag}_c${conc}.json"
        local gpu_csv="$RESULT_DIR/gpu_latency_${scenario_tag}_c${conc}.csv"

        log "  Starting server: CONC=$conc, MTP=3"

        start_gpu_monitor "$gpu_csv"

        PYTHONNOUSERSITE=1 mpirun -n 1 --oversubscribe --allow-run-as-root \
            trtllm-serve "$MODEL" --port="$PORT" \
            --trust_remote_code \
            --backend=pytorch \
            --max_batch_size="$max_batch_size" \
            --max_seq_len="$max_seq_len" \
            --max_num_tokens="$max_num_tokens" \
            --tp_size=$TP --ep_size=4 \
            --extra_llm_api_options="$config_file" \
            > "$server_log" 2>&1 &
        SERVER_PID=$!

        if wait_for_server "$PORT" "$server_log"; then
            run_benchmark "$MODEL" "$PORT" "$isl" "$osl" "$conc" "$result_file"
        else
            log "  SKIP: Server failed to start for CONC=$conc"
        fi

        stop_gpu_monitor
        kill_server
    done
}

run_trt_balanced() {
    local scenario_tag=$1
    local isl=$2
    local osl=$3

    log "========================================================"
    log " CONFIG: trt-balanced | Scenario: $scenario_tag"
    log " TRT-LLM PyTorch, balanced (MTP-1 + attention DP)"
    log " (Ported from SA: dsr1_fp8_h200_trt_mtp.sh DP_ATTENTION=true)"
    log "========================================================"

    local config_file="$RESULT_DIR/config_balanced_${scenario_tag}.yml"

    cat > "$config_file" << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: 64
enable_attention_dp: true
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.80
    enable_block_reuse: false
stream_interval: 10
moe_config:
    backend: CUTLASS
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: 1
attention_dp_config:
    batching_wait_iters: 0
    enable_balance: true
    timeout_iters: 60
EOF

    for conc in "${CONC_SWEEP[@]}"; do
        kill_server

        local max_batch_size=$(( conc / TP ))
        [[ $max_batch_size -lt 1 ]] && max_batch_size=1
        local max_num_tokens=$(( (2 * max_batch_size + isl + 64 + 63) / 64 * 64 ))
        [[ $max_num_tokens -lt 4096 ]] && max_num_tokens=4096
        local max_seq_len=$(( isl + osl + 256 ))
        [[ $max_seq_len -lt 4096 ]] && max_seq_len=4096

        local server_log="$RESULT_DIR/server_balanced_${scenario_tag}_c${conc}.log"
        local result_file="$RESULT_DIR/result_balanced_${scenario_tag}_c${conc}.json"
        local gpu_csv="$RESULT_DIR/gpu_balanced_${scenario_tag}_c${conc}.csv"

        log "  Starting server: CONC=$conc, MTP=1, DP_ATTENTION=true"

        start_gpu_monitor "$gpu_csv"

        PYTHONNOUSERSITE=1 mpirun -n 1 --oversubscribe --allow-run-as-root \
            trtllm-serve "$MODEL" --port="$PORT" \
            --trust_remote_code \
            --backend=pytorch \
            --max_batch_size="$max_batch_size" \
            --max_seq_len="$max_seq_len" \
            --max_num_tokens="$max_num_tokens" \
            --tp_size=$TP --ep_size=4 \
            --extra_llm_api_options="$config_file" \
            > "$server_log" 2>&1 &
        SERVER_PID=$!

        if wait_for_server "$PORT" "$server_log"; then
            run_benchmark "$MODEL" "$PORT" "$isl" "$osl" "$conc" "$result_file"
        else
            log "  SKIP: Server failed to start for CONC=$conc"
        fi

        stop_gpu_monitor
        kill_server
    done
}

# ======================== Summary Report ======================================

generate_summary() {
    log "========================================================"
    log " GENERATING SUMMARY REPORT"
    log "========================================================"

    local summary_file="$RESULT_DIR/summary.md"

    cat > "$summary_file" << 'HEADER'
# SemiAnalysis InferenceX-style Benchmark Results
## DeepSeek R1 FP8 on 8×H20 (96GB/GPU)

| Config | Scenario | Concurrency | Req Throughput (req/s) | Output TPS | Per-GPU TPS | Per-User TPS | Avg Latency (ms) |
|--------|----------|-------------|----------------------|------------|-------------|-------------|-----------------|
HEADER

    for f in "$RESULT_DIR"/result_*.json; do
        [[ -f "$f" ]] || continue
        local fname
        fname=$(basename "$f" .json)

        python3 -c "
import json, sys
try:
    with open('$f') as fh:
        data = json.load(fh)
    perf = data.get('performance', data)
    lat = data.get('latency_ms', data)
    fname = '$fname'
    parts = fname.replace('result_', '').split('_')
    config = parts[0]
    scenario = parts[1] if len(parts) > 1 else '-'
    conc = parts[-1].replace('c','') if parts[-1].startswith('c') else '-'

    req_tps = perf.get('request_throughput_rps', 0)
    out_tps = perf.get('output_throughput_tps', 0)
    gpu_tps = perf.get('per_gpu_output_tps', out_tps/8 if out_tps else 0)
    usr_tps = perf.get('per_user_output_tps', 0)
    avg_lat = lat.get('average', 0) if isinstance(lat, dict) else 0

    print(f'| {config} | {scenario} | {conc} | {req_tps:.2f} | {out_tps:.1f} | {gpu_tps:.1f} | {usr_tps:.1f} | {avg_lat:.0f} |')
except Exception as e:
    print(f'| $fname | ERROR | - | - | - | - | - | {e} |', file=sys.stderr)
" 2>/dev/null >> "$summary_file" || true
    done

    log "Summary written to: $summary_file"
    echo ""
    cat "$summary_file"
}

# ======================== Main Execution ======================================

trap 'kill_server; stop_gpu_monitor 2>/dev/null; exit' INT TERM

log "============================================================"
log "  DeepSeek R1 Benchmark Suite (SA InferenceX-style)"
log "  Adapted for 8×H20 GPUs"
log "============================================================"
log "  Model:       $MODEL"
log "  Configs:     $CONFIGS"
log "  Port:        $PORT"
log "  Result Dir:  $RESULT_DIR"
log "  TP:          $TP"
log "  Scenarios:   ${SCENARIOS[*]}"
log "  Concurrency: ${CONC_SWEEP[*]}"
log "============================================================"
echo ""

for scenario_entry in "${SCENARIOS[@]}"; do
    IFS=':' read -r isl osl tag <<< "$scenario_entry"

    case "$CONFIGS" in
        trt-throughput)
            run_trt_throughput "$tag" "$isl" "$osl"
            ;;
        trt-latency)
            run_trt_latency "$tag" "$isl" "$osl"
            ;;
        trt-balanced)
            run_trt_balanced "$tag" "$isl" "$osl"
            ;;
        all)
            run_trt_throughput "$tag" "$isl" "$osl"
            run_trt_latency "$tag" "$isl" "$osl"
            run_trt_balanced "$tag" "$isl" "$osl"
            ;;
        *)
            log "ERROR: Unknown config '$CONFIGS'"
            log "Available: trt-throughput, trt-latency, trt-balanced, all"
            exit 1
            ;;
    esac
done

generate_summary

log "============================================================"
log "  ALL BENCHMARKS COMPLETE"
log "  Results in: $RESULT_DIR/"
log "============================================================"
