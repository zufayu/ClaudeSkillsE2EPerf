#!/usr/bin/env bash
# =============================================================================
# InferenceX/InferenceMAX-style Benchmark Suite for DeepSeek R1 on 8×H200
#
# Ported from:
#   - InferenceX:   dsr1_fp8_h200_trt.sh, dsr1_fp8_h200_trt_mtp.sh
#   - InferenceMAX: dsr1_fp8_h200_trt_slurm.sh
#
# Configs:
#   fp8-throughput   FP8, no MTP, max throughput
#   fp8-latency      FP8, MTP-3 (or MTP-1 with DP), min latency
#   all              Run all configs
#
# Usage:
#   bash sa_bench_h200.sh \
#     --model /path/to/DeepSeek-R1-FP8 \
#     [--configs all] [--port 8888] [--ep-sizes "4 8"]
#
# Prerequisites:
#   - 8× NVIDIA H200 GPUs (141GB each)
#   - TensorRT-LLM >= 1.2.0rc4 with PyTorch backend
#   - DeepSeek R1 FP8 model weights
#   - benchmark_serving.py available (from InferenceX utils/)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/benchmark_lib.sh"
source "$SCRIPT_DIR/result_dir_validate.sh"

# ======================== Argument Parsing ====================================
MODEL=""
CONFIGS="all"
PORT=8888
RESULT_DIR="./results_h200"
RANDOM_RANGE_RATIO=0.8
EP_SIZES="4 8"
BENCH_SERVING_DIR=""
TP=8

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)             MODEL="$2"; shift 2 ;;
        --configs)           CONFIGS="$2"; shift 2 ;;
        --port)              PORT="$2"; shift 2 ;;
        --result-dir)        RESULT_DIR="$2"; shift 2 ;;
        --ep-sizes)          EP_SIZES="$2"; shift 2 ;;
        --bench-serving-dir) BENCH_SERVING_DIR="$2"; shift 2 ;;
        --tp)                TP="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash sa_bench_h200.sh --model <path> [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH            FP8 model path (required)"
            echo "  --configs CONFIG        fp8-throughput|fp8-latency|all (default: all)"
            echo "  --port PORT             Server port (default: 8888)"
            echo "  --result-dir DIR        Results directory (default: ./results_h200)"
            echo "  --ep-sizes \"4 8\"        EP sizes to sweep (default: \"4 8\")"
            echo "  --bench-serving-dir DIR Directory containing benchmark_serving.py"
            echo "  --tp N                  Tensor parallelism (default: 8)"
            exit 0
            ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required"
    echo "Usage: bash sa_bench_h200.sh --model /path/to/DeepSeek-R1-FP8 [--configs all]"
    exit 1
fi

case "$CONFIGS" in
    fp8-throughput|fp8-latency|all) ;;
    *)
        echo "ERROR: Unknown config '$CONFIGS'"
        echo "Available: fp8-throughput, fp8-latency, all"
        exit 1
        ;;
esac

validate_result_dir "$RESULT_DIR" || exit 1
mkdir -p "$RESULT_DIR"

# ======================== Hardware Constants (8×H200) ==========================
# H200: 141GB HBM3e per GPU, 1.1TB total
# Compared to B200 (192GB): less memory, no FP4, simpler adaptive logic.

# ======================== ISL/OSL Test Matrix =================================
declare -a SCENARIOS=(
    "1024:1024:chat"
    "1024:8192:reasoning"
    "8192:1024:summarize"
)

# ======================== Concurrency Sweep ===================================
# H200 uses max 128 concurrency (vs B200's 256) due to memory constraints
declare -a CONC_SWEEP=(1 4 8 16 32 64 128)

# ======================== Utility =============================================
TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(TS)] $*"; }

# ======================== H200 Adaptive Parameters ============================
# H200 is simpler than B200:
#   - MOE backend always CUTLASS (no TRTLLM/DEEPGEMM switching)
#   - No piecewise CUDA graphs
#   - No delay batching
#   - Fixed cuda_graph max_batch_size=128
#   - KV cache free fraction = 0.75 (vs B200's 0.80)
#
# Arguments: $1=ISL $2=OSL $3=CONC $4=DP_ATTENTION $5=has_mtp
compute_h200_params() {
    local isl=$1 osl=$2 conc=$3 dp_attn=$4 has_mtp=$5

    # --- H200 defaults (from InferenceX dsr1_fp8_h200_trt*.sh) ---
    MOE_BACKEND="CUTLASS"
    CUDA_GRAPH_MAX_BATCH_SIZE=128
    KV_CACHE_FREE_MEM_FRACTION=0.75
    MTP_LAYERS=0
    MAX_BATCH_SIZE=$conc
    ALLOC_CONF_OVERRIDE=""

    if [[ "$has_mtp" == "true" ]]; then
        # MTP-3 default, MTP-1 when DP attention
        if [[ "$dp_attn" == "true" ]]; then
            MTP_LAYERS=1
            MAX_BATCH_SIZE=$(( conc / TP ))
            [[ $MAX_BATCH_SIZE -lt 1 ]] && MAX_BATCH_SIZE=1
        else
            MTP_LAYERS=3
            MAX_BATCH_SIZE=$conc
        fi

        # ISL=8192 + DP needs CUDA alloc config to avoid OOM
        if [[ "$isl" == "8192" && "$dp_attn" == "true" ]]; then
            ALLOC_CONF_OVERRIDE="max_split_size_mb:8192"
        fi
    fi
}

# ======================== Config Runner =======================================

run_single_point() {
    local model=$1 config_name=$2 scenario_tag=$3
    local isl=$4 osl=$5 conc=$6 dp_attn=$7 ep_size=$8

    local has_mtp="false"
    [[ "$config_name" == "latency" ]] && has_mtp="true"

    compute_h200_params "$isl" "$osl" "$conc" "$dp_attn" "$has_mtp"

    local tag="fp8_${config_name}_${scenario_tag}_ep${ep_size}_c${conc}"
    [[ "$dp_attn" == "true" ]] && tag="${tag}_dp"

    log "  ---- $tag ----"
    log "    MOE=$MOE_BACKEND, MTP=$MTP_LAYERS, MAX_BS=$MAX_BATCH_SIZE"
    log "    CUDA_GRAPH_BS=$CUDA_GRAPH_MAX_BATCH_SIZE, KV_FREE=$KV_CACHE_FREE_MEM_FRACTION"

    kill_server

    # --- Generate extra config YAML ---
    local config_file="$RESULT_DIR/config_${tag}.yml"

    cat > "$config_file" << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: $CUDA_GRAPH_MAX_BATCH_SIZE
enable_attention_dp: $dp_attn
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: $KV_CACHE_FREE_MEM_FRACTION
    enable_block_reuse: false
stream_interval: 10
moe_config:
    backend: $MOE_BACKEND
EOF

    if [[ $MTP_LAYERS -gt 0 ]]; then
        cat >> "$config_file" << EOF
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: $MTP_LAYERS
EOF
    fi

    if [[ "$dp_attn" == "true" ]]; then
        cat >> "$config_file" << EOF
attention_dp_config:
    batching_wait_iters: 0
    enable_balance: true
    timeout_iters: 60
EOF
    fi

    # --- Compute MAX_NUM_TOKENS and MAX_MODEL_LEN ---
    local max_num_tokens
    if [[ $MTP_LAYERS -gt 0 ]]; then
        max_num_tokens=$(( ((MTP_LAYERS + 1) * MAX_BATCH_SIZE + isl + 64 + 63) / 64 * 64 ))
    else
        max_num_tokens=$(( (conc + isl + 64 + 63) / 64 * 64 ))
    fi
    local max_model_len=$(( isl + osl + 256 ))

    # Enforce minimums (from InferenceX)
    [[ $max_model_len -lt 8192 ]] && max_model_len=8192
    [[ $max_num_tokens -lt 8192 ]] && max_num_tokens=8192

    # --- Optional CUDA alloc config for large ISL + DP ---
    if [[ -n "$ALLOC_CONF_OVERRIDE" ]]; then
        export PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF_OVERRIDE"
        log "    PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF_OVERRIDE"
    else
        unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
    fi

    # --- Start server ---
    local server_log="$RESULT_DIR/server_${tag}.log"
    local gpu_csv="$RESULT_DIR/gpu_${tag}.csv"
    local result_filename="result_${tag}"

    start_gpu_monitor --output "$gpu_csv"

    log "  Starting server: TP=$TP, EP=$ep_size, MAX_NUM_TOKENS=$max_num_tokens"

    local serve_args=(
        trtllm-serve "$model" --port="$PORT"
        --trust_remote_code
        --backend=pytorch
        --max_seq_len="$max_model_len"
        --max_num_tokens="$max_num_tokens"
        --tp_size=$TP --ep_size=$ep_size
        --extra_llm_api_options="$config_file"
    )

    # For MTP/latency configs, also pass --max_batch_size
    if [[ $MTP_LAYERS -gt 0 ]]; then
        serve_args+=(--max_batch_size="$MAX_BATCH_SIZE")
    fi

    PYTHONNOUSERSITE=1 mpirun -n 1 --oversubscribe --allow-run-as-root \
        "${serve_args[@]}" \
        > "$server_log" 2>&1 &
    SERVER_PID=$!

    if wait_for_server_ready --port "$PORT" --server-log "$server_log" --server-pid "$SERVER_PID"; then
        local num_prompts=$(( conc * 10 ))
        [[ $num_prompts -lt 20 ]] && num_prompts=20

        local bench_args=(
            --model "$model"
            --port "$PORT"
            --backend openai
            --input-len "$isl"
            --output-len "$osl"
            --random-range-ratio "$RANDOM_RANGE_RATIO"
            --num-prompts "$num_prompts"
            --max-concurrency "$conc"
            --result-filename "$result_filename"
            --result-dir "$RESULT_DIR"
        )
        if [[ -n "$BENCH_SERVING_DIR" ]]; then
            bench_args+=(--bench-serving-dir "$BENCH_SERVING_DIR")
        fi
        if [[ $MTP_LAYERS -gt 0 ]]; then
            bench_args+=(--use-chat-template)
        fi

        run_benchmark_serving "${bench_args[@]}" || \
            log "  WARN: Benchmark failed for $tag"
    else
        log "  SKIP: Server failed to start for $tag"
    fi

    stop_gpu_monitor
    kill_server
}

# ======================== Config Dispatchers ===================================

run_config() {
    local config_name=$1

    for scenario_entry in "${SCENARIOS[@]}"; do
        IFS=':' read -r isl osl scenario_tag <<< "$scenario_entry"

        log "========================================================"
        log " CONFIG: fp8-${config_name} | Scenario: $scenario_tag (ISL=$isl, OSL=$osl)"
        log "========================================================"

        for ep_size in $EP_SIZES; do
            local dp_attn="false"
            if [[ $ep_size -gt 1 ]]; then
                dp_attn="true"
            fi

            log "  EP_SIZE=$ep_size, DP_ATTENTION=$dp_attn"

            for conc in "${CONC_SWEEP[@]}"; do
                # For latency config, limit concurrency
                if [[ "$config_name" == "latency" && $conc -gt 64 ]]; then
                    log "  SKIP: CONC=$conc too high for H200 latency config"
                    continue
                fi

                run_single_point "$MODEL" "$config_name" "$scenario_tag" \
                    "$isl" "$osl" "$conc" "$dp_attn" "$ep_size"
            done
        done
    done
}

# ======================== Summary Report ======================================

generate_summary() {
    log "========================================================"
    log " GENERATING SUMMARY REPORT"
    log "========================================================"

    local summary_file="$RESULT_DIR/summary.md"

    cat > "$summary_file" << 'HEADER'
# DeepSeek R1 Benchmark Results (InferenceX-style)
## H200 8×GPU (141GB/GPU)

| Config | Scenario | EP | CONC | DP | MTP | Total Tput | Output Tput | Interac. | TPOT (ms) | TTFT (ms) | DAR |
|--------|----------|-----|------|----|-----|------------|-------------|----------|-----------|-----------|-----|
HEADER

    for f in "$RESULT_DIR"/result_*.json; do
        [[ -f "$f" ]] || continue
        python3 -c "
import json, sys, os
try:
    with open('$f') as fh:
        data = json.load(fh)

    fname = os.path.basename('$f').replace('result_', '').replace('.json', '')
    parts = fname.split('_')

    # Parse tag: fp8_config_scenario_epN_cN[_dp]
    config = parts[1] if len(parts) > 1 else '-'
    scenario = parts[2] if len(parts) > 2 else '-'
    ep = parts[3].replace('ep','') if len(parts) > 3 else '-'
    conc_part = [p for p in parts if p.startswith('c') and p[1:].isdigit()]
    conc = conc_part[0].replace('c','') if conc_part else '-'
    dp = 'Y' if 'dp' in parts else 'N'
    mtp = '3' if config == 'latency' and dp == 'N' else ('1' if config == 'latency' else '-')

    out_tps = data.get('output_throughput', 0)
    in_tps = data.get('input_throughput', data.get('total_token_throughput', 0) - out_tps if data.get('total_token_throughput') else 0)
    total_tps = data.get('total_token_throughput', in_tps + out_tps)
    ttft_p50 = data.get('ttft_p50', data.get('median_ttft_ms', 0))
    tpot_p50 = data.get('tpot_p50', data.get('median_tpot_ms', 0))
    # Interactivity = 1000/TPOT tok/s/user (SA InferenceX format)
    interactivity = 1000.0 / tpot_p50 if tpot_p50 > 0 else 0

    dar = data.get('dar_p50')
    dar_str = f'{dar:.2%}' if dar is not None else '-'
    print(f'| {config} | {scenario} | {ep} | {conc} | {dp} | {mtp} | {total_tps:.1f} | {out_tps:.1f} | {interactivity:.2f} | {tpot_p50:.1f} | {ttft_p50:.0f} | {dar_str} |')
except Exception as e:
    print(f'| ERROR | $f | - | - | - | - | - | - | - | - | - | {e} |', file=sys.stderr)
" >> "$summary_file" 2>/dev/null || true
    done

    log "Summary written to: $summary_file"
    echo ""
    cat "$summary_file"
}

# ======================== Main Execution ======================================

trap 'kill_server; stop_gpu_monitor 2>/dev/null; exit' INT TERM

log "============================================================"
log "  DeepSeek R1 Benchmark Suite (InferenceX/MAX-style)"
log "  Target: 8×H200 GPUs (141GB/GPU)"
log "============================================================"
log "  Model:       $MODEL"
log "  Configs:     $CONFIGS"
log "  Port:        $PORT"
log "  Result Dir:  $RESULT_DIR"
log "  TP:          $TP"
log "  EP Sizes:    $EP_SIZES"
log "  Scenarios:   ${SCENARIOS[*]}"
log "  Concurrency: ${CONC_SWEEP[*]}"
log "  Range Ratio: $RANDOM_RANGE_RATIO"
log "============================================================"
echo ""

case "$CONFIGS" in
    fp8-throughput)
        run_config "throughput"
        ;;
    fp8-latency)
        run_config "latency"
        ;;
    all)
        run_config "throughput"
        run_config "latency"
        ;;
esac

generate_summary

log "============================================================"
log "  ALL BENCHMARKS COMPLETE"
log "  Results in: $RESULT_DIR/"
log "============================================================"
