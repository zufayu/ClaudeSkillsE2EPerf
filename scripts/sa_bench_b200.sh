#!/usr/bin/env bash
# =============================================================================
# InferenceX/InferenceMAX-style Benchmark Suite for DeepSeek R1 on 8×B200
#
# Ported from:
#   - InferenceX:   dsr1_fp4_b200_trt.sh, dsr1_fp4_b200_trt_mtp.sh
#                   dsr1_fp8_b200_trt.sh, dsr1_fp8_b200_trt_mtp.sh
#   - InferenceMAX: dsr1_fp4_b200_trt_slurm.sh, dsr1_fp8_b200_trt_slurm.sh
#
# Configs:
#   fp4-throughput   FP4, no MTP, max throughput
#   fp4-latency      FP4, MTP-3 (or MTP-1 with DP), min latency
#   fp8-throughput   FP8, no MTP, max throughput
#   fp8-latency      FP8, MTP-3 (or MTP-1 with DP), min latency
#   all              Run all configs
#
# Usage:
#   # Run everything (original behavior):
#   bash sa_bench_b200.sh --model-fp4 /data/model [--configs all]
#
#   # Run ONE specific point (reproduce SA result):
#   bash sa_bench_b200.sh \
#     --model-fp4 /data/model \
#     --configs fp4-throughput \
#     --scenario chat \
#     --concurrency 256 \
#     --ep-sizes "8"
#
# Prerequisites:
#   - 8× NVIDIA B200 GPUs
#   - TensorRT-LLM >= 1.2.0rc4 with PyTorch backend
#   - DeepSeek R1 model weights (FP4 and/or FP8)
#   - benchmark_serving.py available (from InferenceX utils/)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/benchmark_lib.sh"

# ======================== Argument Parsing ====================================
MODEL_FP4=""
MODEL_FP8=""
CONFIGS="all"
PORT=8888
RESULT_DIR="./results_b200"
RANDOM_RANGE_RATIO=0.8
EP_SIZES="1 8"
BENCH_SERVING_DIR=""
TP=8
SCENARIO_FILTER="all"        # all | chat | reasoning | summarize
CONC_FILTER=""               # empty = use default sweep; "256" or "4 128 256" = specific values
NUM_WARMUPS=""                # empty = auto (SA default: 8); or explicit number

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-fp4)        MODEL_FP4="$2"; shift 2 ;;
        --model-fp8)        MODEL_FP8="$2"; shift 2 ;;
        --configs)          CONFIGS="$2"; shift 2 ;;
        --port)             PORT="$2"; shift 2 ;;
        --result-dir)       RESULT_DIR="$2"; shift 2 ;;
        --ep-sizes)         EP_SIZES="$2"; shift 2 ;;
        --bench-serving-dir) BENCH_SERVING_DIR="$2"; shift 2 ;;
        --tp)               TP="$2"; shift 2 ;;
        --scenario)         SCENARIO_FILTER="$2"; shift 2 ;;
        --concurrency)      CONC_FILTER="$2"; shift 2 ;;
        --num-warmups)      NUM_WARMUPS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash sa_bench_b200.sh --model-fp4 <path> --model-fp8 <path> [options]"
            echo ""
            echo "Options:"
            echo "  --model-fp4 PATH        FP4 model path (e.g., nvidia/DeepSeek-R1-0528-NVFP4-v2)"
            echo "  --model-fp8 PATH        FP8 model path"
            echo "  --configs CONFIG        fp4-throughput|fp4-latency|fp8-throughput|fp8-latency|all (default: all)"
            echo "  --port PORT             Server port (default: 8888)"
            echo "  --result-dir DIR        Results directory (default: ./results_b200)"
            echo "  --ep-sizes \"1 8\"        EP sizes to sweep (default: \"1 8\")"
            echo "  --bench-serving-dir DIR Directory containing benchmark_serving.py"
            echo "  --tp N                  Tensor parallelism (default: 8)"
            echo ""
            echo "Filtering (run a subset):"
            echo "  --scenario SCENARIO     chat|reasoning|summarize|all (default: all)"
            echo "  --concurrency \"C1 C2\"   Concurrency values to run (default: sweep 1 4 8 16 32 64 128 256)"
            echo "  --num-warmups N         Number of warmup requests (default: 8, matching SA)"
            echo ""
            echo "Examples:"
            echo "  # Reproduce SA B200 TRT FP4 c=256 EP=8 chat point:"
            echo "  bash sa_bench_b200.sh --model-fp4 /data/model \\"
            echo "    --configs fp4-throughput --scenario chat --concurrency 256 --ep-sizes 8"
            echo ""
            echo "  # Run only chat scenario with c=4 and c=128:"
            echo "  bash sa_bench_b200.sh --model-fp4 /data/model \\"
            echo "    --configs fp4-throughput --scenario chat --concurrency \"4 128\""
            exit 0
            ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Validate model paths based on selected configs
case "$CONFIGS" in
    fp4-throughput|fp4-latency)
        if [[ -z "$MODEL_FP4" ]]; then
            echo "ERROR: --model-fp4 is required for $CONFIGS"
            exit 1
        fi
        ;;
    fp8-throughput|fp8-latency)
        if [[ -z "$MODEL_FP8" ]]; then
            echo "ERROR: --model-fp8 is required for $CONFIGS"
            exit 1
        fi
        ;;
    all)
        if [[ -z "$MODEL_FP4" && -z "$MODEL_FP8" ]]; then
            echo "ERROR: at least one of --model-fp4 or --model-fp8 is required"
            exit 1
        fi
        ;;
    *)
        echo "ERROR: Unknown config '$CONFIGS'"
        echo "Available: fp4-throughput, fp4-latency, fp8-throughput, fp8-latency, all"
        exit 1
        ;;
esac

mkdir -p "$RESULT_DIR"

# ======================== ISL/OSL Test Matrix =================================
# SemiAnalysis tests: chat(1k/1k), reasoning(1k/8k), summarize(8k/1k)
declare -a ALL_SCENARIOS=(
    "1024:1024:chat"
    "1024:8192:reasoning"
    "8192:1024:summarize"
)

# Apply --scenario filter
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

# ======================== Concurrency Sweep ===================================
if [[ -n "$CONC_FILTER" ]]; then
    # User specified exact concurrency values
    declare -a CONC_SWEEP=($CONC_FILTER)
else
    declare -a CONC_SWEEP=(1 4 8 16 32 64 128 256)
fi

# ======================== Utility =============================================
TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(TS)] $*"; }

# ======================== Auto-Adaptive Parameters ============================
# Determine MOE_BACKEND, CUDA graph settings, delay batching based on
# quantization, ISL, OSL, CONC, DP_ATTENTION, EP_SIZE.
#
# Arguments: $1=quant(fp4|fp8) $2=ISL $3=OSL $4=CONC $5=DP_ATTENTION $6=EP_SIZE $7=has_mtp
compute_adaptive_params() {
    local quant=$1 isl=$2 osl=$3 conc=$4 dp_attn=$5 ep_size=$6 has_mtp=$7

    # --- Defaults ---
    MOE_BACKEND="TRTLLM"
    PIECEWISE_CUDA_GRAPHS="false"
    CUDA_GRAPH_MAX_BATCH_SIZE=$conc
    KV_CACHE_FREE_MEM_FRACTION=0.8
    DELAY_BATCHING="false"
    MTP_LAYERS=0
    ENABLE_CONFIGURABLE_MOE_FLAG=""

    if [[ "$has_mtp" == "true" ]]; then
        MTP_LAYERS=3
    fi

    # --- FP4 logic (from dsr1_fp4_b200_trt.sh / dsr1_fp4_b200_trt_mtp.sh) ---
    if [[ "$quant" == "fp4" ]]; then
        if [[ "$has_mtp" == "false" ]]; then
            # fp4-throughput: from dsr1_fp4_b200_trt.sh
            if [[ "$dp_attn" == "true" ]]; then
                MOE_BACKEND="CUTLASS"
                CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc < 4 ? conc : conc / 4 ))
            fi
            # Piecewise CUDA graphs for EP=8 + 1k/1k
            if [[ "$isl" == "1024" && "$osl" == "1024" ]]; then
                if [[ "$TP" == "8" && "$ep_size" == "8" ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                fi
            fi
        else
            # fp4-latency: from dsr1_fp4_b200_trt_mtp.sh
            if [[ "$dp_attn" == "true" ]]; then
                CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc < 4 ? conc : conc / 4 ))
                MOE_BACKEND="CUTLASS"
                MTP_LAYERS=1
            fi
            # Piecewise for specific conc values
            if [[ "$isl" == "1024" && "$osl" == "1024" ]]; then
                if [[ $conc == 32 || $conc == 64 ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                elif [[ $conc == 128 && "$dp_attn" == "false" ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                fi
            elif [[ "$isl" == "1024" && "$osl" == "8192" ]]; then
                if [[ $conc == 64 ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                fi
            fi
        fi

    # --- FP8 logic (from dsr1_fp8_b200_trt.sh / dsr1_fp8_b200_trt_mtp.sh) ---
    elif [[ "$quant" == "fp8" ]]; then
        if [[ "$has_mtp" == "false" ]]; then
            # fp8-throughput: from dsr1_fp8_b200_trt.sh
            if [[ "$isl" == "1024" && "$osl" == "1024" ]]; then
                if [[ $conc -ge 128 ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                    DELAY_BATCHING="true"
                    KV_CACHE_FREE_MEM_FRACTION=0.7
                elif [[ $conc -ge 64 ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                    DELAY_BATCHING="true"
                fi
            elif [[ "$isl" == "1024" && "$osl" == "8192" ]]; then
                if [[ $conc -ge 256 ]]; then
                    CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc / 8 ))
                    MOE_BACKEND="DEEPGEMM"
                    KV_CACHE_FREE_MEM_FRACTION=0.7
                elif [[ $conc -ge 128 ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                fi
            elif [[ "$isl" == "8192" && "$osl" == "1024" ]]; then
                if [[ $conc -ge 64 ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                fi
                if [[ "$TP" == "4" ]]; then
                    KV_CACHE_FREE_MEM_FRACTION=0.75
                fi
            fi
            if [[ "$dp_attn" == "true" ]]; then
                MOE_BACKEND="CUTLASS"
            fi
        else
            # fp8-latency: from dsr1_fp8_b200_trt_mtp.sh
            PIECEWISE_CUDA_GRAPHS="true"
            if [[ "$dp_attn" == "true" ]]; then
                MOE_BACKEND="DEEPGEMM"
                PIECEWISE_CUDA_GRAPHS="false"
                CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc < 8 ? conc : conc / 8 ))
                KV_CACHE_FREE_MEM_FRACTION=0.7
                ENABLE_CONFIGURABLE_MOE_FLAG="1"
                MTP_LAYERS=1
            fi
            # Disable PW CUDA for narrow conc
            if [[ "$isl" == "1024" && "$osl" == "1024" ]]; then
                if [[ $conc -le 4 ]]; then
                    PIECEWISE_CUDA_GRAPHS="false"
                fi
            elif [[ "$isl" == "1024" && "$osl" == "8192" ]]; then
                if [[ $conc -le 8 ]]; then
                    PIECEWISE_CUDA_GRAPHS="false"
                fi
            elif [[ "$isl" == "8192" && "$osl" == "1024" ]]; then
                if [[ $conc -le 16 ]]; then
                    PIECEWISE_CUDA_GRAPHS="false"
                fi
            fi
        fi
    fi
}

# ======================== Config Runner =======================================

# Run a single config point: start server, benchmark, stop server.
#
# Arguments:
#   $1 = model path
#   $2 = quant (fp4|fp8)
#   $3 = config_name (throughput|latency)
#   $4 = scenario_tag
#   $5 = ISL
#   $6 = OSL
#   $7 = CONC
#   $8 = DP_ATTENTION (true|false)
#   $9 = EP_SIZE
run_single_point() {
    local model=$1 quant=$2 config_name=$3 scenario_tag=$4
    local isl=$5 osl=$6 conc=$7 dp_attn=$8 ep_size=$9

    local has_mtp="false"
    [[ "$config_name" == "latency" ]] && has_mtp="true"

    compute_adaptive_params "$quant" "$isl" "$osl" "$conc" "$dp_attn" "$ep_size" "$has_mtp"

    local tag="${quant}_${config_name}_${scenario_tag}_ep${ep_size}_c${conc}"
    [[ "$dp_attn" == "true" ]] && tag="${tag}_dp"

    log "  ---- $tag ----"
    log "    MOE=$MOE_BACKEND, PW_CUDA=$PIECEWISE_CUDA_GRAPHS, MTP=$MTP_LAYERS"
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

    if [[ "$DELAY_BATCHING" == "true" ]]; then
        cat >> "$config_file" << EOF
batch_wait_timeout_iters: 40
batch_wait_max_tokens_ratio: 0.8
EOF
    fi

    # --- Compute MAX_NUM_TOKENS and MAX_MODEL_LEN ---
    local max_batch_size=$CUDA_GRAPH_MAX_BATCH_SIZE
    local max_num_tokens
    if [[ $MTP_LAYERS -gt 0 ]]; then
        max_num_tokens=$(( ((MTP_LAYERS + 1) * max_batch_size + isl + 64 + 63) / 64 * 64 ))
    else
        max_num_tokens=$(( (conc + isl + 64 + 63) / 64 * 64 ))
    fi
    # Match ATOM CI max_model_len for fair cross-platform comparison:
    #   chat (1k/1k):      ATOM unset (defaults 163840), we use 8192 (tighter = better)
    #   reasoning (1k/8k): ATOM uses 10240
    #   summarize (8k/1k): ATOM uses 10240
    local max_model_len
    if [[ $((isl + osl)) -le 2048 ]]; then
        max_model_len=8192
    else
        max_model_len=10240
    fi
    [[ $max_num_tokens -lt 8192 ]] && max_num_tokens=8192

    # --- Piecewise CUDA Graphs ---
    if [[ "$PIECEWISE_CUDA_GRAPHS" == "true" ]]; then
        local capture_tokens=(1 2 4 8 16 32 64 128)
        capture_tokens+=( $(seq 256 256 $max_num_tokens) )
        if [[ $((max_num_tokens % 256)) -ne 0 ]]; then
            capture_tokens+=($max_num_tokens)
        fi
        local capture_list
        capture_list=$(printf "%s, " "${capture_tokens[@]}")

        cat >> "$config_file" << EOF
torch_compile_config:
    capture_num_tokens: [${capture_list%, }]
    enable_piecewise_cuda_graph: true
EOF
    fi

    # --- Optional env vars ---
    if [[ -n "$ENABLE_CONFIGURABLE_MOE_FLAG" ]]; then
        export ENABLE_CONFIGURABLE_MOE=1
    else
        unset ENABLE_CONFIGURABLE_MOE 2>/dev/null || true
    fi

    # --- Build server command ---
    local server_log="$RESULT_DIR/server_${tag}.log"
    local gpu_csv="$RESULT_DIR/gpu_${tag}.csv"
    local result_filename="result_${tag}"

    local server_cmd="PYTHONNOUSERSITE=1 mpirun -n 1 --oversubscribe --allow-run-as-root"
    server_cmd+=" trtllm-serve $model --port=$PORT"
    server_cmd+=" --trust_remote_code --backend=pytorch"
    server_cmd+=" --max_seq_len=$max_model_len --max_num_tokens=$max_num_tokens"
    server_cmd+=" --tp_size=$TP --ep_size=$ep_size"
    server_cmd+=" --extra_llm_api_options=$config_file"

    start_gpu_monitor --output "$gpu_csv"

    log "  Starting server: TP=$TP, EP=$ep_size, MAX_NUM_TOKENS=$max_num_tokens"

    PYTHONNOUSERSITE=1 mpirun -n 1 --oversubscribe --allow-run-as-root \
        trtllm-serve "$model" --port="$PORT" \
        --trust_remote_code \
        --backend=pytorch \
        --max_seq_len="$max_model_len" \
        --max_num_tokens="$max_num_tokens" \
        --tp_size=$TP --ep_size=$ep_size \
        --extra_llm_api_options="$config_file" \
        > "$server_log" 2>&1 &
    SERVER_PID=$!

    if wait_for_server_ready --port "$PORT" --server-log "$server_log" --server-pid "$SERVER_PID"; then
        local num_prompts=$(( conc * 10 ))
        [[ $num_prompts -lt 20 ]] && num_prompts=20

        # SA uses 8 warmups by default, not 2*concurrency
        local warmups="${NUM_WARMUPS:-8}"

        local bench_args=(
            --model "$model"
            --port "$PORT"
            --backend openai
            --input-len "$isl"
            --output-len "$osl"
            --random-range-ratio "$RANDOM_RANGE_RATIO"
            --num-prompts "$num_prompts"
            --max-concurrency "$conc"
            --num-warmups "$warmups"
            --result-filename "$result_filename"
            --result-dir "$RESULT_DIR"
            --metadata
                "max_model_len=$max_model_len"
                "kv_cache_dtype=fp8"
                "kv_cache_free_mem_fraction=$KV_CACHE_FREE_MEM_FRACTION"
                "tensor_parallel_size=$TP"
                "ep_size=$ep_size"
                "dp_attention=$dp_attn"
                "moe_backend=$MOE_BACKEND"
                "mtp_layers=$MTP_LAYERS"
                "piecewise_cuda_graphs=$PIECEWISE_CUDA_GRAPHS"
                "random_range_ratio=$RANDOM_RANGE_RATIO"
        )
        if [[ -n "$BENCH_SERVING_DIR" ]]; then
            bench_args+=(--bench-serving-dir "$BENCH_SERVING_DIR")
        fi
        if [[ $MTP_LAYERS -gt 0 ]]; then
            bench_args+=(--use-chat-template)
        fi

        run_benchmark_serving "${bench_args[@]}" || \
            log "  WARN: Benchmark failed for $tag"

        # --- Inject reproduce commands into result JSON ---
        local result_json="$RESULT_DIR/${result_filename}.json"
        if [[ -f "$result_json" ]]; then
            # Detect docker image from env or label
            local docker_image="${DOCKER_IMAGE:-${TRTLLM_IMAGE:-}}"
            if [[ -z "$docker_image" && -f /.dockerenv ]]; then
                docker_image="unknown-docker"
            fi

            # Merge reproduce info into result JSON
            RESULT_JSON="$result_json" \
            SERVER_CMD="$server_cmd" \
            BENCH_ARGS="${bench_args[*]}" \
            CONFIG_FILE="$config_file" \
            DOCKER_IMG="$docker_image" \
            python3 << 'PYEOF'
import json, sys, os

result_path = os.environ["RESULT_JSON"]
server_cmd = os.environ["SERVER_CMD"]
bench_args_str = os.environ["BENCH_ARGS"]
config_file = os.environ["CONFIG_FILE"]
docker_image = os.environ["DOCKER_IMG"]

with open(result_path) as f:
    data = json.load(f)

with open(config_file) as f:
    config_yaml = f.read()

data["server_cmd"] = server_cmd
data["benchmark_cmd"] = "benchmark_serving.py " + bench_args_str
data["config_yaml"] = config_yaml
if docker_image:
    data["docker_image"] = docker_image

with open(result_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"  Injected reproduce info into {result_path}")
PYEOF
        fi
    else
        log "  SKIP: Server failed to start for $tag"
    fi

    stop_gpu_monitor
    kill_server
}

# ======================== Config Dispatchers ===================================

run_config() {
    local model=$1 quant=$2 config_name=$3

    for scenario_entry in "${SCENARIOS[@]}"; do
        IFS=':' read -r isl osl scenario_tag <<< "$scenario_entry"

        log "========================================================"
        log " CONFIG: ${quant}-${config_name} | Scenario: $scenario_tag (ISL=$isl, OSL=$osl)"
        log "========================================================"

        for ep_size in $EP_SIZES; do
            # Determine DP_ATTENTION based on EP_SIZE
            # EP=1 with TP=8: no DP attention (pure TP)
            # EP=8 with TP=8: can use DP attention for throughput
            local dp_attn="false"
            if [[ $ep_size -gt 1 ]]; then
                dp_attn="true"
            fi

            log "  EP_SIZE=$ep_size, DP_ATTENTION=$dp_attn"

            for conc in "${CONC_SWEEP[@]}"; do
                # For latency config, limit concurrency
                if [[ "$config_name" == "latency" && $conc -gt 128 ]]; then
                    log "  SKIP: CONC=$conc too high for latency config"
                    continue
                fi

                run_single_point "$model" "$quant" "$config_name" "$scenario_tag" \
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
    local gpu_count="$TP"

    cat > "$summary_file" <<EOF
# DeepSeek R1 Benchmark Results (InferenceX-style)
## B200 ${gpu_count}×GPU

| Config | Quant | Scenario | EP | CONC | Output TPS | Out TPS/GPU | Total TPS/GPU | Interactivity | TTFT p50 (ms) | TPOT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) |
|--------|-------|----------|-----|------|------------|-------------|---------------|---------------|---------------|---------------|--------------|--------------|
EOF

    for f in "$RESULT_DIR"/result_*.json; do
        [[ -f "$f" ]] || continue
        python3 -c "
import json, sys, os
try:
    gpu_count = $gpu_count
    with open('$f') as fh:
        data = json.load(fh)

    fname = os.path.basename('$f').replace('result_', '').replace('.json', '')
    parts = fname.split('_')

    # Parse tag: quant_config_scenario_epN_cN[_dp]
    quant = parts[0] if len(parts) > 0 else '-'
    config = parts[1] if len(parts) > 1 else '-'
    scenario = parts[2] if len(parts) > 2 else '-'
    ep = parts[3].replace('ep','') if len(parts) > 3 else '-'
    conc_part = [p for p in parts if p.startswith('c') and p[1:].isdigit()]
    conc = int(conc_part[0].replace('c','')) if conc_part else 0

    # Throughput metrics
    out_tps = data.get('output_throughput', 0)
    in_tps = data.get('input_throughput', data.get('total_token_throughput', 0) - out_tps if data.get('total_token_throughput') else 0)
    total_tps = data.get('total_token_throughput', in_tps + out_tps)

    # Per-GPU metrics (SA InferenceX format)
    out_tps_gpu = out_tps / gpu_count if out_tps else 0
    total_tps_gpu = total_tps / gpu_count if total_tps else 0

    # Interactivity = tok/s/user (SA format)
    interactivity = out_tps / conc if (out_tps and conc) else 0

    # Latency metrics
    ttft_p50 = data.get('ttft_p50', data.get('median_ttft_ms', 0))
    tpot_p50 = data.get('tpot_p50', data.get('median_tpot_ms', 0))
    itl_p50 = data.get('itl_p50', data.get('median_itl_ms', 0))
    e2e_p50 = data.get('e2el_p50', data.get('median_e2el_ms', 0))

    print(f'| {config} | {quant} | {scenario} | {ep} | {conc} | {out_tps:.1f} | {out_tps_gpu:.1f} | {total_tps_gpu:.1f} | {interactivity:.2f} | {ttft_p50:.0f} | {tpot_p50:.1f} | {itl_p50:.1f} | {e2e_p50:.0f} |')
except Exception as e:
    print(f'| ERROR | - | $f | - | - | - | - | - | - | - | - | - | {e} |', file=sys.stderr)
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
log "  Target: 8×B200 GPUs"
log "============================================================"
log "  Model FP4:   ${MODEL_FP4:-<not set>}"
log "  Model FP8:   ${MODEL_FP8:-<not set>}"
log "  Configs:     $CONFIGS"
log "  Port:        $PORT"
log "  Result Dir:  $RESULT_DIR"
log "  TP:          $TP"
log "  EP Sizes:    $EP_SIZES"
log "  Scenario:    $SCENARIO_FILTER -> ${SCENARIOS[*]}"
log "  Concurrency: ${CONC_SWEEP[*]}"
log "  Warmups:     ${NUM_WARMUPS:-8}"
log "  Range Ratio: $RANDOM_RANGE_RATIO"
log "  Docker Img:  ${DOCKER_IMAGE:-<not set, set DOCKER_IMAGE env to record>}"
log "============================================================"
echo ""

run_configs() {
    case "$CONFIGS" in
        fp4-throughput)
            run_config "$MODEL_FP4" "fp4" "throughput"
            ;;
        fp4-latency)
            run_config "$MODEL_FP4" "fp4" "latency"
            ;;
        fp8-throughput)
            run_config "$MODEL_FP8" "fp8" "throughput"
            ;;
        fp8-latency)
            run_config "$MODEL_FP8" "fp8" "latency"
            ;;
        all)
            if [[ -n "$MODEL_FP4" ]]; then
                run_config "$MODEL_FP4" "fp4" "throughput"
                run_config "$MODEL_FP4" "fp4" "latency"
            fi
            if [[ -n "$MODEL_FP8" ]]; then
                run_config "$MODEL_FP8" "fp8" "throughput"
                run_config "$MODEL_FP8" "fp8" "latency"
            fi
            ;;
    esac
}

run_configs
generate_summary

log "============================================================"
log "  ALL BENCHMARKS COMPLETE"
log "  Results in: $RESULT_DIR/"
log "============================================================"
