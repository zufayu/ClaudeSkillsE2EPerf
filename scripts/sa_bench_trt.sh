#!/usr/bin/env bash
# =============================================================================
# Unified TRT-LLM Benchmark Suite for DeepSeek R1
#
# Supports all NVIDIA GPU platforms (B200, H20, H200, etc.)
# Platform-specific adaptive params loaded from configs/adaptive/{platform}_trt.sh
#
# Replaces: sa_bench_b200.sh, sa_bench_h20.sh, sa_bench_h200.sh
# (Hermes pattern: one script per framework, hardware as config)
#
# Usage:
#   bash sa_bench_trt.sh --platform b200 --model-fp4 /data/model --configs fp4-throughput
#   bash sa_bench_trt.sh --platform h20  --model-fp8 /data/model --configs fp8-throughput
#   bash sa_bench_trt.sh --platform h200 --model-fp8 /data/model --configs fp8-latency
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/benchmark_lib.sh"
source "$SCRIPT_DIR/result_dir_validate.sh"

# ======================== Argument Parsing ====================================
MODEL_FP4=""
MODEL_FP8=""
CONFIGS="all"
PORT=8888
PLATFORM="b200"
RESULT_DIR="./results_trt"
RANDOM_RANGE_RATIO=0.8
EP_SIZES="1 8"
BENCH_SERVING_DIR=""
TP=8
SCENARIO_FILTER="all"        # all | chat | reasoning | summarize
CONC_FILTER=""               # empty = use default sweep; "256" or "4 128 256" = specific values
DP_OVERRIDE=""               # empty = auto (EP>1 → DP=true); "true" or "false" = force
NUM_WARMUPS=""                # empty = auto (SA default: conc*2); or explicit number
DAR_NUM_REQUESTS=200          # number of requests for DAR measurement
DAR_CONCURRENCY=32            # concurrency for DAR measurement
DAR_WARMUP=5                  # warmup requests for DAR measurement
GPUS=""                        # empty = all GPUs; "0,1,2,3" = specific GPUs
CONTAINER_IMAGE=""             # original docker image name (passed from host)
FWBRINGUP_MODE=false           # true = fw-bringup methodology (no warmup, no num_prompts floor)

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-fp4)        MODEL_FP4="$2"; shift 2 ;;
        --model-fp8)        MODEL_FP8="$2"; shift 2 ;;
        --platform)         PLATFORM="$2"; shift 2 ;;
        --configs)          CONFIGS="$2"; shift 2 ;;
        --port)             PORT="$2"; shift 2 ;;
        --result-dir)       RESULT_DIR="$2"; shift 2 ;;
        --ep-sizes)         EP_SIZES="$2"; shift 2 ;;
        --bench-serving-dir) BENCH_SERVING_DIR="$2"; shift 2 ;;
        --tp)               TP="$2"; shift 2 ;;
        --scenario)         SCENARIO_FILTER="$2"; shift 2 ;;
        --concurrency)      CONC_FILTER="$2"; shift 2 ;;
        --num-warmups)      NUM_WARMUPS="$2"; shift 2 ;;
        --dar-num-requests) DAR_NUM_REQUESTS="$2"; shift 2 ;;
        --dar-concurrency)  DAR_CONCURRENCY="$2"; shift 2 ;;
        --dar-warmup)       DAR_WARMUP="$2"; shift 2 ;;
        --dp)               DP_OVERRIDE="$2"; shift 2 ;;
        --gpus)             GPUS="$2"; shift 2 ;;
        --container-image)  CONTAINER_IMAGE="$2"; shift 2 ;;
        --fwbringup)        FWBRINGUP_MODE=true; shift ;;
        -h|--help)
            echo "Usage: bash sa_bench_trt.sh --platform <gpu> --model-fp4 <path> [options]"
            echo ""
            echo "Options:"
            echo "  --platform PLATFORM     GPU platform: b200|h20|h200 (default: b200)"
            echo "  --model-fp4 PATH        FP4 model path"
            echo "  --model-fp8 PATH        FP8 model path"
            echo "  --configs CONFIG        fp4-throughput|fp4-latency|fp8-throughput|fp8-latency|all (default: all)"
            echo "  --port PORT             Server port (default: 8888)"
            echo "  --result-dir DIR        Results directory (default: ./results_trt)"
            echo "  --ep-sizes \"1 8\"        EP sizes to sweep (default: \"1 8\")"
            echo "  --bench-serving-dir DIR Directory containing benchmark_serving.py"
            echo "  --tp N                  Tensor parallelism (default: 8)"
            echo ""
            echo "Filtering (run a subset):"
            echo "  --scenario SCENARIO     chat|reasoning|summarize|all (default: all)"
            echo "  --concurrency \"C1 C2\"   Concurrency values to run (default: sweep 1 4 8 16 32 64 128 256)"
            echo "  --num-warmups N         Number of warmup requests (default: conc*2, matching SA)"
            echo "  --fwbringup             Use fw-bringup methodology (no warmup, no num_prompts floor) for cross-comparison with MI450 perf-tracking"
            echo "  --dp true|false         Force DP attention on/off (default: auto, EP>1 → true)"
            echo "  --gpus \"0,1,2,3\"        CUDA_VISIBLE_DEVICES — use subset of GPUs (default: all)"
            echo ""
            echo "DAR collection (for latency/MTP configs):"
            echo "  --dar-num-requests N    Number of requests for DAR measurement (default: 200)"
            echo "  --dar-concurrency N     Concurrency for DAR trtllm-bench run (default: 32)"
            echo "  --dar-warmup N          Warmup requests for DAR measurement (default: 5)"
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

validate_result_dir "$RESULT_DIR" || exit 1
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
# log() and TS() inherited from benchmark_lib.sh (no inline definition needed)

# ======================== Platform-Adaptive Parameters ========================
# Load compute_adaptive_params() from configs/adaptive/{platform}_trt.sh
# Each platform has its own adaptive tuning rules (MoE backend, CUDA graphs, etc.)
# Adding a new platform = create configs/adaptive/{platform}_trt.sh (two-file pattern)

ADAPTIVE_FILE="$SCRIPT_DIR/../configs/adaptive/${PLATFORM}_trt.sh"
if [[ -f "$ADAPTIVE_FILE" ]]; then
    source "$ADAPTIVE_FILE"
    log "Loaded adaptive config: ${PLATFORM}_trt.sh"
else
    # Hard-fail: silent fallback to default_trt.sh has historically swapped
    # MoE backend out from under users (B300 2026-04-29: 21-41% perf loss
    # from CUTLASS vs TRTLLM). Force an explicit choice.
    echo "ERROR: No adaptive config for platform '$PLATFORM'." >&2
    echo "  Expected: $ADAPTIVE_FILE" >&2
    echo "  Available: $(ls "$SCRIPT_DIR/../configs/adaptive/" 2>/dev/null | grep _trt.sh | tr '\n' ' ')" >&2
    echo "  Either add configs/adaptive/${PLATFORM}_trt.sh, or pass --platform <known>." >&2
    exit 1
fi

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
    if [[ -n "${ENABLE_CONFIGURABLE_MOE_FLAG:-}" ]]; then
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
        if [[ "$FWBRINGUP_MODE" != "true" ]]; then
            [[ $num_prompts -lt 20 ]] && num_prompts=20
        fi

        # SA InferenceX: --num-warmups "$((2 * max_concurrency))"; fwbringup: 0 (no warmup)
        local default_warmups=$(( conc * 2 ))
        [[ "$FWBRINGUP_MODE" == "true" ]] && default_warmups=0
        local warmups="${NUM_WARMUPS:-$default_warmups}"

        # Capture software version
        local trtllm_version
        trtllm_version=$(python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)" 2>/dev/null || echo "unknown")

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
                "gpu_count=$GPU_COUNT"
                "ep_size=$ep_size"
                "dp_attention=$dp_attn"
                "moe_backend=$MOE_BACKEND"
                "mtp_layers=$MTP_LAYERS"
                "piecewise_cuda_graphs=$PIECEWISE_CUDA_GRAPHS"
                "random_range_ratio=$RANDOM_RANGE_RATIO"
                "trtllm_version=$trtllm_version"
                "container_image=$CONTAINER_IMAGE"
                "bench_methodology=$([[ $FWBRINGUP_MODE == true ]] && echo fwbringup || echo claudeskills)"
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
            # Detect docker image from --container-image arg, env, or label
            local docker_image="${CONTAINER_IMAGE:-${DOCKER_IMAGE:-${TRTLLM_IMAGE:-}}}"
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
import json, os

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

# ======================== DAR Collection via trtllm-bench =====================
# Collect Draft Acceptance Rate by running trtllm-bench throughput with the
# same MTP config. This uses the LLM API directly (not HTTP server), which
# gives access to decoding_iter for DAR calculation.
#
# Arguments:
#   $1 = model path
#   $2 = quant (fp4|fp8)
#   $3 = ep_size
#   $4 = mtp_layers
collect_dar() {
    local model=$1 quant=$2 ep_size=$3 mtp_layers=$4

    if [[ $mtp_layers -le 0 ]]; then
        return 0
    fi

    local dp_attn="false"
    [[ $ep_size -gt 1 ]] && dp_attn="true"

    # Use same MOE backend logic
    local moe_backend="TRTLLM"
    if [[ "$quant" == "fp4" ]]; then
        [[ "$dp_attn" == "true" ]] && moe_backend="CUTLASS"
    elif [[ "$quant" == "fp8" ]]; then
        [[ "$dp_attn" == "true" ]] && moe_backend="DEEPGEMM"
    fi

    log "========================================================"
    log " DAR COLLECTION: ${quant} mtp=${mtp_layers} ep=${ep_size}"
    log "   concurrency=$DAR_CONCURRENCY  requests=$DAR_NUM_REQUESTS  warmup=$DAR_WARMUP"
    log "========================================================"

    kill_server

    # Loop over scenarios (respects --scenario filter)
    for scenario_entry in "${SCENARIOS[@]}"; do
        IFS=':' read -r isl osl scenario_tag <<< "$scenario_entry"
        local dar_tag="${quant}_mtp${mtp_layers}_ep${ep_size}_${scenario_tag}"

        log "  ---- DAR: $scenario_tag (ISL=$isl, OSL=$osl) ----"

        # --- Generate dataset with random tokens (avoids inflated DAR from repeated text) ---
        local dar_dataset="$RESULT_DIR/dar_dataset_${dar_tag}.jsonl"
        python3 "$SCRIPT_DIR/gen_dataset.py" \
            --tokenizer "$model" \
            --fixed_input_len "$isl" \
            --output_tokens "$osl" \
            --num_requests "$DAR_NUM_REQUESTS" \
            --input_mode random \
            --output "$dar_dataset"

        # --- Build config YAML ---
        local dar_config="$RESULT_DIR/dar_config_${dar_tag}.yml"
        local dar_max_seq_len=10240
        if [[ $((isl + osl)) -le 2048 ]]; then
            dar_max_seq_len=8192
        fi

        cat > "$dar_config" << EOF
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.8
    enable_block_reuse: false
moe_config:
    backend: $moe_backend
enable_attention_dp: $dp_attn
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: $mtp_layers
cuda_graph_config:
    enable_padding: true
    max_batch_size: $DAR_CONCURRENCY
EOF

        # Piecewise CUDA graphs
        local dar_max_num_tokens=$(( ((mtp_layers + 1) * DAR_CONCURRENCY + isl + 64 + 63) / 64 * 64 ))
        [[ $dar_max_num_tokens -lt 8192 ]] && dar_max_num_tokens=8192
        local dar_capture_tokens=(1 2 4 8 16 32 64 128)
        dar_capture_tokens+=( $(seq 256 256 $dar_max_num_tokens) )
        if [[ $((dar_max_num_tokens % 256)) -ne 0 ]]; then
            dar_capture_tokens+=($dar_max_num_tokens)
        fi
        local dar_capture_list
        dar_capture_list=$(printf "%s, " "${dar_capture_tokens[@]}")
        cat >> "$dar_config" << EOF
torch_compile_config:
    capture_num_tokens: [${dar_capture_list%, }]
    enable_piecewise_cuda_graph: true
EOF

        if [[ "$dp_attn" == "true" ]]; then
            cat >> "$dar_config" << EOF
attention_dp_config:
    batching_wait_iters: 0
    enable_balance: true
    timeout_iters: 60
EOF
        fi

        # --- Run trtllm-bench ---
        local dar_report="$RESULT_DIR/dar_report_${dar_tag}.json"
        local dar_log="$RESULT_DIR/dar_bench_${dar_tag}.log"

        log "  Running trtllm-bench throughput for DAR ($scenario_tag)..."
        trtllm-bench --model "$model" \
            --model_path "$model" \
            throughput \
            --backend pytorch \
            --extra_llm_api_options "$dar_config" \
            --max_seq_len "$dar_max_seq_len" \
            --tp $TP \
            --ep $ep_size \
            --dataset "$dar_dataset" \
            --concurrency $DAR_CONCURRENCY \
            --num_requests "$DAR_NUM_REQUESTS" \
            --report_json "$dar_report" \
            --warmup "$DAR_WARMUP" \
            > "$dar_log" 2>&1
        local rc=$?

        if [[ $rc -ne 0 ]]; then
            log "  WARN: trtllm-bench DAR failed for $scenario_tag (rc=$rc)"
            tail -20 "$dar_log"
            continue
        fi

        if [[ ! -f "$dar_report" ]]; then
            log "  WARN: DAR report not generated for $scenario_tag"
            continue
        fi

        # --- Parse DAR and inject into matching result JSONs ---
        python3 - "$dar_report" "$RESULT_DIR" "$quant" "$mtp_layers" "$ep_size" "$scenario_tag" << 'DAREOF'
import json, sys, glob, os

report_path = sys.argv[1]
result_dir = sys.argv[2]
quant = sys.argv[3]
mtp_layers = int(sys.argv[4])
ep_size = sys.argv[5]
scenario_tag = sys.argv[6]

with open(report_path) as f:
    report = json.load(f)

decoding = report.get("decoding_stats")
if not decoding:
    print("  WARN: No decoding_stats in trtllm-bench report")
    sys.exit(0)

dar_pct = decoding.get("draft_acceptance_rate_percentiles")
al_pct = decoding.get("acceptance_length_percentiles")
if not dar_pct:
    print("  WARN: No draft_acceptance_rate_percentiles in report")
    sys.exit(0)

dar_avg = dar_pct.get("average", 0)
dar_p50 = dar_pct.get("p50", 0)
dar_p99 = dar_pct.get("p99", 0)
al_avg = al_pct.get("average", 0) if al_pct else 0

print(f"  DAR [{scenario_tag}]: avg={dar_avg:.4f}, p50={dar_p50:.4f}, p99={dar_p99:.4f}, acceptance_length={al_avg:.2f}")

# Inject into result JSONs matching this specific scenario
# Pattern: result_{quant}_latency_{scenario}_ep{ep_size}_c*.json
pattern = os.path.join(result_dir, f"result_{quant}_latency_{scenario_tag}_ep{ep_size}_c*.json")
matches = glob.glob(pattern)

if not matches:
    print(f"  WARN: No result files matched pattern {pattern}")
    sys.exit(0)

injected = 0
for result_path in matches:
    try:
        with open(result_path) as f:
            data = json.load(f)
        data["dar_avg"] = round(dar_avg, 4)
        data["dar_p50"] = round(dar_p50, 4)
        data["dar_p99"] = round(dar_p99, 4)
        data["dar_acceptance_length"] = round(al_avg, 2)
        data["dar_source"] = "trtllm-bench"
        with open(result_path, "w") as f:
            json.dump(data, f, indent=2)
        injected += 1
    except Exception as e:
        print(f"  WARN: Failed to inject DAR into {result_path}: {e}")

print(f"  Injected DAR into {injected}/{len(matches)} result files")
DAREOF

        log "  DAR complete for $scenario_tag"
    done

    log "  DAR collection complete for ${quant} ep=${ep_size}"
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
            if [[ -n "$DP_OVERRIDE" ]]; then
                dp_attn="$DP_OVERRIDE"
            elif [[ $ep_size -gt 1 ]]; then
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
    local gpu_count="$GPU_COUNT"

    cat > "$summary_file" <<EOF
# DeepSeek R1 Benchmark Results (InferenceX-style)
## B200 ${gpu_count}×GPU

| Config | Quant | Scenario | EP | CONC | Total Tput | Output Tput | Interac. | TPOT (ms) | TTFT (ms) | DAR |
|--------|-------|----------|-----|------|------------|-------------|----------|-----------|-----------|-----|
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

    # Latency metrics
    ttft_p50 = data.get('ttft_p50', data.get('median_ttft_ms', 0))
    tpot_p50 = data.get('tpot_p50', data.get('median_tpot_ms', 0))

    # Interactivity = 1000/TPOT tok/s/user (SA InferenceX format)
    interactivity = 1000.0 / tpot_p50 if tpot_p50 > 0 else 0

    dar = data.get('dar_p50')
    dar_str = f'{dar:.2%}' if dar is not None else '-'
    print(f'| {config} | {quant} | {scenario} | {ep} | {conc} | {total_tps:.1f} | {out_tps:.1f} | {interactivity:.2f} | {tpot_p50:.1f} | {ttft_p50:.0f} | {dar_str} |')
except Exception as e:
    print(f'| ERROR | - | $f | - | - | - | - | - | - | - | {e} |', file=sys.stderr)
" >> "$summary_file" 2>/dev/null || true
    done

    log "Summary written to: $summary_file"
    echo ""
    cat "$summary_file"
}

# ======================== Main Execution ======================================

# ======================== GPU Selection =======================================
if [[ -n "$GPUS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
    GPU_COUNT=$(echo "$GPUS" | tr ',' '\n' | wc -l)
else
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    [[ $GPU_COUNT -eq 0 ]] && GPU_COUNT=$TP
fi

trap 'kill_server; stop_gpu_monitor 2>/dev/null; exit' INT TERM

log "============================================================"
log "  DeepSeek R1 Benchmark Suite (InferenceX/MAX-style)"
log "  Target: ${GPU_COUNT}×B200 GPUs"
log "============================================================"
log "  Model FP4:   ${MODEL_FP4:-<not set>}"
log "  Model FP8:   ${MODEL_FP8:-<not set>}"
log "  GPUs:        ${GPUS:-all} (${GPU_COUNT} GPUs)"
log "  Configs:     $CONFIGS"
log "  Port:        $PORT"
log "  Result Dir:  $RESULT_DIR"
log "  TP:          $TP"
log "  EP Sizes:    $EP_SIZES"
log "  Scenario:    $SCENARIO_FILTER -> ${SCENARIOS[*]}"
log "  Concurrency: ${CONC_SWEEP[*]}"
log "  Warmups:     ${NUM_WARMUPS:-conc*2}"
log "  Range Ratio: $RANDOM_RANGE_RATIO"
log "  Container:   ${CONTAINER_IMAGE:-<not set, use --container-image>}"
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

# Collect DAR for latency (MTP) configs after all serving benchmarks.
# Runs trtllm-bench throughput once per quant×EP combo to measure DAR.
collect_dar_for_configs() {
    case "$CONFIGS" in
        fp4-latency)
            for ep_size in $EP_SIZES; do
                local mtp=3; [[ $ep_size -gt 1 ]] && mtp=1
                collect_dar "$MODEL_FP4" "fp4" "$ep_size" "$mtp"
            done
            ;;
        fp8-latency)
            for ep_size in $EP_SIZES; do
                local mtp=3; [[ $ep_size -gt 1 ]] && mtp=1
                collect_dar "$MODEL_FP8" "fp8" "$ep_size" "$mtp"
            done
            ;;
        all)
            if [[ -n "$MODEL_FP4" ]]; then
                for ep_size in $EP_SIZES; do
                    local mtp=3; [[ $ep_size -gt 1 ]] && mtp=1
                    collect_dar "$MODEL_FP4" "fp4" "$ep_size" "$mtp"
                done
            fi
            if [[ -n "$MODEL_FP8" ]]; then
                for ep_size in $EP_SIZES; do
                    local mtp=3; [[ $ep_size -gt 1 ]] && mtp=1
                    collect_dar "$MODEL_FP8" "fp8" "$ep_size" "$mtp"
                done
            fi
            ;;
        # throughput configs have no MTP, skip DAR
    esac
}

run_configs
collect_dar_for_configs
generate_summary

# Trim server logs for repo storage
log "Trimming server logs..."
python3 "$SCRIPT_DIR/trim_logs.py" "$RESULT_DIR"

# Regression comparison with previous commit
log "Running regression comparison..."
python3 "$SCRIPT_DIR/compare_results.py" "$RESULT_DIR" \
    --output "$RESULT_DIR/regression_report.txt" || true

log "============================================================"
log "  ALL BENCHMARKS COMPLETE"
log "  Results in: $RESULT_DIR/"
log "============================================================"
