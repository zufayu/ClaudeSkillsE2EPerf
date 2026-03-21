#!/usr/bin/env bash
# =============================================================================
# SemiAnalysis InferenceX-style Benchmark Suite for DeepSeek R1 on 8×H20
#
# Adapted from sa_bench_b200.sh for H20 GPUs (96GB HBM3e per GPU).
#
# Key H20 differences vs B200:
#   - 96 GB VRAM (vs 192 GB on B200), tighter memory constraints
#   - EP=4 typical (vs EP=8 on B200) since H20 has lower NVLink BW
#   - Lower concurrency limits for reasoning (8K output) scenarios
#   - No FP4 support (FP8 only)
#
# Configs:
#   fp8-throughput   FP8, no MTP, max throughput
#   fp8-latency      FP8, MTP-3, min latency
#   all              Run all configs
#
# Usage:
#   bash sa_bench_h20.sh --model /path/to/DeepSeek-R1-FP8 [--configs all]
#
#   # Run ONE specific point:
#   bash sa_bench_h20.sh --model /path/to/model \
#     --configs fp8-throughput --scenario chat --concurrency 64 --ep-sizes "4"
#
# Prerequisites:
#   - 8× NVIDIA H20 GPUs
#   - TensorRT-LLM with PyTorch backend
#   - DeepSeek R1 FP8 model weights
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/benchmark_lib.sh"

# ======================== Argument Parsing ====================================
MODEL=""
CONFIGS="all"
PORT=8888
RESULT_DIR="./results_h20"
RANDOM_RANGE_RATIO=0.8
EP_SIZES="4"
BENCH_SERVING_DIR=""
TP=8
SCENARIO_FILTER="all"        # all | chat | reasoning | summarize
CONC_FILTER=""               # empty = use default sweep
NUM_WARMUPS=""               # empty = auto (8)

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)            MODEL="$2"; shift 2 ;;
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
            echo "Usage: bash sa_bench_h20.sh --model <path> [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH            FP8 model path (required)"
            echo "  --configs CONFIG        fp8-throughput|fp8-latency|all (default: all)"
            echo "  --port PORT             Server port (default: 8888)"
            echo "  --result-dir DIR        Results directory (default: ./results_h20)"
            echo "  --ep-sizes \"4\"          EP sizes to sweep (default: \"4\")"
            echo "  --bench-serving-dir DIR Directory containing benchmark_serving.py"
            echo "  --tp N                  Tensor parallelism (default: 8)"
            echo ""
            echo "Filtering (run a subset):"
            echo "  --scenario SCENARIO     chat|reasoning|summarize|all (default: all)"
            echo "  --concurrency \"C1 C2\"   Concurrency values (default: 1 4 8 16 32 64)"
            echo "  --num-warmups N         Number of warmup requests (default: 8)"
            exit 0
            ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required"
    exit 1
fi

mkdir -p "$RESULT_DIR"

# ======================== ISL/OSL Test Matrix =================================
# SemiAnalysis tests: chat(1k/1k), reasoning(1k/8k), summarize(8k/1k)
# H20 with 96GB may struggle with reasoning(1k/8k) at high concurrency
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
    declare -a CONC_SWEEP=($CONC_FILTER)
else
    # H20 has less memory, cap at 64 by default (can go higher for chat)
    declare -a CONC_SWEEP=(1 4 8 16 32 64)
fi

# ======================== Utility =============================================
TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(TS)] $*"; }

# ======================== H20 Adaptive Parameters =============================
# H20 has 96GB VRAM vs B200's 192GB, so memory is tighter.
# EP=4 is more typical than EP=8 due to lower NVLink bandwidth.
#
# Arguments: $1=ISL $2=OSL $3=CONC $4=DP_ATTENTION $5=EP_SIZE $6=has_mtp
compute_adaptive_params() {
    local isl=$1 osl=$2 conc=$3 dp_attn=$4 ep_size=$5 has_mtp=$6

    # --- Defaults for H20 ---
    MOE_BACKEND="CUTLASS"
    PIECEWISE_CUDA_GRAPHS="false"
    CUDA_GRAPH_MAX_BATCH_SIZE=$conc
    KV_CACHE_FREE_MEM_FRACTION=0.8
    DELAY_BATCHING="false"
    MTP_LAYERS=0

    if [[ "$has_mtp" == "true" ]]; then
        MTP_LAYERS=3
    fi

    # H20 FP8 throughput (no MTP)
    if [[ "$has_mtp" == "false" ]]; then
        if [[ "$dp_attn" == "true" ]]; then
            MOE_BACKEND="CUTLASS"
            CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc < 4 ? conc : conc / 4 ))
        fi
        # Enable piecewise CUDA graphs for high concurrency chat
        if [[ "$isl" == "1024" && "$osl" == "1024" && $conc -ge 32 ]]; then
            PIECEWISE_CUDA_GRAPHS="true"
        fi
        # Reasoning at high conc: reduce KV fraction to avoid OOM
        if [[ "$osl" == "8192" && $conc -ge 32 ]]; then
            KV_CACHE_FREE_MEM_FRACTION=0.7
        fi
        # Summarize (8K input): needs more headroom
        if [[ "$isl" == "8192" && $conc -ge 32 ]]; then
            PIECEWISE_CUDA_GRAPHS="true"
            KV_CACHE_FREE_MEM_FRACTION=0.75
        fi

    # H20 FP8 latency (MTP)
    else
        if [[ "$dp_attn" == "true" ]]; then
            MTP_LAYERS=1
            MOE_BACKEND="CUTLASS"
            CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc < 4 ? conc : conc / 4 ))
            KV_CACHE_FREE_MEM_FRACTION=0.7
        fi
        # Piecewise for medium concurrency
        if [[ "$isl" == "1024" && "$osl" == "1024" ]]; then
            if [[ $conc -ge 16 && $conc -le 64 ]]; then
                PIECEWISE_CUDA_GRAPHS="true"
            fi
        fi
    fi
}

# ======================== Config Runner =======================================

# Run a single config point: start server, benchmark, stop server.
run_single_point() {
    local model=$1 config_name=$2 scenario_tag=$3
    local isl=$4 osl=$5 conc=$6 dp_attn=$7 ep_size=$8

    local has_mtp="false"
    [[ "$config_name" == "latency" ]] && has_mtp="true"

    compute_adaptive_params "$isl" "$osl" "$conc" "$dp_attn" "$ep_size" "$has_mtp"

    local tag="fp8_${config_name}_${scenario_tag}_ep${ep_size}_c${conc}"
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
            local docker_image="${DOCKER_IMAGE:-${TRTLLM_IMAGE:-}}"
            if [[ -z "$docker_image" && -f /.dockerenv ]]; then
                docker_image="unknown-docker"
            fi

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
#   $2 = ep_size
#   $3 = mtp_layers
collect_dar() {
    local model=$1 ep_size=$2 mtp_layers=$3

    if [[ $mtp_layers -le 0 ]]; then
        return 0
    fi

    local dar_tag="fp8_mtp${mtp_layers}_ep${ep_size}"
    log "========================================================"
    log " DAR COLLECTION: $dar_tag (trtllm-bench throughput)"
    log "========================================================"

    kill_server

    # Generate synthetic dataset (1000 prompts, ISL~1024 / OSL=1024 to match real workload)
    local dar_dataset="$RESULT_DIR/dar_dataset_${dar_tag}.jsonl"
    python3 -c "
import json
for i in range(1000):
    print(json.dumps({'task_id': i, 'prompt': 'Write a detailed story about ' + 'adventure ' * 100, 'output_tokens': 1024}))
" > "$dar_dataset"

    # Build config YAML for trtllm-bench (minimal, with MTP)
    local dar_config="$RESULT_DIR/dar_config_${dar_tag}.yml"
    local dp_attn="false"
    [[ $ep_size -gt 1 ]] && dp_attn="true"

    local moe_backend="CUTLASS"

    cat > "$dar_config" << EOF
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.7
    enable_block_reuse: false
moe_config:
    backend: $moe_backend
enable_attention_dp: $dp_attn
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: $mtp_layers
EOF

    if [[ "$dp_attn" == "true" ]]; then
        cat >> "$dar_config" << EOF
attention_dp_config:
    batching_wait_iters: 0
    enable_balance: true
    timeout_iters: 60
EOF
    fi

    local dar_report="$RESULT_DIR/dar_report_${dar_tag}.json"
    local dar_log="$RESULT_DIR/dar_bench_${dar_tag}.log"

    log "  Running trtllm-bench throughput for DAR measurement..."
    PYTHONNOUSERSITE=1 trtllm-bench \
        -m "$model" \
        throughput \
        --backend pytorch \
        --dataset "$dar_dataset" \
        --tp $TP --ep $ep_size \
        --max_seq_len 8192 \
        --num_requests 1000 \
        --warmup 2 \
        --concurrency 8 \
        --streaming \
        --kv_cache_free_gpu_mem_fraction 0.7 \
        --extra_llm_api_options "$dar_config" \
        --report_json "$dar_report" \
        > "$dar_log" 2>&1
    local rc=$?

    if [[ $rc -ne 0 ]]; then
        log "  WARN: trtllm-bench DAR collection failed (rc=$rc)"
        tail -20 "$dar_log"
        return 0
    fi

    if [[ ! -f "$dar_report" ]]; then
        log "  WARN: DAR report not generated"
        return 0
    fi

    # Parse DAR from report and inject into all matching result JSONs
    python3 - "$dar_report" "$RESULT_DIR" "$mtp_layers" "$ep_size" << 'DAREOF'
import json, sys, glob, os

report_path = sys.argv[1]
result_dir = sys.argv[2]
mtp_layers = int(sys.argv[3])
ep_size = sys.argv[4]

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

print(f"  DAR from trtllm-bench: avg={dar_avg:.4f}, p50={dar_p50:.4f}, p99={dar_p99:.4f}, acceptance_length={al_avg:.2f}")

# Find all latency result JSONs for this ep_size
pattern = os.path.join(result_dir, f"result_fp8_latency_*_ep{ep_size}_c*.json")
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

    log "  DAR collection complete for $dar_tag"
}

# ======================== Config Dispatchers ===================================

run_config() {
    local model=$1 config_name=$2

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
                # H20 memory limits: skip high conc for reasoning
                if [[ "$scenario_tag" == "reasoning" && $conc -gt 32 ]]; then
                    log "  SKIP: CONC=$conc too high for H20 reasoning (8K output)"
                    continue
                fi
                if [[ "$config_name" == "latency" && $conc -gt 64 ]]; then
                    log "  SKIP: CONC=$conc too high for latency config"
                    continue
                fi

                run_single_point "$model" "$config_name" "$scenario_tag" \
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
## H20 ${gpu_count}×GPU (96GB/GPU)

| Config | Scenario | EP | CONC | Output TPS | Out TPS/GPU | Total TPS/GPU | Interactivity | TTFT p50 (ms) | TPOT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) | DAR |
|--------|----------|-----|------|------------|-------------|---------------|---------------|---------------|---------------|--------------|--------------|-----|
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

    config = parts[1] if len(parts) > 1 else '-'
    scenario = parts[2] if len(parts) > 2 else '-'
    ep = parts[3].replace('ep','') if len(parts) > 3 and parts[3].startswith('ep') else '-'
    conc_part = [p for p in parts if p.startswith('c') and p[1:].isdigit()]
    conc = int(conc_part[0].replace('c','')) if conc_part else 0

    out_tps = data.get('output_throughput', 0)
    in_tps = data.get('input_throughput', data.get('total_token_throughput', 0) - out_tps if data.get('total_token_throughput') else 0)
    total_tps = data.get('total_token_throughput', in_tps + out_tps)

    out_tps_gpu = out_tps / gpu_count if out_tps else 0
    total_tps_gpu = total_tps / gpu_count if total_tps else 0
    interactivity = out_tps / conc if (out_tps and conc) else 0

    ttft_p50 = data.get('ttft_p50', data.get('median_ttft_ms', 0))
    tpot_p50 = data.get('tpot_p50', data.get('median_tpot_ms', 0))
    itl_p50 = data.get('itl_p50', data.get('median_itl_ms', 0))
    e2e_p50 = data.get('e2el_p50', data.get('median_e2el_ms', 0))

    mtp = '+MTP' if 'latency' in config else ''
    dp = '+DP' if 'dp' in fname else ''
    label = f'{config}{mtp}{dp}'
    dar = data.get('dar_p50')
    dar_str = f'{dar:.2%}' if dar is not None else '-'
    print(f'| {label} | {scenario} | {ep} | {conc} | {out_tps:.1f} | {out_tps_gpu:.1f} | {total_tps_gpu:.1f} | {interactivity:.2f} | {ttft_p50:.0f} | {tpot_p50:.1f} | {itl_p50:.1f} | {e2e_p50:.0f} | {dar_str} |')
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
log "  DeepSeek R1 Benchmark Suite (InferenceX-style)"
log "  Target: 8×H20 GPUs (96GB/GPU)"
log "============================================================"
log "  Model:       $MODEL"
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
        fp8-throughput)
            run_config "$MODEL" "throughput"
            ;;
        fp8-latency)
            run_config "$MODEL" "latency"
            ;;
        all)
            run_config "$MODEL" "throughput"
            run_config "$MODEL" "latency"
            ;;
        *)
            echo "ERROR: Unknown config '$CONFIGS'"
            echo "Available: fp8-throughput, fp8-latency, all"
            exit 1
            ;;
    esac
}

# Collect DAR for latency (MTP) configs after all serving benchmarks.
collect_dar_for_configs() {
    case "$CONFIGS" in
        fp8-latency|all)
            for ep_size in $EP_SIZES; do
                local mtp=3; [[ $ep_size -gt 1 ]] && mtp=1
                collect_dar "$MODEL" "$ep_size" "$mtp"
            done
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

log "============================================================"
log "  ALL BENCHMARKS COMPLETE"
log "  Results in: $RESULT_DIR/"
log "============================================================"
