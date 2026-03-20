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

    # Enable perf metrics for DAR collection when MTP is active
    if [[ $MTP_LAYERS -gt 0 ]]; then
        local perf_metrics_max=$(( conc * 10 + 100 ))
        cat >> "$config_file" << EOF
return_perf_metrics: true
perf_metrics_max_requests: $perf_metrics_max
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

    # Set TRTLLM_KVCACHE_TIME_OUTPUT_PATH to enable per-request perf metrics
    # (needed for DAR collection via /perf_metrics endpoint on completions API)
    TRTLLM_KVCACHE_TIME_OUTPUT_PATH=/tmp/dar_metrics \
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

        run_benchmark_serving "${bench_args[@]}" || \
            log "  WARN: Benchmark failed for $tag"

        # --- Collect DAR (Draft Acceptance Rate) for MTP configs ---
        if [[ $MTP_LAYERS -gt 0 ]]; then
            log "  Collecting DAR from /perf_metrics..."
            local dar_file="$RESULT_DIR/perf_metrics_${tag}.json"
            local dar_result_path="$RESULT_DIR/${result_filename}.json"
            curl -s "http://0.0.0.0:${PORT}/perf_metrics" > "$dar_file" 2>/dev/null || echo "[]" > "$dar_file"
            log "  DAR response saved to $dar_file ($(wc -c < "$dar_file") bytes)"

            python3 - "$dar_file" "$dar_result_path" << 'DAREOF'
import json, sys

dar_file = sys.argv[1]
result_path = sys.argv[2]

try:
    with open(dar_file) as f:
        metrics = json.load(f)
except (json.JSONDecodeError, FileNotFoundError) as e:
    print(f"  WARN: Failed to read /perf_metrics: {e}")
    sys.exit(0)

print(f"  /perf_metrics returned {len(metrics)} items")

# Extract per-request acceptance rates
acceptance_rates = []
total_accepted = 0
total_draft = 0
for item in metrics:
    pm = item.get("perf_metrics", {})
    if pm is None:
        continue
    sd = pm.get("speculative_decoding", {})
    if sd and sd.get("total_draft_tokens", 0) > 0:
        acceptance_rates.append(sd["acceptance_rate"])
        total_accepted += sd["total_accepted_draft_tokens"]
        total_draft += sd["total_draft_tokens"]

if not acceptance_rates:
    # Debug: show first item structure
    if metrics:
        pm = metrics[0].get("perf_metrics")
        print(f"  DEBUG: first item perf_metrics type={type(pm).__name__}, keys={list(pm.keys()) if isinstance(pm, dict) else 'N/A'}")
    print("  WARN: No speculative decoding metrics found in /perf_metrics")
    sys.exit(0)

# Compute statistics
acceptance_rates.sort()
n = len(acceptance_rates)
dar_mean = sum(acceptance_rates) / n
dar_p50 = acceptance_rates[n // 2]
dar_p99 = acceptance_rates[int(n * 0.99)]
dar_overall = total_accepted / total_draft if total_draft > 0 else 0

print(f"  DAR: mean={dar_mean:.4f}, p50={dar_p50:.4f}, p99={dar_p99:.4f}, "
      f"overall={dar_overall:.4f} ({total_accepted}/{total_draft} tokens, {n} requests)")

# Inject into result JSON
try:
    with open(result_path) as f:
        data = json.load(f)
    data["dar_mean"] = round(dar_mean, 4)
    data["dar_p50"] = round(dar_p50, 4)
    data["dar_p99"] = round(dar_p99, 4)
    data["dar_overall"] = round(dar_overall, 4)
    data["dar_total_accepted_tokens"] = total_accepted
    data["dar_total_draft_tokens"] = total_draft
    data["dar_num_requests"] = n
    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Injected DAR into {result_path}")
except Exception as e:
    print(f"  WARN: Failed to inject DAR: {e}")
DAREOF
        fi

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

run_configs
generate_summary

# Trim server logs for repo storage
log "Trimming server logs..."
python3 "$SCRIPT_DIR/trim_logs.py" "$RESULT_DIR"

log "============================================================"
log "  ALL BENCHMARKS COMPLETE"
log "  Results in: $RESULT_DIR/"
log "============================================================"
