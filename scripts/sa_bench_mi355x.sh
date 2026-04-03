#!/usr/bin/env bash
# =============================================================================
# ATOM Benchmark Suite for DeepSeek R1 on 8×MI355X
#
# Based on ATOM CI benchmarks (ROCm/ATOM GitHub Actions #23115155923)
# and adapted from sa_bench_b200.sh for the ROCm + ATOM stack.
#
# ATOM is a lightweight vLLM implementation from ROCm (https://github.com/ROCm/ATOM).
# Server:    python3 -m atom.entrypoints.openai_server
# Benchmark: python3 -m atom.benchmarks.benchmark_serving
#
# Configs:
#   bf16-throughput   BF16, no MTP, max throughput
#   bf16-latency      BF16, MTP-3, min latency
#   fp8-throughput    FP8, no MTP, max throughput
#   fp8-latency       FP8, MTP-3, min latency
#   mxfp4-throughput  MXFP4, no MTP, max throughput
#   mxfp4-latency     MXFP4, MTP-3, min latency
#   all               Run all configs (requires model paths)
#
# Usage:
#   # Run everything:
#   bash sa_bench_mi355x.sh --model /data/DeepSeek-R1-0528 [--configs all]
#
#   # Run ONE specific point (reproduce ATOM FP8 c=128 chat):
#   bash sa_bench_mi355x.sh \
#     --model-fp8 /data/DeepSeek-R1-0528 \
#     --configs fp8-throughput \
#     --scenario chat \
#     --concurrency 128
#
# Prerequisites:
#   - 8× AMD Instinct MI355X GPUs
#   - ROCm software stack
#   - ATOM installed: pip install -e /app/ATOM (or equivalent)
#   - DeepSeek R1 model weights (BF16 and/or FP8 via config.json)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/benchmark_lib.sh"

# ======================== Argument Parsing ====================================
MODEL=""
MODEL_FP8=""
MODEL_MXFP4=""
MODEL_MTP=""
CONFIGS="all"
RESULT_DIR="./results_mi355x"
RANDOM_RANGE_RATIO=0.8
TP=8
SCENARIO_FILTER="all"        # all | chat | reasoning | summarize
CONC_FILTER=""               # empty = use default sweep; "128" or "4 128 256" = specific values
MAX_MODEL_LEN_OVERRIDE=""    # optional: override computed max_model_len
SERVER_PORT=8000             # ATOM API server port (--server-port)
EXPERT_PARALLEL="false"      # --enable-expert-parallel for ATOM MoE

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)             MODEL="$2"; shift 2 ;;
        --model-fp8)         MODEL_FP8="$2"; shift 2 ;;
        --model-mxfp4)       MODEL_MXFP4="$2"; shift 2 ;;
        --model-mtp)         MODEL_MTP="$2"; shift 2 ;;
        --configs)           CONFIGS="$2"; shift 2 ;;
        --port)              SERVER_PORT="$2"; shift 2 ;;
        --result-dir)        RESULT_DIR="$2"; shift 2 ;;
        --tp)                TP="$2"; shift 2 ;;
        --ep)                EXPERT_PARALLEL="true"; shift 1 ;;
        --scenario)          SCENARIO_FILTER="$2"; shift 2 ;;
        --concurrency)       CONC_FILTER="$2"; shift 2 ;;
        --max-model-len)     MAX_MODEL_LEN_OVERRIDE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash sa_bench_mi355x.sh --model <path> [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH            BF16 model path"
            echo "  --model-fp8 PATH        FP8 model path (model with quantization_config in config.json)"
            echo "  --model-mxfp4 PATH      MXFP4 model path (e.g. DeepSeek-R1-0528-MXFP4)"
            echo "  --model-mtp PATH        MTP-3 model path (for latency config)"
            echo "  --configs CONFIG        bf16-throughput|bf16-latency|fp8-throughput|fp8-latency|mxfp4-throughput|mxfp4-latency|all (default: all)"
            echo "  --port PORT             ATOM API server port (default: 8000)"
            echo "  --result-dir DIR        Results directory (default: ./results_mi355x)"
            echo "  --tp N                  Tensor parallelism (default: 8)"
            echo "  --ep                    Enable expert parallelism (--enable-expert-parallel)"
            echo "  --max-model-len N       Override max model length"
            echo ""
            echo "Filtering (run a subset):"
            echo "  --scenario SCENARIO     chat|reasoning|summarize|all (default: all)"
            echo "  --concurrency \"C1 C2\"   Concurrency values to run (default: sweep 1 4 8 16 32 64 128 256)"
            echo ""
            echo "Examples:"
            echo "  # Run FP8 chat c=128 (reproduce ATOM result):"
            echo "  bash sa_bench_mi355x.sh --model-fp8 /data/DeepSeek-R1-0528 \\"
            echo "    --configs fp8-throughput --scenario chat --concurrency 128"
            echo ""
            echo "  # Run full BF16 throughput sweep:"
            echo "  bash sa_bench_mi355x.sh --model /data/DeepSeek-R1-0528 \\"
            echo "    --configs bf16-throughput"
            exit 0
            ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Validate model paths based on selected configs
case "$CONFIGS" in
    bf16-throughput|bf16-latency)
        if [[ -z "$MODEL" ]]; then
            echo "ERROR: --model is required for $CONFIGS"
            exit 1
        fi
        ;;
    fp8-throughput|fp8-latency)
        if [[ -z "$MODEL_FP8" ]]; then
            echo "ERROR: --model-fp8 is required for $CONFIGS"
            exit 1
        fi
        ;;
    mxfp4-throughput|mxfp4-latency)
        if [[ -z "$MODEL_MXFP4" ]]; then
            echo "ERROR: --model-mxfp4 is required for $CONFIGS"
            exit 1
        fi
        ;;
    all)
        if [[ -z "$MODEL" && -z "$MODEL_FP8" && -z "$MODEL_MXFP4" ]]; then
            echo "ERROR: at least one of --model, --model-fp8, or --model-mxfp4 is required"
            exit 1
        fi
        ;;
    *)
        echo "ERROR: Unknown config '$CONFIGS'"
        echo "Available: bf16-throughput, bf16-latency, fp8-throughput, fp8-latency, mxfp4-throughput, mxfp4-latency, all"
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
    declare -a CONC_SWEEP=($CONC_FILTER)
else
    declare -a CONC_SWEEP=(1 4 8 16 32 64 128 256)
fi

# ======================== Utility =============================================
TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(TS)] $*"; }

# ======================== ROCm GPU Monitoring =================================
# Override benchmark_lib.sh gpu monitor for ROCm (rocm-smi instead of nvidia-smi)

start_gpu_monitor() {
    local output="$GPU_METRICS_CSV"
    local interval=2

    while [[ $# -gt 0 ]]; do
        case $1 in
            --output)   output="$2"; shift 2 ;;
            --interval) interval="$2"; shift 2 ;;
            *)          shift ;;
        esac
    done

    GPU_METRICS_CSV="$output"

    if command -v rocm-smi &>/dev/null; then
        # rocm-smi CSV loop: timestamp, GPU index, temperature, power, VRAM used, GPU util
        (
            echo "timestamp,index,temperature.gpu,power.draw,memory.used,utilization.gpu"
            while true; do
                rocm-smi --showtemp --showpower --showmemuse --showuse --csv 2>/dev/null \
                    | tail -n +2 \
                    | while IFS=',' read -r gpu temp _ power _ mem_used _ gpu_util _rest; do
                        echo "$(date '+%Y/%m/%d %H:%M:%S'),$gpu,$temp,$power,$mem_used,$gpu_util"
                    done
                sleep "$interval"
            done
        ) > "$output" 2>/dev/null &
        GPU_MONITOR_PID=$!
        echo "[GPU Monitor] Started rocm-smi (PID=$GPU_MONITOR_PID, interval=${interval}s, output=$output)"
    elif command -v nvidia-smi &>/dev/null; then
        # Fallback to nvidia-smi (in case running on mixed setup)
        nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.current.sm,utilization.gpu,memory.used \
            --format=csv -l "$interval" > "$output" 2>/dev/null &
        GPU_MONITOR_PID=$!
        echo "[GPU Monitor] Started nvidia-smi fallback (PID=$GPU_MONITOR_PID)"
    else
        echo "[GPU Monitor] No GPU monitor found (rocm-smi / nvidia-smi), skipping"
        return 0
    fi
}

# ======================== Server Lifecycle (ATOM) =============================

# Kill any running ATOM server processes and free the port
kill_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[Server] Stopping PID=$SERVER_PID..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    # Kill ATOM server processes
    pkill -f "atom.entrypoints" 2>/dev/null || true
    # Kill anything still holding the server port (e.g. orphaned workers)
    local port_pids
    port_pids=$(lsof -ti :"$SERVER_PORT" 2>/dev/null) || true
    if [[ -n "$port_pids" ]]; then
        echo "[Server] Killing processes on port $SERVER_PORT: $port_pids"
        echo "$port_pids" | xargs kill -9 2>/dev/null || true
    fi
    sleep 3
    # Verify port is free
    if curl --output /dev/null --silent --fail "http://0.0.0.0:${SERVER_PORT}/health" 2>/dev/null; then
        echo "[Server] WARNING: Port $SERVER_PORT still responding after cleanup!"
    fi
}

# ======================== MI355X Adaptive Parameters ===========================
# MI355X with ATOM/vLLM is simpler than B200 TRT-LLM:
#   - No MOE_BACKEND / CUDA graph switching (handled by vLLM internally)
#   - No piecewise CUDA graphs or delay batching
#   - KV cache managed by vLLM's --gpu-memory-utilization
#   - MTP via --speculative-model / --num-speculative-tokens
#
# Arguments: $1=ISL $2=OSL $3=CONC $4=has_mtp $5=quant(bf16|fp8|mxfp4)
compute_mi355x_params() {
    local isl=$1 osl=$2 conc=$3 has_mtp=$4 quant=${5:-bf16}

    # --- MI355X defaults ---
    # ATOM default max_num_seqs=512, must be >= cudagraph capture sizes (default max 512)
    # Do NOT reduce below 512 unless also setting --cudagraph-capture-sizes
    GPU_MEMORY_UTILIZATION=0.90
    MTP_LAYERS=0
    ENFORCE_EAGER="false"
    MAX_NUM_SEQS=512

    if [[ "$has_mtp" == "true" ]]; then
        MTP_LAYERS=3
    fi

    # Summarize (8K input) needs more KV cache headroom
    if [[ "$isl" == "8192" ]]; then
        GPU_MEMORY_UTILIZATION=0.85
    fi

    # Reasoning (8K output) at high concurrency: reduce memory util
    if [[ "$osl" == "8192" && $conc -gt 64 ]]; then
        GPU_MEMORY_UTILIZATION=0.85
        ENFORCE_EAGER="true"
    fi

    # FP8 uses less memory per parameter, can afford slightly higher utilization
    if [[ "$quant" == "fp8" ]]; then
        GPU_MEMORY_UTILIZATION=0.92
        if [[ "$isl" == "8192" ]]; then
            GPU_MEMORY_UTILIZATION=0.88
        fi
        if [[ "$osl" == "8192" && $conc -gt 64 ]]; then
            GPU_MEMORY_UTILIZATION=0.88
            ENFORCE_EAGER="true"
        fi
    fi

    # MXFP4: MoE weights are FP4 but attention is FP8, so memory footprint
    # is between FP4 and FP8. CUDA graph capture at TP=8 is tight on MI355X.
    # Start conservative and enforce eager to skip graph capture (~1-2GB/GPU).
    if [[ "$quant" == "mxfp4" ]]; then
        GPU_MEMORY_UTILIZATION=0.90
        ENFORCE_EAGER="false"
        if [[ "$isl" == "8192" || "$osl" == "8192" ]]; then
            GPU_MEMORY_UTILIZATION=0.85
        fi
        if [[ "$osl" == "8192" && $conc -gt 32 ]]; then
            GPU_MEMORY_UTILIZATION=0.80
        fi
    fi
}

# ======================== Config Runner =======================================

# Run a single config point: start ATOM server, benchmark, stop server.
#
# Arguments:
#   $1 = model path
#   $2 = quant (bf16|fp8)
#   $3 = config_name (throughput|latency)
#   $4 = scenario_tag
#   $5 = ISL
#   $6 = OSL
#   $7 = CONC
run_single_point() {
    local model=$1 quant=$2 config_name=$3 scenario_tag=$4
    local isl=$5 osl=$6 conc=$7

    local has_mtp="false"
    [[ "$config_name" == "latency" ]] && has_mtp="true"

    compute_mi355x_params "$isl" "$osl" "$conc" "$has_mtp" "$quant"

    local tag="${quant}_${config_name}_${scenario_tag}_c${conc}"
    [[ "$has_mtp" == "true" ]] && tag="${tag}_mtp3"

    log "  ---- $tag ----"
    log "    QUANT=$quant, MTP=$MTP_LAYERS, GPU_MEM_UTIL=$GPU_MEMORY_UTILIZATION"
    log "    MAX_NUM_SEQS=$MAX_NUM_SEQS, ENFORCE_EAGER=$ENFORCE_EAGER"

    kill_server

    # --- Compute MAX_MODEL_LEN ---
    local max_model_len
    if [[ -n "$MAX_MODEL_LEN_OVERRIDE" ]]; then
        max_model_len="$MAX_MODEL_LEN_OVERRIDE"
    else
        max_model_len=$(( isl + osl + 200 ))
    fi

    # --- Start ATOM server ---
    local server_log="$RESULT_DIR/server_${tag}.log"
    local gpu_csv="$RESULT_DIR/gpu_${tag}.csv"
    local result_filename="result_${tag}"

    start_gpu_monitor --output "$gpu_csv"

    log "  Starting ATOM server: TP=$TP, QUANT=$quant, MAX_MODEL_LEN=$max_model_len"

    local serve_args=(
        python3 -m atom.entrypoints.openai_server
        --model "$model"
        --server-port "$SERVER_PORT"
        --tensor-parallel-size "$TP"
        --max-model-len "$max_model_len"
        --max-num-seqs "$MAX_NUM_SEQS"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --kv_cache_dtype fp8
    )

    # ATOM auto-detects FP8 weights from model config.json quantization_config
    # --kv_cache_dtype fp8: use FP8 KV cache (matches ATOM CI config)

    # Enforce eager mode if needed
    if [[ "$ENFORCE_EAGER" == "true" ]]; then
        serve_args+=(--enforce-eager)
    fi

    # MTP/speculative decoding
    if [[ $MTP_LAYERS -gt 0 ]]; then
        serve_args+=(
            --method mtp
            --num-speculative-tokens "$MTP_LAYERS"
        )
    fi

    # Expert parallelism for MoE models
    if [[ "$EXPERT_PARALLEL" == "true" ]]; then
        serve_args+=(--enable-expert-parallel)
    fi

    # ROCm-specific env vars
    export NCCL_SOCKET_IFNAME=lo
    export VLLM_USE_TRITON_FLASH_ATTN=0

    "${serve_args[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    if wait_for_server_ready --port "$SERVER_PORT" --server-log "$server_log" --server-pid "$SERVER_PID"; then
        # Auto-detect served model name (ATOM may register a different name
        # from config.json _name_or_path, e.g. MXFP4 model registers as
        # "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8")
        local served_model
        served_model=$(curl -s "http://0.0.0.0:${SERVER_PORT}/v1/models" \
            | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null) \
            || served_model="$model"
        if [[ "$served_model" != "$model" ]]; then
            log "  NOTE: Served model name differs from --model path"
            log "    --model:        $model"
            log "    served as:      $served_model"
        fi

        local num_prompts=$(( conc * 10 ))
        [[ $num_prompts -lt 20 ]] && num_prompts=20
        local num_warmups=$(( conc * 2 ))

        log "  Running benchmark: ISL=$isl, OSL=$osl, CONC=$conc, NUM_PROMPTS=$num_prompts, WARMUPS=$num_warmups"

        # Capture ATOM and aiter versions
        local atom_version aiter_version
        atom_version=$(python3 -c "import atom; print(atom.__version__)" 2>/dev/null || echo "unknown")
        aiter_version=$(python3 -c "import aiter; print(aiter.__version__)" 2>/dev/null || echo "unknown")

        local bench_cmd=(
            python3 -m atom.benchmarks.benchmark_serving
            --model "$served_model"
            --backend vllm
            --base-url "http://0.0.0.0:${SERVER_PORT}"
            --dataset-name random
            --random-input-len "$isl"
            --random-output-len "$osl"
            --random-range-ratio "$RANDOM_RANGE_RATIO"
            --num-prompts "$num_prompts"
            --max-concurrency "$conc"
            --num-warmups "$num_warmups"
            --request-rate inf
            --ignore-eos
            --save-result
            --percentile-metrics "ttft,tpot,itl,e2el"
            --result-dir "$RESULT_DIR"
            --result-filename "${result_filename}.json"
            --metadata
                "max_model_len=$max_model_len"
                "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
                "enforce_eager=$ENFORCE_EAGER"
                "kv_cache_dtype=fp8"
                "tensor_parallel_size=$TP"
                "expert_parallel=$EXPERT_PARALLEL"
                "max_num_seqs=$MAX_NUM_SEQS"
                "mtp_layers=$MTP_LAYERS"
                "random_range_ratio=$RANDOM_RANGE_RATIO"
                "atom_version=$atom_version"
                "aiter_version=$aiter_version"
        )

        set -x
        "${bench_cmd[@]}" || log "  WARN: Benchmark failed for $tag"
        set +x
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

        for conc in "${CONC_SWEEP[@]}"; do
            # Reasoning scenario: ATOM CI only ran up to c=32
            if [[ "$scenario_tag" == "reasoning" && $conc -gt 32 ]]; then
                log "  SKIP: CONC=$conc exceeds MI355X reasoning limit (max c=32)"
                continue
            fi

            # For latency config with MTP, limit concurrency to avoid OOM
            if [[ "$config_name" == "latency" && $conc -gt 128 ]]; then
                log "  SKIP: CONC=$conc too high for MI355X MTP latency config"
                continue
            fi

            run_single_point "$model" "$quant" "$config_name" "$scenario_tag" \
                "$isl" "$osl" "$conc"
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

    cat > "$summary_file" << 'HEADER'
# DeepSeek R1 Benchmark Results (ATOM/vLLM)
## MI355X 8×GPU

| Config | Quant | Scenario | CONC | Total Tput | Output Tput | Interac. | TPOT (ms) | TTFT (ms) | DAR |
|--------|-------|----------|------|------------|-------------|----------|-----------|-----------|-----|
HEADER

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

    # Parse tag: quant_config_scenario_cN[_mtp3]
    quant = parts[0] if len(parts) > 0 else '-'
    config = parts[1] if len(parts) > 1 else '-'
    scenario = parts[2] if len(parts) > 2 else '-'
    conc_part = [p for p in parts if p.startswith('c') and p[1:].isdigit()]
    conc = int(conc_part[0].replace('c','')) if conc_part else 0
    mtp = '+MTP3' if 'mtp3' in parts else ''

    out_tps = data.get('output_throughput', 0)
    in_tps = data.get('input_throughput', 0)
    total_tps = data.get('total_token_throughput', in_tps + out_tps)

    ttft_p50 = data.get('ttft_p50', data.get('median_ttft_ms', 0))
    tpot_p50 = data.get('tpot_p50', data.get('median_tpot_ms', 0))
    # Interactivity = 1000/TPOT tok/s/user (SA InferenceX format)
    interactivity = 1000.0 / tpot_p50 if tpot_p50 > 0 else 0

    label = f'{config}{mtp}'
    dar = data.get('dar_p50')
    dar_str = f'{dar:.2%}' if dar is not None else '-'
    print(f'| {label} | {quant.upper()} | {scenario} | {conc} | {total_tps:.1f} | {out_tps:.1f} | {interactivity:.2f} | {tpot_p50:.1f} | {ttft_p50:.1f} | {dar_str} |')
except Exception as e:
    print(f'| ERROR | - | $f | - | - | - | - | - | - | {e} |', file=sys.stderr)
" >> "$summary_file" 2>/dev/null || true
    done

    log "Summary written to: $summary_file"
    echo ""
    cat "$summary_file"
}

# ======================== Main Execution ======================================

trap 'kill_server; stop_gpu_monitor 2>/dev/null; exit' INT TERM

log "============================================================"
log "  DeepSeek R1 Benchmark Suite (ATOM)"
log "  Target: 8×MI355X GPUs (ROCm)"
log "============================================================"
log "  Model BF16:  ${MODEL:-<not set>}"
log "  Model FP8:   ${MODEL_FP8:-<not set>}"
log "  Model MXFP4: ${MODEL_MXFP4:-<not set>}"
log "  Model MTP:   ${MODEL_MTP:-<not set>}"
log "  Configs:     $CONFIGS"
log "  Port:        $SERVER_PORT"
log "  Result Dir:  $RESULT_DIR"
log "  TP:          $TP"
log "  Scenario:    $SCENARIO_FILTER -> ${SCENARIOS[*]}"
log "  Concurrency: ${CONC_SWEEP[*]}"
log "  Range Ratio: $RANDOM_RANGE_RATIO"
log "============================================================"
echo ""

run_configs() {
    case "$CONFIGS" in
        bf16-throughput)
            run_config "$MODEL" "bf16" "throughput"
            ;;
        bf16-latency)
            run_config "$MODEL" "bf16" "latency"
            ;;
        fp8-throughput)
            run_config "$MODEL_FP8" "fp8" "throughput"
            ;;
        fp8-latency)
            run_config "$MODEL_FP8" "fp8" "latency"
            ;;
        mxfp4-throughput)
            run_config "$MODEL_MXFP4" "mxfp4" "throughput"
            ;;
        mxfp4-latency)
            run_config "$MODEL_MXFP4" "mxfp4" "latency"
            ;;
        all)
            if [[ -n "$MODEL" ]]; then
                run_config "$MODEL" "bf16" "throughput"
                if [[ -n "$MODEL_MTP" ]]; then
                    run_config "$MODEL" "bf16" "latency"
                else
                    log "SKIP: bf16-latency requires --model-mtp, skipping"
                fi
            fi
            if [[ -n "$MODEL_FP8" ]]; then
                run_config "$MODEL_FP8" "fp8" "throughput"
                if [[ -n "$MODEL_MTP" ]]; then
                    run_config "$MODEL_FP8" "fp8" "latency"
                else
                    log "SKIP: fp8-latency requires --model-mtp, skipping"
                fi
            fi
            if [[ -n "$MODEL_MXFP4" ]]; then
                run_config "$MODEL_MXFP4" "mxfp4" "throughput"
                if [[ -n "$MODEL_MTP" ]]; then
                    run_config "$MODEL_MXFP4" "mxfp4" "latency"
                else
                    log "SKIP: mxfp4-latency requires --model-mtp, skipping"
                fi
            fi
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
