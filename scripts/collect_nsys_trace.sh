#!/usr/bin/env bash
# =============================================================================
# Nsight Systems Trace Capture for TRT-LLM Inference
#
# Captures GPU kernel timelines for kernel-level performance analysis.
# Supports two modes:
#   - bench: trtllm-bench offline (simpler, recommended for kernel analysis)
#   - serve: trtllm-serve + benchmark_serving (for scheduler/batching analysis)
#
# Usage:
#   bash scripts/collect_nsys_trace.sh \
#     --model /home/models/models--DeepSeek-R1-0528 \
#     --mode bench --scenario chat --concurrency 32 \
#     --quant fp8 --config latency \
#     --tp 8 --ep 1 --iter-range 100-150
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/benchmark_lib.sh"

# ======================== Defaults ============================================
MODEL=""
MODE="bench"
SCENARIO="chat"
CONCURRENCY=32
QUANT="fp8"
CONFIG="latency"
TP=8
EP=1
TRACE_DIR="./traces"
ITER_RANGE="100-150"
NUM_REQUESTS=200
WARMUP=0
PORT=8888
SKIP_EXPORT=false
EXTRA_NSYS_ARGS=""

# ======================== Argument Parsing ====================================
usage() {
    cat <<EOF
Usage: bash $(basename "$0") [options]

Nsight Systems trace capture for TRT-LLM kernel-level profiling.

Required:
  --model PATH              Model path (e.g., /home/models/models--DeepSeek-R1-0528)

Options:
  --mode MODE               bench (trtllm-bench) or serve (trtllm-serve) [default: bench]
  --scenario SCENARIO       chat (1K/1K), reasoning (1K/8K), summarize (8K/1K) [default: chat]
  --concurrency N           Request concurrency [default: 32]
  --quant QUANT             Quantization: fp4, fp8 [default: fp8]
  --config CONFIG           latency (MTP3) or throughput (MTP0) [default: latency]
  --tp N                    Tensor parallelism [default: 8]
  --ep N                    Expert parallelism [default: 1]
  --trace-dir DIR           Output directory for traces [default: ./traces]
  --iter-range START-STOP   TLLM_PROFILE_START_STOP range [default: 100-150]
  --num-requests N          Number of requests (bench mode) [default: 200]
  --warmup N                Warmup requests (bench mode) [default: 0]
  --port PORT               Server port (serve mode) [default: 8888]
  --skip-export             Skip post-processing (just capture .nsys-rep)
  --extra-nsys-args ARGS    Additional nsys profile arguments
  -h, --help                Show this help message

Examples:
  # trtllm-bench mode (recommended for kernel analysis)
  bash $(basename "$0") \\
    --model /home/models/models--DeepSeek-R1-0528 \\
    --mode bench --scenario chat --concurrency 32 \\
    --quant fp8 --config latency --iter-range 100-150

  # trtllm-serve mode (for scheduler/batching analysis)
  bash $(basename "$0") \\
    --model /home/models/models--DeepSeek-R1-0528 \\
    --mode serve --scenario reasoning --concurrency 16 \\
    --quant fp8 --config latency --iter-range 100-150
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)           MODEL="$2"; shift 2 ;;
        --mode)            MODE="$2"; shift 2 ;;
        --scenario)        SCENARIO="$2"; shift 2 ;;
        --concurrency)     CONCURRENCY="$2"; shift 2 ;;
        --quant)           QUANT="$2"; shift 2 ;;
        --config)          CONFIG="$2"; shift 2 ;;
        --tp)              TP="$2"; shift 2 ;;
        --ep)              EP="$2"; shift 2 ;;
        --trace-dir)       TRACE_DIR="$2"; shift 2 ;;
        --iter-range)      ITER_RANGE="$2"; shift 2 ;;
        --num-requests)    NUM_REQUESTS="$2"; shift 2 ;;
        --warmup)          WARMUP="$2"; shift 2 ;;
        --port)            PORT="$2"; shift 2 ;;
        --skip-export)     SKIP_EXPORT=true; shift ;;
        --extra-nsys-args) EXTRA_NSYS_ARGS="$2"; shift 2 ;;
        -h|--help)         usage ;;
        *)                 echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ======================== Validation ==========================================
if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required"
    usage
fi

case "$MODE" in
    bench|serve) ;;
    *) echo "ERROR: --mode must be bench or serve"; exit 1 ;;
esac

case "$SCENARIO" in
    chat)      ISL=1024; OSL=1024 ;;
    reasoning) ISL=1024; OSL=8192 ;;
    summarize) ISL=8192; OSL=1024 ;;
    *) echo "ERROR: --scenario must be chat, reasoning, or summarize"; exit 1 ;;
esac

case "$QUANT" in
    fp4|fp8) ;;
    *) echo "ERROR: --quant must be fp4 or fp8"; exit 1 ;;
esac

case "$CONFIG" in
    latency|throughput) ;;
    *) echo "ERROR: --config must be latency or throughput"; exit 1 ;;
esac

# ======================== Utilities ===========================================
TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(TS)] $*"; }

# ======================== Tag & Output Setup ==================================
TAG="nsys_${QUANT}_${CONFIG}_${SCENARIO}_tp${TP}_ep${EP}_c${CONCURRENCY}_iter${ITER_RANGE}"
mkdir -p "$TRACE_DIR"

log "============================================================"
log "  Nsight Systems Trace Capture"
log "============================================================"
log "  Model:       $MODEL"
log "  Mode:        $MODE"
log "  Scenario:    $SCENARIO (ISL=$ISL, OSL=$OSL)"
log "  Concurrency: $CONCURRENCY"
log "  Quant:       $QUANT"
log "  Config:      $CONFIG"
log "  TP:          $TP"
log "  EP:          $EP"
log "  Iter Range:  $ITER_RANGE"
log "  Trace Dir:   $TRACE_DIR"
log "  Tag:         $TAG"
log "============================================================"

# ======================== Adaptive Parameters =================================
# Reused from sa_bench_b200.sh compute_adaptive_params()
compute_adaptive_params() {
    local quant=$1 isl=$2 osl=$3 conc=$4 dp_attn=$5 ep_size=$6 has_mtp=$7

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

    if [[ "$quant" == "fp4" ]]; then
        if [[ "$has_mtp" == "false" ]]; then
            if [[ "$dp_attn" == "true" ]]; then
                MOE_BACKEND="CUTLASS"
                CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc < 4 ? conc : conc / 4 ))
            fi
            if [[ "$isl" == "1024" && "$osl" == "1024" ]]; then
                if [[ "$TP" == "8" && "$ep_size" == "8" ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                fi
            fi
        else
            if [[ "$dp_attn" == "true" ]]; then
                CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc < 4 ? conc : conc / 4 ))
                MOE_BACKEND="CUTLASS"
                MTP_LAYERS=1
            fi
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
    elif [[ "$quant" == "fp8" ]]; then
        if [[ "$has_mtp" == "false" ]]; then
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
            PIECEWISE_CUDA_GRAPHS="true"
            if [[ "$dp_attn" == "true" ]]; then
                MOE_BACKEND="DEEPGEMM"
                PIECEWISE_CUDA_GRAPHS="false"
                CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc < 8 ? conc : conc / 8 ))
                KV_CACHE_FREE_MEM_FRACTION=0.7
                ENABLE_CONFIGURABLE_MOE_FLAG="1"
                MTP_LAYERS=1
            fi
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

# ======================== Generate Config YAML ================================
generate_config_yaml() {
    local has_mtp="false"
    [[ "$CONFIG" == "latency" ]] && has_mtp="true"

    local dp_attn="false"
    [[ $EP -gt 1 ]] && dp_attn="true"

    compute_adaptive_params "$QUANT" "$ISL" "$OSL" "$CONCURRENCY" "$dp_attn" "$EP" "$has_mtp"

    log "  Adaptive params: MOE=$MOE_BACKEND, PW_CUDA=$PIECEWISE_CUDA_GRAPHS, MTP=$MTP_LAYERS"
    log "  CUDA_GRAPH_BS=$CUDA_GRAPH_MAX_BATCH_SIZE, KV_FREE=$KV_CACHE_FREE_MEM_FRACTION"

    CONFIG_YAML="$TRACE_DIR/config_${TAG}.yml"

    cat > "$CONFIG_YAML" << EOF
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
        cat >> "$CONFIG_YAML" << EOF
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: $MTP_LAYERS
EOF
    fi

    if [[ "$dp_attn" == "true" ]]; then
        cat >> "$CONFIG_YAML" << EOF
attention_dp_config:
    batching_wait_iters: 0
    enable_balance: true
    timeout_iters: 60
EOF
    fi

    if [[ "$DELAY_BATCHING" == "true" ]]; then
        cat >> "$CONFIG_YAML" << EOF
batch_wait_timeout_iters: 40
batch_wait_max_tokens_ratio: 0.8
EOF
    fi

    # Compute MAX_NUM_TOKENS and MAX_SEQ_LEN
    local max_batch_size=$CUDA_GRAPH_MAX_BATCH_SIZE
    if [[ $MTP_LAYERS -gt 0 ]]; then
        MAX_NUM_TOKENS=$(( ((MTP_LAYERS + 1) * max_batch_size + ISL + 64 + 63) / 64 * 64 ))
    else
        MAX_NUM_TOKENS=$(( (CONCURRENCY + ISL + 64 + 63) / 64 * 64 ))
    fi
    if [[ $((ISL + OSL)) -le 2048 ]]; then
        MAX_SEQ_LEN=8192
    else
        MAX_SEQ_LEN=10240
    fi
    [[ $MAX_NUM_TOKENS -lt 8192 ]] && MAX_NUM_TOKENS=8192

    # Piecewise CUDA Graphs
    if [[ "$PIECEWISE_CUDA_GRAPHS" == "true" ]]; then
        local capture_tokens=(1 2 4 8 16 32 64 128)
        capture_tokens+=( $(seq 256 256 $MAX_NUM_TOKENS) )
        if [[ $((MAX_NUM_TOKENS % 256)) -ne 0 ]]; then
            capture_tokens+=($MAX_NUM_TOKENS)
        fi
        local capture_list
        capture_list=$(printf "%s, " "${capture_tokens[@]}")

        cat >> "$CONFIG_YAML" << EOF
torch_compile_config:
    capture_num_tokens: [${capture_list%, }]
    enable_piecewise_cuda_graph: true
EOF
    fi

    # Optional env var
    if [[ -n "$ENABLE_CONFIGURABLE_MOE_FLAG" ]]; then
        export ENABLE_CONFIGURABLE_MOE=1
    else
        unset ENABLE_CONFIGURABLE_MOE 2>/dev/null || true
    fi

    log "  Config YAML: $CONFIG_YAML"
}

# ======================== Post-Processing =====================================
run_post_processing() {
    local trace_file="$TRACE_DIR/${TAG}.nsys-rep"

    if [[ ! -f "$trace_file" ]]; then
        log "WARN: Trace file not found: $trace_file"
        return 1
    fi

    local size_mb
    size_mb=$(du -m "$trace_file" | cut -f1)
    log "Trace file: $trace_file (${size_mb} MB)"

    # Export to SQLite
    log "Exporting to SQLite..."
    nsys export --type sqlite -o "$TRACE_DIR/${TAG}.sqlite" "$trace_file" 2>&1 || \
        log "WARN: SQLite export failed"

    # Export kernel CSV
    log "Exporting kernel trace CSV..."
    nsys stats --report cuda_gpu_trace --format csv \
        -o "$TRACE_DIR/${TAG}_kernels" "$trace_file" 2>&1 || \
        log "WARN: Kernel CSV export failed"

    # Print summary
    if [[ -f "$TRACE_DIR/${TAG}.sqlite" ]]; then
        log "=== Top 10 Kernels by Total Duration ==="
        sqlite3 -header -column "$TRACE_DIR/${TAG}.sqlite" \
            "SELECT shortName, COUNT(*) as count,
                    SUM(end-start) as total_ns,
                    AVG(end-start) as avg_ns,
                    ROUND(CAST(SUM(end-start) AS FLOAT) / 1e6, 2) as total_ms
             FROM CUPTI_ACTIVITY_KIND_KERNEL
             GROUP BY shortName
             ORDER BY total_ns DESC
             LIMIT 10;" 2>/dev/null || \
            log "WARN: Could not query kernel summary"
    fi
}

# ======================== Bench Mode ==========================================
run_bench_mode() {
    log "=== BENCH MODE ==="

    # Generate dataset
    local dataset="$TRACE_DIR/dataset_${TAG}.jsonl"
    log "Generating dataset..."
    python3 "$SCRIPT_DIR/gen_dataset.py" \
        --tokenizer "$MODEL" \
        --fixed_input_len "$ISL" \
        --output_tokens "$OSL" \
        --num_requests "$NUM_REQUESTS" \
        --input_mode random \
        --output "$dataset"

    # Generate config YAML
    generate_config_yaml

    # Run nsys-wrapped trtllm-bench
    log "Starting nsys-profiled trtllm-bench..."
    log "  Profiling iterations: $ITER_RANGE"

    local nsys_cmd=(
        nsys profile
        -o "$TRACE_DIR/${TAG}" -f true
        -t 'cuda,nvtx,python-gil' -c cudaProfilerApi
        --cuda-graph-trace node
        -e TLLM_PROFILE_RECORD_GC=1,TLLM_LLMAPI_ENABLE_NVTX=1
        --trace-fork-before-exec=true
    )
    if [[ -n "$EXTRA_NSYS_ARGS" ]]; then
        read -ra extra_args <<< "$EXTRA_NSYS_ARGS"
        nsys_cmd+=("${extra_args[@]}")
    fi

    local bench_cmd=(
        trtllm-bench --model "$MODEL" --model_path "$MODEL"
        throughput --backend pytorch
        --extra_llm_api_options "$CONFIG_YAML"
        --max_seq_len "$MAX_SEQ_LEN"
        --tp "$TP" --ep "$EP"
        --dataset "$dataset"
        --concurrency "$CONCURRENCY"
        --num_requests "$NUM_REQUESTS"
        --warmup "$WARMUP"
    )

    log "  nsys cmd: ${nsys_cmd[*]}"
    log "  bench cmd: ${bench_cmd[*]}"

    TLLM_PROFILE_START_STOP="$ITER_RANGE" \
        "${nsys_cmd[@]}" "${bench_cmd[@]}"
    local rc=$?

    if [[ $rc -ne 0 ]]; then
        log "ERROR: trtllm-bench exited with rc=$rc"
        return $rc
    fi

    log "Trace capture complete."
}

# ======================== Serve Mode ==========================================
run_serve_mode() {
    log "=== SERVE MODE ==="

    # Generate config YAML
    generate_config_yaml

    kill_server

    # Start nsys-wrapped trtllm-serve
    log "Starting nsys-profiled trtllm-serve..."

    local server_log="$TRACE_DIR/server_${TAG}.log"

    local nsys_cmd=(
        nsys profile
        -o "$TRACE_DIR/${TAG}" -f true
        -t 'cuda,nvtx,python-gil' -c cudaProfilerApi
        --cuda-graph-trace node
        -e TLLM_PROFILE_RECORD_GC=1,TLLM_LLMAPI_ENABLE_NVTX=1
        --trace-fork-before-exec=true
    )
    if [[ -n "$EXTRA_NSYS_ARGS" ]]; then
        read -ra extra_args <<< "$EXTRA_NSYS_ARGS"
        nsys_cmd+=("${extra_args[@]}")
    fi

    TLLM_PROFILE_START_STOP="$ITER_RANGE" \
    PYTHONNOUSERSITE=1 \
        "${nsys_cmd[@]}" \
        mpirun -n 1 --oversubscribe --allow-run-as-root \
        trtllm-serve "$MODEL" --port="$PORT" \
        --trust_remote_code --backend=pytorch \
        --max_seq_len="$MAX_SEQ_LEN" --max_num_tokens="$MAX_NUM_TOKENS" \
        --tp_size="$TP" --ep_size="$EP" \
        --extra_llm_api_options="$CONFIG_YAML" \
        > "$server_log" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server_ready --port "$PORT" --server-log "$server_log" --server-pid "$SERVER_PID"; then
        log "ERROR: Server failed to start"
        kill_server
        return 1
    fi

    # Run benchmark client
    local num_prompts=$(( CONCURRENCY * 10 ))
    [[ $num_prompts -lt 20 ]] && num_prompts=20

    local dp_attn="false"
    [[ $EP -gt 1 ]] && dp_attn="true"

    local bench_args=(
        --model "$MODEL"
        --port "$PORT"
        --backend openai
        --input-len "$ISL"
        --output-len "$OSL"
        --random-range-ratio 0.8
        --num-prompts "$num_prompts"
        --max-concurrency "$CONCURRENCY"
        --num-warmups 0
        --result-filename "nsys_serving_${TAG}"
        --result-dir "$TRACE_DIR"
    )

    log "Running benchmark_serving..."
    run_benchmark_serving "${bench_args[@]}" || \
        log "WARN: benchmark_serving failed"

    # Kill server to trigger nsys flush
    log "Stopping server (triggers nsys trace flush)..."
    kill_server
    # Give nsys time to finalize the trace
    sleep 5

    log "Trace capture complete."
}

# ======================== Main ================================================
trap 'kill_server 2>/dev/null; exit' INT TERM

case "$MODE" in
    bench)
        run_bench_mode
        ;;
    serve)
        run_serve_mode
        ;;
esac

# Post-processing
if [[ "$SKIP_EXPORT" == "false" ]]; then
    log "=== POST-PROCESSING ==="
    run_post_processing
else
    log "Skipping post-processing (--skip-export)"
fi

log "============================================================"
log "  TRACE CAPTURE COMPLETE"
log "  Output: $TRACE_DIR/${TAG}.nsys-rep"
log "============================================================"
