#!/usr/bin/env bash
# =============================================================================
# Nsight Compute (ncu) Trace Capture
#
# Captures decode-phase GPU kernels with full hardware metrics
# including PM Sampling (SM utilization over time) for PDL analysis.
#
# Supports two modes:
#   - offline: ncu wraps offline inference (BS=1, single prompt)
#   - serve:   ncu wraps server + concurrent benchmark requests
#
# Two-phase approach:
#   Phase 1 (nsys dry-run): quick nsys trace to count inference kernels
#                           during warmup, determining --launch-skip value
#   Phase 2 (ncu capture):  full ncu metrics for decode iteration(s)
#
# Usage (offline):
#   bash scripts/collect_ncu_trace.sh \
#     --model /path/to/model \
#     --backend sglang --tp 8 --ep 8 \
#     --quantization modelopt_fp4 \
#     --result-dir ./results/ncu_profiling
#
# Usage (serve, concurrent):
#   bash scripts/collect_ncu_trace.sh \
#     --model /path/to/model \
#     --backend sglang --tp 8 --ep 8 \
#     --quantization modelopt_fp4 \
#     --mode serve --concurrency 64 --scenario chat \
#     --result-dir ./results/ncu_profiling
#
# Then open in Nsight Compute GUI:
#   ncu-ui results/ncu_profiling/ncu/ncu_decode.ncu-rep
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ======================== Defaults ============================================
MODEL=""
BACKEND="sglang"  # sglang | trtllm
MODE="offline"    # offline | serve
TP=4
EP=4
QUANTIZATION=""
RESULT_DIR=""
NCU_SET="full"             # full | detailed | basic | pmsampling
NCU_MODE="launch"          # launch | attach
LAUNCH_SKIP=""             # auto-detect from nsys dry-run if empty
LAUNCH_COUNT=50            # kernels to capture
REPORT_NAME="ncu_decode"
SKIP_DRY_RUN=false
NSYS_REP=""               # reuse existing nsys-rep for decode region detection
WARMUP_PROMPTS=1
ISL=64
OSL=4

# Serve mode defaults
CONCURRENCY=64
SCENARIO="chat"
PORT=8888
NUM_PROMPTS=0              # 0 = auto (concurrency * 10)
DRY_RUN_PROMPTS=0          # 0 = auto (concurrency * 3 for serve, same as main for offline)

# Kernel name regex — matches inference kernels, skips loading kernels
# Covers: GEMM (nvjet/cutlass), attention (fmha/flash), MoE, comm (nccl/allreduce)
KERNEL_REGEX="nvjet|fmha|cutlass|flash_attn|kernel_mha|allreduce|reduce_scatter|all_gather|nccl|deep_gemm"

# SGLang-specific defaults
MEM_FRACTION=0.85
CHUNKED_PREFILL=16384
KV_CACHE_DTYPE="fp8_e4m3"
CUDA_GRAPH_MAX_BS=256
MAX_RUNNING_REQUESTS=256

# ======================== CLI Parsing =========================================

usage() {
    cat <<EOF
Usage: $0 --model PATH --result-dir DIR [options]

Nsight Compute trace capture for LLM inference decode kernels.

Required:
  --model PATH            Model path
  --result-dir DIR        Output directory (ncu/ subfolder created automatically)

Options:
  --backend BACKEND       sglang | trtllm (default: sglang)
  --mode MODE             offline | serve (default: offline)
  --tp N                  Tensor parallel size (default: 4)
  --ep N                  Expert parallel size (default: 4)
  --quantization Q        Quantization method (e.g. modelopt_fp4)
  --ncu-set SET           full|detailed|basic|pmsampling (default: full)
  --ncu-mode MODE         launch|attach (default: launch)
                          launch: ncu wraps the process (traditional)
                          attach: start server normally, attach ncu after warmup
                          (bypasses IPC timeout issues with multi-TP frameworks)
  --launch-skip N         Skip first N matching kernel launches (auto if omitted)
  --launch-count N        Number of kernel launches to capture (default: 50)
  --skip-dry-run          Skip nsys dry-run, requires --launch-skip to be set
  --nsys-rep PATH         Reuse existing .nsys-rep for decode region detection
  --report-name NAME      Output report name (default: ncu_decode)
  --warmup-prompts N      Number of warmup prompts (default: 1)
  --isl N                 Input sequence length (default: 64)
  --osl N                 Output sequence length (default: 4)
  --kernel-regex REGEX    Kernel name filter regex (default: built-in)
  --concurrency N         Request concurrency for serve mode (default: 64)
  --scenario SCENARIO     chat|reasoning|summarize for serve mode (default: chat)
  --port N                Server port for serve mode (default: 8888)
  --num-prompts N         Benchmark prompts for serve mode (default: concurrency*10)
  --dry-run-prompts N     Prompts for Phase 1 nsys dry-run (default: concurrency*3 for serve)
  -h, --help              Show this help

Section sets:
  full       ~7800 metrics, all sections including PM Sampling
  detailed   ~900 metrics, compute + memory analysis
  basic      ~200 metrics, SpeedOfLight + occupancy
  pmsampling PM Sampling only (SM utilization timeline)
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)            MODEL="$2"; shift 2 ;;
        --backend)          BACKEND="$2"; shift 2 ;;
        --mode)             MODE="$2"; shift 2 ;;
        --tp)               TP="$2"; shift 2 ;;
        --ep)               EP="$2"; shift 2 ;;
        --quantization)     QUANTIZATION="$2"; shift 2 ;;
        --result-dir)       RESULT_DIR="$2"; shift 2 ;;
        --ncu-set)          NCU_SET="$2"; shift 2 ;;
        --ncu-mode)         NCU_MODE="$2"; shift 2 ;;
        --launch-skip)      LAUNCH_SKIP="$2"; shift 2 ;;
        --launch-count)     LAUNCH_COUNT="$2"; shift 2 ;;
        --skip-dry-run)     SKIP_DRY_RUN=true; shift ;;
        --nsys-rep)         NSYS_REP="$2"; shift 2 ;;
        --report-name)      REPORT_NAME="$2"; shift 2 ;;
        --warmup-prompts)   WARMUP_PROMPTS="$2"; shift 2 ;;
        --isl)              ISL="$2"; shift 2 ;;
        --osl)              OSL="$2"; shift 2 ;;
        --kernel-regex)     KERNEL_REGEX="$2"; shift 2 ;;
        --concurrency)      CONCURRENCY="$2"; shift 2 ;;
        --scenario)         SCENARIO="$2"; shift 2 ;;
        --port)             PORT="$2"; shift 2 ;;
        --num-prompts)      NUM_PROMPTS="$2"; shift 2 ;;
        --dry-run-prompts)  DRY_RUN_PROMPTS="$2"; shift 2 ;;
        -h|--help)          usage ;;
        *)                  echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$MODEL" ]] && { echo "ERROR: --model is required"; usage; }
[[ -z "$RESULT_DIR" ]] && { echo "ERROR: --result-dir is required"; usage; }

# ======================== Setup ===============================================

NCU_DIR="$RESULT_DIR/ncu"
mkdir -p "$NCU_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "============================================================"
log "  Nsight Compute Trace Capture"
log "============================================================"
log "  Backend:       $BACKEND"
log "  Mode:          $MODE"
log "  NCU Mode:      $NCU_MODE"
log "  Model:         $MODEL"
log "  TP=$TP  EP=$EP  quant=${QUANTIZATION:-none}"
if [[ "$MODE" == "serve" ]]; then
    log "  Scenario=$SCENARIO  Concurrency=$CONCURRENCY  Port=$PORT"
    log "  Warmup=$WARMUP_PROMPTS  NumPrompts=${NUM_PROMPTS:-auto}"
else
    log "  ISL=$ISL  OSL=$OSL  warmup=$WARMUP_PROMPTS"
fi
log "  ncu set:       $NCU_SET"
log "  kernel:        $KERNEL_REGEX"
log "  launch-skip:   ${LAUNCH_SKIP:-auto}"
log "  launch-count:  $LAUNCH_COUNT"
log "  Result Dir:    $RESULT_DIR"
log "============================================================"

NCU_VERSION=$(ncu --version 2>/dev/null | head -1 || echo "ncu not found")
log "ncu: $NCU_VERSION"

# ======================== Build inference command ==============================

build_infer_cmd() {
    local CMD="python3 $SCRIPT_DIR/ncu_infer.py --backend $BACKEND --mode $MODE --model $MODEL --tp $TP --ep $EP --warmup-prompts $WARMUP_PROMPTS"
    if [[ "$MODE" == "serve" ]]; then
        CMD="$CMD --concurrency $CONCURRENCY --scenario $SCENARIO --port $PORT"
        if [[ "$NUM_PROMPTS" -gt 0 ]] 2>/dev/null; then CMD="$CMD --num-prompts $NUM_PROMPTS"; fi
    else
        CMD="$CMD --isl $ISL --osl $OSL"
    fi
    if [[ -n "${QUANTIZATION:-}" ]]; then CMD="$CMD --quantization $QUANTIZATION"; fi
    if [[ "$BACKEND" == "sglang" ]]; then
        CMD="$CMD --mem-fraction-static $MEM_FRACTION --chunked-prefill-size $CHUNKED_PREFILL --kv-cache-dtype $KV_CACHE_DTYPE --cuda-graph-max-bs $CUDA_GRAPH_MAX_BS --max-running-requests $MAX_RUNNING_REQUESTS"
    fi
    echo "$CMD"
}

INFER_CMD=$(build_infer_cmd)
log "Inference command:"
log "  $INFER_CMD"

# ======================== Phase 1: nsys dry-run ===============================

if [[ "$NCU_MODE" == "attach" ]]; then
    log "Attach mode: skipping Phase 1 nsys dry-run (warmup done after server start)"
elif [[ -z "$LAUNCH_SKIP" ]] && [[ "$SKIP_DRY_RUN" != "true" ]]; then
    log ""
    log "============================================================"
    log "  Phase 1: nsys dry-run → find steady-state decode region"
    log "============================================================"

    # Reuse existing nsys-rep if provided, otherwise run a fresh dry-run
    if [[ -n "$NSYS_REP" ]] && [[ -f "$NSYS_REP" ]]; then
        log "Reusing existing nsys trace: $NSYS_REP"
        NSYS_REP_FILE="$NSYS_REP"
    else
        NSYS_REPORT="$NCU_DIR/dry_run"

        # For serve mode, use fewer prompts to keep the trace small
        DRY_RUN_CMD="$INFER_CMD"
        if [[ "$MODE" == "serve" ]]; then
            DRY_RUN_NUM=${DRY_RUN_PROMPTS:-0}
            if [[ "$DRY_RUN_NUM" -eq 0 ]] 2>/dev/null; then
                DRY_RUN_NUM=$((CONCURRENCY * 3))
            fi
            DRY_RUN_CMD="$DRY_RUN_CMD --num-prompts $DRY_RUN_NUM"
            log "Dry-run with reduced prompts: $DRY_RUN_NUM (vs full benchmark)"
        fi

        log "Running nsys trace..."
        log "  $DRY_RUN_CMD"
        nsys profile --trace cuda -o "$NSYS_REPORT" --force-overwrite true $DRY_RUN_CMD > "$NCU_DIR/dry_run_stdout.log" 2>&1 || true
        NSYS_REP_FILE="${NSYS_REPORT}.nsys-rep"
    fi

    if [[ -f "$NSYS_REP_FILE" ]]; then
        NSYS_SIZE=$(du -h "$NSYS_REP_FILE" | cut -f1)
        log "nsys trace captured ($NSYS_SIZE). Finding steady-state decode region..."

        # Run with stderr captured for debugging (not swallowed)
        FIND_RESULT=$(python3 "$SCRIPT_DIR/find_decode_region.py" --nsys-rep "$NSYS_REP_FILE" --kernel-regex "$KERNEL_REGEX" --launch-count "$LAUNCH_COUNT" --json 2>"$NCU_DIR/find_decode_stderr.log" | tail -1)

        if [[ -n "$FIND_RESULT" ]]; then
            LAUNCH_SKIP=$(echo "$FIND_RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['launch_skip'])" 2>/dev/null || echo "0")
            DETECTED_COUNT=$(echo "$FIND_RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['launch_count'])" 2>/dev/null || echo "$LAUNCH_COUNT")
            DECODE_DUR=$(echo "$FIND_RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"{d.get('decode_duration_ms',0):.3f}\")" 2>/dev/null || echo "0")
            log "Steady-state decode found:"
            log "  launch-skip:  $LAUNCH_SKIP"
            log "  launch-count: $DETECTED_COUNT"
            log "  decode kernel time: ${DECODE_DUR}ms"
        else
            log "ERROR: find_decode_region.py failed to produce JSON output"
            log "  stderr: $(cat "$NCU_DIR/find_decode_stderr.log" 2>/dev/null | tail -5)"
            LAUNCH_SKIP=0
        fi

        # Also run full analysis for logging
        python3 "$SCRIPT_DIR/find_decode_region.py" --nsys-rep "$NSYS_REP_FILE" --kernel-regex "$KERNEL_REGEX" --launch-count "$LAUNCH_COUNT" > "$NCU_DIR/decode_region.log" 2>&1 || true
    else
        log "WARNING: nsys trace failed, using launch-skip=0"
        LAUNCH_SKIP=0
    fi
fi

[[ -z "$LAUNCH_SKIP" ]] && LAUNCH_SKIP=0

if [[ "$LAUNCH_SKIP" -eq 0 ]] && [[ "$MODE" == "serve" ]] && [[ "$NCU_MODE" != "attach" ]]; then
    log "WARNING: launch-skip=0 in serve mode — ncu will profile from server startup (likely wrong)"
    log "  Consider using --launch-skip N or --nsys-rep to provide skip value"
fi

# ======================== Phase 2: ncu capture ================================

log ""
log "============================================================"
log "  Phase 2: ncu Capture"
log "  --launch-skip=$LAUNCH_SKIP --launch-count=$LAUNCH_COUNT"
log "============================================================"

NCU_OPTS=(
    --target-processes all
    --graph-profiling node
    --pm-sampling-interval 1000
    -k "regex:$KERNEL_REGEX"
    --launch-skip "$LAUNCH_SKIP"
    --launch-count "$LAUNCH_COUNT"
    -f
    -o "$NCU_DIR/$REPORT_NAME"
)

# Section set
if [[ "$NCU_SET" == "pmsampling" ]]; then
    NCU_OPTS+=(--section PmSampling --section PmSampling_WarpStates)
else
    NCU_OPTS+=(--set "$NCU_SET")
fi

# Always add NVLink sections for multi-GPU configurations
NCU_OPTS+=(--section Nvlink --section Nvlink_Tables --section Nvlink_Topology)

if [[ "$NCU_MODE" == "attach" ]] && [[ "$MODE" == "serve" ]]; then
    # ── Attach mode (serve only) ──────────────────────────────────────────
    # Uses CUDA_INJECTION64_PATH to pre-inject NCU library at server startup.
    # The injection library loads in passthrough mode (no profiling overhead).
    # After server is ready + warmed up, `ncu --mode attach` connects to the
    # injection library's IPC endpoint and triggers profiling.
    #
    # This bypasses IPC timeout issues that crash multi-TP frameworks
    # (e.g., TRT-LLM's _create_ipc_executor) under ncu launch mode,
    # because the injection library in passthrough mode has negligible overhead.
    #
    # Flow:
    #   1. Query NCU injection library path
    #   2. Start server with CUDA_INJECTION64_PATH (passive injection)
    #   3. Wait for server ready + warmup to steady state
    #   4. ncu --mode attach (connects to injection IPC, starts profiling)
    #   5. Send benchmark requests → ncu captures decode kernels
    #   6. Collect .ncu-rep

    # Kill any leftover server and free GPU/port
    log "Cleaning up leftover processes..."
    pids=$(pgrep -f "python3.*sglang.launch_server" 2>/dev/null | grep -v $$ || true); [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
    pids=$(pgrep -f "python3.*sglang" 2>/dev/null | grep -v $$ || true); [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
    pids=$(pgrep -f "trtllm-serve" 2>/dev/null | grep -v $$ || true); [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
    fuser -k -9 "${PORT}/tcp" 2>/dev/null || true
    sleep 5
    log "GPU processes after cleanup:"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null | head -5 || echo "  (none)"

    # Phase 2a: Query injection path and start server with passive injection
    log "=== Attach mode: Phase 2a — Start server with NCU injection (passive) ==="
    NCU_INJECTION_PATH=$(ncu --query-injection-path 2>/dev/null || true)
    if [[ -z "$NCU_INJECTION_PATH" ]]; then
        log "ERROR: ncu --query-injection-path returned empty"
        log "  Trying fallback paths..."
        for p in /usr/local/cuda/nsight-compute/host/linux-desktop-glibc_2_11_3-x64/libcuinj64.so /usr/local/cuda/extras/CUPTI/lib64/libcuinj64.so; do
            if [[ -f "$p" ]]; then NCU_INJECTION_PATH="$p"; break; fi
        done
    fi
    if [[ -z "$NCU_INJECTION_PATH" ]]; then
        log "ERROR: Could not find NCU injection library"
        exit 1
    fi
    log "NCU injection library: $NCU_INJECTION_PATH"

    # Launch server with injection library (passive — no profiling yet)
    ATTACH_SERVER_CMD="$INFER_CMD --server-only"
    log "Launching server with CUDA_INJECTION64_PATH: $ATTACH_SERVER_CMD"
    CUDA_INJECTION64_PATH="$NCU_INJECTION_PATH" $ATTACH_SERVER_CMD &
    ATTACH_SERVER_PID=$!
    log "Server launcher PID=$ATTACH_SERVER_PID"

    # Wait for server ready via health endpoint
    log "Waiting for server to be ready..."
    ATTACH_WAIT=0
    ATTACH_TIMEOUT=900
    while true; do
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            log "Server is ready (${ATTACH_WAIT}s)"
            break
        fi
        if ! kill -0 $ATTACH_SERVER_PID 2>/dev/null; then
            log "ERROR: Server process died during startup"
            exit 1
        fi
        sleep 5
        ATTACH_WAIT=$((ATTACH_WAIT + 5))
        if [[ $ATTACH_WAIT -ge $ATTACH_TIMEOUT ]]; then
            log "ERROR: Server not ready after ${ATTACH_TIMEOUT}s"
            kill $ATTACH_SERVER_PID 2>/dev/null || true
            exit 1
        fi
        [[ $((ATTACH_WAIT % 30)) -eq 0 ]] && log "  Still waiting... (${ATTACH_WAIT}s)"
    done

    # Phase 2b: Warmup to steady state
    case "$SCENARIO" in
        chat)      ATTACH_ISL=1024; ATTACH_OSL=1024 ;;
        reasoning) ATTACH_ISL=1024; ATTACH_OSL=8192 ;;
        summarize) ATTACH_ISL=8192; ATTACH_OSL=1024 ;;
    esac
    ATTACH_WARMUP_PROMPTS=$((CONCURRENCY * 2))
    log "=== Attach mode: Phase 2b — Warmup ($ATTACH_WARMUP_PROMPTS prompts, c=$CONCURRENCY) ==="
    python3 -m sglang.bench_serving --model "$MODEL" --port "$PORT" --backend vllm --dataset-name random --random-input-len "$ATTACH_ISL" --random-output-len "$ATTACH_OSL" --random-range-ratio 0.8 --num-prompts "$ATTACH_WARMUP_PROMPTS" --max-concurrency "$CONCURRENCY" --warmup-requests 0 --output-file /tmp/ncu_warmup.jsonl 2>&1 || log "WARNING: warmup benchmark returned non-zero"
    log "Warmup complete. Server in steady state."

    # Phase 2c: Attach NCU profiler
    log "=== Attach mode: Phase 2c — Attach NCU profiler ==="

    NCU_ATTACH_OPTS=(
        --mode attach
        --target-processes all
        --graph-profiling node
        --pm-sampling-interval 1000
        -k "regex:$KERNEL_REGEX"
        --launch-count "$LAUNCH_COUNT"
        -f
        -o "$NCU_DIR/$REPORT_NAME"
    )

    # Section set
    if [[ "$NCU_SET" == "pmsampling" ]]; then
        NCU_ATTACH_OPTS+=(--section PmSampling --section PmSampling_WarpStates)
    else
        NCU_ATTACH_OPTS+=(--set "$NCU_SET")
    fi
    NCU_ATTACH_OPTS+=(--section Nvlink --section Nvlink_Tables --section Nvlink_Topology)

    log "ncu command: ncu ${NCU_ATTACH_OPTS[*]}"

    # Start ncu attach in background — it connects to the injection library IPC
    ncu "${NCU_ATTACH_OPTS[@]}" > "$NCU_DIR/${REPORT_NAME}.log" 2>&1 &
    NCU_PID=$!
    log "ncu attach PID=$NCU_PID"
    sleep 5

    # Verify ncu is still running
    if ! kill -0 $NCU_PID 2>/dev/null; then
        log "ERROR: ncu attach failed immediately. Log:"
        cat "$NCU_DIR/${REPORT_NAME}.log" 2>/dev/null | tail -20
        log "Falling back: trying ncu --mode launch-and-attach with server restart..."
        # If attach fails, we'll exit and let the user decide
        kill $ATTACH_SERVER_PID 2>/dev/null || true
        exit 1
    fi
    log "ncu attached successfully, waiting for benchmark to trigger kernels..."

    # Phase 2d: Send benchmark requests to trigger decode kernels
    NUM_PROMPTS_ACTUAL=${NUM_PROMPTS}
    [[ "$NUM_PROMPTS_ACTUAL" -eq 0 ]] && NUM_PROMPTS_ACTUAL=$((CONCURRENCY * 10))
    log "=== Attach mode: Phase 2d — Benchmark ($NUM_PROMPTS_ACTUAL prompts, c=$CONCURRENCY) ==="

    python3 -m sglang.bench_serving --model "$MODEL" --port "$PORT" --backend vllm --dataset-name random --random-input-len "$ATTACH_ISL" --random-output-len "$ATTACH_OSL" --random-range-ratio 0.8 --num-prompts "$NUM_PROMPTS_ACTUAL" --max-concurrency "$CONCURRENCY" --warmup-requests 0 --output-file /tmp/ncu_profiled.jsonl 2>&1 || log "WARNING: benchmark returned non-zero"

    # Wait for ncu to finish collecting
    log "Waiting for ncu to complete (may take a while for kernel replay)..."
    wait $NCU_PID 2>/dev/null
    NCU_EXIT=$?
    log "ncu exited with code $NCU_EXIT"

    # Show ncu log tail
    log "ncu log (last 20 lines):"
    tail -20 "$NCU_DIR/${REPORT_NAME}.log" 2>/dev/null

    # Cleanup server
    log "Stopping server..."
    kill $ATTACH_SERVER_PID 2>/dev/null || true
    wait $ATTACH_SERVER_PID 2>/dev/null || true
    pids=$(pgrep -f "python3.*sglang.launch_server" 2>/dev/null | grep -v $$ || true); [ -n "$pids" ] && kill $pids 2>/dev/null || true
    pids=$(pgrep -f "trtllm-serve" 2>/dev/null | grep -v $$ || true); [ -n "$pids" ] && kill $pids 2>/dev/null || true
    sleep 3

elif [[ "$MODE" == "serve" ]]; then
    # ── Launch mode (serve) ───────────────────────────────────────────────
    # ncu wraps the entire ncu_infer.py process (server + warmup + benchmark)

    # Kill any leftover server
    pids=$(pgrep -f "python3.*sglang.launch_server" 2>/dev/null | grep -v $$ || true); [ -n "$pids" ] && kill $pids 2>/dev/null || true
    pids=$(pgrep -f "trtllm-serve" 2>/dev/null | grep -v $$ || true); [ -n "$pids" ] && kill $pids 2>/dev/null || true
    sleep 2

    export NCU_SERVER_TIMEOUT=3600
    log "Serve mode (launch): ncu wraps full ncu_infer.py (server + warmup + benchmark)"
    log "  NCU_SERVER_TIMEOUT=${NCU_SERVER_TIMEOUT}s for server startup under ncu"
    NCU_CMD="$INFER_CMD"
    log "Command:"
    log "  ncu ${NCU_OPTS[*]} $NCU_CMD"
    log ""
    ncu "${NCU_OPTS[@]}" $NCU_CMD > "$NCU_DIR/${REPORT_NAME}.log" 2>&1
    NCU_EXIT=$?

    # Cleanup any remaining server processes
    log "Stopping server..."
    pids=$(pgrep -f "python3.*sglang.launch_server" 2>/dev/null | grep -v $$ || true); [ -n "$pids" ] && kill $pids 2>/dev/null || true
    pids=$(pgrep -f "trtllm-serve" 2>/dev/null | grep -v $$ || true); [ -n "$pids" ] && kill $pids 2>/dev/null || true
    sleep 3
else
    # ── Launch mode (offline) ─────────────────────────────────────────────
    log "Command:"
    log "  ncu ${NCU_OPTS[*]} $INFER_CMD"
    log ""
    ncu "${NCU_OPTS[@]}" $INFER_CMD > "$NCU_DIR/${REPORT_NAME}.log" 2>&1
    NCU_EXIT=$?
fi

# ======================== Results =============================================

REPORT_FILE="$NCU_DIR/${REPORT_NAME}.ncu-rep"
if [[ -f "$REPORT_FILE" ]]; then
    REPORT_SIZE=$(du -h "$REPORT_FILE" | cut -f1)
    log ""
    log "============================================================"
    log "  NCU CAPTURE COMPLETE"
    log "============================================================"
    log "  Report:  $REPORT_FILE ($REPORT_SIZE)"
    log "  Log:     $NCU_DIR/${REPORT_NAME}.log"
    log "  Params:  launch-skip=$LAUNCH_SKIP  launch-count=$LAUNCH_COUNT"
    log "  Filter:  $KERNEL_REGEX"
    log ""
    log "  Open in GUI: ncu-ui $REPORT_FILE"
    log "  Export CSV:  ncu -i $REPORT_FILE --page details --csv > details.csv"
    log "  PM Sampling: ncu -i $REPORT_FILE --page raw --csv --print-metric-instances details > pm_raw.csv"
    log "============================================================"
else
    log "ERROR: No .ncu-rep file generated (exit code: $NCU_EXIT)"
    ls -la "$NCU_DIR/" 2>/dev/null
    tail -30 "$NCU_DIR/${REPORT_NAME}.log" 2>/dev/null
fi
