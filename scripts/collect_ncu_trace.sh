#!/usr/bin/env bash
# =============================================================================
# Nsight Compute (ncu) Trace Capture — Offline Engine Mode
#
# Captures decode-phase GPU kernels with full hardware metrics
# including PM Sampling (SM utilization over time) for PDL analysis.
#
# Approach:
#   Uses kernel-name regex filter to skip loading kernels entirely,
#   then --launch-skip to skip warmup inference kernels,
#   and --launch-count to capture exactly N decode kernels.
#
# Two-phase approach:
#   Phase 1 (nsys dry-run): quick nsys trace to count inference kernels
#                           during warmup, determining --launch-skip value
#   Phase 2 (ncu capture):  full ncu metrics for decode iteration(s)
#
# Flow:
#   1. ncu wraps the offline inference script (ncu_infer.py)
#   2. Kernel-name filter (-k regex) skips all loading kernels
#   3. --launch-skip skips warmup inference kernels
#   4. --launch-count captures decode kernel(s)
#   5. Output: .ncu-rep for GUI inspection
#
# Usage:
#   bash scripts/collect_ncu_trace.sh \
#     --model /path/to/model \
#     --backend sglang --tp 8 --ep 8 \
#     --quantization modelopt_fp4 \
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
TP=4
EP=4
QUANTIZATION=""
RESULT_DIR=""
NCU_SET="full"             # full | detailed | basic | pmsampling
LAUNCH_SKIP=""             # auto-detect from nsys dry-run if empty
LAUNCH_COUNT=50            # kernels to capture
REPORT_NAME="ncu_decode"
SKIP_DRY_RUN=false
NSYS_REP=""               # reuse existing nsys-rep for decode region detection
WARMUP_PROMPTS=1
ISL=64
OSL=4

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
  --tp N                  Tensor parallel size (default: 4)
  --ep N                  Expert parallel size (default: 4)
  --quantization Q        Quantization method (e.g. modelopt_fp4)
  --ncu-set SET           full|detailed|basic|pmsampling (default: full)
  --launch-skip N         Skip first N matching kernel launches (auto if omitted)
  --launch-count N        Number of kernel launches to capture (default: 50)
  --skip-dry-run          Skip nsys dry-run, requires --launch-skip to be set
  --nsys-rep PATH         Reuse existing .nsys-rep for decode region detection
  --report-name NAME      Output report name (default: ncu_decode)
  --warmup-prompts N      Number of warmup prompts (default: 1)
  --isl N                 Input sequence length (default: 64)
  --osl N                 Output sequence length (default: 4)
  --kernel-regex REGEX    Kernel name filter regex (default: built-in)
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
        --tp)               TP="$2"; shift 2 ;;
        --ep)               EP="$2"; shift 2 ;;
        --quantization)     QUANTIZATION="$2"; shift 2 ;;
        --result-dir)       RESULT_DIR="$2"; shift 2 ;;
        --ncu-set)          NCU_SET="$2"; shift 2 ;;
        --launch-skip)      LAUNCH_SKIP="$2"; shift 2 ;;
        --launch-count)     LAUNCH_COUNT="$2"; shift 2 ;;
        --skip-dry-run)     SKIP_DRY_RUN=true; shift ;;
        --nsys-rep)         NSYS_REP="$2"; shift 2 ;;
        --report-name)      REPORT_NAME="$2"; shift 2 ;;
        --warmup-prompts)   WARMUP_PROMPTS="$2"; shift 2 ;;
        --isl)              ISL="$2"; shift 2 ;;
        --osl)              OSL="$2"; shift 2 ;;
        --kernel-regex)     KERNEL_REGEX="$2"; shift 2 ;;
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
log "  Model:         $MODEL"
log "  TP=$TP  EP=$EP  quant=${QUANTIZATION:-none}"
log "  ISL=$ISL  OSL=$OSL  warmup=$WARMUP_PROMPTS"
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
    local CMD="python3 $SCRIPT_DIR/ncu_infer.py --backend $BACKEND --model $MODEL --tp $TP --ep $EP --warmup-prompts $WARMUP_PROMPTS --isl $ISL --osl $OSL"
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

if [[ -z "$LAUNCH_SKIP" ]] && [[ "$SKIP_DRY_RUN" != "true" ]]; then
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
        log "Running nsys trace..."
        nsys profile --trace cuda -o "$NSYS_REPORT" --force-overwrite true $INFER_CMD > "$NCU_DIR/dry_run_stdout.log" 2>&1 || true
        NSYS_REP_FILE="${NSYS_REPORT}.nsys-rep"
    fi

    if [[ -f "$NSYS_REP_FILE" ]]; then
        log "nsys trace captured. Finding steady-state decode region..."
        FIND_RESULT=$(python3 "$SCRIPT_DIR/find_decode_region.py" --nsys-rep "$NSYS_REP_FILE" --kernel-regex "$KERNEL_REGEX" --launch-count "$LAUNCH_COUNT" --json 2>/dev/null | tail -1)

        if [[ -n "$FIND_RESULT" ]]; then
            LAUNCH_SKIP=$(echo "$FIND_RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['launch_skip'])" 2>/dev/null || echo "0")
            DETECTED_COUNT=$(echo "$FIND_RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['launch_count'])" 2>/dev/null || echo "$LAUNCH_COUNT")
            DECODE_DUR=$(echo "$FIND_RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"{d.get('decode_duration_ms',0):.3f}\")" 2>/dev/null || echo "0")
            log "Steady-state decode found:"
            log "  launch-skip:  $LAUNCH_SKIP"
            log "  launch-count: $DETECTED_COUNT"
            log "  decode kernel time: ${DECODE_DUR}ms"
        else
            log "WARNING: find_decode_region.py failed, using launch-skip=0"
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

log "Command:"
log "  ncu ${NCU_OPTS[*]} $INFER_CMD"
log ""

ncu "${NCU_OPTS[@]}" $INFER_CMD > "$NCU_DIR/${REPORT_NAME}.log" 2>&1
NCU_EXIT=$?

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
