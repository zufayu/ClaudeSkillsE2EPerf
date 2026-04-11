#!/usr/bin/env bash
# =============================================================================
# Nsight Compute (ncu) Trace Capture for Offline Inference
#
# Captures per-kernel GPU hardware metrics including PM Sampling (SM
# utilization over time) for PDL analysis. Uses cudaProfilerStart/Stop
# in ncu_infer.py to precisely capture steady-state decode kernels.
#
# Works with both SGLang and TRT-LLM offline engines.
#
# Flow:
#   1. ncu wraps the offline inference script with --profile-from-start off
#   2. Script loads model, warmups (ncu idle)
#   3. Script calls cudaProfilerStart → ncu begins capturing
#   4. Script runs one inference → all decode kernels profiled
#   5. Script calls cudaProfilerStop → ncu stops
#   6. Output: .ncu-rep for GUI inspection
#
# Usage:
#   bash scripts/collect_ncu_trace.sh \
#     --model /path/to/DeepSeek-R1-0528-NVFP4-v2 \
#     --tp 8 --ep 8 --quantization modelopt_fp4 \
#     --result-dir ./results/ncu_profiling
#
#   # Then open in Nsight Compute GUI:
#   ncu-ui results/ncu_profiling/ncu_decode.ncu-rep
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
ISL=1024
OSL=64          # keep short — ncu replays each kernel
WARMUP=5
MEM_FRACTION=0.85
CHUNKED_PREFILL=16384
KV_CACHE_DTYPE="fp8_e4m3"
CUDA_GRAPH_MAX_BS=256
MAX_RUNNING_REQUESTS=256
NCU_SET="full"  # full | detailed | basic | pmsampling
KERNEL_FILTER="gemm|fmha|allreduce|moe|Norm|silu|routing|quantize|cvt_fp|allgather|reduce_scatter"  # default: inference kernels only (skips loading kernels like CatArrayBatchedCopy/vectorized_elementwise/memcpy)
REPORT_NAME="ncu_decode"

# ======================== CLI Parsing =========================================

usage() {
    cat <<EOF
Usage: $0 --model PATH --result-dir DIR [options]

Nsight Compute trace capture for offline LLM inference (SGLang/TRT-LLM).

Required:
  --model PATH            Model path
  --result-dir DIR        Output directory (ncu/ subfolder created automatically)

Options:
  --backend BACKEND       sglang | trtllm (default: sglang)
  --tp N                  Tensor parallel size (default: 4)
  --ep N                  Expert parallel size (default: 4)
  --quantization Q        Quantization method (e.g. modelopt_fp4)
  --isl N                 Input sequence length (default: 1024)
  --osl N                 Output sequence length (default: 64, keep short)
  --warmup N              Warmup prompts (default: 5)
  --ncu-set SET           ncu section set: full|detailed|basic|pmsampling (default: full)
  --kernel-filter REGEX   Only profile matching kernels (default: all)
  --report-name NAME      Output report name (default: ncu_decode)
  -h, --help              Show this help

Section sets:
  full       ~7800 metrics, all sections including PM Sampling (slowest)
  detailed   ~900 metrics, compute + memory analysis
  basic      ~200 metrics, SpeedOfLight + occupancy (fastest)
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
        --isl)              ISL="$2"; shift 2 ;;
        --osl)              OSL="$2"; shift 2 ;;
        --warmup)           WARMUP="$2"; shift 2 ;;
        --ncu-set)          NCU_SET="$2"; shift 2 ;;
        --kernel-filter)    KERNEL_FILTER="$2"; shift 2 ;;
        --report-name)      REPORT_NAME="$2"; shift 2 ;;
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
log "  Backend:     $BACKEND"
log "  Model:       $MODEL"
log "  TP=$TP  EP=$EP  quant=${QUANTIZATION:-none}"
log "  ISL=$ISL  OSL=$OSL  warmup=$WARMUP"
log "  ncu set:     $NCU_SET"
log "  kernel:      ${KERNEL_FILTER:-all}"
log "  Result Dir:  $RESULT_DIR"
log "============================================================"

# Check ncu version
NCU_VERSION=$(ncu --version 2>/dev/null | head -1 || echo "ncu not found")
log "ncu: $NCU_VERSION"

# ======================== Build ncu Command ====================================

NCU_OPTS=(
    --graph-profiling node
    --target-processes all
    --pm-sampling-interval 1000
    -f                         # force overwrite
    -o "$NCU_DIR/$REPORT_NAME"
)

# Both SGLang and TRT-LLM use multi-process execution. cudaProfilerStart
# in the main process doesn't propagate to GPU worker subprocesses, so we
# cannot use --profile-from-start off. Instead, we rely on --kernel-filter
# to skip loading kernels (CatArrayBatchedCopy, memcpy, etc.) and only
# profile inference kernels (gemm, fmha, allreduce, moe, etc.).

# Section set
if [[ "$NCU_SET" == "pmsampling" ]]; then
    NCU_OPTS+=(--section PmSampling --section PmSampling_WarpStates)
else
    NCU_OPTS+=(--set "$NCU_SET")
fi

# Kernel filter
if [[ -n "$KERNEL_FILTER" ]]; then
    NCU_OPTS+=(-k "regex:$KERNEL_FILTER")
fi

# Build inference script args
INFER_ARGS=(
    --backend "$BACKEND"
    --model "$MODEL"
    --tp "$TP"
    --ep "$EP"
    --warmup-prompts "$WARMUP"
    --isl "$ISL"
    --osl "$OSL"
    --mem-fraction-static "$MEM_FRACTION"
    --chunked-prefill-size "$CHUNKED_PREFILL"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
)
if [[ -n "$QUANTIZATION" ]]; then
    INFER_ARGS+=(--quantization "$QUANTIZATION")
fi

# ======================== Run =================================================

FULL_CMD="ncu ${NCU_OPTS[*]} python3 $SCRIPT_DIR/ncu_infer.py ${INFER_ARGS[*]}"
log "Command:"
log "  $FULL_CMD"
log ""

ncu "${NCU_OPTS[@]}" python3 "$SCRIPT_DIR/ncu_infer.py" "${INFER_ARGS[@]}" 2>&1 | tee "$NCU_DIR/${REPORT_NAME}.log"

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
    log ""
    log "  Open in GUI: ncu-ui $REPORT_FILE"
    log "  Export CSV:  ncu -i $REPORT_FILE --page details --csv > details.csv"
    log "  PM Sampling: ncu -i $REPORT_FILE --page raw --csv --print-metric-instances details > pm_raw.csv"
    log "============================================================"
else
    log "ERROR: No .ncu-rep file generated"
    ls -la "$NCU_DIR/" 2>/dev/null
fi
