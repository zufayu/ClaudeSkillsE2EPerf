#!/usr/bin/env bash
# =============================================================================
# Nsight Compute (ncu) Trace Capture — Server Mode
#
# Precisely captures decode-phase GPU kernels with full hardware metrics
# including PM Sampling (SM utilization over time) for PDL analysis.
#
# Two-phase approach:
#   Phase 1 (dry-run): lightweight summary to determine --launch-skip value
#   Phase 2 (capture): full metrics for exactly 1 decode iteration
#
# Flow:
#   1. ncu wraps the server process (sglang/trtllm)
#   2. Server loads model normally (ncu doesn't replay loading kernels)
#   3. External client sends warmup + 1 profiled request
#   4. --launch-skip skips prefill/warmup kernels
#   5. --launch-count captures exactly 1 decode iteration
#   6. Output: .ncu-rep for GUI inspection
#
# Usage:
#   bash scripts/collect_ncu_trace.sh \
#     --model /path/to/DeepSeek-R1-0528-NVFP4-v2 \
#     --tp 8 --ep 8 --quantization modelopt_fp4 \
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
PORT=30000
NCU_SET="full"             # full | detailed | basic | pmsampling
LAUNCH_SKIP=""             # auto-detect from dry-run if empty
LAUNCH_COUNT=50            # ~1 decode iteration for DeepSeek-R1 (40-50 kernels)
REPORT_NAME="ncu_decode"
SKIP_DRY_RUN=false
MEM_FRACTION=0.85
CHUNKED_PREFILL=16384
KV_CACHE_DTYPE="fp8_e4m3"

# ======================== CLI Parsing =========================================

usage() {
    cat <<EOF
Usage: $0 --model PATH --result-dir DIR [options]

Nsight Compute trace capture for LLM inference (SGLang/TRT-LLM server mode).

Required:
  --model PATH            Model path
  --result-dir DIR        Output directory (ncu/ subfolder created automatically)

Options:
  --backend BACKEND       sglang | trtllm (default: sglang)
  --tp N                  Tensor parallel size (default: 4)
  --ep N                  Expert parallel size (default: 4)
  --quantization Q        Quantization method (e.g. modelopt_fp4)
  --port N                Server port (default: 30000)
  --ncu-set SET           full|detailed|basic|pmsampling (default: full)
  --launch-skip N         Skip first N kernel launches (auto from dry-run if omitted)
  --launch-count N        Number of kernel launches to capture (default: 50)
  --skip-dry-run          Skip dry-run, requires --launch-skip to be set
  --report-name NAME      Output report name (default: ncu_decode)
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
        --port)             PORT="$2"; shift 2 ;;
        --ncu-set)          NCU_SET="$2"; shift 2 ;;
        --launch-skip)      LAUNCH_SKIP="$2"; shift 2 ;;
        --launch-count)     LAUNCH_COUNT="$2"; shift 2 ;;
        --skip-dry-run)     SKIP_DRY_RUN=true; shift ;;
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
log "  Nsight Compute Trace Capture (Server Mode)"
log "============================================================"
log "  Backend:       $BACKEND"
log "  Model:         $MODEL"
log "  TP=$TP  EP=$EP  quant=${QUANTIZATION:-none}"
log "  Port:          $PORT"
log "  ncu set:       $NCU_SET"
log "  launch-skip:   ${LAUNCH_SKIP:-auto}"
log "  launch-count:  $LAUNCH_COUNT"
log "  Result Dir:    $RESULT_DIR"
log "============================================================"

NCU_VERSION=$(ncu --version 2>/dev/null | head -1 || echo "ncu not found")
log "ncu: $NCU_VERSION"

# ======================== Build server command =================================

build_server_cmd() {
    if [[ "$BACKEND" == "sglang" ]]; then
        local CMD="python3 -m sglang.launch_server --model-path $MODEL --tp $TP --port $PORT --mem-fraction-static $MEM_FRACTION --chunked-prefill-size $CHUNKED_PREFILL --kv-cache-dtype $KV_CACHE_DTYPE --trust-remote-code --log-level warning"
        if [[ $EP -gt 1 ]]; then CMD="$CMD --ep-size $EP"; fi
        if [[ -n "${QUANTIZATION:-}" ]]; then CMD="$CMD --quantization $QUANTIZATION"; fi
    elif [[ "$BACKEND" == "trtllm" ]]; then
        local CMD="python3 -m tensorrt_llm.commands.serve --model $MODEL --tp_size $TP --port $PORT --trust_remote_code"
        if [[ -n "${QUANTIZATION:-}" ]]; then CMD="$CMD --quantization $QUANTIZATION"; fi
    fi
    echo "$CMD"
}

SERVER_CMD=$(build_server_cmd)

# ======================== Helper: wait for server ready ========================

wait_for_server() {
    local max_wait=${1:-600}
    local elapsed=0
    log "Waiting for server on port $PORT (max ${max_wait}s)..."
    while [[ $elapsed -lt $max_wait ]]; do
        if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1 || curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            log "Server ready (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log "ERROR: Server not ready after ${max_wait}s"
    return 1
}

# ======================== Helper: send request =================================

send_request() {
    local prompt="Explain the architecture of modern large language models."
    if [[ "$BACKEND" == "sglang" ]]; then
        curl -s "http://localhost:$PORT/generate" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"$prompt\", \"sampling_params\": {\"max_new_tokens\": 8, \"temperature\": 0}}" \
            -o /dev/null -w "%{http_code}"
    elif [[ "$BACKEND" == "trtllm" ]]; then
        curl -s "http://localhost:$PORT/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"default\", \"prompt\": \"$prompt\", \"max_tokens\": 8, \"temperature\": 0}" \
            -o /dev/null -w "%{http_code}"
    fi
}

# ======================== Helper: cleanup ======================================

cleanup_server() {
    log "Cleaning up server processes..."
    pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
    pkill -f "tensorrt_llm.*serve.*--port $PORT" 2>/dev/null || true
    sleep 2
    pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
    pkill -9 -f "tensorrt_llm.*serve.*--port $PORT" 2>/dev/null || true
}

trap cleanup_server EXIT

# ======================== Phase 1: Dry-run ====================================

if [[ -z "$LAUNCH_SKIP" ]] && [[ "$SKIP_DRY_RUN" != "true" ]]; then
    log ""
    log "============================================================"
    log "  Phase 1: Dry-run (determine --launch-skip)"
    log "============================================================"

    DRY_RUN_CSV="$NCU_DIR/dry_run.csv"

    # Start server under ncu summary mode
    log "Starting server under ncu --mode summary..."
    ncu --target-processes all --metrics gpu__time_duration.sum --csv \
        $SERVER_CMD > "$DRY_RUN_CSV" 2>"$NCU_DIR/dry_run_stderr.log" &
    NCU_PID=$!

    if wait_for_server 600; then
        log "Sending 1 warmup request..."
        send_request
        sleep 2

        log "Sending 1 profiled request..."
        send_request
        sleep 5

        # Kill server — ncu writes CSV on exit
        log "Stopping server..."
        kill $NCU_PID 2>/dev/null || true
        wait $NCU_PID 2>/dev/null || true
        sleep 3

        # Parse CSV to find decode start
        if [[ -f "$DRY_RUN_CSV" ]] && [[ -s "$DRY_RUN_CSV" ]]; then
            log "Analyzing dry-run CSV..."
            # Find first kernel with short duration (decode) after prefill
            LAUNCH_SKIP=$(python3 -c "
import csv, sys
kernels = []
with open('$DRY_RUN_CSV') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        dur = float(row.get('gpu__time_duration.sum', 0))
        kernels.append((i, row.get('Kernel Name',''), dur))

# Decode kernels are short (<50us typically), prefill are long
# Find the transition point
decode_start = 0
for i, (idx, name, dur) in enumerate(kernels):
    if dur > 0 and dur < 50000:  # < 50us in ns
        # Check if next few are also short (decode pattern)
        if i + 3 < len(kernels):
            next_durs = [kernels[j][2] for j in range(i, min(i+5, len(kernels)))]
            if all(d < 50000 for d in next_durs if d > 0):
                decode_start = idx
                break
if decode_start > 0:
    print(decode_start)
else:
    # Fallback: skip first half of kernels
    print(len(kernels) // 2)
" 2>/dev/null || echo "")
            log "Detected decode start at launch #$LAUNCH_SKIP"
        else
            log "WARNING: dry-run CSV empty, using fallback launch-skip=0"
            LAUNCH_SKIP=0
        fi
    else
        log "WARNING: server failed to start for dry-run, using launch-skip=0"
        LAUNCH_SKIP=0
        kill $NCU_PID 2>/dev/null || true
        wait $NCU_PID 2>/dev/null || true
    fi

    cleanup_server
    sleep 5
fi

[[ -z "$LAUNCH_SKIP" ]] && LAUNCH_SKIP=0

# ======================== Phase 2: Full capture ================================

log ""
log "============================================================"
log "  Phase 2: Full Capture (--launch-skip=$LAUNCH_SKIP --launch-count=$LAUNCH_COUNT)"
log "============================================================"

NCU_OPTS=(
    --target-processes all
    --graph-profiling node
    --pm-sampling-interval 1000
    --launch-skip "$LAUNCH_SKIP"
    --launch-count "$LAUNCH_COUNT"
    --force-launch
    -f
    -o "$NCU_DIR/$REPORT_NAME"
)

# Section set
if [[ "$NCU_SET" == "pmsampling" ]]; then
    NCU_OPTS+=(--section PmSampling --section PmSampling_WarpStates)
else
    NCU_OPTS+=(--set "$NCU_SET")
fi

FULL_CMD="ncu ${NCU_OPTS[*]} $SERVER_CMD"
log "Command:"
log "  $FULL_CMD"
log ""

# Start server under ncu full capture
ncu "${NCU_OPTS[@]}" $SERVER_CMD > "$NCU_DIR/${REPORT_NAME}.log" 2>&1 &
NCU_PID=$!

if wait_for_server 600; then
    log "Server ready. Sending request to trigger decode..."
    sleep 2
    HTTP_CODE=$(send_request)
    log "Request sent (HTTP $HTTP_CODE). Waiting for ncu to finish capturing..."

    # Wait for ncu to finish (it auto-stops after --launch-count kernels)
    wait $NCU_PID 2>/dev/null || true
    log "ncu process exited."
else
    log "ERROR: Server failed to start under ncu"
    kill $NCU_PID 2>/dev/null || true
    wait $NCU_PID 2>/dev/null || true
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
    log ""
    log "  Open in GUI: ncu-ui $REPORT_FILE"
    log "  Export CSV:  ncu -i $REPORT_FILE --page details --csv > details.csv"
    log "  PM Sampling: ncu -i $REPORT_FILE --page raw --csv --print-metric-instances details > pm_raw.csv"
    log "============================================================"
else
    log "ERROR: No .ncu-rep file generated"
    ls -la "$NCU_DIR/" 2>/dev/null
    tail -30 "$NCU_DIR/${REPORT_NAME}.log" 2>/dev/null
fi
