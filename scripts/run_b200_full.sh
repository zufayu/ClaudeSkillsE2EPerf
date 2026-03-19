#!/usr/bin/env bash
# =============================================================================
# Run all 6 B200 FP8 benchmark configs and import results.
#
# Configs:
#   mtp0 (fp8-throughput) × EP=1, EP=4, EP=8
#   mtp3 (fp8-latency)    × EP=1, EP=4, EP=8
#
# Usage:
#   bash scripts/run_b200_full.sh --model-fp8 /home/models/models--DeepSeek-R1-0528/
#   bash scripts/run_b200_full.sh --model-fp8 /path/to/model --port 8888 --dry-run
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ======================== Defaults ============================================
MODEL_FP8=""
PORT=8888
BASE_RESULT_DIR="./results_b200"
FRAMEWORK="TRT-LLM 1.2.0rc6.post3"
DRY_RUN=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-fp8)    MODEL_FP8="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --result-dir)   BASE_RESULT_DIR="$2"; shift 2 ;;
        --framework)    FRAMEWORK="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: bash scripts/run_b200_full.sh --model-fp8 <path> [options]"
            echo ""
            echo "Options:"
            echo "  --model-fp8 PATH     FP8 model path (required)"
            echo "  --port PORT          Server port (default: 8888)"
            echo "  --result-dir DIR     Base results directory (default: ./results_b200)"
            echo "  --framework STR      Framework version string (default: TRT-LLM 1.2.0rc6.post3)"
            echo "  --dry-run            Print commands without executing"
            exit 0
            ;;
        *)  EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [[ -z "$MODEL_FP8" ]]; then
    echo "ERROR: --model-fp8 is required"
    exit 1
fi

TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(TS)] $*"; }

# ======================== Job Matrix ==========================================
#   config          env-tag    ep-sizes
JOBS=(
    "fp8-throughput  mtp0-ep1   1"
    "fp8-throughput  mtp0-ep4   4"
    "fp8-throughput  mtp0-ep8   8"
    "fp8-latency    mtp3-ep1   1"
    "fp8-latency    mtp3-ep4   4"
    "fp8-latency    mtp3-ep8   8"
)

TOTAL=${#JOBS[@]}
PASS=0
FAIL=0

log "============================================================"
log "  B200 Full Benchmark Suite (6 configs)"
log "  Model: $MODEL_FP8"
log "  Port:  $PORT"
log "============================================================"
echo ""

for i in "${!JOBS[@]}"; do
    read -r config env_tag ep_size <<< "${JOBS[$i]}"
    result_dir="${BASE_RESULT_DIR}_${env_tag}"
    step="[$((i+1))/$TOTAL]"

    log "$step ===== $env_tag ($config, EP=$ep_size) ====="

    # --- Benchmark ---
    bench_cmd=(
        bash "$SCRIPT_DIR/sa_bench_b200.sh"
        --model-fp8 "$MODEL_FP8"
        --configs "$config"
        --ep-sizes "$ep_size"
        --port "$PORT"
        --result-dir "$result_dir"
    )

    if $DRY_RUN; then
        echo "  [DRY-RUN] ${bench_cmd[*]}"
        echo ""
        continue
    fi

    log "  Running: ${bench_cmd[*]}"
    if "${bench_cmd[@]}"; then
        log "  Benchmark OK"
    else
        log "  WARN: Benchmark had failures (continuing)"
        ((FAIL++))
        continue
    fi

    # --- Import ---
    import_cmd=(
        python3 "$SCRIPT_DIR/import_results.py"
        --results-dir "$result_dir"
        --platform "8xB200"
        --framework "$FRAMEWORK"
        --quantization FP8
        --env-tag "$env_tag"
    )

    log "  Importing: ${import_cmd[*]}"
    if "${import_cmd[@]}"; then
        log "  Import OK"
        ((PASS++))
    else
        log "  WARN: Import failed"
        ((FAIL++))
    fi

    echo ""
done

log "============================================================"
log "  DONE: $PASS passed, $FAIL failed (of $TOTAL)"
log "============================================================"
