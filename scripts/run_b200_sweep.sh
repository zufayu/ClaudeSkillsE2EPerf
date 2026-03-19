#!/usr/bin/env bash
# =============================================================================
# B200 FP8 benchmark sweep — all MTP × EP combinations.
#
# Configs (6 total):
#   mtp0 (fp8-throughput) × EP=1, EP=4, EP=8
#   mtp3 (fp8-latency)    × EP=1, EP=4, EP=8
#
# Usage:
#   bash scripts/run_b200_sweep.sh --model-fp8 /home/models/models--DeepSeek-R1-0528/
#   bash scripts/run_b200_sweep.sh --model-fp8 /path/to/model --dry-run
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ======================== Defaults ============================================
MODEL_FP8=""
PORT=8888
TP=8
BASE_RESULT_DIR="./results_b200"
FRAMEWORK="TRT-LLM 1.2.0rc6.post3"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-fp8)    MODEL_FP8="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --tp)           TP="$2"; shift 2 ;;
        --result-dir)   BASE_RESULT_DIR="$2"; shift 2 ;;
        --framework)    FRAMEWORK="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: bash scripts/run_b200_sweep.sh --model-fp8 <path> [options]"
            echo ""
            echo "Options:"
            echo "  --model-fp8 PATH     FP8 model path (required)"
            echo "  --port PORT          Server port (default: 8888)"
            echo "  --tp N               Tensor parallelism (default: 8)"
            echo "  --result-dir DIR     Base results directory (default: ./results_b200)"
            echo "  --framework STR      Framework version string (default: TRT-LLM 1.2.0rc6.post3)"
            echo "  --dry-run            Print commands without executing"
            exit 0
            ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_FP8" ]]; then
    echo "ERROR: --model-fp8 is required"
    exit 1
fi

TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(TS)] $*"; }

# ======================== Job Matrix ==========================================
# Each row: config  ep_size  mtp_tag  dp_tag
# DP is derived from EP: EP>1 → dp_on, EP=1 → dp_off
JOBS=(
    "fp8-throughput  1  mtp0  dp_off"
    "fp8-throughput  4  mtp0  dp_on"
    "fp8-throughput  8  mtp0  dp_on"
    "fp8-latency    1  mtp3  dp_off"
    "fp8-latency    4  mtp3  dp_on"
    "fp8-latency    8  mtp3  dp_on"
)

TOTAL=${#JOBS[@]}
PASS=0
FAIL=0

log "============================================================"
log "  B200 FP8 Benchmark Sweep (${TOTAL} configs)"
log "  Model: $MODEL_FP8"
log "  TP:    $TP"
log "  Port:  $PORT"
log "============================================================"
echo ""

for i in "${!JOBS[@]}"; do
    read -r config ep_size mtp_tag dp_tag <<< "${JOBS[$i]}"

    # e.g. results_b200_fp8_tp8_ep1_dp_off_mtp0
    run_tag="fp8_tp${TP}_ep${ep_size}_${dp_tag}_${mtp_tag}"
    result_dir="${BASE_RESULT_DIR}_${run_tag}"

    # env-tag for import: fp8-tp8-ep1-dp_off-mtp0
    env_tag="fp8-tp${TP}-ep${ep_size}-${dp_tag}-${mtp_tag}"

    step="[$((i+1))/$TOTAL]"

    log "$step ===== ${run_tag} ($config, EP=$ep_size) ====="

    # --- Benchmark ---
    bench_cmd=(
        bash "$SCRIPT_DIR/sa_bench_b200.sh"
        --model-fp8 "$MODEL_FP8"
        --configs "$config"
        --ep-sizes "$ep_size"
        --tp "$TP"
        --port "$PORT"
        --result-dir "$result_dir"
    )

    if $DRY_RUN; then
        echo "  [DRY-RUN] ${bench_cmd[*]}"
        echo "  [DRY-RUN] import → --env-tag $env_tag"
        echo "  [DRY-RUN] result_dir → $result_dir"
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
        --gpu-count "$TP"
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
