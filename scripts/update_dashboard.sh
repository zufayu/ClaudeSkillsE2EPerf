#!/usr/bin/env bash
# =============================================================================
# One-click dashboard update after benchmark runs.
#
# Usage:
#   bash scripts/update_dashboard.sh --results-dir ./results_mi355x \
#     --platform "8×MI355X" --framework "ATOM 0.1.1" --quantization FP8
#
#   # Skip competitor fetch:
#   bash scripts/update_dashboard.sh --results-dir ./results_mi355x \
#     --platform "8×MI355X" --framework "ATOM 0.1.1" --quantization FP8 \
#     --no-fetch
#
#   # Skip git push (local only):
#   bash scripts/update_dashboard.sh --results-dir ./results_mi355x \
#     --platform "8×MI355X" --framework "ATOM 0.1.1" --quantization FP8 \
#     --no-push
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
RESULTS_DIR=""
PLATFORM=""
FRAMEWORK=""
QUANTIZATION=""
DO_FETCH=true
DO_PUSH=true
COMMIT_MSG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --results-dir)    RESULTS_DIR="$2"; shift 2 ;;
        --platform)       PLATFORM="$2"; shift 2 ;;
        --framework)      FRAMEWORK="$2"; shift 2 ;;
        --quantization)   QUANTIZATION="$2"; shift 2 ;;
        --no-fetch)       DO_FETCH=false; shift ;;
        --no-push)        DO_PUSH=false; shift ;;
        --message|-m)     COMMIT_MSG="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/update_dashboard.sh [options]"
            echo ""
            echo "Required:"
            echo "  --results-dir DIR       Benchmark results directory (e.g. ./results_mi355x)"
            echo "  --platform PLATFORM     Platform name (e.g. \"8×MI355X\")"
            echo "  --framework FRAMEWORK   Framework name (e.g. \"ATOM 0.1.1\")"
            echo "  --quantization QUANT    Quantization type (e.g. FP8)"
            echo ""
            echo "Optional:"
            echo "  --no-fetch              Skip fetching competitor data"
            echo "  --no-push               Skip git commit & push (local preview only)"
            echo "  --message, -m MSG       Custom commit message"
            exit 0
            ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$RESULTS_DIR" || -z "$PLATFORM" || -z "$FRAMEWORK" || -z "$QUANTIZATION" ]]; then
    echo "ERROR: --results-dir, --platform, --framework, --quantization are all required"
    echo "Run with -h for usage"
    exit 1
fi

cd "$REPO_DIR"

echo "============================================================"
echo "  Dashboard Update Pipeline"
echo "============================================================"
echo "  Results:      $RESULTS_DIR"
echo "  Platform:     $PLATFORM"
echo "  Framework:    $FRAMEWORK"
echo "  Quantization: $QUANTIZATION"
echo "  Fetch competitors: $DO_FETCH"
echo "  Git push:     $DO_PUSH"
echo "============================================================"
echo ""

# Step 1: Import results
echo "[1/4] Importing results..."
python3 scripts/import_results.py \
    --results-dir "$RESULTS_DIR" \
    --platform "$PLATFORM" \
    --framework "$FRAMEWORK" \
    --quantization "$QUANTIZATION"
echo ""

# Step 2: Fetch competitor data (optional)
if [[ "$DO_FETCH" == "true" ]]; then
    echo "[2/4] Fetching competitor data..."
    python3 scripts/fetch_competitors.py || echo "WARN: Competitor fetch failed, continuing with existing data"
    echo ""
else
    echo "[2/4] Skipping competitor fetch (--no-fetch)"
fi

# Step 3: Regenerate dashboard
echo "[3/4] Generating docs/data.js..."
python3 scripts/generate_dashboard.py
echo ""

# Step 4: Commit & push
if [[ "$DO_PUSH" == "true" ]]; then
    echo "[4/4] Committing and pushing..."
    git add runs/ docs/data.js
    if git diff --cached --quiet; then
        echo "  No changes to commit"
    else
        msg="${COMMIT_MSG:-data: update ${PLATFORM} ${QUANTIZATION} benchmark results}"
        git commit -m "$msg"
        git push
        echo ""
        echo "Pushed. Dashboard will update in ~1-2 minutes at:"
        echo "  https://zufayu.github.io/ClaudeSkillsE2EPerf/"
    fi
else
    echo "[4/4] Skipping git push (--no-push)"
    echo "  Preview locally: cd dashboard && python3 -m http.server 8899"
fi

echo ""
echo "Done."
