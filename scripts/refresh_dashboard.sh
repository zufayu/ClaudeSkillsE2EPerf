#!/usr/bin/env bash
# =============================================================================
# One-command dashboard refresh: import results + regenerate dashboard
#
# Usage (on B200 machine):
#   bash scripts/refresh_dashboard.sh
#
# What it does:
#   1. Import new B200 FP8 EP=1 results → replaces old mtp0 data
#   2. Fetch latest competitor data (ATOM MI355X)
#   3. Regenerate dashboard/data.js
#   4. Show summary of what changed
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

echo "============================================================"
echo "  Dashboard Refresh"
echo "============================================================"
echo ""

# ---- Step 1: Import B200 FP8 EP=1 (mtp0) results ----
# New data from results_b200_fp8_ep1 replaces old incomplete mtp0 data
B200_FP8_EP1_DIR=""
for candidate in \
    "./results_b200_fp8_ep1" \
    "$HOME/zufa/ClaudeSkillsE2EPerf/results_b200_fp8_ep1" \
    "../results_b200_fp8_ep1"; do
    if [[ -d "$candidate" ]] && ls "$candidate"/result_*.json &>/dev/null; then
        B200_FP8_EP1_DIR="$(cd "$candidate" && pwd)"
        break
    fi
done

if [[ -n "$B200_FP8_EP1_DIR" ]]; then
    echo "[1/3] Importing B200 FP8 mtp0 results from: $B200_FP8_EP1_DIR"
    n_files=$(ls "$B200_FP8_EP1_DIR"/result_*.json 2>/dev/null | wc -l)
    echo "  Found $n_files result files"

    python3 "$SCRIPT_DIR/import_results.py" \
        --results-dir "$B200_FP8_EP1_DIR" \
        --platform "8×B200" \
        --framework "TRT-LLM 1.2.0rc6.post3" \
        --quantization FP8 \
        --env-tag mtp0

    echo ""
else
    echo "[1/3] SKIP: No B200 FP8 EP=1 results found"
    echo "  Looked for: ./results_b200_fp8_ep1, ~/zufa/.../results_b200_fp8_ep1"
    echo ""
fi

# ---- Step 2: Fetch competitor data ----
echo "[2/3] Fetching competitor data (ATOM MI355X)..."
if python3 "$SCRIPT_DIR/fetch_competitors.py" 2>/dev/null; then
    echo "  Done"
else
    echo "  WARN: fetch_competitors.py failed (network?), using existing data"
fi
echo ""

# ---- Step 3: Regenerate dashboard ----
echo "[3/3] Regenerating dashboard..."
python3 "$SCRIPT_DIR/generate_dashboard.py"
echo ""

# ---- Summary ----
echo "============================================================"
echo "  Dashboard refresh complete"
echo "============================================================"
echo ""
echo "Current runs/ contents:"
for f in runs/*.json; do
    [[ -f "$f" ]] || continue
    python3 -c "
import json, os
d = json.load(open('$f'))
n = len(d.get('results', []))
tag = d.get('env_tag', '')
tag_str = f' [{tag}]' if tag else ''
print(f\"  {os.path.basename('$f'):50s} {d['platform']} {d['quantization']}{tag_str}  {n} points  ({d.get('date','?')})\")
" 2>/dev/null
done
echo ""
echo "To view: cd dashboard && python3 -m http.server 8899"
