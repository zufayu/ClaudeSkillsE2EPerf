#!/usr/bin/env bash
# =============================================================================
# One-command dashboard refresh: trim logs + import results + regenerate dashboard
#
# Usage (on B200 machine):
#   bash scripts/refresh_dashboard.sh
#
# What it does:
#   1. Trim server logs in all results_*/ directories (for repo storage)
#   2. Import B200 FP8 results from all known result directories
#   3. Fetch latest competitor data (ATOM MI355X)
#   4. Regenerate dashboard/data.js
#   5. Show summary of what changed
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

echo "============================================================"
echo "  Dashboard Refresh"
echo "============================================================"
echo ""

# ---- Step 1: Trim server logs ----
echo "[1/4] Trimming server logs..."
if ls -d results_*/ &>/dev/null; then
    python3 "$SCRIPT_DIR/trim_logs.py" --all
else
    echo "  No results_*/ directories found"
fi
echo ""

# ---- Step 2: Import B200 FP8 results ----
# Import map: directory pattern → env-tag
# Each entry: "dir_pattern env_tag"
IMPORT_MAP=(
    "results_b200_fp8_mtp0_ep1  mtp0-ep1"
    "results_b200_fp8_mtp3_ep1  mtp3-ep1"
    "results_b200_fp8_mtp0_ep8  mtp0-ep8"
    "results_b200_fp8_mtp3_ep8  mtp3-ep8"
)

imported=0
skipped=0
for entry in "${IMPORT_MAP[@]}"; do
    dir_pattern="${entry%% *}"
    env_tag="${entry##* }"

    # Search for the directory
    found_dir=""
    for candidate in \
        "./$dir_pattern" \
        "$HOME/zufa/ClaudeSkillsE2EPerf/$dir_pattern" \
        "../$dir_pattern"; do
        if [[ -d "$candidate" ]] && ls "$candidate"/result_*.json &>/dev/null; then
            found_dir="$(cd "$candidate" && pwd)"
            break
        fi
    done

    if [[ -n "$found_dir" ]]; then
        n_files=$(ls "$found_dir"/result_*.json 2>/dev/null | wc -l)
        echo "[2/4] Importing $dir_pattern ($n_files files, env=$env_tag)"

        python3 "$SCRIPT_DIR/import_results.py" \
            --results-dir "$found_dir" \
            --platform "8×B200" \
            --framework "TRT-LLM 1.2.0rc6.post3" \
            --quantization FP8 \
            --env-tag "$env_tag"

        ((imported++)) || true
    else
        ((skipped++)) || true
    fi
done

echo ""
if [[ $imported -eq 0 && $skipped -gt 0 ]]; then
    echo "[2/4] SKIP: No B200 FP8 result directories found"
    echo "  Expected: results_b200_fp8_{mtp0,mtp3}_{ep1,ep8}/"
else
    echo "[2/4] Imported $imported directories ($skipped not found)"
fi
echo ""

# ---- Step 3: Fetch competitor data ----
echo "[3/4] Fetching competitor data (ATOM MI355X)..."
if python3 "$SCRIPT_DIR/fetch_competitors.py" 2>/dev/null; then
    echo "  Done"
else
    echo "  WARN: fetch_competitors.py failed (network?), using existing data"
fi
echo ""

# ---- Step 4: Regenerate dashboard ----
echo "[4/4] Regenerating dashboard..."
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

# Show trimmed log stats
trimmed_count=$(find results_*/ -name "*.trimmed.log" 2>/dev/null | wc -l)
trimmed_size=$(du -sh results_*/*.trimmed.log 2>/dev/null | tail -1 | awk '{print $1}' || echo "0")
echo "Trimmed logs: $trimmed_count files"
echo ""
echo "To view: cd docs && python3 -m http.server 8899"
echo "To commit: git add results_*/*.trimmed.log runs/ docs/data.js && git commit"
