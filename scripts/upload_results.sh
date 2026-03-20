#!/usr/bin/env bash
# Upload benchmark results after human verification.
# Usage: bash scripts/upload_results.sh [-m "commit message"] [--yes]
#
# Steps:
#   1. Run refresh_dashboard.sh (trim + import + dashboard)
#   2. Show summary of changes (git diff --stat)
#   3. Ask for confirmation (unless --yes)
#   4. git add + commit + push

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# Parse arguments
COMMIT_MSG=""
AUTO_YES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m)
            COMMIT_MSG="$2"
            shift 2
            ;;
        --yes)
            AUTO_YES=true
            shift
            ;;
        *)
            echo "Usage: bash scripts/upload_results.sh [-m \"commit message\"] [--yes]"
            exit 1
            ;;
    esac
done

# Step 1: Refresh dashboard (trim logs + import results + regenerate dashboard)
echo "============================================================"
echo "  Step 1: Refresh Dashboard"
echo "============================================================"
bash "$SCRIPT_DIR/refresh_dashboard.sh"
echo ""

# Step 2: Stage files and show changes
echo "============================================================"
echo "  Step 2: Review Changes"
echo "============================================================"

# Stage the specific file types we want to commit
git add -f results_*/*.trimmed.log 2>/dev/null || true
git add results_*/result_*.json 2>/dev/null || true
git add results_*/config_*.yml 2>/dev/null || true
git add results_*/summary.md 2>/dev/null || true
git add runs/*.json 2>/dev/null || true
git add docs/data.js 2>/dev/null || true

# Show what would be committed
echo ""
echo "Staged changes:"
git diff --cached --stat
echo ""

# Check if there's anything to commit
if git diff --cached --quiet; then
    echo "Nothing to commit. All results are already up to date."
    exit 0
fi

# Step 3: Confirm
if [[ "$AUTO_YES" != "true" ]]; then
    echo "============================================================"
    echo "  Step 3: Confirm Upload"
    echo "============================================================"
    echo ""
    read -r -p "Commit and push these changes? [y/N] " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted. Changes remain staged."
        exit 0
    fi
fi

# Step 4: Commit and push
if [[ -z "$COMMIT_MSG" ]]; then
    # Auto-generate commit message from changed result dirs
    changed_dirs=$(git diff --cached --name-only | grep -oP 'results_[^/]+' | sort -u | tr '\n' ', ' | sed 's/,$//')
    COMMIT_MSG="Update benchmark results: ${changed_dirs:-results}"
fi

echo ""
echo "Committing: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"

echo ""
echo "Pushing to remote..."
git push

echo ""
echo "============================================================"
echo "  Upload complete!"
echo "============================================================"
