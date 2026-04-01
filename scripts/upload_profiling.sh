#!/usr/bin/env bash
# =============================================================================
# Upload profiling results to GitHub
#
# Automates: fix permissions → git add (skip traces/logs/large files) → commit → push
#
# Usage:
#   bash scripts/upload_profiling.sh <result_dir> [--message "msg"]
#
# Examples:
#   bash scripts/upload_profiling.sh results_mi355x_mxfp4_mtp0_ep1_tp4_profiling_more
#   bash scripts/upload_profiling.sh results_mi355x_mxfp4_mtp0_ep1_tp4
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMMIT_MSG=""
MAX_SIZE_MB=50

# --- Parse args ---
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <result_dir> [--message \"msg\"]"
    exit 1
fi
RESULT_DIR="$1"; shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --message) COMMIT_MSG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Resolve to absolute path if relative
if [[ ! "$RESULT_DIR" = /* ]]; then
    RESULT_DIR="$REPO_DIR/$RESULT_DIR"
fi
if [[ ! -d "$RESULT_DIR" ]]; then
    echo "ERROR: Directory not found: $RESULT_DIR"
    exit 1
fi

DIR_NAME="$(basename "$RESULT_DIR")"
echo "=== Upload profiling: $DIR_NAME ==="

# --- Step 1: Fix permissions (common issue on shared machines) ---
echo ">>> Fixing file permissions..."
chmod -R u+rw "$RESULT_DIR" 2>/dev/null || true

# --- Step 2: Git pull + add (skip large files) ---
cd "$REPO_DIR"
git pull --ff-only 2>/dev/null || true

echo ">>> Staging files (skipping traces, logs, large files)..."
STAGED=0
SKIPPED=0
while IFS= read -r f; do
    BASENAME="$(basename "$f")"
    # Skip trace files (.json.gz), log files, and libkineto config
    case "$BASENAME" in
        *.trace.json.gz|*.log|libkineto.conf)
            echo "  SKIP (trace/log): $BASENAME"
            SKIPPED=$((SKIPPED + 1))
            continue
            ;;
    esac
    # Skip files >50MB
    SIZE_BYTES=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    SIZE_MB=$(( SIZE_BYTES / 1048576 ))
    REL_PATH="${f#$REPO_DIR/}"
    if [[ $SIZE_MB -gt $MAX_SIZE_MB ]]; then
        echo "  SKIP (${SIZE_MB}MB): $BASENAME"
        SKIPPED=$((SKIPPED + 1))
    else
        git add "$REL_PATH"
        echo "  ADD: $BASENAME"
        STAGED=$((STAGED + 1))
    fi
done < <(find "$RESULT_DIR" -type f)

echo ""
echo "Staged: $STAGED files, Skipped: $SKIPPED large files"

if git diff --cached --quiet; then
    echo "Nothing new to upload."
    exit 0
fi

# --- Step 3: Commit & push ---
if [[ -z "$COMMIT_MSG" ]]; then
    COMMIT_MSG="Add $DIR_NAME profiling outputs"
fi

echo ""
git diff --cached --stat
echo ""
echo ">>> Committing: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"
echo ">>> Pushing..."
git push

echo ""
echo "=== Done: $STAGED files uploaded ==="
