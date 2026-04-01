#!/usr/bin/env bash
# =============================================================================
# Upload profiling results to GitHub
#
# Automates: regenerate breakdown (if trace exists) → git add → commit → push
# Skips large files (>50MB, e.g. trace .json.gz) automatically.
# Fixes file permissions before staging.
#
# Usage:
#   bash scripts/upload_profiling.sh <result_dir> [--layer N] [--message "msg"]
#
# Examples:
#   bash scripts/upload_profiling.sh results_mi355x_mxfp4_mtp0_ep1_tp4_profiling_more
#   bash scripts/upload_profiling.sh results_mi355x_mxfp4_mtp0_ep1_tp4 --layer 40
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LAYER=40
COMMIT_MSG=""
MAX_SIZE_MB=50

# --- Parse args ---
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <result_dir> [--layer N] [--message \"msg\"]"
    exit 1
fi
RESULT_DIR="$1"; shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --layer)   LAYER="$2"; shift 2 ;;
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

# --- Step 2: Regenerate breakdown if trace exists ---
TRACE_FILE=$(ls "$RESULT_DIR"/*.pt.trace.json.gz 2>/dev/null | grep -v capture_graph | head -1) || true
RUN_PARSE="$SCRIPT_DIR/run_parse_trace.py"

if [[ -n "$TRACE_FILE" && -f "$RUN_PARSE" ]]; then
    echo ">>> Regenerating kernel breakdown (layer=$LAYER)..."
    (cd "$RESULT_DIR" && ATOM_TOOLS=/app/ATOM/tools python3 "$RUN_PARSE" "$TRACE_FILE" --layer "$LAYER") || echo "WARNING: run_parse_trace.py failed, using existing xlsx"
    echo ""
fi

# --- Step 3: Git pull + add (skip large files) ---
cd "$REPO_DIR"
git pull --ff-only 2>/dev/null || true

echo ">>> Staging files (skipping >${MAX_SIZE_MB}MB)..."
STAGED=0
SKIPPED=0
while IFS= read -r f; do
    SIZE_BYTES=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    SIZE_MB=$(( SIZE_BYTES / 1048576 ))
    REL_PATH="${f#$REPO_DIR/}"
    if [[ $SIZE_MB -gt $MAX_SIZE_MB ]]; then
        echo "  SKIP (${SIZE_MB}MB): $(basename "$f")"
        SKIPPED=$((SKIPPED + 1))
    else
        git add "$REL_PATH"
        echo "  ADD: $(basename "$f")"
        STAGED=$((STAGED + 1))
    fi
done < <(find "$RESULT_DIR" -type f)

echo ""
echo "Staged: $STAGED files, Skipped: $SKIPPED large files"

if git diff --cached --quiet; then
    echo "Nothing new to upload."
    exit 0
fi

# --- Step 4: Commit & push ---
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
