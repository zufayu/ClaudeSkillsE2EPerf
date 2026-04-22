#!/usr/bin/env bash
# =============================================================================
# Unified Dashboard Pipeline
#
# Replaces: deploy_dashboard.sh, refresh_dashboard.sh, update_dashboard.sh
# (Hermes pattern: one script with --mode, not 3 copy-paste wrappers)
#
# Modes:
#   refresh   Trim logs + import + fetch + generate (local only, no push)
#   update    Import + fetch + generate + git push to main
#   deploy    Import + fetch + generate + deploy to gh-pages
#
# Usage:
#   bash scripts/dashboard.sh --mode refresh
#   bash scripts/dashboard.sh --mode update --results-dir ./results --platform "8×B200" --framework "TRT-LLM" --quantization FP4
#   bash scripts/dashboard.sh --mode deploy --import ./results_b200 --fetch-competitors
#   bash scripts/dashboard.sh --mode deploy --deploy-only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/benchmark_lib.sh"

# ======================== Defaults ============================================
MODE=""
RESULTS_DIR=""
PLATFORM=""
FRAMEWORK=""
QUANTIZATION=""
FETCH_COMPETITORS=true
NO_PUSH=false
DEPLOY_ONLY=false
COMMIT_MSG=""

# ======================== CLI Parsing =========================================
usage() {
    cat <<EOF
Usage: bash scripts/dashboard.sh --mode <refresh|update|deploy> [options]

Modes:
  refresh     Trim logs + import all known results + fetch competitors + regenerate (no push)
  update      Import results + fetch competitors + regenerate + git push
  deploy      Full pipeline: import + fetch + generate + deploy to gh-pages

Options:
  --results-dir DIR     Results directory to import
  --platform NAME       Platform name (e.g., "8×B200")
  --framework NAME      Framework name (e.g., "TRT-LLM 1.3.0rc10")
  --quantization NAME   Quantization (e.g., "NVFP4", "FP8")
  --no-fetch            Skip competitor data fetch
  --no-push             Skip git push (update mode)
  --deploy-only         Skip import/fetch, just deploy existing data (deploy mode)
  --message MSG         Custom commit message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)           MODE="$2"; shift 2 ;;
        --results-dir)    RESULTS_DIR="$2"; shift 2 ;;
        --platform)       PLATFORM="$2"; shift 2 ;;
        --framework)      FRAMEWORK="$2"; shift 2 ;;
        --quantization)   QUANTIZATION="$2"; shift 2 ;;
        --no-fetch)       FETCH_COMPETITORS=false; shift ;;
        --no-push)        NO_PUSH=true; shift ;;
        --deploy-only)    DEPLOY_ONLY=true; shift ;;
        --message|-m)     COMMIT_MSG="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *)                echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$MODE" ]] && { echo "ERROR: --mode is required"; usage; }

# ======================== Shared Steps ========================================

step_trim_logs() {
    log "Trimming server logs..."
    python3 "$SCRIPT_DIR/trim_logs.py" --all --base-dir results 2>/dev/null || \
        log "WARN: trim_logs.py failed (non-critical)"
}

step_import() {
    if [[ -n "$RESULTS_DIR" ]]; then
        local count
        count=$(find "$RESULTS_DIR" -name "result_*.json" 2>/dev/null | wc -l)
        if [[ $count -eq 0 ]]; then
            log "WARN: No result_*.json in $RESULTS_DIR"
            return
        fi
        log "Importing $count results from $RESULTS_DIR"
        python3 "$SCRIPT_DIR/import_results.py" \
            --results-dir "$RESULTS_DIR" \
            ${PLATFORM:+--platform "$PLATFORM"} \
            ${FRAMEWORK:+--framework "$FRAMEWORK"} \
            ${QUANTIZATION:+--quantization "$QUANTIZATION"} \
            --output-dir runs/
    else
        log "No --results-dir specified, using existing runs/"
    fi
}

step_fetch() {
    if [[ "$FETCH_COMPETITORS" == "true" ]]; then
        log "Fetching competitor data..."
        python3 "$SCRIPT_DIR/fetch_competitors.py" 2>/dev/null || \
            log "WARN: fetch_competitors.py failed (network?), using existing data"
    else
        log "Skipping competitor fetch (--no-fetch)"
    fi
}

step_generate() {
    local run_count
    run_count=$(find runs/ -name "*.json" 2>/dev/null | wc -l)
    log "Generating dashboard from $run_count runs..."
    python3 "$SCRIPT_DIR/generate_dashboard.py"
}

step_push() {
    if [[ "$NO_PUSH" == "true" ]]; then
        log "Skipping push (--no-push)"
        log "Preview: cd docs && python3 -m http.server 8899"
        return
    fi
    local msg="${COMMIT_MSG:-data: dashboard update ($(date +%Y-%m-%d))}"
    git add runs/ docs/data.js
    if git diff --cached --quiet; then
        log "Nothing to commit"
    else
        git commit -m "$msg"
        git push
        log "Pushed to main"
    fi
}

step_deploy_ghpages() {
    log "Deploying to gh-pages..."
    local tmpdir
    tmpdir=$(mktemp -d)
    cp -r docs/* "$tmpdir/"

    if git rev-parse --verify gh-pages >/dev/null 2>&1; then
        git checkout gh-pages --quiet
    else
        git checkout --orphan gh-pages --quiet
    fi
    git rm -rf . --quiet 2>/dev/null || true
    cp -r "$tmpdir"/* .
    git add .
    git commit -m "deploy: dashboard $(date +%Y-%m-%d)" --quiet
    git push origin gh-pages --quiet
    git checkout main --quiet
    rm -rf "$tmpdir"
    log "Deployed to gh-pages"
}

# ======================== Mode Dispatch =======================================

case "$MODE" in
    refresh)
        log "=== Dashboard Refresh ==="
        step_trim_logs
        step_import
        step_fetch
        step_generate
        log "Done. Preview: cd docs && python3 -m http.server 8899"
        ;;
    update)
        log "=== Dashboard Update ==="
        step_import
        step_fetch
        step_generate
        step_push
        ;;
    deploy)
        log "=== Dashboard Deploy ==="
        if [[ "$DEPLOY_ONLY" == "false" ]]; then
            step_import
            step_fetch
            step_generate
        fi
        step_deploy_ghpages
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'. Use: refresh|update|deploy"
        exit 1
        ;;
esac
