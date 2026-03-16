#!/usr/bin/env bash
# =============================================================================
# One-stop dashboard deploy script
#
# Does everything: import results, fetch competitors, generate data, deploy.
# Run from anywhere — the script finds the repo root automatically.
#
# Usage:
#   # Full pipeline: import B200 results + fetch competitors + deploy
#   bash scripts/deploy_dashboard.sh \
#     --import ./results_b200 --platform "8×B200" --framework "TRT-LLM 1.2.0rc4" --quantization NVFP4
#
#   # Just refresh competitor data and redeploy
#   bash scripts/deploy_dashboard.sh --fetch-competitors
#
#   # Just redeploy existing data (no data changes)
#   bash scripts/deploy_dashboard.sh --deploy-only
#
#   # Import multiple result dirs
#   bash scripts/deploy_dashboard.sh \
#     --import ./results_b200 --platform "8×B200" --framework "TRT-LLM 1.2.0rc4" --quantization NVFP4 \
#     --import ./results_h200 --platform "8×H200" --framework "TRT-LLM 1.2.0rc4" --quantization FP8
#
#   # Skip deploy (just update data locally)
#   bash scripts/deploy_dashboard.sh --fetch-competitors --no-deploy
# =============================================================================

set -euo pipefail

# ======================== Find repo root ======================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ======================== Defaults ============================================
FETCH_COMPETITORS=false
DEPLOY_ONLY=false
NO_DEPLOY=false
SERVE_AFTER=false
SERVE_PORT=8899

# Import jobs: arrays of (results_dir, platform, framework, quantization)
declare -a IMPORT_DIRS=()
declare -a IMPORT_PLATFORMS=()
declare -a IMPORT_FRAMEWORKS=()
declare -a IMPORT_QUANTS=()
IMPORT_IDX=-1

# ======================== Parse args ==========================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --import)
            IMPORT_IDX=$(( IMPORT_IDX + 1 ))
            IMPORT_DIRS+=("$2")
            IMPORT_PLATFORMS+=("")
            IMPORT_FRAMEWORKS+=("")
            IMPORT_QUANTS+=("")
            shift 2
            ;;
        --platform)
            [[ $IMPORT_IDX -ge 0 ]] && IMPORT_PLATFORMS[$IMPORT_IDX]="$2"
            shift 2
            ;;
        --framework)
            [[ $IMPORT_IDX -ge 0 ]] && IMPORT_FRAMEWORKS[$IMPORT_IDX]="$2"
            shift 2
            ;;
        --quantization|--quant)
            [[ $IMPORT_IDX -ge 0 ]] && IMPORT_QUANTS[$IMPORT_IDX]="$2"
            shift 2
            ;;
        --fetch-competitors)
            FETCH_COMPETITORS=true
            shift
            ;;
        --deploy-only)
            DEPLOY_ONLY=true
            shift
            ;;
        --no-deploy)
            NO_DEPLOY=true
            shift
            ;;
        --serve)
            SERVE_AFTER=true
            shift
            ;;
        --port)
            SERVE_PORT="$2"
            shift 2
            ;;
        -h|--help)
            head -25 "$0" | tail -22
            exit 0
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

# If no explicit action, default to fetch-competitors + deploy
if [[ $IMPORT_IDX -lt 0 && "$FETCH_COMPETITORS" == "false" && "$DEPLOY_ONLY" == "false" ]]; then
    FETCH_COMPETITORS=true
fi

TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(TS)] $*"; }

# ======================== Step 1: Import results ==============================
if [[ $IMPORT_IDX -ge 0 && "$DEPLOY_ONLY" == "false" ]]; then
    for i in $(seq 0 $IMPORT_IDX); do
        dir="${IMPORT_DIRS[$i]}"
        platform="${IMPORT_PLATFORMS[$i]}"
        framework="${IMPORT_FRAMEWORKS[$i]}"
        quant="${IMPORT_QUANTS[$i]}"

        if [[ -z "$platform" || -z "$framework" || -z "$quant" ]]; then
            echo "ERROR: --import $dir requires --platform, --framework, --quantization"
            exit 1
        fi

        if [[ ! -d "$dir" ]]; then
            echo "ERROR: Results directory not found: $dir"
            exit 1
        fi

        count=$(ls "$dir"/result_*.json 2>/dev/null | wc -l)
        if [[ $count -eq 0 ]]; then
            log "WARN: No result_*.json in $dir, skipping"
            continue
        fi

        log "Importing $count results from $dir ($platform $quant)"
        python3 "$REPO_ROOT/scripts/import_results.py" \
            --results-dir "$dir" \
            --platform "$platform" \
            --framework "$framework" \
            --quantization "$quant"
    done
fi

# ======================== Step 2: Fetch competitor data ========================
if [[ "$FETCH_COMPETITORS" == "true" && "$DEPLOY_ONLY" == "false" ]]; then
    log "Fetching competitor data..."
    python3 "$REPO_ROOT/scripts/fetch_competitors.py" || \
        log "WARN: Failed to fetch competitor data (network issue?), continuing with existing data"
fi

# ======================== Step 3: Generate dashboard data =====================
if [[ "$DEPLOY_ONLY" == "false" ]]; then
    run_count=$(ls "$REPO_ROOT/runs/"*.json 2>/dev/null | wc -l)
    if [[ $run_count -eq 0 ]]; then
        echo "ERROR: No run data in runs/. Import results or fetch competitors first."
        exit 1
    fi
    log "Generating dashboard data from $run_count runs..."
    python3 "$REPO_ROOT/scripts/generate_dashboard.py"
fi

# ======================== Step 4: Deploy to gh-pages ==========================
if [[ "$NO_DEPLOY" == "true" ]]; then
    log "Skipping deploy (--no-deploy). Dashboard updated locally at dashboard/index.html"
else
    log "Deploying to gh-pages..."

    # Copy dashboard files to temp before any git operations
    TMPDIR=$(mktemp -d)
    trap "rm -rf $TMPDIR" EXIT
    cp "$REPO_ROOT/dashboard/index.html" "$TMPDIR/"
    cp "$REPO_ROOT/dashboard/data.js" "$TMPDIR/"

    # Save current branch and stash if needed
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
    CURRENT_SHA=$(git rev-parse HEAD)
    STASH_NEEDED=false
    if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
        STASH_NEEDED=true
        git stash push -m "deploy_dashboard auto-stash" --quiet
    fi

    # Switch to gh-pages
    if git rev-parse --verify gh-pages >/dev/null 2>&1; then
        git checkout gh-pages --quiet
        # Clean any leftover files
        git ls-files -z | xargs -0 rm -f 2>/dev/null || true
    else
        git checkout --orphan gh-pages --quiet
        git rm -rf . --quiet 2>/dev/null || true
    fi
    git clean -fd --quiet 2>/dev/null || true

    # Place files at root
    cp "$TMPDIR/index.html" .
    cp "$TMPDIR/data.js" .

    git add index.html data.js
    if git diff --cached --quiet 2>/dev/null; then
        log "No changes to deploy"
    else
        git commit -m "deploy: update dashboard $(date +%Y-%m-%d)" --quiet
        git push origin gh-pages --quiet
        log "Deployed to gh-pages"
    fi

    # Return to original branch
    git checkout "$CURRENT_BRANCH" --quiet 2>/dev/null || git checkout "$CURRENT_SHA" --quiet
    if [[ "$STASH_NEEDED" == "true" ]]; then
        git stash pop --quiet 2>/dev/null || true
    fi

    log "Dashboard URL: https://zufayu.github.io/ClaudeSkillsE2EPerf/"
fi

# ======================== Step 5: Optional local server =======================
if [[ "$SERVE_AFTER" == "true" ]]; then
    log "Starting local server on port $SERVE_PORT..."
    cd "$REPO_ROOT/dashboard"
    python3 -m http.server "$SERVE_PORT" &
    log "Local preview: http://localhost:$SERVE_PORT/"
fi

log "Done."
