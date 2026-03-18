#!/usr/bin/env bash
# =============================================================================
# MI355X MTP comparison: run FP8 throughput (MTP=0) and latency (MTP=3)
# with matching configs, then auto-import + generate dashboard + push.
#
# Usage: bash scripts/run_mi355x_mtp_compare.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_FP8="/data/DeepSeek-R1-0528"

cd "$REPO_DIR"

TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(TS)] $*"; }

log "============================================================"
log "  MI355X MTP Comparison Benchmark"
log "  MTP=0 (throughput) vs MTP=3 (latency)"
log "============================================================"

# --- Step 1: MTP=0 (fp8-throughput) ---
# Matching configs: chat c=1,2,4,8,16,32,64,128,256 + reasoning c=2,4,8,16
log ""
log "===== PHASE 1: FP8 Throughput (MTP=0) ====="

bash "$SCRIPT_DIR/sa_bench_mi355x.sh" \
  --model-fp8 "$MODEL_FP8" \
  --configs fp8-throughput \
  --scenario chat \
  --concurrency "1 2 4 8 16 32 64 128 256" \
  --result-dir ./results_mi355x_mtp0

bash "$SCRIPT_DIR/sa_bench_mi355x.sh" \
  --model-fp8 "$MODEL_FP8" \
  --configs fp8-throughput \
  --scenario reasoning \
  --concurrency "2 4 8 16" \
  --result-dir ./results_mi355x_mtp0

log "===== PHASE 1 COMPLETE ====="

# --- Step 2: MTP=3 (fp8-latency) ---
# fp8-latency limits c<=64, so trim chat to c<=64
log ""
log "===== PHASE 2: FP8 Latency (MTP=3) ====="

bash "$SCRIPT_DIR/sa_bench_mi355x.sh" \
  --model-fp8 "$MODEL_FP8" \
  --configs fp8-latency \
  --scenario chat \
  --concurrency "1 2 4 8 16 32 64" \
  --result-dir ./results_mi355x_mtp3

bash "$SCRIPT_DIR/sa_bench_mi355x.sh" \
  --model-fp8 "$MODEL_FP8" \
  --configs fp8-latency \
  --scenario reasoning \
  --concurrency "2 4 8 16" \
  --result-dir ./results_mi355x_mtp3

log "===== PHASE 2 COMPLETE ====="

# --- Step 3: Import results ---
log ""
log "===== PHASE 3: Import & Upload ====="

python3 "$SCRIPT_DIR/import_results.py" \
  --results-dir ./results_mi355x_mtp0 \
  --platform "8×MI355X" --framework "ATOM" --quantization FP8 \
  --env-tag "mtp0"

python3 "$SCRIPT_DIR/import_results.py" \
  --results-dir ./results_mi355x_mtp3 \
  --platform "8×MI355X" --framework "ATOM" --quantization FP8 \
  --env-tag "mtp3"

# --- Step 4: Regenerate dashboard ---
python3 "$SCRIPT_DIR/generate_dashboard.py"

# --- Step 5: Commit & push ---
git add runs/ docs/data.js
if git diff --cached --quiet; then
    log "No changes to commit"
else
    git commit -m "data: MI355X FP8 MTP=0 vs MTP=3 comparison benchmark results"
    git push
    log "Pushed to remote. Dashboard will update at:"
    log "  https://zufayu.github.io/ClaudeSkillsE2EPerf/"
fi

log ""
log "============================================================"
log "  ALL DONE. Results:"
log "    MTP=0: ./results_mi355x_mtp0/"
log "    MTP=3: ./results_mi355x_mtp3/"
log "  Dashboard: https://zufayu.github.io/ClaudeSkillsE2EPerf/"
log "============================================================"
