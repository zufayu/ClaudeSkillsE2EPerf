#!/usr/bin/env bash
# =============================================================================
# Nsight Systems Trace Analysis
#
# Companion to collect_nsys_trace.sh. Exports traces to SQLite and runs
# kernel breakdown queries for cross-platform performance comparison.
#
# Usage:
#   bash scripts/analyze_nsys_trace.sh --trace traces/foo.nsys-rep
#   bash scripts/analyze_nsys_trace.sh --trace traces/foo.nsys-rep --top 30
#   bash scripts/analyze_nsys_trace.sh --trace traces/foo.nsys-rep --export-only
# =============================================================================

set -euo pipefail

# ======================== Defaults ============================================
TRACE_FILE=""
TOP_N=20
EXPORT_ONLY=false
TIMERANGE=""

# ======================== Argument Parsing ====================================
usage() {
    cat <<EOF
Usage: bash $(basename "$0") [options]

Analyze Nsight Systems traces for TRT-LLM kernel-level performance.

Required:
  --trace PATH          Path to .nsys-rep trace file

Options:
  --top N               Number of top kernels to show [default: 20]
  --timerange NS-NS     Filter by time range in nanoseconds (e.g., 1000000000-2000000000)
  --export-only         Export to SQLite/CSV only, skip analysis queries
  -h, --help            Show this help message

Examples:
  # Quick kernel summary
  bash $(basename "$0") --trace traces/nsys_fp8_latency_chat_tp8_ep1_c32_iter100-150.nsys-rep

  # Top 30 kernels with time range filter
  bash $(basename "$0") --trace traces/foo.nsys-rep --top 30 --timerange 1000000000-2000000000

  # Export only
  bash $(basename "$0") --trace traces/foo.nsys-rep --export-only
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --trace)       TRACE_FILE="$2"; shift 2 ;;
        --top)         TOP_N="$2"; shift 2 ;;
        --timerange)   TIMERANGE="$2"; shift 2 ;;
        --export-only) EXPORT_ONLY=true; shift ;;
        -h|--help)     usage ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$TRACE_FILE" ]]; then
    echo "ERROR: --trace is required"
    usage
fi

if [[ ! -f "$TRACE_FILE" ]]; then
    echo "ERROR: Trace file not found: $TRACE_FILE"
    exit 1
fi

# ======================== Utilities ===========================================
TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(TS)] $*"; }

BASENAME="$(basename "$TRACE_FILE" .nsys-rep)"
TRACE_DIR="$(dirname "$TRACE_FILE")"
SQLITE_FILE="$TRACE_DIR/${BASENAME}.sqlite"

# ======================== Export ==============================================
log "=== Nsight Systems Trace Analysis ==="
log "  Trace: $TRACE_FILE"

# File size
SIZE_MB=$(du -m "$TRACE_FILE" | cut -f1)
log "  Size: ${SIZE_MB} MB"

# Export to SQLite if needed
if [[ ! -f "$SQLITE_FILE" ]]; then
    log "Exporting to SQLite..."
    nsys export --type sqlite -o "$SQLITE_FILE" "$TRACE_FILE"
    log "  -> $SQLITE_FILE"
else
    log "SQLite file exists: $SQLITE_FILE"
fi

# Export kernel CSV
CSV_PREFIX="$TRACE_DIR/${BASENAME}_kernels"
if [[ ! -f "${CSV_PREFIX}_cuda_gpu_trace.csv" ]]; then
    log "Exporting kernel trace CSV..."
    nsys stats --report cuda_gpu_trace --format csv \
        -o "$CSV_PREFIX" "$TRACE_FILE" 2>&1 || \
        log "WARN: Kernel CSV export failed"
else
    log "Kernel CSV exists: ${CSV_PREFIX}_cuda_gpu_trace.csv"
fi

if [[ "$EXPORT_ONLY" == "true" ]]; then
    log "Export complete (--export-only)."
    exit 0
fi

# ======================== Time Range Filter ===================================
TIME_FILTER=""
if [[ -n "$TIMERANGE" ]]; then
    IFS='-' read -r T_START T_END <<< "$TIMERANGE"
    TIME_FILTER="AND start >= $T_START AND end <= $T_END"
    log "  Time range filter: ${T_START}ns - ${T_END}ns"
fi

# ======================== Analysis Queries ====================================

echo ""
log "=== 1. Top $TOP_N Kernels by Total GPU Time ==="
sqlite3 -header -column "$SQLITE_FILE" \
    "SELECT shortName AS kernel,
            COUNT(*) AS count,
            ROUND(CAST(SUM(end-start) AS FLOAT) / 1e6, 2) AS total_ms,
            ROUND(CAST(AVG(end-start) AS FLOAT) / 1e3, 2) AS avg_us,
            ROUND(CAST(MIN(end-start) AS FLOAT) / 1e3, 2) AS min_us,
            ROUND(CAST(MAX(end-start) AS FLOAT) / 1e3, 2) AS max_us
     FROM CUPTI_ACTIVITY_KIND_KERNEL
     WHERE 1=1 $TIME_FILTER
     GROUP BY shortName
     ORDER BY SUM(end-start) DESC
     LIMIT $TOP_N;" 2>/dev/null || \
    log "WARN: Query failed (table may not exist)"

echo ""
log "=== 2. NVTX Event Durations (Layer-Level Breakdown) ==="
sqlite3 -header -column "$SQLITE_FILE" \
    "SELECT text AS event,
            COUNT(*) AS count,
            ROUND(CAST(SUM(end-start) AS FLOAT) / 1e6, 2) AS total_ms,
            ROUND(CAST(AVG(end-start) AS FLOAT) / 1e3, 2) AS avg_us
     FROM NVTX_EVENTS
     WHERE (text LIKE '%layer%'
            OR text LIKE '%MoE%'
            OR text LIKE '%attention%'
            OR text LIKE '%moe%'
            OR text LIKE '%expert%'
            OR text LIKE '%allreduce%')
           $TIME_FILTER
     GROUP BY text
     ORDER BY SUM(end-start) DESC
     LIMIT 30;" 2>/dev/null || \
    log "WARN: NVTX query failed (table may not exist)"

echo ""
log "=== 3. MoE vs Attention vs NCCL Time Split ==="
sqlite3 -header -column "$SQLITE_FILE" \
    "SELECT CASE
       WHEN shortName LIKE '%moe%' OR shortName LIKE '%expert%'
            OR shortName LIKE '%MoE%' THEN 'MoE'
       WHEN shortName LIKE '%attention%' OR shortName LIKE '%flash%'
            OR shortName LIKE '%fmha%' THEN 'Attention'
       WHEN shortName LIKE '%nccl%' OR shortName LIKE '%allreduce%'
            OR shortName LIKE '%AllReduce%' THEN 'NCCL'
       WHEN shortName LIKE '%gemm%' OR shortName LIKE '%cutlass%'
            OR shortName LIKE '%cublas%' THEN 'GEMM'
       WHEN shortName LIKE '%memcpy%' OR shortName LIKE '%memset%' THEN 'Memory'
       ELSE 'Other'
     END AS category,
     COUNT(*) AS kernel_count,
     ROUND(CAST(SUM(end-start) AS FLOAT) / 1e6, 2) AS total_ms,
     ROUND(100.0 * SUM(end-start) / (SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE 1=1 $TIME_FILTER), 1) AS pct
     FROM CUPTI_ACTIVITY_KIND_KERNEL
     WHERE 1=1 $TIME_FILTER
     GROUP BY category
     ORDER BY SUM(end-start) DESC;" 2>/dev/null || \
    log "WARN: Category query failed"

echo ""
log "=== 4. Overall Statistics ==="
sqlite3 -header -column "$SQLITE_FILE" \
    "SELECT COUNT(*) AS total_kernels,
            COUNT(DISTINCT shortName) AS unique_kernels,
            ROUND(CAST(SUM(end-start) AS FLOAT) / 1e9, 3) AS total_gpu_time_s,
            ROUND(CAST(AVG(end-start) AS FLOAT) / 1e3, 2) AS avg_kernel_us
     FROM CUPTI_ACTIVITY_KIND_KERNEL
     WHERE 1=1 $TIME_FILTER;" 2>/dev/null || \
    log "WARN: Stats query failed"

echo ""
log "=== Analysis Complete ==="
log "  SQLite: $SQLITE_FILE"
log "  For custom queries: sqlite3 $SQLITE_FILE"
