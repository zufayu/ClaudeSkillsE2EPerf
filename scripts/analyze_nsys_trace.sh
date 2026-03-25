#!/usr/bin/env bash
# =============================================================================
# Nsight Systems Trace Analysis
#
# Exports .nsys-rep to SQLite (via nsys export) then analyzes kernel breakdown
# using Python sqlite3 (no sqlite3 CLI needed).
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
  --trace PATH          Path to .nsys-rep or .sqlite trace file

Options:
  --top N               Number of top kernels to show [default: 20]
  --timerange NS-NS     Filter by time range in nanoseconds (e.g., 1000000000-2000000000)
  --export-only         Export to SQLite/CSV only, skip analysis queries
  -h, --help            Show this help message

Examples:
  bash $(basename "$0") --trace traces/foo.nsys-rep
  bash $(basename "$0") --trace traces/foo.sqlite --top 30
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

# ======================== Determine SQLite path ===============================
if [[ "$TRACE_FILE" == *.sqlite ]]; then
    SQLITE_FILE="$TRACE_FILE"
    log "=== Nsight Systems Trace Analysis ==="
    log "  SQLite input: $SQLITE_FILE"
else
    BASENAME="$(basename "$TRACE_FILE" .nsys-rep)"
    TRACE_DIR="$(dirname "$TRACE_FILE")"
    SQLITE_FILE="$TRACE_DIR/${BASENAME}.sqlite"

    log "=== Nsight Systems Trace Analysis ==="
    log "  Trace: $TRACE_FILE"
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
fi

if [[ "$EXPORT_ONLY" == "true" ]]; then
    log "Export complete (--export-only)."
    exit 0
fi

# ======================== Python Analysis =====================================
log "Running analysis (python3 sqlite3)..."

python3 - "$SQLITE_FILE" "$TOP_N" "$TIMERANGE" <<'PYEOF'
import sqlite3
import sys
import os

db_path = sys.argv[1]
top_n = int(sys.argv[2])
timerange = sys.argv[3] if len(sys.argv) > 3 else ""

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# ---- Auto-detect kernel table name ----
tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

KERNEL_TABLE = None
for candidate in ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME",
                   "TARGET_INFO_CUDA_GPU_TRACE", "CUDA_GPU_TRACE"]:
    if candidate in tables:
        KERNEL_TABLE = candidate
        break
# Fallback: find any table with 'kernel' or 'gpu' in name
if not KERNEL_TABLE:
    for t in tables:
        tl = t.lower()
        if 'kernel' in tl or 'gpu_trace' in tl:
            KERNEL_TABLE = t
            break

NVTX_TABLE = None
for candidate in ["NVTX_EVENTS", "NVTX_RANGES"]:
    if candidate in tables:
        NVTX_TABLE = candidate
        break
if not NVTX_TABLE:
    for t in tables:
        if 'nvtx' in t.lower():
            NVTX_TABLE = t
            break

print(f"\n=== Database Info ===")
print(f"  Tables ({len(tables)}): {', '.join(sorted(tables))}")
print(f"  Kernel table: {KERNEL_TABLE or 'NOT FOUND'}")
print(f"  NVTX table:   {NVTX_TABLE or 'NOT FOUND'}")

if not KERNEL_TABLE:
    print("\nERROR: No kernel table found. Available tables:")
    for t in sorted(tables):
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info('{t}')").fetchall()]
        print(f"  {t}: {', '.join(cols[:8])}{'...' if len(cols)>8 else ''}")
    sys.exit(1)

# ---- Detect column names ----
kcols = {r[1].lower(): r[1] for r in cur.execute(f"PRAGMA table_info('{KERNEL_TABLE}')").fetchall()}
print(f"  Kernel columns: {', '.join(kcols.keys())}")

# Map columns - nsys versions use different names
name_col = kcols.get('shortname') or kcols.get('name') or kcols.get('demangledname') or kcols.get('kernel_name') or 'shortName'
start_col = kcols.get('start') or kcols.get('startns') or kcols.get('start_time') or 'start'
end_col = kcols.get('end') or kcols.get('endns') or kcols.get('end_time') or 'end'
dur_col = kcols.get('duration') or kcols.get('dur') or None

if dur_col:
    dur_expr = dur_col
else:
    dur_expr = f"({end_col}-{start_col})"

# ---- Time filter ----
time_filter = ""
if timerange:
    parts = timerange.split("-")
    if len(parts) == 2:
        time_filter = f"AND {start_col} >= {parts[0]} AND {end_col} <= {parts[1]}"
        print(f"  Time range: {parts[0]}ns - {parts[1]}ns")

# ---- Helper ----
def print_table(headers, rows, widths=None):
    if not rows:
        print("  (no data)")
        return
    if not widths:
        widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-"*w for w in widths))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))

# ======================== Query 1: Top N Kernels =============================
print(f"\n=== 1. Top {top_n} Kernels by Total GPU Time ===")
q = f"""SELECT {name_col} AS kernel,
        COUNT(*) AS count,
        ROUND(CAST(SUM({dur_expr}) AS FLOAT) / 1e6, 2) AS total_ms,
        ROUND(CAST(AVG({dur_expr}) AS FLOAT) / 1e3, 2) AS avg_us,
        ROUND(CAST(MIN({dur_expr}) AS FLOAT) / 1e3, 2) AS min_us,
        ROUND(CAST(MAX({dur_expr}) AS FLOAT) / 1e3, 2) AS max_us,
        ROUND(100.0 * SUM({dur_expr}) / (SELECT SUM({dur_expr}) FROM {KERNEL_TABLE} WHERE 1=1 {time_filter}), 1) AS pct
 FROM {KERNEL_TABLE}
 WHERE 1=1 {time_filter}
 GROUP BY {name_col}
 ORDER BY SUM({dur_expr}) DESC
 LIMIT {top_n}"""
try:
    rows = cur.execute(q).fetchall()
    print_table(["kernel", "count", "total_ms", "avg_us", "min_us", "max_us", "pct%"], rows)
except Exception as e:
    print(f"  WARN: Query failed: {e}")

# ======================== Query 2: Category Breakdown ========================
print(f"\n=== 2. MoE vs Attention vs NCCL vs GEMM Time Split ===")
q = f"""SELECT CASE
       WHEN {name_col} LIKE '%moe%' OR {name_col} LIKE '%MoE%'
            OR {name_col} LIKE '%expert%' OR {name_col} LIKE '%Expert%'
            OR {name_col} LIKE '%expandInput%' OR {name_col} LIKE '%doActivation%'
            OR {name_col} LIKE '%topk%' OR {name_col} LIKE '%buildExpert%'
            OR {name_col} LIKE '%computeStrides%' THEN 'MoE'
       WHEN {name_col} LIKE '%attention%' OR {name_col} LIKE '%flash%'
            OR {name_col} LIKE '%fmha%' OR {name_col} LIKE '%Fmha%' THEN 'Attention'
       WHEN {name_col} LIKE '%nccl%' OR {name_col} LIKE '%allreduce%'
            OR {name_col} LIKE '%AllReduce%' OR {name_col} LIKE '%alltoall%'
            OR {name_col} LIKE '%AllToAll%' THEN 'NCCL/Comm'
       WHEN {name_col} LIKE '%gemm%' OR {name_col} LIKE '%Gemm%'
            OR {name_col} LIKE '%cutlass%' OR {name_col} LIKE '%cublas%'
            OR {name_col} LIKE '%nvjet%' THEN 'GEMM'
       WHEN {name_col} LIKE '%Norm%' OR {name_col} LIKE '%norm%' THEN 'Norm'
       WHEN {name_col} LIKE '%rope%' OR {name_col} LIKE '%Rope%'
            OR {name_col} LIKE '%RoPE%' THEN 'RoPE'
       WHEN {name_col} LIKE '%quantize%' OR {name_col} LIKE '%Quantize%'
            OR {name_col} LIKE '%dequant%' THEN 'Quantize'
       WHEN {name_col} LIKE '%memcpy%' OR {name_col} LIKE '%memset%'
            OR {name_col} LIKE '%Memcpy%' OR {name_col} LIKE '%Memset%' THEN 'Memory'
       ELSE 'Other'
     END AS category,
     COUNT(*) AS kernel_count,
     ROUND(CAST(SUM({dur_expr}) AS FLOAT) / 1e6, 2) AS total_ms,
     ROUND(100.0 * SUM({dur_expr}) / (SELECT SUM({dur_expr}) FROM {KERNEL_TABLE} WHERE 1=1 {time_filter}), 1) AS pct
     FROM {KERNEL_TABLE}
     WHERE 1=1 {time_filter}
     GROUP BY category
     ORDER BY SUM({dur_expr}) DESC"""
try:
    rows = cur.execute(q).fetchall()
    print_table(["category", "kernel_count", "total_ms", "pct%"], rows)
except Exception as e:
    print(f"  WARN: Query failed: {e}")

# ======================== Query 3: NVTX Events ===============================
if NVTX_TABLE:
    print(f"\n=== 3. NVTX Event Durations (Layer-Level Breakdown) ===")
    ncols = {r[1].lower(): r[1] for r in cur.execute(f"PRAGMA table_info('{NVTX_TABLE}')").fetchall()}
    text_col = ncols.get('text') or ncols.get('name') or ncols.get('message') or 'text'
    nstart = ncols.get('start') or ncols.get('startns') or 'start'
    nend = ncols.get('end') or ncols.get('endns') or 'end'
    ndur = ncols.get('duration') or ncols.get('dur') or None
    ndur_expr = ndur if ndur else f"({nend}-{nstart})"

    q = f"""SELECT {text_col} AS event,
            COUNT(*) AS count,
            ROUND(CAST(SUM({ndur_expr}) AS FLOAT) / 1e6, 2) AS total_ms,
            ROUND(CAST(AVG({ndur_expr}) AS FLOAT) / 1e3, 2) AS avg_us
     FROM {NVTX_TABLE}
     WHERE ({text_col} LIKE '%layer%' OR {text_col} LIKE '%Layer%'
            OR {text_col} LIKE '%MoE%' OR {text_col} LIKE '%moe%'
            OR {text_col} LIKE '%attention%' OR {text_col} LIKE '%Attention%'
            OR {text_col} LIKE '%expert%' OR {text_col} LIKE '%Expert%'
            OR {text_col} LIKE '%allreduce%' OR {text_col} LIKE '%decode%'
            OR {text_col} LIKE '%prefill%' OR {text_col} LIKE '%forward%')
     GROUP BY {text_col}
     ORDER BY SUM({ndur_expr}) DESC
     LIMIT 30"""
    try:
        rows = cur.execute(q).fetchall()
        print_table(["event", "count", "total_ms", "avg_us"], rows)
    except Exception as e:
        print(f"  WARN: NVTX query failed: {e}")
else:
    print(f"\n=== 3. NVTX Events === (no NVTX table found)")

# ======================== Query 4: Overall Stats =============================
print(f"\n=== 4. Overall Statistics ===")
q = f"""SELECT COUNT(*) AS total_kernels,
        COUNT(DISTINCT {name_col}) AS unique_kernels,
        ROUND(CAST(SUM({dur_expr}) AS FLOAT) / 1e9, 3) AS total_gpu_time_s,
        ROUND(CAST(AVG({dur_expr}) AS FLOAT) / 1e3, 2) AS avg_kernel_us
 FROM {KERNEL_TABLE}
 WHERE 1=1 {time_filter}"""
try:
    rows = cur.execute(q).fetchall()
    print_table(["total_kernels", "unique_kernels", "total_gpu_time_s", "avg_kernel_us"], rows)
except Exception as e:
    print(f"  WARN: Stats query failed: {e}")

print(f"\n=== Analysis Complete ===")
print(f"  SQLite: {db_path}")
print(f"  Custom queries: python3 -c \"import sqlite3; c=sqlite3.connect('{db_path}'); ...\"")

conn.close()
PYEOF

log "Done."
