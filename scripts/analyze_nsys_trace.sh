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

db_path = sys.argv[1]
top_n = int(sys.argv[2])
timerange = sys.argv[3] if len(sys.argv) > 3 else ""

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# ---- Auto-detect tables ----
tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

KERNEL_TABLE = None
for c in ["CUPTI_ACTIVITY_KIND_KERNEL", "CUDA_GPU_TRACE"]:
    if c in tables:
        KERNEL_TABLE = c
        break
if not KERNEL_TABLE:
    for t in tables:
        if 'kernel' in t.lower() or 'gpu_trace' in t.lower():
            KERNEL_TABLE = t
            break

NVTX_TABLE = "NVTX_EVENTS" if "NVTX_EVENTS" in tables else None
HAS_STRINGS = "StringIds" in tables

print(f"\n=== Database Info ===")
print(f"  Kernel table: {KERNEL_TABLE or 'NOT FOUND'}")
print(f"  NVTX table:   {NVTX_TABLE or 'NOT FOUND'}")
print(f"  StringIds:    {'YES' if HAS_STRINGS else 'NO'}")

if not KERNEL_TABLE:
    print("\nERROR: No kernel table found. Tables:", ", ".join(sorted(tables)))
    sys.exit(1)

# ---- Detect columns ----
kcols = {r[1].lower(): r[1] for r in cur.execute(f"PRAGMA table_info('{KERNEL_TABLE}')").fetchall()}

name_col = kcols.get('shortname') or kcols.get('name') or kcols.get('demangledname') or 'shortName'
start_col = kcols.get('start') or 'start'
end_col = kcols.get('end') or 'end'
dur_col = kcols.get('duration') or kcols.get('dur') or None
dur_expr = dur_col if dur_col else f"({end_col}-{start_col})"

# ---- Check if name column stores string IDs (integers) ----
sample = cur.execute(f"SELECT {name_col} FROM {KERNEL_TABLE} LIMIT 1").fetchone()
name_is_id = HAS_STRINGS and sample and isinstance(sample[0], int)

if name_is_id:
    scols = {r[1].lower(): r[1] for r in cur.execute("PRAGMA table_info('StringIds')").fetchall()}
    sid_col = scols.get('id') or scols.get('rowid') or 'id'
    sval_col = scols.get('value') or scols.get('string') or scols.get('name') or 'value'
    print(f"  StringIds columns: {', '.join(scols.keys())}")
    print(f"  Name column '{name_col}' stores integer IDs -> joining with StringIds.{sval_col}")
    # Also resolve demangledName for better classification
    demangled_col = kcols.get('demangledname')
    if demangled_col:
        K = f"""(SELECT k.*, sn.{sval_col} AS _kname, sd.{sval_col} AS _dname
                 FROM {KERNEL_TABLE} k
                 JOIN StringIds sn ON k.{name_col} = sn.{sid_col}
                 JOIN StringIds sd ON k.{demangled_col} = sd.{sid_col})"""
        resolved_name = "_kname"
        resolved_demangled = "_dname"
        print(f"  Also resolving demangledName for classification")
    else:
        K = f"(SELECT k.*, s.{sval_col} AS _kname FROM {KERNEL_TABLE} k JOIN StringIds s ON k.{name_col} = s.{sid_col})"
        resolved_name = "_kname"
        resolved_demangled = "_kname"
else:
    K = KERNEL_TABLE
    resolved_name = name_col
    resolved_demangled = kcols.get('demangledname') or name_col
    print(f"  Name column '{name_col}' stores strings directly")

# ---- Time filter ----
time_filter = ""
if timerange:
    parts = timerange.split("-")
    if len(parts) == 2:
        time_filter = f"AND {start_col} >= {parts[0]} AND {end_col} <= {parts[1]}"
        print(f"  Time range: {parts[0]}ns - {parts[1]}ns")

# Total GPU time for percentage calc
total_gpu = cur.execute(f"SELECT SUM({dur_expr}) FROM {KERNEL_TABLE} WHERE 1=1 {time_filter}").fetchone()[0]

# ---- Helper ----
def print_table(headers, rows):
    if not rows:
        print("  (no data)")
        return
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    # Cap kernel name width at 90 chars for readability
    if headers[0] in ('kernel', 'event'):
        widths[0] = min(widths[0], 90)
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-"*w for w in widths))
    for row in rows:
        cells = [str(v) for v in row]
        if headers[0] in ('kernel', 'event') and len(cells[0]) > 90:
            cells[0] = cells[0][:87] + "..."
        print(fmt.format(*cells))

# ======================== Query 1: Top N Kernels =============================
print(f"\n=== 1. Top {top_n} Kernels by Total GPU Time ===")
q = f"""SELECT {resolved_name} AS kernel,
        COUNT(*) AS count,
        ROUND(CAST(SUM({dur_expr}) AS FLOAT) / 1e6, 2) AS total_ms,
        ROUND(CAST(AVG({dur_expr}) AS FLOAT) / 1e3, 2) AS avg_us,
        ROUND(CAST(MIN({dur_expr}) AS FLOAT) / 1e3, 2) AS min_us,
        ROUND(CAST(MAX({dur_expr}) AS FLOAT) / 1e3, 2) AS max_us,
        ROUND(100.0 * SUM({dur_expr}) / {total_gpu}, 1) AS pct
 FROM {K}
 WHERE 1=1 {time_filter}
 GROUP BY {resolved_name}
 ORDER BY SUM({dur_expr}) DESC
 LIMIT {top_n}"""
try:
    rows = cur.execute(q).fetchall()
    print_table(["kernel", "count", "total_ms", "avg_us", "min_us", "max_us", "pct%"], rows)
except Exception as e:
    print(f"  WARN: Query failed: {e}")

# ======================== Query 2: Category Breakdown ========================
print(f"\n=== 2. MoE vs Attention vs NCCL vs GEMM Time Split ===")
# Use demangledName for classification (has full kernel signature like cutlass_sm100_mxf4...)
# shortName can be generic (e.g. "device_kernel" for all CUTLASS kernels)
n = resolved_name
d = resolved_demangled
q = f"""SELECT CASE
       WHEN {n} LIKE '%moe%' OR {n} LIKE '%MoE%'
            OR {n} LIKE '%expert%' OR {n} LIKE '%Expert%'
            OR {n} LIKE '%expandInput%' OR {n} LIKE '%doActivation%'
            OR {n} LIKE '%topk%' OR {n} LIKE '%buildExpert%'
            OR {n} LIKE '%computeStrides%' OR {n} LIKE '%Dispatch%'
            OR {n} LIKE '%Combine%' OR {n} LIKE '%Prepare%'
            OR {n} LIKE '%Sanitize%'
            OR {d} LIKE '%moe%' OR {d} LIKE '%MoE%'
            OR {d} LIKE '%expert%' OR {d} LIKE '%Expert%'
            OR {d} LIKE '%PtrArray%' THEN 'MoE'
       WHEN {n} LIKE '%fmha%' OR {n} LIKE '%Fmha%'
            OR {n} LIKE '%flash%' OR {n} LIKE '%attention%'
            OR {d} LIKE '%fmha%' OR {d} LIKE '%Fmha%' THEN 'Attention'
       WHEN {n} LIKE '%nccl%' OR {n} LIKE '%allreduce%'
            OR {n} LIKE '%AllReduce%'
            OR {d} LIKE '%nccl%' THEN 'NCCL/Comm'
       WHEN {n} LIKE '%gemm%' OR {n} LIKE '%Gemm%'
            OR {n} LIKE '%cutlass%' OR {n} LIKE '%cublas%'
            OR {n} LIKE '%nvjet%' OR {n} LIKE '%splitKreduce%'
            OR {d} LIKE '%gemm%' OR {d} LIKE '%Gemm%'
            OR {d} LIKE '%cutlass%' OR {d} LIKE '%cublas%'
            OR {d} LIKE '%nvjet%' THEN 'GEMM'
       WHEN {n} LIKE '%Norm%' OR {n} LIKE '%norm%' THEN 'Norm'
       WHEN {n} LIKE '%rope%' OR {n} LIKE '%Rope%'
            OR {n} LIKE '%RoPE%' THEN 'RoPE'
       WHEN {n} LIKE '%quantize%' OR {n} LIKE '%Quantize%'
            OR {n} LIKE '%dequant%' THEN 'Quantize'
       WHEN {n} LIKE '%memcpy%' OR {n} LIKE '%memset%'
            OR {n} LIKE '%Memcpy%' OR {n} LIKE '%Memset%' THEN 'Memory'
       ELSE 'Other'
     END AS category,
     COUNT(*) AS kernel_count,
     ROUND(CAST(SUM({dur_expr}) AS FLOAT) / 1e6, 2) AS total_ms,
     ROUND(100.0 * SUM({dur_expr}) / {total_gpu}, 1) AS pct
     FROM {K}
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
    ncols = {r[1].lower(): r[1] for r in cur.execute(f"PRAGMA table_info('{NVTX_TABLE}')").fetchall()}
    text_raw = ncols.get('text') or ncols.get('name') or ncols.get('message') or 'text'
    nstart = ncols.get('start') or 'start'
    nend = ncols.get('end') or 'end'
    ndur = ncols.get('duration') or ncols.get('dur') or None
    ndur_expr = ndur if ndur else f"({nend}-{nstart})"

    # Check if NVTX text column also stores string IDs
    nsample = cur.execute(f"SELECT {text_raw} FROM {NVTX_TABLE} LIMIT 1").fetchone()
    nvtx_text_is_id = HAS_STRINGS and nsample and isinstance(nsample[0], int)

    if nvtx_text_is_id:
        NV = f"(SELECT nv.*, s.{sval_col} AS _ntext FROM {NVTX_TABLE} nv JOIN StringIds s ON nv.{text_raw} = s.{sid_col})"
        resolved_text = "_ntext"
    else:
        NV = NVTX_TABLE
        resolved_text = text_raw

    print(f"\n=== 3. NVTX Event Durations (Layer-Level Breakdown) ===")
    q = f"""SELECT {resolved_text} AS event,
            COUNT(*) AS count,
            ROUND(CAST(SUM({ndur_expr}) AS FLOAT) / 1e6, 2) AS total_ms,
            ROUND(CAST(AVG({ndur_expr}) AS FLOAT) / 1e3, 2) AS avg_us
     FROM {NV}
     WHERE ({resolved_text} LIKE '%layer%' OR {resolved_text} LIKE '%Layer%'
            OR {resolved_text} LIKE '%MoE%' OR {resolved_text} LIKE '%moe%'
            OR {resolved_text} LIKE '%attention%' OR {resolved_text} LIKE '%Attention%'
            OR {resolved_text} LIKE '%expert%' OR {resolved_text} LIKE '%Expert%'
            OR {resolved_text} LIKE '%allreduce%' OR {resolved_text} LIKE '%decode%'
            OR {resolved_text} LIKE '%prefill%' OR {resolved_text} LIKE '%forward%')
     GROUP BY {resolved_text}
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
        ROUND(CAST({total_gpu} AS FLOAT) / 1e9, 3) AS total_gpu_time_s,
        ROUND(CAST(SUM({dur_expr}) AS FLOAT) / COUNT(*) / 1e3, 2) AS avg_kernel_us
 FROM {KERNEL_TABLE}
 WHERE 1=1 {time_filter}"""
try:
    rows = cur.execute(q).fetchall()
    print_table(["total_kernels", "unique_kernels", "total_gpu_time_s", "avg_kernel_us"], rows)
except Exception as e:
    print(f"  WARN: Stats query failed: {e}")

print(f"\n=== Analysis Complete ===")
print(f"  SQLite: {db_path}")

conn.close()
PYEOF

log "Done."
