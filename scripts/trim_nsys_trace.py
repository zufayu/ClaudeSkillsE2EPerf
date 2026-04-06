#!/usr/bin/env python3
"""
Trim large nsys-rep traces to a steady-state window and analyze.

For SGLang 2.5GB+ nsys-rep files that are too large for Nsight Systems GUI.
Requires a newer nsys version that supports 'nsys filter' (Windows nsys typically does).

Usage:
    # Auto trim to middle 60s of the trace
    python trim_nsys_trace.py trace.nsys-rep

    # Specify time range in seconds
    python trim_nsys_trace.py trace.nsys-rep --start 270 --end 330

    # Just export SQLite with time range (no trim)
    python trim_nsys_trace.py trace.nsys-rep --sqlite-only --start 270 --end 330

    # Analyze existing SQLite
    python trim_nsys_trace.py trace.sqlite --analyze-only --gpu 0

Output:
    trace_trimmed_t270-330.nsys-rep   (small, openable in Nsight Systems GUI)
    trace_t270-330.sqlite             (for query-based analysis)
    Prints kernel breakdown to stdout
"""

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def run_cmd(cmd, check=True):
    """Run a shell command and return output."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr.strip()}")
        if check:
            sys.exit(1)
    return result


def get_nsys_version():
    """Get nsys version string."""
    r = run_cmd("nsys --version", check=False)
    return r.stdout.strip() or r.stderr.strip()


def get_trace_duration_from_sqlite(sqlite_path):
    """Get total trace duration from SQLite."""
    conn = sqlite3.connect(sqlite_path)
    row = conn.execute(
        "SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    conn.close()
    if row and row[0]:
        return row[0], row[1], (row[1] - row[0]) / 1e9
    return None, None, 0


def trim_nsys_rep(input_path, output_path, start_s, end_s):
    """Trim nsys-rep to a time window using nsys filter."""
    print(f"\nTrimming {input_path}")
    print(f"  Window: {start_s}s - {end_s}s ({end_s - start_s}s)")

    # Try nsys filter (newer versions)
    r = run_cmd(
        f'nsys filter -i "{input_path}" -o "{output_path}" '
        f"--time-range={start_s},{end_s} --timeunit sec",
        check=False,
    )
    if r.returncode == 0 and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1e6
        print(f"  Trimmed: {output_path} ({size_mb:.1f} MB)")
        return True

    # Try alternative syntax
    r = run_cmd(
        f'nsys filter --input "{input_path}" --output "{output_path}" '
        f"--time-range {start_s},{end_s} --time-unit seconds",
        check=False,
    )
    if r.returncode == 0 and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1e6
        print(f"  Trimmed: {output_path} ({size_mb:.1f} MB)")
        return True

    print("  nsys filter not available or failed")
    return False


def export_sqlite(input_path, output_path, start_s=None, end_s=None):
    """Export nsys-rep to SQLite, optionally with time range."""
    print(f"\nExporting to SQLite: {output_path}")

    timerange_args = ""
    if start_s is not None and end_s is not None:
        # Try different flag combinations for different nsys versions
        for flags in [
            f"--timeunit sec --timerange {start_s},{end_s}",
            f"--time-unit seconds --time-range {start_s},{end_s}",
        ]:
            r = run_cmd(
                f'nsys export --type sqlite {flags} -o "{output_path}" "{input_path}"',
                check=False,
            )
            if r.returncode == 0 and os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / 1e6
                print(f"  SQLite: {output_path} ({size_mb:.1f} MB)")
                return True
        print("  timerange export failed, doing full export...")

    r = run_cmd(
        f'nsys export --type sqlite -o "{output_path}" "{input_path}"', check=False
    )
    if r.returncode == 0 and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1e6
        print(f"  SQLite: {output_path} ({size_mb:.1f} MB)")
        return True

    print("  SQLite export failed")
    return False


def analyze_sqlite(sqlite_path, gpu_id=None, top_n=30):
    """Analyze kernel breakdown from SQLite."""
    if not os.path.exists(sqlite_path):
        print(f"ERROR: {sqlite_path} not found")
        return

    conn = sqlite3.connect(sqlite_path)

    # Trace time range
    row = conn.execute(
        "SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    if row and row[0]:
        dur_s = (row[1] - row[0]) / 1e9
        print(f"\nKernel time range: {row[0]} ~ {row[1]} ({dur_s:.1f}s)")

    # NVTX markers
    print(f"\n{'='*80}")
    print("NVTX Markers")
    print(f"{'='*80}")
    try:
        cur = conn.execute("""
            SELECT s.value AS text, COUNT(*) AS cnt,
                   ROUND(AVG(n.end - n.start) / 1e6, 3) AS avg_ms
            FROM NVTX_EVENTS n
            JOIN StringIds s ON n.textId = s.id
            WHERE n.end > n.start
            GROUP BY s.value
            ORDER BY cnt DESC
            LIMIT 20
        """)
        rows = cur.fetchall()
        print(f"{'Count':>8}  {'Avg(ms)':>10}  Text")
        print(f"{'-'*8}  {'-'*10}  {'-'*50}")
        for text, cnt, avg in rows:
            print(f"{cnt:>8}  {avg:>10.3f}  {text[:70]}")
    except Exception as e:
        print(f"  NVTX query failed: {e}")

    # Top kernels
    gpu_filter = f"AND k.deviceId = {gpu_id}" if gpu_id is not None else ""
    print(f"\n{'='*80}")
    print(f"Top {top_n} Kernels (GPU {'all' if gpu_id is None else gpu_id})")
    print(f"{'='*80}")
    print(f"{'#':>3} {'Total(ms)':>10} {'Avg(us)':>10} {'Count':>8}  {'%':>6}  Kernel")
    print(f"{'-'*3} {'-'*10} {'-'*10} {'-'*8}  {'-'*6}  {'-'*50}")

    cur = conn.execute(f"""
        SELECT s.value AS kernel_name,
               COUNT(*) AS cnt,
               ROUND(CAST(SUM(k.end - k.start) AS FLOAT) / 1e6, 2) AS total_ms,
               ROUND(CAST(AVG(k.end - k.start) AS FLOAT) / 1e3, 2) AS avg_us
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE 1=1 {gpu_filter}
        GROUP BY s.value
        ORDER BY total_ms DESC
        LIMIT ?
    """, (top_n,))

    rows = cur.fetchall()
    total_ms = sum(r[2] for r in rows)
    for i, (name, cnt, t_ms, avg_us) in enumerate(rows, 1):
        pct = 100 * t_ms / total_ms if total_ms > 0 else 0
        print(f"{i:>3} {t_ms:>10.2f} {avg_us:>10.2f} {cnt:>8}  {pct:>5.1f}%  {name[:60]}")

    print(f"\nTotal kernel time: {total_ms:.1f}ms ({total_ms/1000:.1f}s)")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Trim and analyze large nsys traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto trim to middle 60s, export SQLite, analyze
  python trim_nsys_trace.py trace.nsys-rep

  # Specify exact window (seconds from trace start)
  python trim_nsys_trace.py trace.nsys-rep --start 270 --end 330

  # Just analyze existing SQLite
  python trim_nsys_trace.py trace.sqlite --analyze-only --gpu 0

  # Only export SQLite with time range
  python trim_nsys_trace.py trace.nsys-rep --sqlite-only --start 100 --end 200
        """,
    )
    parser.add_argument("input", help="Path to .nsys-rep or .sqlite file")
    parser.add_argument("--start", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--window", type=float, default=60, help="Window size in seconds if auto-detecting middle (default: 60)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID (default: all)")
    parser.add_argument("--top", type=int, default=30, help="Top N kernels (default: 30)")
    parser.add_argument("--sqlite-only", action="store_true", help="Only export SQLite, skip trim")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing SQLite")
    parser.add_argument("--no-analyze", action="store_true", help="Skip analysis")
    parser.add_argument("--outdir", default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    outdir = args.outdir or os.path.dirname(input_path)
    basename = Path(input_path).stem

    print(f"nsys version: {get_nsys_version()}")
    size_mb = os.path.getsize(input_path) / 1e6
    print(f"Input: {input_path} ({size_mb:.1f} MB)")

    # If input is already SQLite, just analyze
    if input_path.endswith(".sqlite"):
        analyze_sqlite(input_path, args.gpu, args.top)
        return

    # Determine time range
    start_s = args.start
    end_s = args.end

    if start_s is None or end_s is None:
        # Need to figure out trace duration - try quick SQLite export first
        print("\nAuto-detecting trace duration...")
        tmp_sqlite = os.path.join(outdir, f"{basename}_tmp.sqlite")
        if export_sqlite(input_path, tmp_sqlite):
            t0, t1, dur = get_trace_duration_from_sqlite(tmp_sqlite)
            print(f"  Trace duration: {dur:.1f}s")
            mid = dur / 2
            half_win = args.window / 2
            start_s = mid - half_win
            end_s = mid + half_win
            print(f"  Auto-selected window: {start_s:.0f}s - {end_s:.0f}s (middle {args.window:.0f}s)")
            # Remove tmp sqlite, we'll make a proper one
            os.remove(tmp_sqlite)
        else:
            print("  Cannot determine duration, using 270-330s default")
            start_s = 270
            end_s = 330

    tag = f"t{int(start_s)}-{int(end_s)}"

    # Step 1: Trim nsys-rep
    if not args.sqlite_only:
        trimmed_path = os.path.join(outdir, f"{basename}_trimmed_{tag}.nsys-rep")
        if trim_nsys_rep(input_path, trimmed_path, start_s, end_s):
            print(f"\n>>> Open in Nsight Systems GUI: {trimmed_path}")
        else:
            print("\n>>> nsys filter failed — try updating nsys or use --sqlite-only")

    # Step 2: Export SQLite
    sqlite_path = os.path.join(outdir, f"{basename}_{tag}.sqlite")
    if not os.path.exists(sqlite_path):
        export_sqlite(input_path, sqlite_path, start_s, end_s)

    # Step 3: Analyze
    if not args.no_analyze:
        if os.path.exists(sqlite_path):
            analyze_sqlite(sqlite_path, args.gpu, args.top)
        else:
            # Fall back to full SQLite if it exists
            full_sqlite = os.path.join(outdir, f"{basename}.sqlite")
            if os.path.exists(full_sqlite):
                analyze_sqlite(full_sqlite, args.gpu, args.top)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    trimmed = os.path.join(outdir, f"{basename}_trimmed_{tag}.nsys-rep")
    if os.path.exists(trimmed):
        print(f"  Trimmed nsys-rep: {trimmed} ({os.path.getsize(trimmed)/1e6:.1f} MB)")
    if os.path.exists(sqlite_path):
        print(f"  SQLite:           {sqlite_path} ({os.path.getsize(sqlite_path)/1e6:.1f} MB)")
    print(f"  Time window:      {start_s:.0f}s - {end_s:.0f}s ({end_s-start_s:.0f}s)")


if __name__ == "__main__":
    main()
