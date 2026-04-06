#!/usr/bin/env python3
"""
Analyze nsys SQLite traces for per-decode-step kernel breakdown.

For large nsys-rep files (2-3GB) that Nsight Systems can't open, this script
queries the SQLite export to find steady-state decode steps via NVTX markers
and extracts per-step kernel timelines.

Workflow:
  1. nsys export --type sqlite [--timerange start,end] -o trace.sqlite trace.nsys-rep
  2. python3 analyze_nsys_sqlite.py trace.sqlite [options]

Supports both:
  - SGLang: --enable-layerwise-nvtx-marker (layer_0, layer_1, ...)
  - TRT-LLM: TLLM_LLMAPI_ENABLE_NVTX ([Executor] _forward_step N: ...)

Usage:
    # Full analysis: find stable decode steps, extract kernel breakdown
    python3 scripts/analyze_nsys_sqlite.py trace.sqlite --gpu 0 --top-kernels 20

    # Find decode steps matching pattern
    python3 scripts/analyze_nsys_sqlite.py trace.sqlite --find-steps "64 gen reqs"

    # Extract kernels for a specific NVTX time range
    python3 scripts/analyze_nsys_sqlite.py trace.sqlite --time-range 20907845545,20910445115

    # Dump all NVTX events (understand what markers exist)
    python3 scripts/analyze_nsys_sqlite.py trace.sqlite --dump-nvtx

    # Export per-step kernel CSV
    python3 scripts/analyze_nsys_sqlite.py trace.sqlite --csv decode_kernels.csv
"""

import argparse
import csv as csv_mod
import os
import sqlite3
import sys
from collections import OrderedDict, defaultdict


def connect(db_path):
    if not os.path.exists(db_path):
        print(f"ERROR: {db_path} not found")
        sys.exit(1)
    return sqlite3.connect(db_path)


def dump_nvtx(conn, limit=50):
    """Show all distinct NVTX event texts to understand what markers exist."""
    cur = conn.execute("""
        SELECT s.value AS text, COUNT(*) AS cnt,
               ROUND(AVG(n.end - n.start) / 1e6, 3) AS avg_ms,
               ROUND(MIN(n.end - n.start) / 1e6, 3) AS min_ms,
               ROUND(MAX(n.end - n.start) / 1e6, 3) AS max_ms
        FROM NVTX_EVENTS n
        JOIN StringIds s ON n.textId = s.id
        WHERE n.end > n.start
        GROUP BY s.value
        ORDER BY cnt DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    print(f"\n{'='*80}")
    print(f"NVTX Events (top {limit} by count)")
    print(f"{'='*80}")
    print(f"{'Count':>8}  {'Avg(ms)':>10}  {'Min(ms)':>10}  {'Max(ms)':>10}  Text")
    print(f"{'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*40}")
    for text, cnt, avg, mn, mx in rows:
        print(f"{cnt:>8}  {avg:>10.3f}  {mn:>10.3f}  {mx:>10.3f}  {text[:80]}")
    return rows


def _find_sglang_module_steps(conn, skip_first=5, max_steps=20):
    """Find decode steps from SGLang module-level NVTX markers.

    SGLang with --enable-layerwise-nvtx-marker produces markers like:
      :{'Module': 'model.model', 'Inputs': [[256], [256]]}
    where the number in Inputs is the batch dimension (token count).

    Strategy: find top-level 'model.model' markers (not model.model.layers.*),
    group by batch size, identify decode steps as the most frequent batch size
    (decode steps repeat many times, prefill is fewer with larger batch).
    """
    cur = conn.execute("""
        SELECT s.value AS text, n.start, n.end,
               ROUND((n.end - n.start) / 1e6, 3) AS dur_ms
        FROM NVTX_EVENTS n
        JOIN StringIds s ON n.textId = s.id
        WHERE s.value LIKE '%model.model%Inputs%'
          AND s.value NOT LIKE '%layers%'
          AND n.end > n.start
        ORDER BY n.start
    """)
    all_steps = cur.fetchall()
    if not all_steps:
        return []

    # Parse batch size from Inputs: [[N], [N]] → extract N
    import re
    bs_steps = {}  # batch_size -> list of steps
    for step in all_steps:
        m = re.search(r"'Inputs':\s*\[\[(\d+)\]", step[0])
        if m:
            bs = int(m.group(1))
            bs_steps.setdefault(bs, []).append(step)

    if not bs_steps:
        return []

    # Print batch size distribution
    print("\nSGLang module-level NVTX markers detected (model.model top-level):")
    for bs in sorted(bs_steps.keys()):
        steps = bs_steps[bs]
        avg_dur = sum(s[3] for s in steps) / len(steps)
        print(f"  bs={bs}: {len(steps)} events, avg={avg_dur:.1f}ms")

    # Decode steps = the batch size with the most events (decode repeats many times)
    decode_bs = max(bs_steps.keys(), key=lambda b: len(bs_steps[b]))
    steps = bs_steps[decode_bs]
    print(f"\nUsing bs={decode_bs} as decode steps ({len(steps)} events)")

    return steps


def find_decode_steps(conn, pattern=None, gen_reqs_min=None, skip_first=5, max_steps=20):
    """Find steady-state decode NVTX events.

    For TRT-LLM: looks for '[Executor] _forward_step N: 0 ctx reqs, K gen reqs'
    For SGLang: looks for 'decode[bs=N]' or module-level markers
    """
    # First, detect what kind of NVTX markers we have
    cur = conn.execute("""
        SELECT s.value, COUNT(*) as cnt
        FROM NVTX_EVENTS n JOIN StringIds s ON n.textId = s.id
        WHERE s.value LIKE '%forward_step%' OR s.value LIKE '%decode%'
           OR s.value LIKE '%gen reqs%'
        GROUP BY s.value ORDER BY cnt DESC LIMIT 10
    """)
    marker_types = cur.fetchall()

    if not marker_types:
        # Try SGLang module-level markers
        print("No forward_step/decode NVTX markers found, trying SGLang module markers...")
        steps = _find_sglang_module_steps(conn, skip_first, max_steps)
        if not steps:
            print("No SGLang module markers found either.")
            print("Available NVTX markers:")
            dump_nvtx(conn, 20)
            return []
    else:
        # Build query based on detected markers
        if pattern:
            where_clause = f"s.value LIKE '%{pattern}%'"
        elif any('forward_step' in m[0] for m in marker_types):
            # TRT-LLM style
            if gen_reqs_min:
                where_clause = f"s.value LIKE '%forward_step%' AND s.value LIKE '%{gen_reqs_min} gen reqs%' AND s.value LIKE '%0 ctx reqs%'"
            else:
                where_clause = "s.value LIKE '%forward_step%' AND s.value LIKE '%0 ctx reqs%'"
        elif any('decode' in m[0] for m in marker_types):
            # SGLang style
            where_clause = "s.value LIKE 'decode%'"
        else:
            where_clause = "1=1"

        cur = conn.execute(f"""
            SELECT s.value AS text, n.start, n.end,
                   ROUND((n.end - n.start) / 1e6, 3) AS dur_ms
            FROM NVTX_EVENTS n
            JOIN StringIds s ON n.textId = s.id
            WHERE {where_clause} AND n.end > n.start
            ORDER BY n.start
        """)
        steps = cur.fetchall()

        if not steps:
            print(f"No NVTX events matching: {where_clause}")
            # Fallback to SGLang module markers
            print("Trying SGLang module markers as fallback...")
            steps = _find_sglang_module_steps(conn, skip_first, max_steps)
            if not steps:
                return []

    print(f"\nFound {len(steps)} decode step NVTX events")

    # Skip first N steps (warmup/prefill transition)
    if skip_first and len(steps) > skip_first:
        steps = steps[skip_first:]
        print(f"Skipped first {skip_first}, using {len(steps)} remaining")

    if max_steps and len(steps) > max_steps:
        steps = steps[:max_steps]
        print(f"Limited to {max_steps} steps")

    # Show duration stats
    durs = [s[3] for s in steps]
    if durs:
        avg_dur = sum(durs) / len(durs)
        print(f"Step duration: avg={avg_dur:.3f}ms, min={min(durs):.3f}ms, max={max(durs):.3f}ms")

    return steps


def get_kernels_in_range(conn, start_ns, end_ns, device_id=None):
    """Get all GPU kernels within a time range."""
    device_filter = f"AND k.deviceId = {device_id}" if device_id is not None else ""
    cur = conn.execute(f"""
        SELECT s.value AS kernel_name,
               k.start, k.end,
               ROUND((k.end - k.start) / 1000.0, 3) AS dur_us,
               k.deviceId, k.streamId,
               k.gridX || 'x' || k.gridY || 'x' || k.gridZ AS grid,
               k.blockX || 'x' || k.blockY || 'x' || k.blockZ AS block
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE k.start >= ? AND k.end <= ? {device_filter}
        ORDER BY k.start
    """, (start_ns, end_ns))
    return cur.fetchall()


def classify_kernel_short(name):
    """Classify kernel short name into logical operator category."""
    nl = name.lower()
    if 'allreduce' in nl or 'lamport' in nl or 'nccl' in nl or 'userbuffers' in nl:
        return 'Communication'
    if 'rmsnorm' in nl or 'layernorm' in nl:
        return 'Normalization'
    if 'fmha' in nl or 'flash' in nl or 'attention' in nl:
        return 'Attention'
    if 'moe' in nl or 'routing' in nl:
        if 'routing' in nl:
            return 'MoE/Router'
        if 'finalize' in nl:
            return 'MoE/Finalize'
        return 'MoE/Expert'
    if 'bmm' in nl:
        if 'swiglu' in nl or 'silu' in nl:
            return 'MoE/Expert (gate_up)'
        return 'MoE/Expert (down)'
    if 'rope' in nl:
        return 'RoPE'
    if 'quantize' in nl or 'cvt_fp' in nl or 'dequant' in nl:
        return 'Quantization'
    if 'gemm' in nl or 'splitk' in nl or 'nvjet' in nl or 'cutlass' in nl:
        return 'GEMM/MatMul'
    if 'elementwise' in nl or 'vectorized' in nl or 'fill' in nl:
        return 'Elementwise'
    if 'copy' in nl or 'memcpy' in nl or 'memset' in nl:
        return 'Memory'
    if 'set_mla' in nl or 'concat' in nl or 'cat' in nl:
        return 'KV Cache'
    if 'sample' in nl or 'topk' in nl or 'softmax' in nl:
        return 'Sampling'
    return 'Other'


def analyze_step_kernels(conn, steps, device_id=None, top_n=30):
    """Analyze kernel breakdown across decode steps."""
    all_kernel_stats = defaultdict(lambda: {
        'count': 0, 'total_us': 0.0, 'category': '', 'steps_seen': 0
    })
    step_wall_times = []
    step_kernel_sums = []
    n_steps = len(steps)

    for i, (text, start_ns, end_ns, dur_ms) in enumerate(steps):
        kernels = get_kernels_in_range(conn, start_ns, end_ns, device_id)
        step_total_us = sum(k[3] for k in kernels)
        step_kernel_sums.append(step_total_us)
        step_wall_times.append(dur_ms * 1000)  # ms to us

        seen_this_step = set()
        for kname, kstart, kend, kdur, kdev, kstream, kgrid, kblock in kernels:
            cat = classify_kernel_short(kname)
            all_kernel_stats[kname]['count'] += 1
            all_kernel_stats[kname]['total_us'] += kdur
            all_kernel_stats[kname]['category'] = cat
            if kname not in seen_this_step:
                all_kernel_stats[kname]['steps_seen'] += 1
                seen_this_step.add(kname)

    # Print per-category breakdown
    cat_stats = defaultdict(lambda: {'count': 0, 'total_us': 0.0})
    for kname, stats in all_kernel_stats.items():
        cat = stats['category']
        cat_stats[cat]['count'] += stats['count']
        cat_stats[cat]['total_us'] += stats['total_us']

    total_kernel_us = sum(s['total_us'] for s in cat_stats.values())

    print(f"\n{'='*80}")
    print(f"Category Breakdown (averaged over {n_steps} steps, GPU {'all' if device_id is None else device_id})")
    print(f"{'='*80}")
    print(f"{'#':>3} {'Category':<30} {'Avg/Step(us)':>12} {'%':>7} {'Count/Step':>10}")
    print(f"{'-'*3} {'-'*30} {'-'*12} {'-'*7} {'-'*10}")

    sorted_cats = sorted(cat_stats.items(), key=lambda x: x[1]['total_us'], reverse=True)
    for rank, (cat, stats) in enumerate(sorted_cats, 1):
        avg_us = stats['total_us'] / n_steps
        pct = 100 * stats['total_us'] / total_kernel_us if total_kernel_us > 0 else 0
        avg_count = stats['count'] / n_steps
        print(f"{rank:>3} {cat:<30} {avg_us:>12.1f} {pct:>6.1f}% {avg_count:>10.1f}")

    avg_kernel_sum = sum(step_kernel_sums) / n_steps if n_steps else 0
    avg_wall = sum(step_wall_times) / n_steps if n_steps else 0
    print(f"\n  Avg kernel sum/step: {avg_kernel_sum:.1f}μs ({avg_kernel_sum/1000:.2f}ms)")
    if avg_wall > 0:
        print(f"  Avg wall-clock/step: {avg_wall:.1f}μs ({avg_wall/1000:.2f}ms)")
        print(f"  GPU utilization: {100*avg_kernel_sum/avg_wall:.1f}% (kernel_sum/wall)")

    # Top kernels
    print(f"\n{'='*80}")
    print(f"Top {top_n} Kernels by Total Duration")
    print(f"{'='*80}")
    print(f"{'#':>3} {'Category':<20} {'Avg/Step(us)':>12} {'%':>7} {'Cnt/Step':>8}  Kernel")
    print(f"{'-'*3} {'-'*20} {'-'*12} {'-'*7} {'-'*8}  {'-'*40}")

    sorted_kernels = sorted(all_kernel_stats.items(), key=lambda x: x[1]['total_us'], reverse=True)
    for rank, (kname, stats) in enumerate(sorted_kernels[:top_n], 1):
        avg_us = stats['total_us'] / n_steps
        pct = 100 * stats['total_us'] / total_kernel_us if total_kernel_us > 0 else 0
        avg_count = stats['count'] / n_steps
        short_name = kname[:60] if len(kname) > 60 else kname
        print(f"{rank:>3} {stats['category']:<20} {avg_us:>12.1f} {pct:>6.1f}% {avg_count:>8.1f}  {short_name}")

    return all_kernel_stats, cat_stats


def get_nvtx_layers_in_range(conn, start_ns, end_ns):
    """Get NVTX layer markers within a decode step for per-layer analysis."""
    cur = conn.execute("""
        SELECT s.value AS text, n.start, n.end,
               ROUND((n.end - n.start) / 1e6, 3) AS dur_ms
        FROM NVTX_EVENTS n
        JOIN StringIds s ON n.textId = s.id
        WHERE n.start >= ? AND n.end <= ?
          AND (s.value LIKE 'layer_%' OR s.value LIKE 'Layer %'
               OR s.value LIKE '%DecoderLayer%')
        ORDER BY n.start
    """, (start_ns, end_ns))
    return cur.fetchall()


def analyze_per_layer(conn, steps, device_id=None, show_layers=5):
    """Analyze kernel breakdown per transformer layer using NVTX layer markers."""
    # Pick a representative step (median)
    mid = len(steps) // 2
    text, start_ns, end_ns, dur_ms = steps[mid]
    print(f"\n{'='*80}")
    print(f"Per-Layer Analysis (step {mid}: {text[:60]})")
    print(f"{'='*80}")

    layers = get_nvtx_layers_in_range(conn, start_ns, end_ns)
    if not layers:
        print("No per-layer NVTX markers found in this step.")
        print("SGLang needs --enable-layerwise-nvtx-marker")
        print("TRT-LLM needs TLLM_LLMAPI_ENABLE_NVTX=1")
        return

    print(f"Found {len(layers)} layer markers")
    for i, (ltext, lstart, lend, ldur) in enumerate(layers[:show_layers]):
        print(f"\n  --- {ltext} ({ldur:.3f}ms) ---")
        kernels = get_kernels_in_range(conn, lstart, lend, device_id)
        for kname, kstart, kend, kdur, kdev, kstream, kgrid, kblock in kernels[:15]:
            cat = classify_kernel_short(kname)
            short = kname[:50]
            print(f"    {kdur:>10.1f}μs  [{cat:<15}]  {short}")
        if len(kernels) > 15:
            print(f"    ... +{len(kernels)-15} more kernels")


def top_kernels_global(conn, device_id=None, limit=20):
    """Quick global top-N kernels (no step filtering)."""
    device_filter = f"AND k.deviceId = {device_id}" if device_id is not None else ""
    cur = conn.execute(f"""
        SELECT s.value AS kernel_name,
               COUNT(*) AS cnt,
               ROUND(SUM(k.end - k.start) / 1e6, 2) AS total_ms,
               ROUND(AVG(k.end - k.start) / 1e3, 2) AS avg_us
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE 1=1 {device_filter}
        GROUP BY s.value
        ORDER BY total_ms DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    print(f"\n{'='*80}")
    print(f"Top {limit} Kernels (global, GPU {'all' if device_id is None else device_id})")
    print(f"{'='*80}")
    print(f"{'#':>3} {'Total(ms)':>10} {'Avg(us)':>10} {'Count':>8}  Kernel")
    print(f"{'-'*3} {'-'*10} {'-'*10} {'-'*8}  {'-'*50}")
    for i, (name, cnt, total, avg) in enumerate(rows, 1):
        print(f"{i:>3} {total:>10.2f} {avg:>10.2f} {cnt:>8}  {name[:70]}")


def write_csv(all_kernel_stats, n_steps, csv_path):
    """Write kernel breakdown to CSV."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv_mod.writer(f)
        writer.writerow(['rank', 'name', 'category', 'count', 'total_ms', 'pct', 'avg_us', 'steps_seen'])
        total_us = sum(s['total_us'] for s in all_kernel_stats.values())
        sorted_k = sorted(all_kernel_stats.items(), key=lambda x: x[1]['total_us'], reverse=True)
        for rank, (name, stats) in enumerate(sorted_k, 1):
            pct = 100 * stats['total_us'] / total_us if total_us > 0 else 0
            writer.writerow([
                rank, name, stats['category'],
                stats['count'],
                round(stats['total_us'] / 1000, 3),
                round(pct, 2),
                round(stats['total_us'] / stats['count'] if stats['count'] > 0 else 0, 3),
                stats['steps_seen'],
            ])
    print(f"\nCSV written: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze nsys SQLite trace for per-decode-step kernel breakdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick global top kernels
  python3 %(prog)s trace.sqlite --global-top 30

  # Full decode step analysis
  python3 %(prog)s trace.sqlite --gpu 0 --skip-first 10 --max-steps 20

  # Find specific decode steps (TRT-LLM)
  python3 %(prog)s trace.sqlite --find-steps "64 gen reqs"

  # Dump NVTX markers to understand trace structure
  python3 %(prog)s trace.sqlite --dump-nvtx

  # Extract kernels in a specific time range
  python3 %(prog)s trace.sqlite --time-range 20907845545,20910445115 --gpu 0

Preprocessing (for large nsys-rep files):
  # Full export
  nsys export --type sqlite -o trace.sqlite trace.nsys-rep

  # Export only 20-40 second window (much smaller)
  nsys export --type sqlite --timeunit sec --timerange 20,40 -o trace_20s.sqlite trace.nsys-rep

  # Export kernel CSV (for grep-based analysis)
  nsys stats --report cuda_gpu_trace --format csv -o kernels trace.nsys-rep
        """)
    parser.add_argument("sqlite", help="Path to nsys SQLite file")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID to analyze (default: all)")
    parser.add_argument("--skip-first", type=int, default=5, help="Skip first N decode steps (default: 5)")
    parser.add_argument("--max-steps", type=int, default=20, help="Max decode steps to analyze (default: 20)")
    parser.add_argument("--find-steps", default=None, help="NVTX pattern to find decode steps (e.g. '64 gen reqs')")
    parser.add_argument("--dump-nvtx", action="store_true", help="Dump all NVTX event types")
    parser.add_argument("--global-top", type=int, default=None, help="Show global top-N kernels (no step filtering)")
    parser.add_argument("--top-kernels", type=int, default=30, help="Number of top kernels to show per-step (default: 30)")
    parser.add_argument("--time-range", default=None, help="start_ns,end_ns — extract kernels in this range")
    parser.add_argument("--per-layer", action="store_true", help="Show per-layer breakdown using NVTX layer markers")
    parser.add_argument("--show-layers", type=int, default=3, help="Number of layers to show in per-layer mode (default: 3)")
    parser.add_argument("--csv", default=None, help="Output CSV path for kernel breakdown")
    args = parser.parse_args()

    conn = connect(args.sqlite)

    if args.dump_nvtx:
        dump_nvtx(conn, 100)
        return

    if args.global_top:
        top_kernels_global(conn, args.gpu, args.global_top)
        return

    if args.time_range:
        start_ns, end_ns = [int(x) for x in args.time_range.split(',')]
        print(f"Extracting kernels in range [{start_ns}, {end_ns}] ({(end_ns-start_ns)/1e6:.3f}ms)")
        kernels = get_kernels_in_range(conn, start_ns, end_ns, args.gpu)
        print(f"Found {len(kernels)} kernels")
        total_us = sum(k[3] for k in kernels)
        for kname, kstart, kend, kdur, kdev, kstream, kgrid, kblock in kernels:
            cat = classify_kernel_short(kname)
            pct = 100 * kdur / total_us if total_us > 0 else 0
            print(f"  {kdur:>10.1f}μs {pct:>5.1f}% GPU{kdev} [{cat:<15}] {kname[:60]}")
        print(f"\nTotal: {total_us:.1f}μs ({total_us/1000:.2f}ms)")
        return

    # Find decode steps
    steps = find_decode_steps(conn, pattern=args.find_steps,
                              skip_first=args.skip_first, max_steps=args.max_steps)
    if not steps:
        print("\nFalling back to global kernel analysis...")
        top_kernels_global(conn, args.gpu, args.top_kernels)
        return

    # Analyze kernel breakdown
    all_kernel_stats, cat_stats = analyze_step_kernels(conn, steps, args.gpu, args.top_kernels)

    # Per-layer analysis if requested
    if args.per_layer:
        analyze_per_layer(conn, steps, args.gpu, args.show_layers)

    # Write CSV
    if args.csv:
        write_csv(all_kernel_stats, len(steps), args.csv)

    conn.close()


if __name__ == "__main__":
    main()
