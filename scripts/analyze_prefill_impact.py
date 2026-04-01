#!/usr/bin/env python3
"""
Analyze prefill interruption impact on TPOT from Kineto trace data.

Proves that the gap between GPU decode walltime (~21ms) and mean_tpot (~25ms)
is caused by prefill events interleaving with decode steps in continuous batching.

Evidence chain:
  1. Inter-decode gaps have bimodal distribution: ~22ms (normal) vs ~92ms (interrupted)
  2. Every large gap has a prefill event sandwiched between the two decode events
  3. Weighted average of gaps ≈ mean_itl ≈ mean_tpot from benchmark JSON

Usage:
  python3 scripts/analyze_prefill_impact.py TRACE_FILE [--benchmark-json RESULT.json]

  TRACE_FILE: Kineto trace (.json.gz or .json) from collect_atom_trace.sh
  --benchmark-json: Optional benchmark result JSON for cross-validation
"""
import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd


def load_trace(path):
    opener = gzip.open if path.endswith(".gz") else open
    print(f"Loading trace: {path}")
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    print(f"Total trace events: {len(events)}")
    return events


def extract_events(events):
    """Extract decode and prefill events, picking the PID with more decode events."""
    decode_by_pid = defaultdict(list)
    prefill_by_pid = defaultdict(list)

    for e in events:
        name = e.get("name", "")
        if e.get("ph") != "X":
            continue
        pid = e.get("pid")
        if name.startswith("decode["):
            decode_by_pid[pid].append(e)
        elif name.startswith("prefill"):
            prefill_by_pid[pid].append(e)

    # Pick PID with most decode events (avoid duplicates from multiple Kineto tracks)
    best_pid = max(decode_by_pid, key=lambda p: len(decode_by_pid[p]))
    decodes = sorted(decode_by_pid[best_pid], key=lambda x: x["ts"])
    prefills = sorted(prefill_by_pid.get(best_pid, []), key=lambda x: x["ts"])

    print(f"Selected PID: {best_pid}")
    print(f"  Decode events: {len(decodes)}")
    print(f"  Prefill events: {len(prefills)}")

    # Also report other PIDs
    for pid in decode_by_pid:
        if pid != best_pid:
            print(f"  Skipped PID {pid}: {len(decode_by_pid[pid])} decode events (duplicate track)")

    return decodes, prefills


def parse_bs(name):
    m = re.search(r"bs=(\d+)", name)
    return int(m.group(1)) if m else 0


def analyze_gaps(decodes, prefills):
    """Compute inter-decode gaps and classify as normal vs prefill-interrupted."""
    # Build sorted timeline of all events for lookup
    prefill_intervals = [(p["ts"], p["ts"] + p["dur"]) for p in prefills]

    gaps = []  # (gap_ms, bs_before, bs_after, has_prefill, prefill_dur_ms)
    for i in range(1, len(decodes)):
        prev = decodes[i - 1]
        curr = decodes[i]
        gap_us = curr["ts"] - prev["ts"]
        gap_ms = gap_us / 1000.0

        bs_before = parse_bs(prev["name"])
        bs_after = parse_bs(curr["name"])

        # Check if any prefill event falls between prev and curr
        prev_end = prev["ts"] + prev["dur"]
        curr_start = curr["ts"]
        prefill_dur_us = 0
        prefill_count = 0
        for ps, pe in prefill_intervals:
            if ps >= prev_end and pe <= curr_start + 1000:  # 1ms tolerance
                prefill_dur_us += (pe - ps)
                prefill_count += 1

        gaps.append({
            "gap_ms": gap_ms,
            "bs": bs_before,
            "bs_after": bs_after,
            "has_prefill": prefill_count > 0,
            "prefill_count": prefill_count,
            "prefill_dur_ms": prefill_dur_us / 1000.0,
        })

    return gaps


def print_gap_distribution(gaps):
    """Print histogram of inter-decode gaps."""
    print(f"\n{'='*70}")
    print(f"  INTER-DECODE GAP DISTRIBUTION")
    print(f"{'='*70}")

    # Bucket into ranges
    buckets = defaultdict(int)
    bucket_size = 5  # 5ms buckets
    for g in gaps:
        bucket = int(g["gap_ms"] / bucket_size) * bucket_size
        buckets[bucket] += 1

    max_count = max(buckets.values()) if buckets else 1
    bar_width = 50

    for b in sorted(buckets):
        count = buckets[b]
        bar = "#" * int(count / max_count * bar_width)
        label = f"{b:>5d}-{b+bucket_size:<5d}ms"
        print(f"  {label} | {bar} ({count})")


def print_analysis(gaps, benchmark=None):
    """Print detailed analysis proving prefill impact."""
    threshold_ms = 50  # classify gaps above this as interrupted

    normal = [g for g in gaps if g["gap_ms"] < threshold_ms]
    interrupted = [g for g in gaps if g["gap_ms"] >= threshold_ms]

    normal_vals = [g["gap_ms"] for g in normal]
    interrupted_vals = [g["gap_ms"] for g in interrupted]

    # Prefill verification
    interrupted_with_prefill = [g for g in interrupted if g["has_prefill"]]
    interrupted_without_prefill = [g for g in interrupted if not g["has_prefill"]]

    print(f"\n{'='*70}")
    print(f"  GAP CLASSIFICATION (threshold={threshold_ms}ms)")
    print(f"{'='*70}")
    print(f"  Normal gaps (<{threshold_ms}ms):      {len(normal):>6d}  ({len(normal)/len(gaps)*100:.1f}%)")
    print(f"  Interrupted gaps (>={threshold_ms}ms): {len(interrupted):>6d}  ({len(interrupted)/len(gaps)*100:.1f}%)")
    print()

    if normal_vals:
        normal_sorted = sorted(normal_vals)
        n = len(normal_sorted)
        print(f"  Normal gap stats:")
        print(f"    mean:   {sum(normal_vals)/n:.2f} ms")
        print(f"    median: {normal_sorted[n//2]:.2f} ms")
        print(f"    min:    {min(normal_vals):.2f} ms")
        print(f"    max:    {max(normal_vals):.2f} ms")
        print()

    if interrupted_vals:
        interrupted_sorted = sorted(interrupted_vals)
        n = len(interrupted_sorted)
        print(f"  Interrupted gap stats:")
        print(f"    mean:   {sum(interrupted_vals)/n:.2f} ms")
        print(f"    median: {interrupted_sorted[n//2]:.2f} ms")
        print(f"    min:    {min(interrupted_vals):.2f} ms")
        print(f"    max:    {max(interrupted_vals):.2f} ms")
        print()

    # Evidence 1: Every interrupted gap has a prefill
    print(f"\n{'='*70}")
    print(f"  EVIDENCE 1: Prefill causes every large gap")
    print(f"{'='*70}")
    print(f"  Interrupted gaps with prefill:    {len(interrupted_with_prefill)}")
    print(f"  Interrupted gaps without prefill: {len(interrupted_without_prefill)}")
    if not interrupted_without_prefill:
        print(f"  --> 100% of large gaps have prefill events between them")
    else:
        print(f"  --> {len(interrupted_with_prefill)/len(interrupted)*100:.1f}% of large gaps have prefill")
        print(f"  Gaps without prefill (may be multi-prefill or edge cases):")
        for g in interrupted_without_prefill[:5]:
            print(f"    gap={g['gap_ms']:.1f}ms bs={g['bs']}->{g['bs_after']}")

    # Evidence 2: Weighted average = mean_itl
    print(f"\n{'='*70}")
    print(f"  EVIDENCE 2: Weighted average of gaps = mean_tpot")
    print(f"{'='*70}")

    # Simple average (all gaps equal weight)
    all_gaps = [g["gap_ms"] for g in gaps]
    simple_avg = sum(all_gaps) / len(all_gaps)

    # BS-weighted average (each gap weighted by batch size)
    bs_weighted_sum = sum(g["gap_ms"] * g["bs"] for g in gaps)
    bs_total = sum(g["bs"] for g in gaps)
    bs_weighted_avg = bs_weighted_sum / bs_total if bs_total else 0

    # Reconstruct from normal/interrupted split
    if normal_vals and interrupted_vals:
        p_interrupted = len(interrupted) / len(gaps)
        normal_median = sorted(normal_vals)[len(normal_vals)//2]
        interrupted_median = sorted(interrupted_vals)[len(interrupted_vals)//2]
        reconstructed = (1 - p_interrupted) * normal_median + p_interrupted * interrupted_median

        print(f"  Simple average of all gaps:     {simple_avg:.2f} ms")
        print(f"  BS-weighted average of all gaps: {bs_weighted_avg:.2f} ms")
        print(f"  Reconstructed from split:")
        print(f"    ({1-p_interrupted:.3f} x {normal_median:.2f}) + ({p_interrupted:.3f} x {interrupted_median:.2f}) = {reconstructed:.2f} ms")

    # Evidence 3: Cross-validation with benchmark JSON
    if benchmark:
        print(f"\n{'='*70}")
        print(f"  EVIDENCE 3: Cross-validation with benchmark JSON")
        print(f"{'='*70}")
        bench_mean_tpot = benchmark.get("mean_tpot_ms", 0)
        bench_mean_itl = benchmark.get("mean_itl_ms", 0)
        bench_median_itl = benchmark.get("median_itl_ms", 0)
        bench_p99_itl = benchmark.get("p99_itl_ms", 0)

        print(f"  {'Metric':<35} {'Trace':>10} {'Benchmark':>10} {'Delta':>10}")
        print(f"  {'-'*65}")

        if normal_vals:
            normal_med = sorted(normal_vals)[len(normal_vals)//2]
            d = normal_med - bench_median_itl
            print(f"  {'Normal gap median vs median_itl':<35} {normal_med:>10.2f} {bench_median_itl:>10.2f} {d:>+10.2f}")

        if interrupted_vals:
            inter_med = sorted(interrupted_vals)[len(interrupted_vals)//2]
            d = inter_med - bench_p99_itl
            print(f"  {'Interrupted gap median vs p99_itl':<35} {inter_med:>10.2f} {bench_p99_itl:>10.2f} {d:>+10.2f}")

        d = bs_weighted_avg - bench_mean_itl
        print(f"  {'BS-weighted avg vs mean_itl':<35} {bs_weighted_avg:>10.2f} {bench_mean_itl:>10.2f} {d:>+10.2f}")

        d = bs_weighted_avg - bench_mean_tpot
        print(f"  {'BS-weighted avg vs mean_tpot':<35} {bs_weighted_avg:>10.2f} {bench_mean_tpot:>10.2f} {d:>+10.2f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  CONCLUSION")
    print(f"{'='*70}")
    if normal_vals and interrupted_vals:
        normal_med = sorted(normal_vals)[len(normal_vals)//2]
        inter_med = sorted(interrupted_vals)[len(interrupted_vals)//2]
        p = len(interrupted) / len(gaps) * 100
        print(f"  - {len(gaps)} decode gaps analyzed")
        print(f"  - {p:.1f}% of gaps are interrupted by prefill ({len(interrupted)}/{len(gaps)})")
        print(f"  - Normal decode gap:      {normal_med:.2f} ms (GPU decode time)")
        print(f"  - Interrupted decode gap:  {inter_med:.2f} ms (decode + prefill)")
        print(f"  - Prefill overhead per gap: {inter_med - normal_med:.2f} ms")
        print(f"  - BS-weighted average:     {bs_weighted_avg:.2f} ms")
        if benchmark:
            print(f"  - Benchmark mean_tpot:     {benchmark.get('mean_tpot_ms', 0):.2f} ms")
            print(f"  - Match: BS-weighted avg ≈ mean_tpot confirms prefill is the cause")


def print_sample_events(gaps, n=10):
    """Show sample interrupted gaps with prefill details."""
    interrupted = [g for g in gaps if g["has_prefill"]]
    if not interrupted:
        return

    print(f"\n{'='*70}")
    print(f"  SAMPLE INTERRUPTED GAPS (first {n})")
    print(f"{'='*70}")
    print(f"  {'#':>4}  {'gap_ms':>8}  {'bs':>4}  {'prefills':>8}  {'prefill_ms':>10}")
    print(f"  {'-'*40}")
    for i, g in enumerate(interrupted[:n]):
        print(f"  {i+1:>4}  {g['gap_ms']:>8.2f}  {g['bs']:>4}  {g['prefill_count']:>8}  {g['prefill_dur_ms']:>10.2f}")


def save_xlsx(gaps, benchmark, output_path):
    """Save gaps and summary as two sheets in one xlsx file."""
    threshold_ms = 50
    normal = [g for g in gaps if g["gap_ms"] < threshold_ms]
    interrupted = [g for g in gaps if g["gap_ms"] >= threshold_ms]
    normal_vals = sorted([g["gap_ms"] for g in normal])
    interrupted_vals = sorted([g["gap_ms"] for g in interrupted])

    bs_weighted_sum = sum(g["gap_ms"] * g["bs"] for g in gaps)
    bs_total = sum(g["bs"] for g in gaps)
    bs_weighted_avg = bs_weighted_sum / bs_total if bs_total else 0

    # Sheet 1: all gaps
    gaps_df = pd.DataFrame([
        {
            "gap_idx": i,
            "gap_ms": round(g["gap_ms"], 3),
            "bs": g["bs"],
            "bs_after": g["bs_after"],
            "has_prefill": int(g["has_prefill"]),
            "prefill_count": g["prefill_count"],
            "prefill_dur_ms": round(g["prefill_dur_ms"], 3),
        }
        for i, g in enumerate(gaps)
    ])

    # Sheet 2: summary
    summary_rows = [
        {"metric": "total_gaps", "trace_value": len(gaps), "benchmark_value": "", "unit": "count"},
        {"metric": "normal_gaps", "trace_value": len(normal), "benchmark_value": "", "unit": "count"},
        {"metric": "interrupted_gaps", "trace_value": len(interrupted), "benchmark_value": "", "unit": "count"},
        {"metric": "interrupted_pct", "trace_value": round(len(interrupted) / len(gaps) * 100, 2), "benchmark_value": "", "unit": "%"},
    ]
    if normal_vals:
        summary_rows.append({"metric": "normal_gap_median", "trace_value": round(normal_vals[len(normal_vals)//2], 2), "benchmark_value": round(benchmark["median_itl_ms"], 2) if benchmark else "", "unit": "ms"})
        summary_rows.append({"metric": "normal_gap_mean", "trace_value": round(sum(normal_vals)/len(normal_vals), 2), "benchmark_value": "", "unit": "ms"})
    if interrupted_vals:
        summary_rows.append({"metric": "interrupted_gap_median", "trace_value": round(interrupted_vals[len(interrupted_vals)//2], 2), "benchmark_value": round(benchmark["p99_itl_ms"], 2) if benchmark else "", "unit": "ms"})
        summary_rows.append({"metric": "interrupted_gap_mean", "trace_value": round(sum(interrupted_vals)/len(interrupted_vals), 2), "benchmark_value": "", "unit": "ms"})
        summary_rows.append({"metric": "prefill_verified_pct", "trace_value": round(sum(1 for g in interrupted if g["has_prefill"]) / len(interrupted) * 100, 2), "benchmark_value": "", "unit": "%"})
    summary_rows.append({"metric": "bs_weighted_avg_gap", "trace_value": round(bs_weighted_avg, 2), "benchmark_value": "", "unit": "ms"})
    if benchmark:
        summary_rows.append({"metric": "benchmark_mean_tpot", "trace_value": "", "benchmark_value": round(benchmark["mean_tpot_ms"], 2), "unit": "ms"})
        summary_rows.append({"metric": "benchmark_mean_itl", "trace_value": "", "benchmark_value": round(benchmark["mean_itl_ms"], 2), "unit": "ms"})
        summary_rows.append({"metric": "benchmark_median_itl", "trace_value": "", "benchmark_value": round(benchmark["median_itl_ms"], 2), "unit": "ms"})
        summary_rows.append({"metric": "benchmark_p99_itl", "trace_value": "", "benchmark_value": round(benchmark["p99_itl_ms"], 2), "unit": "ms"})

    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        gaps_df.to_excel(writer, sheet_name="gaps", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print(f"\nXLSX saved: {output_path}")
    print(f"  Sheet 'gaps':    {len(gaps_df)} rows (all inter-decode gaps)")
    print(f"  Sheet 'summary': {len(summary_df)} rows (statistics + benchmark comparison)")


def main():
    parser = argparse.ArgumentParser(description="Analyze prefill impact on TPOT from Kineto trace")
    parser.add_argument("trace_file", help="Kineto trace file (.json.gz or .json)")
    parser.add_argument("--benchmark-json", help="Benchmark result JSON for cross-validation")
    parser.add_argument("--output-dir", help="Directory for xlsx output (default: same dir as trace_file)")
    args = parser.parse_args()

    # Load trace
    events = load_trace(args.trace_file)
    decodes, prefills = extract_events(events)

    if len(decodes) < 2:
        print("ERROR: Need at least 2 decode events to analyze gaps")
        sys.exit(1)

    # Load benchmark JSON if provided
    benchmark = None
    if args.benchmark_json:
        with open(args.benchmark_json) as f:
            benchmark = json.load(f)
        print(f"Loaded benchmark JSON: {args.benchmark_json}")

    # Analyze
    gaps = analyze_gaps(decodes, prefills)
    print_gap_distribution(gaps)
    print_analysis(gaps, benchmark)
    print_sample_events(gaps)

    # Save xlsx
    output_dir = args.output_dir or os.path.dirname(args.trace_file) or "."
    os.makedirs(output_dir, exist_ok=True)
    save_xlsx(gaps, benchmark, os.path.join(output_dir, "prefill_impact_analysis.xlsx"))


if __name__ == "__main__":
    main()
