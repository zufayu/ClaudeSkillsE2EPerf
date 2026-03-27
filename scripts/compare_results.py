#!/usr/bin/env python3
"""Compare current benchmark results with the last committed version.

Usage:
    python scripts/compare_results.py
    python scripts/compare_results.py results_b200_fp8_mtp3
    python scripts/compare_results.py --ref HEAD~2
    python scripts/compare_results.py results_b200_fp8_mtp3 --output results_b200_fp8_mtp3/regression_report.txt
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime


def get_old_data(filepath, ref):
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{ref}:{filepath}"],
            stderr=subprocess.DEVNULL,
        )
        return json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def pct_diff(new_val, old_val):
    """Return percentage difference, or 0 if old is zero."""
    return ((new_val - old_val) / old_val * 100) if old_val else 0


def run_compare(dirs, ref="HEAD", threshold=5.0):
    """Run comparison and return (lines, total, flagged)."""
    lines = []
    total = 0
    flagged = 0

    # Metrics aligned with dashboard: Total Tput, Output Tput, TPOT, TTFT, Interactivity
    metrics = [
        ("Total Tput",     "total_token_throughput", True),
        ("Output Tput",    "output_throughput",      True),
        ("TPOT p50 (ms)",  "median_tpot_ms",         False),
        ("TTFT p50 (ms)",  "median_ttft_ms",         False),
        ("Interactivity",  None,                     True),  # computed: 1000/TPOT
    ]

    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "result_*.json")))
        if not files:
            continue

        lines.append(f"\n{'=' * 120}")
        lines.append(f"  {d}")
        lines.append(f"{'=' * 120}")

        # Header row
        hdr = f"{'Config':<40s}"
        for name, _, _ in metrics:
            hdr += f"  {'Old':>8s} {'New':>8s} {'Diff':>7s}"
        lines.append(hdr)

        # Metric names sub-header
        sub = f"{'':40s}"
        for name, _, _ in metrics:
            short = name.split('(')[0].strip()[:10]
            sub += f"  {short:>8s} {short:>8s} {'':>7s}"
        lines.append(sub)
        lines.append("-" * 120)

        for f in files:
            tag = os.path.basename(f).replace("result_", "").replace(".json", "")
            with open(f) as fh:
                new = json.load(fh)

            old = get_old_data(f, ref)

            if old is None:
                row = f"{tag:<40s}"
                for name, key, _ in metrics:
                    if key is None:
                        tpot = new.get("median_tpot_ms", 0)
                        val = 1000.0 / tpot if tpot > 0 else 0
                    else:
                        val = new.get(key, 0) or 0
                    row += f"  {'NEW':>8s} {val:>8.1f} {'':>7s}"
                lines.append(row)
                continue

            row = f"{tag:<40s}"
            max_diff = 0
            for name, key, higher_better in metrics:
                if key is None:
                    # Interactivity = 1000 / TPOT
                    old_tpot = old.get("median_tpot_ms", 0) or 0
                    new_tpot = new.get("median_tpot_ms", 0) or 0
                    old_val = 1000.0 / old_tpot if old_tpot > 0 else 0
                    new_val = 1000.0 / new_tpot if new_tpot > 0 else 0
                else:
                    old_val = old.get(key, 0) or 0
                    new_val = new.get(key, 0) or 0

                diff = pct_diff(new_val, old_val)
                max_diff = max(max_diff, abs(diff))
                row += f"  {old_val:>8.1f} {new_val:>8.1f} {diff:>+6.1f}%"

            flag = " <<<" if max_diff > threshold else ""
            if flag:
                flagged += 1
            total += 1
            lines.append(row + flag)

    lines.append(f"\n{'=' * 120}")
    lines.append(f"Total: {total} points compared, {flagged} flagged (>{threshold}% diff on any metric)")
    if flagged:
        lines.append("  ^^^ Review flagged items above (marked with <<<)")

    return lines, total, flagged


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results with git history")
    parser.add_argument("dirs", nargs="*", help="Result directories to compare (default: auto-detect)")
    parser.add_argument("--ref", default="HEAD", help="Git ref for old data (default: HEAD)")
    parser.add_argument("--threshold", type=float, default=5.0, help="Flag diff above this %% (default: 5)")
    parser.add_argument("--output", "-o", default=None, help="Save report to file")
    args = parser.parse_args()

    if args.dirs:
        dirs = args.dirs
    else:
        dirs = sorted(glob.glob("results_b200_*") + glob.glob("results_h20_*"))
        dirs = [d for d in dirs if os.path.isdir(d) and not d.endswith("_test")]

    if not dirs:
        print("No result directories found.")
        sys.exit(1)

    lines, total, flagged = run_compare(dirs, args.ref, args.threshold)

    # Print to stdout
    for line in lines:
        print(line)

    # Write to file if requested
    if args.output:
        header = f"Regression Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        header += f"\nCompared against: {args.ref}"
        header += f"\nDirectories: {', '.join(dirs)}"
        with open(args.output, "w") as f:
            f.write(header + "\n")
            for line in lines:
                f.write(line + "\n")
        print(f"\nReport saved to {args.output}")

    # Exit code: 1 if any flagged
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
