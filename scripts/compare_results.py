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


def run_compare(dirs, ref="HEAD", threshold=5.0):
    """Run comparison and return (lines, total, flagged)."""
    lines = []
    total = 0
    flagged = 0

    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "result_*.json")))
        if not files:
            continue

        lines.append(f"\n{'=' * 80}")
        lines.append(f"  {d}")
        lines.append(f"{'=' * 80}")
        lines.append(f"{'File':<42s} {'Old TPS':>10s} {'New TPS':>10s} {'Diff':>8s}  "
                      f"{'Old TPOT':>10s} {'New TPOT':>10s} {'Diff':>8s}")
        lines.append("-" * 80)

        for f in files:
            tag = os.path.basename(f).replace("result_", "").replace(".json", "")
            with open(f) as fh:
                new = json.load(fh)

            old = get_old_data(f, ref)
            if old is None:
                lines.append(f"{tag:<42s} {'NEW':>10s} {new.get('output_throughput', 0):>10.1f}")
                continue

            ot_old = old.get("output_throughput", 0)
            ot_new = new.get("output_throughput", 0)
            tp_old = old.get("median_tpot_ms", 0)
            tp_new = new.get("median_tpot_ms", 0)

            ot_diff = ((ot_new - ot_old) / ot_old * 100) if ot_old else 0
            tp_diff = ((tp_new - tp_old) / tp_old * 100) if tp_old else 0

            flag = " <<<" if abs(ot_diff) > threshold else ""
            if flag:
                flagged += 1
            total += 1

            lines.append(f"{tag:<42s} {ot_old:>10.1f} {ot_new:>10.1f} {ot_diff:>+7.1f}%  "
                          f"{tp_old:>10.1f} {tp_new:>10.1f} {tp_diff:>+7.1f}%{flag}")

    lines.append(f"\n{'=' * 80}")
    lines.append(f"Total: {total} points compared, {flagged} flagged (>{threshold}% diff)")
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
