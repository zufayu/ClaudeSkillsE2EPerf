#!/usr/bin/env python3
"""Compare current benchmark results with the last committed version.

Usage:
    python scripts/compare_results.py
    python scripts/compare_results.py results_b200_fp8_mtp3
    python scripts/compare_results.py --ref HEAD~2
"""

import argparse
import glob
import json
import os
import subprocess
import sys


def get_old_data(filepath, ref):
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{ref}:{filepath}"],
            stderr=subprocess.DEVNULL,
        )
        return json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results with git history")
    parser.add_argument("dirs", nargs="*", help="Result directories to compare (default: auto-detect)")
    parser.add_argument("--ref", default="HEAD", help="Git ref for old data (default: HEAD)")
    parser.add_argument("--threshold", type=float, default=5.0, help="Flag diff above this %% (default: 5)")
    args = parser.parse_args()

    if args.dirs:
        dirs = args.dirs
    else:
        dirs = sorted(glob.glob("results_b200_*") + glob.glob("results_h20_*"))
        dirs = [d for d in dirs if os.path.isdir(d) and not d.endswith("_test")]

    if not dirs:
        print("No result directories found.")
        sys.exit(1)

    total = 0
    flagged = 0

    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "result_*.json")))
        if not files:
            continue

        print(f"\n{'=' * 80}")
        print(f"  {d}")
        print(f"{'=' * 80}")
        print(f"{'File':<42s} {'Old TPS':>10s} {'New TPS':>10s} {'Diff':>8s}  "
              f"{'Old TPOT':>10s} {'New TPOT':>10s} {'Diff':>8s}")
        print("-" * 80)

        for f in files:
            tag = os.path.basename(f).replace("result_", "").replace(".json", "")
            with open(f) as fh:
                new = json.load(fh)

            old = get_old_data(f, args.ref)
            if old is None:
                print(f"{tag:<42s} {'NEW':>10s} {new.get('output_throughput', 0):>10.1f}")
                continue

            ot_old = old.get("output_throughput", 0)
            ot_new = new.get("output_throughput", 0)
            tp_old = old.get("median_tpot_ms", 0)
            tp_new = new.get("median_tpot_ms", 0)

            ot_diff = ((ot_new - ot_old) / ot_old * 100) if ot_old else 0
            tp_diff = ((tp_new - tp_old) / tp_old * 100) if tp_old else 0

            flag = " <<<" if abs(ot_diff) > args.threshold else ""
            if flag:
                flagged += 1
            total += 1

            print(f"{tag:<42s} {ot_old:>10.1f} {ot_new:>10.1f} {ot_diff:>+7.1f}%  "
                  f"{tp_old:>10.1f} {tp_new:>10.1f} {tp_diff:>+7.1f}%{flag}")

    print(f"\n{'=' * 80}")
    print(f"Total: {total} points compared, {flagged} flagged (>{args.threshold}% diff)")
    if flagged:
        print("  ^^^ Review flagged items above (marked with <<<)")


if __name__ == "__main__":
    main()
