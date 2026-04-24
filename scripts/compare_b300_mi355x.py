#!/usr/bin/env python3
"""
Compare B300/B200 vs MI355X R8d wide-sample decode breakdowns.

Inputs:
  --b300  results/.../STEADY/decode_breakdown_c<N>.csv
              (from scripts/trace_layer_detail.py; columns: B200_Module,
               B200_Operator, B200_Raw_Kernel, B200_Stream, B200_Avg_us,
               B200_Median_us, B200_P95_us, B200_Std_us, B200_CV_pct,
               B200_N_samples, ..., plus PASS rows at end)
  --mi355x results/.../decode_breakdown_c<N>.csv
              (from scripts/run_parse_trace.py merge_decode_xlsx; columns:
               cpu_module, gpu_kernel, avg_us, median_us, p95_us, std_us,
               n_steps, pct%)

Outputs (stdout, optional --md / --csv flags for files):
  Section A: Totals + B300 Pass-level breakdown (no per-op mapping yet)
  Section B: Top-N kernels each platform, side by side
  Section C: Caveats (sample sizes, methodology version, phase mapping gap)

Usage:
  python3 scripts/compare_b300_mi355x.py \\
      --b300   results/b300_dsr_fp4/.../STEADY/decode_breakdown_c4.csv \\
      --mi355x results/mi355x_dsr_mxfp4/.../decode_breakdown_c4.csv \\
      --top-n 15

Goal: produce data that fits real production decode average AND is
apples-to-apples comparable across NV and AMD (R8d methodology).
"""

import argparse
import csv
import os
import sys


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------

def load_b300_csv(path):
    """Parse B300/B200 R8d csv. Returns (kernel_rows, pass_sums, totals).

    Skips the empty separator + TOTAL/Walltime/Overlap rows + PASS section.
    """
    kernel_rows = []
    pass_sums = {}
    totals = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            op = (r.get("B200_Operator") or "").strip()
            if not op:
                continue
            # Special rows after blank separator
            if op.startswith("B200 TOTAL"):
                try: totals["kernel_sum"] = float(r["B200_Avg_us"])
                except: pass
                continue
            if op == "B200 Walltime":
                try: totals["walltime"] = float(r["B200_Avg_us"])
                except: pass
                continue
            if op == "B200 Overlap":
                try: totals["overlap"] = float(r["B200_Avg_us"])
                except: pass
                continue
            if op == "PASS":  # header row of pass section
                continue
            # Pass-level sum row: "MOE", "MHA", "O_proj", "EP_AR_before_MHA", "EP_AR_before_MOE"
            if op in ("MOE", "MHA", "O_proj", "EP_AR_before_MHA", "EP_AR_before_MOE"):
                # In Pass section, the value is in the B200_Stream column (per layout)
                # Actually format per generator: ["", "", pname, "", f"{b200_val:.1f}", ...]
                # so the value lands in B200_Stream column. Try multiple keys to be tolerant.
                for k in ("B200_Stream", "B200_Avg_us", "B200_Median_us"):
                    v = r.get(k, "").strip()
                    try:
                        pass_sums[op] = float(v)
                        break
                    except (TypeError, ValueError):
                        continue
                continue
            # Regular kernel row
            try:
                kernel_rows.append({
                    "module": r.get("B200_Module", "").strip(),
                    "operator": op,
                    "kernel": r.get("B200_Raw_Kernel", "").strip(),
                    "avg_us": float(r.get("B200_Avg_us", 0) or 0),
                    "median_us": float(r.get("B200_Median_us", 0) or r.get("B200_Avg_us", 0) or 0),
                    "p95_us": float(r.get("B200_P95_us", 0) or r.get("B200_Avg_us", 0) or 0),
                    "cv_pct": float(r.get("B200_CV_pct", 0) or 0),
                    "n_samples": int(r.get("B200_N_samples", 0) or 0),
                })
            except (TypeError, ValueError):
                continue
    return kernel_rows, pass_sums, totals


def load_mi355x_csv(path):
    """Parse MI355X R8d csv (from merge_decode_xlsx companion CSV).

    Returns kernel_rows list. Schema:
      cpu_module, gpu_kernel, avg_us, median_us, p95_us, std_us, n_steps, pct%
    """
    kernel_rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            mod = (r.get("cpu_module") or "").strip()
            kn = (r.get("gpu_kernel") or "").strip()
            if not kn or mod == "TOTAL":
                continue
            try:
                kernel_rows.append({
                    "module": mod,
                    "kernel": kn,
                    "avg_us": float(r.get("avg_us", 0) or 0),
                    "median_us": float(r.get("median_us", 0) or 0),
                    "p95_us": float(r.get("p95_us", 0) or 0),
                    "n_steps": int(r.get("n_steps", 0) or 0),
                    "pct": float(r.get("pct%", 0) or 0),
                })
            except (TypeError, ValueError):
                continue
    return kernel_rows


# ----------------------------------------------------------------------------
# Reporters
# ----------------------------------------------------------------------------

def emit_section_a(b3_rows, b3_pass, b3_totals, mi_rows, out):
    b3_total_avg = sum(r["avg_us"] for r in b3_rows)
    mi_total_avg = sum(r["avg_us"] for r in mi_rows)
    b3_n = b3_rows[0]["n_samples"] if b3_rows else 0
    mi_n = mi_rows[0]["n_steps"] if mi_rows else 0

    out.write("## A. Totals\n\n")
    out.write("| metric | B300/B200 | MI355X | ratio (NV/AMD) |\n")
    out.write("|---|---:|---:|---:|\n")
    out.write(f"| sum of per-kernel mean (μs) | {b3_total_avg:.1f} | {mi_total_avg:.1f} "
              f"| {b3_total_avg/mi_total_avg:.2f}× |\n" if mi_total_avg > 0 else
              f"| sum of per-kernel mean (μs) | {b3_total_avg:.1f} | {mi_total_avg:.1f} | n/a |\n")
    if "walltime" in b3_totals:
        out.write(f"| walltime (NV only, μs) | {b3_totals['walltime']:.1f} | — | — |\n")
    if "overlap" in b3_totals:
        out.write(f"| overlap (NV only, μs) | {b3_totals['overlap']:.1f} | — | — |\n")
    out.write(f"| samples / kernel | {b3_n} | {mi_n} | — |\n")
    out.write(f"| kernels reported | {len(b3_rows)} | {len(mi_rows)} | — |\n\n")

    if b3_pass:
        out.write("### B300 Pass-level breakdown\n\n")
        out.write("| pass | sum mean μs | % of B300 total |\n|---|---:|---:|\n")
        for p in ["MHA", "O_proj", "MOE", "EP_AR_before_MHA", "EP_AR_before_MOE"]:
            v = b3_pass.get(p, 0)
            pct = 100 * v / b3_total_avg if b3_total_avg > 0 else 0
            out.write(f"| {p} | {v:.1f} | {pct:.1f}% |\n")
        out.write("\n")
    out.write("> No per-phase MI355X breakdown yet — kernel name mapping NV↔AMD "
              "needs a curated rules table (different ops, different naming). "
              "Section B (top kernels) is the actionable view until then.\n\n")


def emit_section_b(b3_rows, mi_rows, top_n, out):
    out.write(f"## B. Top {top_n} kernels by mean (each platform)\n\n")
    b3_sorted = sorted(b3_rows, key=lambda r: -r["avg_us"])[:top_n]
    mi_sorted = sorted(mi_rows, key=lambda r: -r["avg_us"])[:top_n]

    out.write("### B300 / B200\n\n")
    out.write("| rank | operator | mean μs | median | p95 | CV% | tail = p95/mean |\n")
    out.write("|---:|---|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(b3_sorted, 1):
        tail = r["p95_us"] / r["avg_us"] if r["avg_us"] > 0 else 0
        op = r["operator"][:42]
        out.write(f"| {i} | {op} | {r['avg_us']:.2f} | {r['median_us']:.2f} | "
                  f"{r['p95_us']:.2f} | {r['cv_pct']:.1f} | {tail:.2f}× |\n")
    out.write("\n")

    out.write("### MI355X\n\n")
    out.write("| rank | module / kernel | mean μs | median | p95 | tail = p95/mean |\n")
    out.write("|---:|---|---:|---:|---:|---:|\n")
    for i, r in enumerate(mi_sorted, 1):
        tail = r["p95_us"] / r["avg_us"] if r["avg_us"] > 0 else 0
        kn = (r["module"] or "?")[:18] + " / " + r["kernel"][:42]
        out.write(f"| {i} | {kn} | {r['avg_us']:.2f} | {r['median_us']:.2f} | "
                  f"{r['p95_us']:.2f} | {tail:.2f}× |\n")
    out.write("\n")


def emit_section_c(b3_rows, mi_rows, b3_path, mi_path, out):
    b3_n = b3_rows[0]["n_samples"] if b3_rows else 0
    mi_n = mi_rows[0]["n_steps"] if mi_rows else 0
    out.write("## C. Caveats / methodology audit\n\n")
    out.write(f"- B300 source: `{os.path.basename(b3_path)}` (N={b3_n} samples/kernel)\n")
    out.write(f"- MI355X source: `{os.path.basename(mi_path)}` (N={mi_n} steps/kernel × ~30 layers ≈ {mi_n*30} samples)\n")
    out.write("- Both should use R8d wide-sample defaults: `--skip-warmup 5 --max-steps 20` "
              f"(target ~600 samples). N={b3_n} vs N={mi_n}: ")
    if 15 <= b3_n <= 25 and 15 <= mi_n <= 25:
        out.write("✅ both inside expected range.\n")
    else:
        out.write("⚠️ outside expected range — verify both used the new methodology.\n")
    out.write("- No per-phase MI355X mapping yet — write a kernel rules table next "
              "if cross-architecture per-phase comparison is needed.\n")
    out.write("- Mean is the primary metric per R8d (fits production decode average); "
              "p95 / tail column flags long-tail kernels worth optimizing.\n\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Cross-platform R8d decode-breakdown comparison "
                    "(B300/B200 vs MI355X).")
    p.add_argument("--b300",   required=True, help="B300/B200 STEADY csv")
    p.add_argument("--mi355x", required=True, help="MI355X aggregated csv")
    p.add_argument("--top-n",  type=int, default=15, help="Top-N kernels per platform (default 15)")
    p.add_argument("--md",     type=str, default=None, help="Also write report to this Markdown file")
    args = p.parse_args()

    b3_rows, b3_pass, b3_totals = load_b300_csv(args.b300)
    mi_rows = load_mi355x_csv(args.mi355x)

    if not b3_rows:
        print(f"ERROR: no kernel rows parsed from {args.b300}", file=sys.stderr); sys.exit(1)
    if not mi_rows:
        print(f"ERROR: no kernel rows parsed from {args.mi355x}", file=sys.stderr); sys.exit(1)

    out = sys.stdout
    out.write("# B300/B200 vs MI355X — R8d decode breakdown comparison\n\n")
    emit_section_a(b3_rows, b3_pass, b3_totals, mi_rows, out)
    emit_section_b(b3_rows, mi_rows, args.top_n, out)
    emit_section_c(b3_rows, mi_rows, args.b300, args.mi355x, out)

    if args.md:
        with open(args.md, "w") as f:
            f.write("# B300/B200 vs MI355X — R8d decode breakdown comparison\n\n")
            emit_section_a(b3_rows, b3_pass, b3_totals, mi_rows, f)
            emit_section_b(b3_rows, mi_rows, args.top_n, f)
            emit_section_c(b3_rows, mi_rows, args.b300, args.mi355x, f)
        print(f"\n[wrote markdown: {args.md}]", file=sys.stderr)


if __name__ == "__main__":
    main()
