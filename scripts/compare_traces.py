#!/usr/bin/env python3
"""Compare two kernel-level trace analysis results and generate delta tables.

Supports:
  - SQLite from nsys (analyze_nsys_trace.sh output)
  - CSV from ncu_kernel_analysis.sh
  - Decode walltime CSV from collect_atom_trace.sh

Usage:
    # Compare two nsys SQLite category breakdowns
    python3 scripts/compare_traces.py \
        --baseline traces/baseline.sqlite \
        --current traces/new.sqlite

    # Compare two ncu metric CSVs
    python3 scripts/compare_traces.py \
        --baseline ncu_reports/baseline_metrics.csv \
        --current ncu_reports/new_metrics.csv

    # Compare decode walltime CSVs (MI355X)
    python3 scripts/compare_traces.py \
        --baseline results_old/decode_walltime_*.csv \
        --current results_new/decode_walltime_*.csv

    # Output as Markdown (for reports/PRs)
    python3 scripts/compare_traces.py --baseline A --current B --md

    # Cross-platform: B200 nsys category vs MI355X category (manual JSON)
    python3 scripts/compare_traces.py \
        --baseline b200_categories.json \
        --current mi355x_categories.json --cross-platform
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
from collections import OrderedDict


# ---- Category classification (same as analyze_nsys_trace.sh) ----
CATEGORY_PATTERNS = OrderedDict([
    ("MoE", ["moe", "MoE", "expert", "Expert", "expandInput", "doActivation",
             "topk", "buildExpert", "computeStrides", "Dispatch", "Combine",
             "Prepare", "Sanitize", "PtrArray"]),
    ("Attention", ["fmha", "Fmha", "flash", "attention"]),
    ("NCCL/Comm", ["nccl", "allreduce", "AllReduce", "allgather", "AllGather",
                   "userbuffers"]),
    ("GEMM", ["gemm", "Gemm", "cutlass", "cublas", "nvjet", "splitKreduce",
              "bmm"]),
    ("Norm", ["Norm", "norm", "rmsnorm"]),
    ("RoPE", ["rope", "Rope", "RoPE", "rotary"]),
    ("Quantize", ["quantize", "Quantize", "dequant"]),
    ("Memory", ["memcpy", "memset", "Memcpy", "Memset"]),
])


def classify_kernel(name):
    """Classify a kernel name into a category."""
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(p in name for p in patterns):
            return category
    return "Other"


# ---- SQLite category extraction ----
def extract_categories_from_sqlite(db_path):
    """Extract kernel category breakdown from nsys SQLite database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    kernel_table = None
    for c in ["CUPTI_ACTIVITY_KIND_KERNEL", "CUDA_GPU_TRACE"]:
        if c in tables:
            kernel_table = c
            break
    if not kernel_table:
        print(f"ERROR: No kernel table in {db_path}", file=sys.stderr)
        conn.close()
        return {}, {}

    kcols = {r[1].lower(): r[1]
             for r in cur.execute(f"PRAGMA table_info('{kernel_table}')").fetchall()}
    name_col = kcols.get("shortname") or kcols.get("name") or "shortName"
    dur_expr = kcols.get("duration") or "(end-start)"

    has_strings = "StringIds" in tables
    sample = cur.execute(f"SELECT {name_col} FROM {kernel_table} LIMIT 1").fetchone()
    name_is_id = has_strings and sample and isinstance(sample[0], int)

    if name_is_id:
        scols = {r[1].lower(): r[1]
                 for r in cur.execute("PRAGMA table_info('StringIds')").fetchall()}
        sid_col = scols.get("id") or "id"
        sval_col = scols.get("value") or scols.get("string") or "value"
        query = f"""
            SELECT s.{sval_col} AS kname,
                   {dur_expr} AS dur
            FROM {kernel_table} k
            JOIN StringIds s ON k.{name_col} = s.{sid_col}
        """
    else:
        query = f"SELECT {name_col} AS kname, {dur_expr} AS dur FROM {kernel_table}"

    categories = {}
    kernels = {}
    for kname, dur in cur.execute(query).fetchall():
        cat = classify_kernel(str(kname))
        categories[cat] = categories.get(cat, 0) + (dur or 0)
        short = str(kname)[:80]
        if short not in kernels:
            kernels[short] = {"total_ns": 0, "count": 0}
        kernels[short]["total_ns"] += (dur or 0)
        kernels[short]["count"] += 1

    conn.close()
    return categories, kernels


# ---- CSV extraction ----
def extract_from_ncu_csv(csv_path):
    """Extract kernel metrics from ncu_kernel_analysis.sh output CSV."""
    kernels = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("kernel", "unknown")
            kernels[name] = {
                "total_us": float(row.get("total_us", 0)),
                "count": int(row.get("count", 0)),
                "avg_us": float(row.get("avg_us", 0)),
                "dram_pct": float(row.get("dram_pct", 0)),
                "sm_pct": float(row.get("sm_pct", 0)),
                "occupancy_pct": float(row.get("occupancy_pct", 0)),
                "diagnosis": row.get("diagnosis", ""),
            }
    return kernels


def extract_from_decode_csv(csv_path):
    """Extract decode walltime from collect_atom_trace.sh output CSV."""
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bs = int(row.get("bs", 0))
            data[bs] = {
                "count": int(row.get("count", 0)),
                "avg_ms": float(row.get("avg_ms", 0)),
                "p50_ms": float(row.get("p50_ms", 0)),
                "p99_ms": float(row.get("p99_ms", 0)),
            }
    return data


def extract_from_json(json_path):
    """Extract category breakdown from a manually-created JSON."""
    with open(json_path) as f:
        return json.load(f)


def extract_from_kernel_breakdown_csv(csv_path):
    """Extract from collect_sglang_trace.sh's kernel_breakdown_*.csv output.

    Columns: rank, operator, avg_us, pct, avg_count, total_us, n_steps_present, kernel_names
    Operator format is 'module: name' (e.g. 'comm: lamport_AR+RMSNorm') — derive
    category from the prefix and aggregate.

    Returns (categories_dict, kernels_dict) compatible with compare_categories
    (scalar ns) / compare_kernels ({total_us, count}).
    """
    categories = {}
    kernels = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            op = row.get("operator", "").strip()
            if not op:
                continue
            total_us = float(row.get("total_us", 0))
            count = float(row.get("avg_count", 0))
            # Derive category: prefix before ':' (e.g. 'comm', 'attn', 'moe', 'other')
            cat = op.split(":", 1)[0].strip() if ":" in op else "Other"
            # compare_categories treats values as nanoseconds (divides by 1e6 → ms)
            categories[cat] = categories.get(cat, 0) + total_us * 1000
            kernels[op] = {
                "total_us": total_us,
                "count": count,
            }
    return categories, kernels


def extract_from_per_layer_csv(csv_path):
    """Extract from collect_sglang_trace.sh's per_layer_breakdown_*.csv output.

    Columns: module, operator, avg_us, pct, module_elapsed_us, avg_count,
             total_us, n_layers, kernel_names
    Has explicit `module` column — use directly as category.
    """
    categories = {}
    kernels = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mod = row.get("module", "").strip() or "Other"
            op = row.get("operator", "").strip() or row.get("kernel_names", "?")[:60]
            total_us = float(row.get("total_us", 0))
            count = float(row.get("avg_count", 0))
            categories[mod] = categories.get(mod, 0) + total_us * 1000
            kernels[op] = {
                "total_us": total_us,
                "count": count,
            }
    return categories, kernels


# ---- Comparison logic ----
def pct_delta(new, old):
    if old == 0:
        return 0
    return (new - old) / old * 100


def compare_categories(base_cats, curr_cats, md=False):
    """Compare category breakdowns. Returns formatted table."""
    all_cats = sorted(set(list(base_cats.keys()) + list(curr_cats.keys())),
                      key=lambda c: -(base_cats.get(c, 0) + curr_cats.get(c, 0)))

    base_total = sum(base_cats.values()) or 1
    curr_total = sum(curr_cats.values()) or 1

    lines = []
    if md:
        lines.append("| Category | Baseline (ms) | Baseline % | Current (ms) | Current % | Delta % |")
        lines.append("|----------|--------------|-----------|-------------|----------|---------|")
    else:
        lines.append(f"{'Category':<16} {'Base(ms)':>10} {'Base%':>7} {'Curr(ms)':>10} {'Curr%':>7} {'Delta':>8}")
        lines.append("-" * 64)

    for cat in all_cats:
        b = base_cats.get(cat, 0)
        c = curr_cats.get(cat, 0)
        b_ms = b / 1e6
        c_ms = c / 1e6
        b_pct = b / base_total * 100
        c_pct = c / curr_total * 100
        delta = pct_delta(c, b)

        if md:
            lines.append(f"| {cat} | {b_ms:.1f} | {b_pct:.1f}% | {c_ms:.1f} | {c_pct:.1f}% | {delta:+.1f}% |")
        else:
            lines.append(f"{cat:<16} {b_ms:>10.1f} {b_pct:>6.1f}% {c_ms:>10.1f} {c_pct:>6.1f}% {delta:>+7.1f}%")

    # Totals
    b_total_ms = base_total / 1e6
    c_total_ms = curr_total / 1e6
    total_delta = pct_delta(curr_total, base_total)
    if md:
        lines.append(f"| **Total** | **{b_total_ms:.1f}** | 100% | **{c_total_ms:.1f}** | 100% | **{total_delta:+.1f}%** |")
    else:
        lines.append("-" * 64)
        lines.append(f"{'TOTAL':<16} {b_total_ms:>10.1f} {'100.0':>6}% {c_total_ms:>10.1f} {'100.0':>6}% {total_delta:>+7.1f}%")

    return "\n".join(lines)


def compare_kernels(base_kernels, curr_kernels, top_n=20, md=False):
    """Compare top kernels between two traces."""
    all_names = set(list(base_kernels.keys()) + list(curr_kernels.keys()))

    # Sort by max total time across both
    ranked = sorted(all_names,
                    key=lambda n: max(
                        base_kernels.get(n, {}).get("total_ns", base_kernels.get(n, {}).get("total_us", 0)),
                        curr_kernels.get(n, {}).get("total_ns", curr_kernels.get(n, {}).get("total_us", 0))
                    ), reverse=True)[:top_n]

    lines = []
    if md:
        lines.append("| Kernel | Base | Curr | Delta % |")
        lines.append("|--------|------|------|---------|")
    else:
        lines.append(f"{'Kernel':<60} {'Base':>10} {'Curr':>10} {'Delta':>8}")
        lines.append("-" * 92)

    for name in ranked:
        b = base_kernels.get(name, {})
        c = curr_kernels.get(name, {})

        # Handle both ns (from SQLite) and us (from ncu CSV)
        b_val = b.get("total_ns", 0) / 1e3 if "total_ns" in b else b.get("total_us", 0)
        c_val = c.get("total_ns", 0) / 1e3 if "total_ns" in c else c.get("total_us", 0)

        delta = pct_delta(c_val, b_val) if b_val else 0
        label = name[:58] if len(name) <= 58 else name[:55] + "..."

        if md:
            lines.append(f"| {label} | {b_val:.1f} us | {c_val:.1f} us | {delta:+.1f}% |")
        else:
            lines.append(f"{label:<60} {b_val:>9.1f}us {c_val:>9.1f}us {delta:>+7.1f}%")

    return "\n".join(lines)


def compare_decode_walltime(base_data, curr_data, md=False):
    """Compare decode walltime CSVs."""
    all_bs = sorted(set(list(base_data.keys()) + list(curr_data.keys())))

    lines = []
    if md:
        lines.append("| BS | Base avg(ms) | Curr avg(ms) | Delta % | Base p50 | Curr p50 | p50 Delta |")
        lines.append("|----|-------------|-------------|---------|---------|---------|-----------|")
    else:
        lines.append(f"{'BS':<6} {'Base avg':>10} {'Curr avg':>10} {'Delta':>8} {'Base p50':>10} {'Curr p50':>10} {'p50 Delta':>10}")
        lines.append("-" * 70)

    for bs in all_bs:
        b = base_data.get(bs, {})
        c = curr_data.get(bs, {})
        b_avg = b.get("avg_ms", 0)
        c_avg = c.get("avg_ms", 0)
        b_p50 = b.get("p50_ms", 0)
        c_p50 = c.get("p50_ms", 0)
        avg_delta = pct_delta(c_avg, b_avg)
        p50_delta = pct_delta(c_p50, b_p50)

        if md:
            lines.append(f"| {bs} | {b_avg:.2f} | {c_avg:.2f} | {avg_delta:+.1f}% | {b_p50:.2f} | {c_p50:.2f} | {p50_delta:+.1f}% |")
        else:
            lines.append(f"{bs:<6} {b_avg:>10.2f} {c_avg:>10.2f} {avg_delta:>+7.1f}% {b_p50:>10.2f} {c_p50:>10.2f} {p50_delta:>+9.1f}%")

    return "\n".join(lines)


# ---- File type detection ----
def detect_type(path):
    if path.endswith(".sqlite"):
        return "sqlite"
    if path.endswith(".json"):
        return "json"
    if path.endswith(".csv"):
        with open(path) as f:
            header = f.readline().strip()
            if "kernel" in header and "dram_pct" in header:
                return "ncu_csv"
            if "bs" in header and "avg_ms" in header:
                return "decode_csv"
            # Project's own torch trace analysis outputs:
            if "module" in header and "operator" in header and "module_elapsed_us" in header:
                return "per_layer_csv"      # per_layer_breakdown_*.csv
            if "operator" in header and "avg_us" in header and "pct" in header:
                return "kernel_breakdown_csv"  # kernel_breakdown_*.csv
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Compare kernel trace analysis results")
    parser.add_argument("--baseline", required=True, help="Baseline trace file (SQLite/CSV/JSON)")
    parser.add_argument("--current", required=True, help="Current trace file (SQLite/CSV/JSON)")
    parser.add_argument("--md", action="store_true", help="Output as Markdown table")
    parser.add_argument("--top", type=int, default=20, help="Top N kernels to compare")
    parser.add_argument("--cross-platform", action="store_true",
                        help="Cross-platform mode (normalize by total time)")
    parser.add_argument("--output", "-o", help="Save report to file")
    args = parser.parse_args()

    for path in [args.baseline, args.current]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}", file=sys.stderr)
            sys.exit(1)

    base_type = detect_type(args.baseline)
    curr_type = detect_type(args.current)

    output_lines = []
    output_lines.append(f"# Kernel Trace Comparison")
    output_lines.append(f"")
    output_lines.append(f"- Baseline: `{os.path.basename(args.baseline)}`")
    output_lines.append(f"- Current:  `{os.path.basename(args.current)}`")
    output_lines.append(f"")

    if base_type == "sqlite" and curr_type == "sqlite":
        base_cats, base_kernels = extract_categories_from_sqlite(args.baseline)
        curr_cats, curr_kernels = extract_categories_from_sqlite(args.current)

        output_lines.append("## Category Breakdown")
        output_lines.append("")
        output_lines.append(compare_categories(base_cats, curr_cats, md=args.md))
        output_lines.append("")
        output_lines.append(f"## Top {args.top} Kernels")
        output_lines.append("")
        output_lines.append(compare_kernels(base_kernels, curr_kernels, top_n=args.top, md=args.md))

    elif base_type == "ncu_csv" and curr_type == "ncu_csv":
        base_kernels = extract_from_ncu_csv(args.baseline)
        curr_kernels = extract_from_ncu_csv(args.current)

        output_lines.append(f"## ncu Kernel Metrics Comparison (Top {args.top})")
        output_lines.append("")
        output_lines.append(compare_kernels(base_kernels, curr_kernels, top_n=args.top, md=args.md))

    elif base_type == "decode_csv" and curr_type == "decode_csv":
        base_data = extract_from_decode_csv(args.baseline)
        curr_data = extract_from_decode_csv(args.current)

        output_lines.append("## Decode Walltime Comparison")
        output_lines.append("")
        output_lines.append(compare_decode_walltime(base_data, curr_data, md=args.md))

    elif base_type == "json" and curr_type == "json":
        base_cats = extract_from_json(args.baseline)
        curr_cats = extract_from_json(args.current)

        output_lines.append("## Category Breakdown (Cross-Platform)")
        output_lines.append("")
        output_lines.append(compare_categories(base_cats, curr_cats, md=args.md))

    elif base_type == "kernel_breakdown_csv" and curr_type == "kernel_breakdown_csv":
        base_cats, base_kernels = extract_from_kernel_breakdown_csv(args.baseline)
        curr_cats, curr_kernels = extract_from_kernel_breakdown_csv(args.current)

        output_lines.append("## Category Breakdown (from kernel_breakdown.csv)")
        output_lines.append("")
        output_lines.append(compare_categories(base_cats, curr_cats, md=args.md))
        output_lines.append("")
        output_lines.append(f"## Top {args.top} Operators")
        output_lines.append("")
        output_lines.append(compare_kernels(base_kernels, curr_kernels, top_n=args.top, md=args.md))

    elif base_type == "per_layer_csv" and curr_type == "per_layer_csv":
        base_cats, base_kernels = extract_from_per_layer_csv(args.baseline)
        curr_cats, curr_kernels = extract_from_per_layer_csv(args.current)

        output_lines.append("## Module Breakdown (from per_layer_breakdown.csv)")
        output_lines.append("")
        output_lines.append(compare_categories(base_cats, curr_cats, md=args.md))
        output_lines.append("")
        output_lines.append(f"## Top {args.top} Operators")
        output_lines.append("")
        output_lines.append(compare_kernels(base_kernels, curr_kernels, top_n=args.top, md=args.md))

    else:
        print(f"ERROR: Cannot compare {base_type} with {curr_type}", file=sys.stderr)
        print(f"Supported: sqlite+sqlite, ncu_csv+ncu_csv, decode_csv+decode_csv, json+json,",
              file=sys.stderr)
        print(f"           kernel_breakdown_csv+kernel_breakdown_csv, per_layer_csv+per_layer_csv",
              file=sys.stderr)
        sys.exit(1)

    report = "\n".join(output_lines)
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"\nReport saved: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
