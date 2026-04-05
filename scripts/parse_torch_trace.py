#!/usr/bin/env python3
"""
Generic Torch Profiler trace parser for kernel-level breakdown.

Works with both SGLang and ATOM/vLLM traces (Chrome trace JSON format).
Extracts GPU kernel events, groups by operator type, and produces
a summary table suitable for cross-platform comparison.

Usage:
    python3 scripts/parse_torch_trace.py <trace.json.gz> [options]

    # Top-20 kernels by total time:
    python3 scripts/parse_torch_trace.py trace.json.gz

    # Output CSV for further analysis:
    python3 scripts/parse_torch_trace.py trace.json.gz --csv kernel_breakdown.csv

    # Filter to decode-only events (ATOM traces with decode[bs=X] annotations):
    python3 scripts/parse_torch_trace.py trace.json.gz --phase decode

Output:
    Prints kernel breakdown table to stdout.
    Optionally writes CSV with --csv flag.
"""

import argparse
import csv
import gzip
import json
import re
import sys
from collections import defaultdict


def load_trace(filepath):
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rt", encoding="utf-8") as f:
        return json.load(f)


def classify_kernel(name):
    """Classify a GPU kernel name into a high-level operator category."""
    n = name.lower()
    # Attention (check before GEMM since fmha kernels may contain "gemm")
    if any(k in n for k in ["fmha", "flash_fwd", "flash_bwd", "flash_attn", "mha_", "merge_attn", "concat_and_cast_mha", "set_mla_kv"]):
        return "Attention"
    # MoE / Expert routing and compute
    # bmm_E2m1 with swiGlu = MoE expert GEMM with fused activation
    # nvjet_sm100_tst = NVIDIA Blackwell MoE tensor kernels
    if any(k in n for k in ["moe::", "expert", "routing"]):
        return "MoE/Expert"
    if "swiglu" in n or "swig" in n:
        return "MoE/Expert (fused GEMM+SwiGLU)"
    if "nvjet_sm100" in n:
        return "MoE/Expert (nvjet)"
    if "bmm_" in n and ("e2m1" in n or "bfloat16_e2m1" in n):
        return "MoE/Expert (BMM)"
    # AllReduce / Communication (check before GEMM)
    if any(k in n for k in ["allreduce", "reduce_scatter", "all_gather", "allgather", "nccl", "rccl", "ncclkernel", "device_load", "device_store"]):
        return "Communication"
    # GEMM / MatMul
    if any(k in n for k in ["gemm", "gemv", "cutlass", "cublas", "matmul", "dot_product", "cijk_", "bf16gemm", "fp8gemm", "splitkreduce"]):
        return "GEMM/MatMul"
    # Normalization
    if any(k in n for k in ["layernorm", "rmsnorm", "batchnorm", "groupnorm"]):
        return "Normalization"
    # Elementwise / Activation
    if any(k in n for k in ["silu", "gelu", "relu", "elementwise", "add_kernel", "mul_kernel", "act_and_mul"]):
        return "Activation/Elementwise"
    # Quantization
    if any(k in n for k in ["quant", "dequant", "cvt_fp16_to_fp4", "cvt_fp4", "fp4", "fp8", "mxfp"]):
        return "Quantization"
    # Copy / Memory
    if any(k in n for k in ["memcpy", "memset", "copy", "transpose"]):
        return "Memory"
    # Embedding / Positional
    if any(k in n for k in ["embedding", "rotary", "rope"]):
        return "Embedding/RoPE"
    # Sampling
    if any(k in n for k in ["sample", "argmax", "topk_sampling", "topp"]):
        return "Sampling"
    return "Other"


def get_phase_ranges(events):
    """Extract decode/prefill phase time ranges from ATOM-style annotations."""
    ranges = {"decode": [], "prefill": []}
    for e in events:
        name = e.get("name", "")
        if e.get("ph") != "X" or e.get("cat") != "gpu_user_annotation":
            continue
        ts = e.get("ts", 0)
        dur = e.get("dur", 0)
        if name.startswith("decode["):
            ranges["decode"].append((ts, ts + dur))
        elif name.startswith("prefill"):
            ranges["prefill"].append((ts, ts + dur))
    return ranges


def in_phase(ts, phase_ranges):
    """Check if timestamp falls within any of the phase ranges."""
    for start, end in phase_ranges:
        if start <= ts <= end:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Parse Torch Profiler trace for kernel breakdown")
    parser.add_argument("filepath", help="Path to trace JSON or JSON.GZ file")
    parser.add_argument("--csv", default=None, help="Output CSV path")
    parser.add_argument("--phase", choices=["decode", "prefill", "all"], default="all",
                        help="Filter to specific phase (ATOM traces only)")
    parser.add_argument("--top", type=int, default=30, help="Show top N kernels")
    parser.add_argument("--by-category", action="store_true", help="Group by category instead of individual kernels")
    parser.add_argument("--gpu-pid", type=int, default=None, help="Filter to specific GPU PID (rank)")
    args = parser.parse_args()

    print(f"Loading: {args.filepath}")
    data = load_trace(args.filepath)
    events = data.get("traceEvents", [])
    print(f"Total events: {len(events)}")

    # Find GPU kernel events
    gpu_kernels = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        # GPU kernels have cat="kernel" or "gpu_memcpy" or similar
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu_kernels.append(e)
        # Also check for cuda_runtime kernels that launched GPU work
        # Some traces put kernels under different categories
        if "kernel" in cat.lower() and cat != "cpu_op":
            gpu_kernels.append(e)

    # Deduplicate (some may match multiple conditions)
    seen = set()
    unique_kernels = []
    for k in gpu_kernels:
        key = (k.get("ts"), k.get("name"), k.get("pid"), k.get("tid"))
        if key not in seen:
            seen.add(key)
            unique_kernels.append(k)
    gpu_kernels = unique_kernels

    print(f"GPU kernel events: {len(gpu_kernels)}")

    # Filter by GPU PID if specified
    if args.gpu_pid is not None:
        gpu_kernels = [k for k in gpu_kernels if k.get("pid") == args.gpu_pid]
        print(f"Filtered to GPU PID {args.gpu_pid}: {len(gpu_kernels)} kernels")

    # If only one GPU, auto-select; if multiple, use rank 0
    if not args.gpu_pid:
        pids = set(k.get("pid") for k in gpu_kernels)
        if len(pids) > 1:
            # Pick smallest PID (usually rank 0)
            min_pid = min(pids)
            gpu_kernels = [k for k in gpu_kernels if k.get("pid") == min_pid]
            print(f"Multiple GPUs detected ({len(pids)} PIDs), using PID {min_pid} (rank 0): {len(gpu_kernels)} kernels")

    # Phase filtering (for ATOM traces)
    if args.phase != "all":
        phase_ranges = get_phase_ranges(events)
        target_ranges = phase_ranges.get(args.phase, [])
        if target_ranges:
            gpu_kernels = [k for k in gpu_kernels if in_phase(k.get("ts", 0), target_ranges)]
            print(f"Filtered to {args.phase} phase: {len(gpu_kernels)} kernels")
        else:
            print(f"Warning: no {args.phase} annotations found, using all kernels")

    if not gpu_kernels:
        print("No GPU kernel events found!")
        sys.exit(1)

    # Aggregate by kernel name or category
    if args.by_category:
        groups = defaultdict(lambda: {"count": 0, "total_us": 0, "min_us": float("inf"), "max_us": 0})
        for k in gpu_kernels:
            cat = classify_kernel(k.get("name", ""))
            dur = k.get("dur", 0)
            groups[cat]["count"] += 1
            groups[cat]["total_us"] += dur
            groups[cat]["min_us"] = min(groups[cat]["min_us"], dur)
            groups[cat]["max_us"] = max(groups[cat]["max_us"], dur)
    else:
        groups = defaultdict(lambda: {"count": 0, "total_us": 0, "min_us": float("inf"), "max_us": 0})
        for k in gpu_kernels:
            name = k.get("name", "unknown")
            dur = k.get("dur", 0)
            groups[name]["count"] += 1
            groups[name]["total_us"] += dur
            groups[name]["min_us"] = min(groups[name]["min_us"], dur)
            groups[name]["max_us"] = max(groups[name]["max_us"], dur)

    total_time = sum(g["total_us"] for g in groups.values())

    # Sort by total time descending
    sorted_groups = sorted(groups.items(), key=lambda x: -x[1]["total_us"])

    # Print table
    label = "Category" if args.by_category else "Kernel"
    print(f"\n{'=' * 100}")
    print(f"Kernel Breakdown (phase={args.phase}, total GPU time={total_time/1000:.1f}ms)")
    print(f"{'=' * 100}")
    print(f"{'#':>3} | {label:<60} | {'Count':>6} | {'Total(ms)':>10} | {'Pct%':>5} | {'Avg(us)':>8}")
    print("-" * 100)

    cumulative = 0
    rows = []
    for i, (name, stats) in enumerate(sorted_groups[:args.top]):
        pct = 100 * stats["total_us"] / total_time if total_time > 0 else 0
        avg = stats["total_us"] / stats["count"] if stats["count"] > 0 else 0
        cumulative += pct
        display_name = name[:60] if len(name) > 60 else name
        print(f"{i+1:>3} | {display_name:<60} | {stats['count']:>6} | {stats['total_us']/1000:>10.2f} | {pct:>5.1f} | {avg:>8.1f}")
        rows.append({
            "rank": i + 1,
            "name": name,
            "category": classify_kernel(name) if not args.by_category else name,
            "count": stats["count"],
            "total_ms": stats["total_us"] / 1000,
            "pct": pct,
            "avg_us": avg,
            "min_us": stats["min_us"],
            "max_us": stats["max_us"],
        })

    remaining = len(sorted_groups) - args.top
    if remaining > 0:
        remaining_pct = 100 - cumulative
        print(f"    | ... {remaining} more kernels | {'':>6} | {'':>10} | {remaining_pct:>5.1f} |")

    print(f"\nTotal: {len(groups)} unique kernels, {sum(g['count'] for g in groups.values())} invocations")

    # Category summary
    if not args.by_category:
        print(f"\n{'=' * 70}")
        print("Category Summary")
        print(f"{'=' * 70}")
        cat_stats = defaultdict(lambda: {"count": 0, "total_us": 0})
        for name, stats in groups.items():
            cat = classify_kernel(name)
            cat_stats[cat]["count"] += stats["count"]
            cat_stats[cat]["total_us"] += stats["total_us"]

        for cat, stats in sorted(cat_stats.items(), key=lambda x: -x[1]["total_us"]):
            pct = 100 * stats["total_us"] / total_time if total_time > 0 else 0
            print(f"  {cat:<25} | {stats['count']:>6} calls | {stats['total_us']/1000:>10.2f}ms | {pct:>5.1f}%")

    # CSV output
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["rank", "name", "category", "count", "total_ms", "pct", "avg_us", "min_us", "max_us"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to: {args.csv}")


if __name__ == "__main__":
    main()
