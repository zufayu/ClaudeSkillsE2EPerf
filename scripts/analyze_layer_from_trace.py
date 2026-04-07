#!/usr/bin/env python3
"""
Per-layer walltime analysis from raw SGLang torch trace.

SGLang uses 2 CUDA streams with overlapping kernels (dual-stream for
shared+routed experts). This script computes true per-layer walltime
by merging intervals across both streams.

Layer boundary definition:
  START: lamport_allreduce_fusion kernel (first occurrence per layer)
  END:   CUDAFunctor_add kernel (vectorized_elementwise...CUDAFunctor_add)

Selects 100 stable-state decode iterations, averages per-layer metrics.

Usage:
    python3 analyze_layer_from_trace.py <trace.json.gz> [options]
    python3 analyze_layer_from_trace.py trace.json.gz --csv output.csv
    python3 analyze_layer_from_trace.py trace.json.gz --decodes 100 --skip 20
"""

import argparse
import csv
import gzip
import json
import re
import sys
from collections import defaultdict, OrderedDict
from statistics import median, stdev


def load_trace(filepath):
    """Load Chrome trace JSON (optionally gzipped)."""
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rt", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    print(f"Loaded {filepath}: {len(events)} events")
    return events


def find_gpu_kernels(events, gpu_pid=None):
    """Extract GPU kernel events for rank 0 (smallest PID), sorted by ts."""
    kernels = []
    pids = set()
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append(e)
            pids.add(e.get("pid"))

    if gpu_pid is not None:
        kernels = [k for k in kernels if k.get("pid") == gpu_pid]
    elif len(pids) > 1:
        min_pid = min(pids)
        kernels = [k for k in kernels if k.get("pid") == min_pid]
        print(f"  Multiple GPU PIDs ({len(pids)}), using PID {min_pid} (rank 0)")

    kernels.sort(key=lambda x: x.get("ts", 0))
    print(f"  GPU kernel events: {len(kernels)}")

    # Show unique TIDs (streams)
    tids = set(k.get("tid") for k in kernels)
    print(f"  GPU streams (TIDs): {sorted(tids)}")
    return kernels


def find_decode_iterations(events, kernels):
    """Find decode iteration boundaries from cudaGraphLaunch events.

    Returns list of (start_ts, end_ts) intervals for each decode iteration.
    """
    launches = []
    for e in events:
        if "cudaGraphLaunch" in e.get("name", "") and e.get("ph") == "X":
            launches.append(e)
    launches.sort(key=lambda x: x.get("ts", 0))
    print(f"  cudaGraphLaunch events: {len(launches)}")

    # Each launch maps to one decode iteration; use inter-launch intervals
    iterations = []
    for i in range(len(launches)):
        start_ts = launches[i]["ts"]
        if i + 1 < len(launches):
            end_ts = launches[i + 1]["ts"]
        else:
            # Last: estimate from median gap
            if i > 0:
                gaps = [launches[j + 1]["ts"] - launches[j]["ts"] for j in range(max(0, i - 10), i)]
                end_ts = start_ts + sorted(gaps)[len(gaps) // 2] * 2
            else:
                end_ts = start_ts + 30000
        iterations.append((start_ts, end_ts))

    return iterations


def get_iteration_kernels(kernels, start_ts, end_ts):
    """Get all GPU kernels within a time window."""
    result = []
    for k in kernels:
        k_ts = k.get("ts", 0)
        if k_ts >= start_ts and k_ts < end_ts:
            result.append(k)
    return result


# Kernel name patterns
RE_LAMPORT = re.compile(r"allreduce_fusion_kernel.*lamport|lamport.*allreduce", re.IGNORECASE)
RE_CUDA_FUNCTOR_ADD = re.compile(r"vectorized_elementwise_kernel.*CUDAFunctor_add|CUDAFunctorOnSelf_add", re.IGNORECASE)


def is_lamport(name):
    return bool(RE_LAMPORT.search(name))


def is_functor_add(name):
    return bool(RE_CUDA_FUNCTOR_ADD.search(name))


def split_into_layers(iter_kernels):
    """Split one decode iteration's kernels into layers.

    Layer boundary: lamport_allreduce_fusion (start) → CUDAFunctor_add (end).

    Each layer starts at a LAMPORT kernel and ends at the next CUDAFunctor_add
    that appears AFTER all the MoE kernels (i.e., the residual add at layer end).

    Returns list of layers, each is a list of kernel dicts with keys:
      name, ts, dur, tid
    """
    # Find all LAMPORT positions
    lamport_indices = [i for i, k in enumerate(iter_kernels) if is_lamport(k.get("name", ""))]
    # Find all CUDAFunctor_add positions
    add_indices = [i for i, k in enumerate(iter_kernels) if is_functor_add(k.get("name", ""))]

    if not lamport_indices or not add_indices:
        return []

    layers = []
    for li, lam_idx in enumerate(lamport_indices):
        lam_ts = iter_kernels[lam_idx].get("ts", 0)

        # Find the next CUDAFunctor_add AFTER this lamport
        # It should be the residual add at the end of the layer
        # (after MoE finalize). Pick the first add that comes after
        # the lamport AND before the next lamport (if any).
        next_lam_ts = None
        if li + 1 < len(lamport_indices):
            next_lam_ts = iter_kernels[lamport_indices[li + 1]].get("ts", 0)

        # Find matching CUDAFunctor_add: last add before next lamport
        best_add_idx = None
        for ai in add_indices:
            add_ts = iter_kernels[ai].get("ts", 0)
            if add_ts <= lam_ts:
                continue
            if next_lam_ts is not None and add_ts >= next_lam_ts:
                break
            best_add_idx = ai

        if best_add_idx is None:
            # Last layer might not have a closing add
            if li == len(lamport_indices) - 1:
                # Use remaining kernels
                best_add_idx = len(iter_kernels) - 1
            else:
                continue

        # Collect all kernels from lamport to add (inclusive),
        # across ALL streams (both main and alt)
        layer_start_ts = iter_kernels[lam_idx].get("ts", 0)
        layer_end_ts = iter_kernels[best_add_idx].get("ts", 0) + iter_kernels[best_add_idx].get("dur", 0)

        layer_kernels = []
        for k in iter_kernels:
            k_ts = k.get("ts", 0)
            k_end = k_ts + k.get("dur", 0)
            # Include kernel if it overlaps with the layer time window
            if k_ts >= layer_start_ts and k_ts < layer_end_ts:
                layer_kernels.append({
                    "name": k.get("name", ""),
                    "ts": k_ts,
                    "dur": k.get("dur", 0),
                    "tid": k.get("tid", 0),
                })

        if layer_kernels:
            layers.append(layer_kernels)

    return layers


def merge_intervals(intervals):
    """Merge overlapping (start, end) intervals. Returns merged list."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def compute_layer_metrics(layer_kernels):
    """Compute metrics for one layer.

    Returns dict with:
      n_kernels: number of kernels
      kernel_sum_us: sum of all kernel durations
      walltime_us: true walltime (merged intervals across streams)
      overlap_us: kernel_sum - walltime
      start_ts: first kernel start
      end_ts: last kernel end
      per_stream: {tid: {n_kernels, kernel_sum_us, walltime_us}}
    """
    if not layer_kernels:
        return None

    kernel_sum = sum(k["dur"] for k in layer_kernels)

    # Build intervals for all kernels
    intervals = [(k["ts"], k["ts"] + k["dur"]) for k in layer_kernels]
    merged = merge_intervals(intervals)
    walltime = sum(e - s for s, e in merged)

    # Per-stream breakdown
    stream_kernels = defaultdict(list)
    for k in layer_kernels:
        stream_kernels[k["tid"]].append(k)

    per_stream = {}
    for tid, sks in stream_kernels.items():
        s_sum = sum(k["dur"] for k in sks)
        s_intervals = [(k["ts"], k["ts"] + k["dur"]) for k in sks]
        s_merged = merge_intervals(s_intervals)
        s_wall = sum(e - s for s, e in s_merged)
        per_stream[tid] = {
            "n_kernels": len(sks),
            "kernel_sum_us": s_sum,
            "walltime_us": s_wall,
        }

    return {
        "n_kernels": len(layer_kernels),
        "kernel_sum_us": kernel_sum,
        "walltime_us": walltime,
        "overlap_us": kernel_sum - walltime,
        "start_ts": min(k["ts"] for k in layer_kernels),
        "end_ts": max(k["ts"] + k["dur"] for k in layer_kernels),
        "per_stream": per_stream,
    }


# Kernel classification for per-operator breakdown
KERNEL_TAGS = OrderedDict([
    (r"allreduce_fusion_kernel.*lamport", "lamport_AR+RMSNorm"),
    (r"nvjet_sm100_tst.*splitK_TNT|splitK_TNT", "splitK_GEMM"),
    (r"splitKreduce.*bf16|splitKreduce.*bfloat|splitKreduce.*Bfloat16", "splitK_reduce"),
    (r"FusedAddRMSNorm|RMSNormKernel", "RMSNorm"),
    (r"nvjet_sm100_tst.*_v_bz_TNN|_v_bz_TNN", "q_b_proj"),
    (r"nvjet_sm100_tst.*_v_bz_TNT|_v_bz_TNT", "uk_gemm"),
    (r"CatArrayBatchedCopy", "k_concat"),
    (r"RopeQuantizeKernel", "RoPE"),
    (r"set_mla_kv_buffer", "set_mla_kv"),
    (r"fmhaSm100|fmhaKernel", "FMHA"),
    (r"_h_bz_TNT(?!.*splitK)", "uv_gemm"),
    (r"_h_bz_splitK_TNT", "o_proj_splitK"),
    (r"DeviceGemmFp4GemmSm100", "FP4_GEMM"),
    (r"cvt_fp16_to_fp4", "FP4_convert"),
    (r"quantize_with_block_size", "FP4_quant"),
    (r"vectorized_elementwise_kernel.*CUDAFunctor_add|CUDAFunctorOnSelf_add", "residual_add"),
    (r"routingMainKernel", "TopK_select"),
    (r"routingIndicesCluster|routingDeepSe", "expert_sort"),
    (r"bmm_E2m1.*E2m1E2m1", "moe_gate_up"),
    (r"bmm_Bfloat16.*E2m1|bmm_.*E2m1.*Bfloat", "moe_down"),
    (r"finalizeKernelVecLoad", "moe_finalize"),
    (r"act_and_mul_kernel|silu_and_mul_kernel", "SiLU_mul"),
    (r"elementwise_kernel.*direct_copy|unrolled_elementwise.*direct_copy", "tensor_copy"),
    (r"allreduce|nccl|ncclDevKernel", "nccl_comm"),
    (r"memcpy|memset", "memop"),
])


def classify(name):
    for pattern, tag in KERNEL_TAGS.items():
        if re.search(pattern, name, re.IGNORECASE):
            return tag
    return f"other:{name[:40]}"


def compute_per_operator_breakdown(all_layer_kernels):
    """Compute averaged per-operator breakdown across all layers.

    all_layer_kernels: list of layer kernel lists
    Returns: list of {tag, avg_us, avg_count, pct}
    """
    n_layers = len(all_layer_kernels)
    if n_layers == 0:
        return []

    tag_totals = defaultdict(lambda: {"total_us": 0.0, "count": 0})
    for layer_kernels in all_layer_kernels:
        for k in layer_kernels:
            tag = classify(k["name"])
            tag_totals[tag]["total_us"] += k["dur"]
            tag_totals[tag]["count"] += 1

    grand_total = sum(v["total_us"] for v in tag_totals.values())
    result = []
    for tag, stats in tag_totals.items():
        avg_us = stats["total_us"] / n_layers
        avg_count = stats["count"] / n_layers
        pct = 100 * stats["total_us"] / grand_total if grand_total > 0 else 0
        result.append({
            "tag": tag,
            "avg_us": avg_us,
            "avg_count": avg_count,
            "pct": pct,
        })

    result.sort(key=lambda x: -x["avg_us"])
    return result


def detect_overlap_pairs(layer_kernels):
    """Detect overlapping kernel pairs across streams for one layer.

    Returns list of {kernel_a, kernel_b, overlap_us, stream_a, stream_b}
    """
    intervals = []
    for k in layer_kernels:
        intervals.append((k["ts"], k["ts"] + k["dur"], k["tid"], classify(k["name"])))
    intervals.sort()

    pairs = []
    for i in range(len(intervals)):
        s1, e1, tid1, tag1 = intervals[i]
        for j in range(i + 1, len(intervals)):
            s2, e2, tid2, tag2 = intervals[j]
            if s2 >= e1:
                break
            if tid1 != tid2:
                overlap_us = min(e1, e2) - s2
                if overlap_us > 0.5:
                    pairs.append({
                        "kernel_a": tag1,
                        "kernel_b": tag2,
                        "overlap_us": overlap_us,
                        "stream_a": tid1,
                        "stream_b": tid2,
                    })
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Per-layer walltime from SGLang torch trace")
    parser.add_argument("filepath", help="Path to trace JSON or JSON.GZ file")
    parser.add_argument("--decodes", type=int, default=100, help="Number of stable decode iterations (default: 100)")
    parser.add_argument("--skip", type=int, default=10, help="Skip first N decode iterations for warmup (default: 10)")
    parser.add_argument("--layer-range", default="5-56", help="Layer range to analyze (default: 5-56, middle layers)")
    parser.add_argument("--csv", default=None, help="Output CSV path")
    parser.add_argument("--gpu-pid", type=int, default=None, help="Filter to specific GPU PID")
    parser.add_argument("--show-layers", type=int, default=3, help="Show kernel details for first N layers (default: 3)")
    parser.add_argument("--json", default=None, help="Output JSON summary path")
    args = parser.parse_args()

    events = load_trace(args.filepath)
    kernels = find_gpu_kernels(events, gpu_pid=args.gpu_pid)

    # Find decode iterations
    iterations = find_decode_iterations(events, kernels)
    print(f"  Total decode iterations: {len(iterations)}")

    # Select stable iterations
    skip = args.skip
    n_use = min(args.decodes, len(iterations) - skip)
    if n_use <= 0:
        print(f"ERROR: Not enough iterations. Have {len(iterations)}, skip={skip}")
        sys.exit(1)

    selected = iterations[skip:skip + n_use]
    print(f"  Using iterations [{skip}, {skip + n_use}) = {n_use} iterations")

    # Parse layer range
    lr = args.layer_range.split("-")
    layer_start = int(lr[0])
    layer_end = int(lr[1]) if len(lr) > 1 else 61

    # Process each iteration
    all_layer_metrics = []  # one entry per layer across all iterations
    all_layer_kernels_for_breakdown = []
    layers_per_iter = []

    # Overlap tracking
    overlap_pair_totals = defaultdict(lambda: {"count": 0, "total_us": 0.0})

    for iter_idx, (start_ts, end_ts) in enumerate(selected):
        iter_kernels = get_iteration_kernels(kernels, start_ts, end_ts)
        layers = split_into_layers(iter_kernels)
        layers_per_iter.append(len(layers))

        # Select middle layers (avoid first/last which may be edge cases)
        sel_layers = layers[layer_start:layer_end] if len(layers) > layer_end else layers[layer_start:]

        for layer_kernels in sel_layers:
            metrics = compute_layer_metrics(layer_kernels)
            if metrics:
                all_layer_metrics.append(metrics)
                all_layer_kernels_for_breakdown.append(layer_kernels)

                # Track overlap pairs
                pairs = detect_overlap_pairs(layer_kernels)
                for p in pairs:
                    key = (p["kernel_a"], p["kernel_b"])
                    overlap_pair_totals[key]["count"] += 1
                    overlap_pair_totals[key]["total_us"] += p["overlap_us"]

    n_layers = len(all_layer_metrics)
    if n_layers == 0:
        print("ERROR: No layers found")
        sys.exit(1)

    # Compute statistics
    walltimes = [m["walltime_us"] for m in all_layer_metrics]
    kernel_sums = [m["kernel_sum_us"] for m in all_layer_metrics]
    overlaps = [m["overlap_us"] for m in all_layer_metrics]
    n_kernels_list = [m["n_kernels"] for m in all_layer_metrics]

    avg_walltime = sum(walltimes) / n_layers
    avg_kernel_sum = sum(kernel_sums) / n_layers
    avg_overlap = sum(overlaps) / n_layers
    avg_n_kernels = sum(n_kernels_list) / n_layers

    med_walltime = median(walltimes)
    med_kernel_sum = median(kernel_sums)
    std_walltime = stdev(walltimes) if n_layers > 1 else 0

    avg_layers_per_iter = sum(layers_per_iter) / len(layers_per_iter)

    # Print results
    print(f"\n{'='*80}")
    print(f"Per-Layer Analysis (lamport→CUDAFunctor_add)")
    print(f"{'='*80}")
    print(f"  Iterations used:      {n_use} (skip first {skip})")
    print(f"  Layers per iteration: avg={avg_layers_per_iter:.1f} (range: {min(layers_per_iter)}-{max(layers_per_iter)})")
    print(f"  Layer range analyzed: [{layer_start}, {layer_end})")
    print(f"  Total layer samples:  {n_layers}")
    print(f"  Avg kernels/layer:    {avg_n_kernels:.1f}")
    print(f"")
    print(f"  === KEY METRICS (per-layer avg) ===")
    print(f"  Kernel sum:    {avg_kernel_sum:.1f} μs  (median: {med_kernel_sum:.1f})")
    print(f"  Walltime:      {avg_walltime:.1f} μs  (median: {med_walltime:.1f}, std: {std_walltime:.1f})")
    print(f"  Overlap:       {avg_overlap:.1f} μs  ({avg_overlap/avg_kernel_sum*100:.1f}% of kernel_sum)")
    print(f"")
    print(f"  === FULL MODEL ESTIMATE ===")
    print(f"  Walltime × {int(avg_layers_per_iter)} layers = {avg_walltime * avg_layers_per_iter / 1000:.2f} ms")
    print(f"  Kernel sum × {int(avg_layers_per_iter)} layers = {avg_kernel_sum * avg_layers_per_iter / 1000:.2f} ms")

    # Per-stream breakdown
    all_tids = set()
    for m in all_layer_metrics:
        all_tids.update(m["per_stream"].keys())

    print(f"\n  === PER-STREAM BREAKDOWN ===")
    for tid in sorted(all_tids):
        s_walls = [m["per_stream"].get(tid, {}).get("walltime_us", 0) for m in all_layer_metrics]
        s_sums = [m["per_stream"].get(tid, {}).get("kernel_sum_us", 0) for m in all_layer_metrics]
        s_counts = [m["per_stream"].get(tid, {}).get("n_kernels", 0) for m in all_layer_metrics]
        avg_s_wall = sum(s_walls) / n_layers
        avg_s_sum = sum(s_sums) / n_layers
        avg_s_cnt = sum(s_counts) / n_layers
        print(f"  Stream {tid}: {avg_s_cnt:.1f} kernels, sum={avg_s_sum:.1f}μs, wall={avg_s_wall:.1f}μs")

    # Overlap pair analysis
    if overlap_pair_totals:
        print(f"\n  === OVERLAP PAIRS (per-layer avg, top 15) ===")
        print(f"  {'Kernel A':<25} {'Kernel B':<25} {'Avg(μs)':>8} {'Freq':>6}")
        print(f"  {'-'*68}")
        sorted_pairs = sorted(overlap_pair_totals.items(), key=lambda x: -x[1]["total_us"])
        total_pair_overlap = 0
        for (tag_a, tag_b), stats in sorted_pairs[:15]:
            avg_ov = stats["total_us"] / n_layers
            freq = stats["count"] / n_layers
            total_pair_overlap += avg_ov
            print(f"  {tag_a:<25} {tag_b:<25} {avg_ov:>8.1f} {freq:>6.1f}")
        print(f"  {'TOTAL pairs overlap':<52} {total_pair_overlap:>8.1f}")

    # Per-operator breakdown
    op_breakdown = compute_per_operator_breakdown(all_layer_kernels_for_breakdown)
    if op_breakdown:
        print(f"\n  === PER-OPERATOR BREAKDOWN (per-layer avg) ===")
        print(f"  {'#':>3} {'Operator':<30} {'Avg(μs)':>8} {'Pct%':>5} {'Count':>5}")
        print(f"  {'-'*55}")
        for i, op in enumerate(op_breakdown):
            print(f"  {i+1:>3} {op['tag']:<30} {op['avg_us']:>8.1f} {op['pct']:>5.1f} {op['avg_count']:>5.1f}")
        total_op = sum(op["avg_us"] for op in op_breakdown)
        print(f"  {'':>3} {'TOTAL':<30} {total_op:>8.1f}")

    # Show sample layers
    if args.show_layers > 0 and all_layer_kernels_for_breakdown:
        n_show = min(args.show_layers, len(all_layer_kernels_for_breakdown))
        print(f"\n  === SAMPLE LAYERS (first {n_show}) ===")
        for li in range(n_show):
            lk = all_layer_kernels_for_breakdown[li]
            m = all_layer_metrics[li]
            base_ts = lk[0]["ts"]
            print(f"\n  Layer sample {li}: {m['n_kernels']} kernels, wall={m['walltime_us']:.1f}μs, sum={m['kernel_sum_us']:.1f}μs, overlap={m['overlap_us']:.1f}μs")
            print(f"  {'#':>3} {'Kernel':<50} {'Stream':>6} {'Offset(μs)':>10} {'Dur(μs)':>8} {'Tag':<20}")
            print(f"  {'-'*100}")
            for ki, k in enumerate(lk):
                tag = classify(k["name"])
                name_short = k["name"][:50]
                offset = k["ts"] - base_ts
                print(f"  {ki+1:>3} {name_short:<50} {k['tid']:>6} {offset:>10.1f} {k['dur']:>8.1f} {tag:<20}")

    # CSV output
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iter_idx", "layer_idx", "n_kernels", "kernel_sum_us", "walltime_us", "overlap_us"])
            idx = 0
            for iter_i in range(n_use):
                n_sel = min(layer_end, layers_per_iter[iter_i]) - layer_start
                for li in range(n_sel):
                    if idx < n_layers:
                        m = all_layer_metrics[idx]
                        writer.writerow([iter_i, li, m["n_kernels"], f"{m['kernel_sum_us']:.1f}", f"{m['walltime_us']:.1f}", f"{m['overlap_us']:.1f}"])
                        idx += 1
        print(f"\nCSV written to: {args.csv}")

    # JSON summary
    if args.json:
        summary = {
            "trace_file": args.filepath,
            "n_iterations": n_use,
            "skip": skip,
            "layer_range": f"{layer_start}-{layer_end}",
            "n_layer_samples": n_layers,
            "avg_layers_per_iter": round(avg_layers_per_iter, 1),
            "per_layer_avg": {
                "kernel_sum_us": round(avg_kernel_sum, 1),
                "walltime_us": round(avg_walltime, 1),
                "overlap_us": round(avg_overlap, 1),
                "overlap_pct": round(avg_overlap / avg_kernel_sum * 100, 1),
                "n_kernels": round(avg_n_kernels, 1),
            },
            "per_layer_median": {
                "kernel_sum_us": round(med_kernel_sum, 1),
                "walltime_us": round(med_walltime, 1),
            },
            "full_model_estimate_ms": {
                "by_walltime": round(avg_walltime * avg_layers_per_iter / 1000, 2),
                "by_kernel_sum": round(avg_kernel_sum * avg_layers_per_iter / 1000, 2),
            },
            "per_operator": [
                {"tag": op["tag"], "avg_us": round(op["avg_us"], 1), "pct": round(op["pct"], 1), "avg_count": round(op["avg_count"], 1)}
                for op in op_breakdown
            ],
        }
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"JSON summary written to: {args.json}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
