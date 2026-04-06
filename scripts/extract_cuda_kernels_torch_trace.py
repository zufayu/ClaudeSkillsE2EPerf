#!/usr/bin/env python3
"""
Extract per-decode-step kernel breakdown from Torch Profiler traces
containing CUDA Graph captures.

Works with SGLang and any framework that uses CUDA Graphs during decode.
The trace viewer shows cudaGraphLaunch as a single block on the GPU timeline,
but the underlying trace JSON contains flow events (ph="s"/"f") linking each
launch to the individual GPU kernels executed inside the graph.

This script:
  1. Finds all cudaGraphLaunch events on the CPU timeline
  2. Follows flow events to map each launch → list of GPU kernels
  3. Falls back to time-window matching if flow events are absent
  4. Maps kernel names to logical operators (configurable per platform)
  5. Outputs per-decode-step and averaged kernel breakdown tables

Usage:
    # Analyze SGLang B200 trace:
    python3 scripts/extract_cuda_graph_kernels.py trace.json.gz --platform b200

    # Analyze ATOM MI355X trace:
    python3 scripts/extract_cuda_graph_kernels.py trace.json.gz --platform mi355x

    # Just dump raw kernels per step (no logical mapping):
    python3 scripts/extract_cuda_graph_kernels.py trace.json.gz --raw

    # Output CSV:
    python3 scripts/extract_cuda_graph_kernels.py trace.json.gz --csv breakdown.csv
"""

import argparse
import csv
import gzip
import json
import re
import sys
from collections import defaultdict, OrderedDict

# =============================================================================
# Kernel name → logical operator mapping
# =============================================================================

# B200 SGLang FP4 kernel mapping
# Kernel names vary between nsys (short) and torch profiler (full C++ mangled).
# Regexes must match both forms.
B200_KERNEL_MAP = OrderedDict([
    # Pre-attention: MoE finalize + residual + lamport allreduce
    (r"finalizeKernelVecLoad|moefinalize", "pre_attn: MoE_finalize+residual"),
    (r"vectorized_elementwise_kernel.*CUDAFunctor_add|elementwise.*add", "pre_attn: residual_add"),
    # Communication: lamport allreduce fusion (includes RMSNorm)
    (r"allreduce_fusion_kernel.*lamport|moefinalize_lamport", "comm: lamport_AR+RMSNorm"),
    # QKV projection
    (r"splitK_TNT|nvjet_splitK_TNT", "qkv_proj: qkv_a_proj_GEMM"),
    (r"splitKreduce.*bf16|splitKreduce.*bfloat|splitKreduce.*Bfloat16", "qkv_proj: qkv_a_splitK_reduce"),
    (r"RMSNormKernel", "qkv_proj: q/k_norm_RMSNorm"),
    (r"_v_bz_TNN|nvjet_tst_TNN", "qkv_proj: q_b_proj_GEMM"),
    (r"_v_bz_TNT|nvjet_sm100_tst_128x64.*TNT", "qkv_proj: uk_gemm"),
    (r"CatArrayBatchedCopy", "qkv_proj: k_concat"),
    # RoPE + Attention
    (r"RopeQuantizeKernel|applyMLARopeAndAssignQKV", "rope_attn: RoPE+KV_write"),
    (r"fmhaSm100|fmhaKernel", "rope_attn: Attention_FMHA"),
    (r"set_mla_kv_buffer", "rope_attn: set_mla_kv"),
    # Output projection
    (r"_h_bz_TNT(?!.*splitK)|nvjet_tst_TNT", "out_proj: uv_gemm"),
    (r"_h_bz_splitK_TNT", "out_proj: o_proj_splitK_GEMM"),
    (r"nvjet_ootst_FP4|DeviceGemmFp4GemmSm100", "out_proj/shared: FP4_GEMM"),
    (r"quantize_with_block_size", "quant: FP4_blockwise_quant"),
    # Post-attention communication
    (r"userbuffers_rmsnorm", "post_attn: TP_AR+RMSNorm"),
    (r"userbuffers_allgather", "post_attn: EP_allgather"),
    # Router
    (r"nvjet_tss_splitK|splitK.*router", "router: router_GEMM"),
    (r"splitKreduce.*fp32|splitKreduce.*float32|splitKreduce.*Fp32", "router: router_splitK_reduce"),
    (r"routingMainKernel", "router: TopK_select"),
    (r"routingIndicesCluster", "router: expert_sort"),
    # MoE expert
    (r"bmm_E2m1.*[Ss]wi[Gg]lu|bmm_E2m1.*E2m1E2m1", "moe: gate_up_GEMM"),
    (r"bmm_Bfloat16|bmm_.*E2m1.*Bfloat", "moe: down_GEMM"),
    # Shared expert
    (r"act_and_mul_kernel|silu_and_mul_kernel", "shared: SiLU_mul"),
    # Elementwise
    (r"cvt_fp16_to_fp4|cvt_fp4", "quant: FP4_convert"),
    # Communication (catch-all after specific lamport pattern)
    (r"allreduce|reduce_scatter|all_gather|nccl", "comm: allreduce/other"),
    # Memory
    (r"memcpy|memset", "mem: copy/set"),
    # Copy kernels
    (r"unrolled_elementwise_kernel.*direct_copy|direct_copy_kernel", "mem: tensor_copy"),
])

# MI355X ATOM MXFP4 kernel mapping
MI355X_KERNEL_MAP = OrderedDict([
    (r"reduce_scatter", "pre_attn_comm: reduce_scatter"),
    (r"rmsnorm|rms_norm", "norm: RMSNorm"),
    (r"dynamic_per_token_scaled_quant", "quant: per_token_quant"),
    (r"fused_rms_fp8_group_quant", "quant: fused_rms_fp8_group_quant"),
    (r"gemm_xdl_preshuffle", "gemm: preshuffle_GEMM"),
    (r"batched_gemm_a8w8", "gemm: batched_a8w8_GEMM"),
    (r"fuse_qk_rope_concat_and_cache_mla", "rope_attn: RoPE+KV_write"),
    (r"mla_a8w8_qh16", "rope_attn: Attention_MLA"),
    (r"kn_mla_reduce", "rope_attn: MLA_reduce"),
    (r"bf16gemm_splitk", "router: router_GEMM"),
    (r"grouped_topk_opt_sort", "router: TopK_select"),
    (r"MoeSorting", "router: MoE_sort"),
    (r"fused_mxfp4_quant_moe_sort", "moe: fused_quant_sort"),
    (r"kernel_moe_mxgemm", "moe: MoE_GEMM"),
    (r"allreduce|all_gather|rccl", "comm: allreduce/other"),
    (r"memcpy|memset", "mem: copy/set"),
])


def classify_kernel(name, kernel_map):
    """Map a kernel name to its logical operator using regex patterns."""
    for pattern, label in kernel_map.items():
        if re.search(pattern, name, re.IGNORECASE):
            return label
    return f"other: {name[:50]}"


# =============================================================================
# Trace parsing
# =============================================================================

def load_trace(filepath):
    """Load Chrome trace JSON (optionally gzipped)."""
    print(f"Loading: {filepath}")
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rt", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    print(f"  Total events: {len(events)}")
    return events


def find_cuda_graph_launches(events):
    """Find all cudaGraphLaunch events on the CPU timeline."""
    launches = []
    for e in events:
        name = e.get("name", "")
        if "cudaGraphLaunch" in name and e.get("ph") == "X":
            launches.append(e)
    launches.sort(key=lambda x: x.get("ts", 0))
    print(f"  cudaGraphLaunch events: {len(launches)}")
    return launches


def find_gpu_kernels(events, gpu_pid=None):
    """Find all GPU kernel events."""
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
        # Use rank 0 (smallest PID)
        min_pid = min(pids)
        kernels = [k for k in kernels if k.get("pid") == min_pid]
        print(f"  Multiple GPU PIDs, using {min_pid} (rank 0)")

    kernels.sort(key=lambda x: x.get("ts", 0))
    print(f"  GPU kernel events: {len(kernels)}")
    return kernels


def build_flow_map(events):
    """Build flow event map: flow_id → (source_event, target_event).

    Flow events in Chrome trace format:
      ph="s" (start): origin of the flow arrow
      ph="f" (finish): destination of the flow arrow
      id: unique flow identifier

    In Torch Profiler traces, flows connect cudaGraphLaunch (CPU) to
    individual GPU kernels inside the graph.
    """
    flow_starts = {}  # id → event
    flow_ends = defaultdict(list)  # id → [events]

    for e in events:
        ph = e.get("ph")
        if ph == "s":
            fid = e.get("id")
            if fid is not None:
                flow_starts[fid] = e
        elif ph == "f":
            fid = e.get("id")
            if fid is not None:
                flow_ends[fid].append(e)

    print(f"  Flow events: {len(flow_starts)} starts, {sum(len(v) for v in flow_ends.values())} ends")
    return flow_starts, flow_ends


def match_kernels_by_flow(launch, all_events, flow_starts, flow_ends):
    """Find GPU kernels linked to a cudaGraphLaunch via flow events.

    Strategy: find flow events that originate from the same (pid,tid)
    and overlap temporally with the launch event.
    """
    launch_pid = launch.get("pid")
    launch_tid = launch.get("tid")
    launch_ts = launch.get("ts", 0)
    launch_dur = launch.get("dur", 0)
    launch_end = launch_ts + launch_dur

    matched_kernels = []

    for fid, start_evt in flow_starts.items():
        # Flow start should be from the same thread as the launch
        if start_evt.get("pid") != launch_pid or start_evt.get("tid") != launch_tid:
            continue
        # Flow start should be within the launch's time window
        flow_ts = start_evt.get("ts", 0)
        if flow_ts < launch_ts or flow_ts > launch_end:
            continue
        # Follow to the flow end(s) — these point to GPU kernel events
        # Filter: only accept actual GPU kernel/memcpy/memset events,
        # not ac2g (async cuda graph correlation) or other non-kernel events
        for end_evt in flow_ends.get(fid, []):
            end_cat = end_evt.get("cat", "")
            if end_cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
                continue
            matched_kernels.append({
                "name": end_evt.get("name", ""),
                "ts": end_evt.get("ts", 0),
                "dur": end_evt.get("dur", 0),
                "tid": end_evt.get("tid"),
                "pid": end_evt.get("pid"),
                "cat": end_cat,
            })

    matched_kernels.sort(key=lambda x: x["ts"])
    return matched_kernels


def match_kernels_by_time(launch, gpu_kernels):
    """Fallback: match GPU kernels that execute during a cudaGraphLaunch window.

    When flow events are absent, we use temporal overlap:
    a GPU kernel belongs to this launch if it starts within the launch's
    duration window (with a small margin).
    """
    launch_ts = launch.get("ts", 0)
    launch_dur = launch.get("dur", 0)
    launch_end = launch_ts + launch_dur
    margin = 100  # μs margin for clock skew between CPU/GPU

    matched = []
    for k in gpu_kernels:
        k_ts = k.get("ts", 0)
        if k_ts >= launch_ts - margin and k_ts <= launch_end + margin:
            matched.append({
                "name": k.get("name", ""),
                "ts": k_ts,
                "dur": k.get("dur", 0),
                "tid": k.get("tid"),
                "pid": k.get("pid"),
                "cat": k.get("cat", ""),
            })

    matched.sort(key=lambda x: x["ts"])
    return matched


def match_kernels_by_interval(launches, gpu_kernels, idx):
    """Match GPU kernels in the interval [launch[idx].ts, launch[idx+1].ts).

    Each cudaGraphLaunch is one decode iteration. GPU kernels execute
    asynchronously well past the CPU launch duration, so we use the
    inter-launch interval (not launch.dur) as the kernel ownership window.
    """
    start_ts = launches[idx].get("ts", 0)
    if idx + 1 < len(launches):
        end_ts = launches[idx + 1].get("ts", 0)
    else:
        # Last launch: use start + 2 * median gap as window
        if idx > 0:
            gaps = [launches[j+1]["ts"] - launches[j]["ts"] for j in range(max(0, idx-5), idx)]
            median_gap = sorted(gaps)[len(gaps)//2] if gaps else 20000
            end_ts = start_ts + 2 * median_gap
        else:
            end_ts = start_ts + 20000  # 20ms default

    matched = []
    for k in gpu_kernels:
        k_ts = k.get("ts", 0)
        if k_ts >= start_ts and k_ts < end_ts:
            matched.append({
                "name": k.get("name", ""),
                "ts": k_ts,
                "dur": k.get("dur", 0),
                "tid": k.get("tid"),
                "pid": k.get("pid"),
                "cat": k.get("cat", ""),
            })

    matched.sort(key=lambda x: x["ts"])
    return matched


def extract_decode_steps(events, max_steps=None, skip_first=5):
    """Extract kernel lists for each decode step (cudaGraphLaunch).

    Each cudaGraphLaunch = one full decode iteration. GPU kernels are
    matched using inter-launch intervals (not flow events or launch dur),
    because GPU execution extends well past the CPU launch call.

    Args:
        events: raw trace events
        max_steps: limit number of steps to process
        skip_first: skip first N launches (warmup/prefill)

    Returns:
        List of (launch_event, [kernel_dicts]) tuples
    """
    launches = find_cuda_graph_launches(events)
    gpu_kernels = find_gpu_kernels(events)

    # Skip first few launches (often prefill or ramp-up)
    if skip_first and len(launches) > skip_first:
        launches = launches[skip_first:]
        print(f"  Skipped first {skip_first} launches, using {len(launches)} remaining")

    if max_steps:
        launches = launches[:max_steps]

    decode_steps = []

    for i, launch in enumerate(launches):
        kernels = match_kernels_by_interval(launches, gpu_kernels, i)

        if kernels:
            decode_steps.append((launch, kernels))

        if i == 0:
            print(f"  Kernel matching method: inter-launch interval")
            if kernels:
                span = kernels[-1]['ts'] - kernels[0]['ts']
                print(f"  First decode step: {len(kernels)} kernels, span={span:.0f}μs ({span/1000:.2f}ms)")
            else:
                print(f"  First decode step: 0 kernels")

    print(f"  Total decode steps extracted: {len(decode_steps)}")
    return decode_steps


# =============================================================================
# Analysis and output
# =============================================================================

def compute_step_breakdown(kernels, kernel_map):
    """Compute per-logical-operator timing for one decode step."""
    ops = OrderedDict()
    for k in kernels:
        label = classify_kernel(k["name"], kernel_map)
        if label not in ops:
            ops[label] = {"count": 0, "total_us": 0, "kernels": []}
        ops[label]["count"] += 1
        ops[label]["total_us"] += k["dur"]
        ops[label]["kernels"].append(k["name"])
    return ops


def compute_average_breakdown(decode_steps, kernel_map):
    """Average kernel breakdown across multiple decode steps."""
    all_ops = defaultdict(lambda: {"count": 0, "total_us": 0, "n_steps": 0, "kernel_names": set()})

    for launch, kernels in decode_steps:
        step_ops = compute_step_breakdown(kernels, kernel_map)
        for label, stats in step_ops.items():
            all_ops[label]["count"] += stats["count"]
            all_ops[label]["total_us"] += stats["total_us"]
            all_ops[label]["n_steps"] += 1
            for kn in stats["kernels"]:
                all_ops[label]["kernel_names"].add(kn)

    n_steps = len(decode_steps)
    result = []
    for label, stats in all_ops.items():
        avg_us = stats["total_us"] / n_steps if n_steps > 0 else 0
        avg_count = stats["count"] / n_steps if n_steps > 0 else 0
        result.append({
            "operator": label,
            "avg_us": avg_us,
            "avg_count": avg_count,
            "total_us": stats["total_us"],
            "n_steps_present": stats["n_steps"],
            "kernel_names": sorted(stats["kernel_names"]),
        })

    result.sort(key=lambda x: -x["avg_us"])
    return result


def print_breakdown_table(breakdown, n_steps):
    """Print formatted breakdown table."""
    total_us = sum(r["avg_us"] for r in breakdown)

    print(f"\n{'='*110}")
    print(f"Per-Decode-Step Kernel Breakdown (averaged over {n_steps} steps, total={total_us:.1f}μs)")
    print(f"{'='*110}")
    print(f"{'#':>3} | {'Logical Operator':<45} | {'Avg(μs)':>8} | {'Pct%':>5} | {'Count':>5} | {'Kernel(s)':<40}")
    print("-" * 110)

    for i, r in enumerate(breakdown):
        pct = 100 * r["avg_us"] / total_us if total_us > 0 else 0
        # Show first kernel name (truncated)
        knames = ", ".join(r["kernel_names"][:2])
        if len(knames) > 40:
            knames = knames[:37] + "..."
        print(f"{i+1:>3} | {r['operator']:<45} | {r['avg_us']:>8.1f} | {pct:>5.1f} | {r['avg_count']:>5.1f} | {knames:<40}")

    print(f"\n  Total per-step: {total_us:.1f}μs ({total_us/1000:.2f}ms)")


def print_raw_step(launch, kernels, step_idx):
    """Print raw kernel list for one decode step."""
    launch_dur = launch.get("dur", 0)
    print(f"\n--- Decode Step {step_idx} (cudaGraphLaunch dur={launch_dur:.0f}μs, {len(kernels)} kernels) ---")
    print(f"{'#':>3} | {'Kernel Name':<60} | {'ts':>12} | {'dur(μs)':>8} | {'stream':>8}")
    print("-" * 100)
    base_ts = kernels[0]["ts"] if kernels else 0
    for i, k in enumerate(kernels):
        name = k["name"][:60]
        rel_ts = k["ts"] - base_ts
        print(f"{i+1:>3} | {name:<60} | {rel_ts:>12.0f} | {k['dur']:>8.1f} | {k['tid']}")


def write_csv(breakdown, n_steps, filepath):
    """Write breakdown to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "operator", "avg_us", "pct", "avg_count",
            "total_us", "n_steps_present", "kernel_names"
        ])
        writer.writeheader()
        total_us = sum(r["avg_us"] for r in breakdown)
        for i, r in enumerate(breakdown):
            pct = 100 * r["avg_us"] / total_us if total_us > 0 else 0
            writer.writerow({
                "rank": i + 1,
                "operator": r["operator"],
                "avg_us": f"{r['avg_us']:.2f}",
                "pct": f"{pct:.1f}",
                "avg_count": f"{r['avg_count']:.1f}",
                "total_us": f"{r['total_us']:.1f}",
                "n_steps_present": r["n_steps_present"],
                "kernel_names": "; ".join(r["kernel_names"]),
            })
    print(f"\nCSV written to: {filepath}")


# =============================================================================
# Main
# =============================================================================

def print_trace_info(events):
    """Print trace metadata: time span, phases, decode steps, layers, kernel counts."""
    # Time span
    all_ts = [e.get("ts", 0) for e in events if e.get("ts")]
    all_dur = [e.get("dur", 0) for e in events if e.get("dur")]
    if not all_ts:
        print("  No timestamped events found.")
        return

    min_ts = min(all_ts)
    max_ts = max(all_ts)
    # Account for duration of last events
    max_end = max(e.get("ts", 0) + e.get("dur", 0) for e in events if e.get("ts"))
    span_us = max_end - min_ts
    span_s = span_us / 1e6

    print(f"\n{'='*80}")
    print(f"TRACE INFO")
    print(f"{'='*80}")
    print(f"  Total events:    {len(events)}")
    print(f"  Time span:       {span_s:.2f}s ({span_us/1e6:.2f}s)")
    print(f"  Time range:      [{min_ts/1e6:.3f}s, {max_end/1e6:.3f}s]")

    # Categorize events
    cats = defaultdict(int)
    phs = defaultdict(int)
    for e in events:
        cats[e.get("cat", "")] += 1
        phs[e.get("ph", "")] += 1

    print(f"\n  Event categories:")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1])[:15]:
        print(f"    {cat or '(none)':<35s} {cnt:>8d}")

    # PIDs (processes/GPUs)
    pids = defaultdict(set)
    for e in events:
        pid = e.get("pid")
        cat = e.get("cat", "")
        if pid is not None:
            pids[pid].add(cat)
    print(f"\n  PIDs ({len(pids)}):")
    for pid in sorted(pids, key=lambda x: str(x)):
        cats_str = ", ".join(sorted(c for c in pids[pid] if c)[:5])
        print(f"    PID {pid}: {cats_str}")

    # cudaGraphLaunch events
    launches = [e for e in events if "cudaGraphLaunch" in e.get("name", "") and e.get("ph") == "X"]
    launches.sort(key=lambda x: x.get("ts", 0))
    print(f"\n  cudaGraphLaunch events: {len(launches)}")
    if launches:
        first_ts = (launches[0]["ts"] - min_ts) / 1e6
        last_ts = (launches[-1]["ts"] - min_ts) / 1e6
        print(f"    First at: {first_ts:.3f}s, Last at: {last_ts:.3f}s")
        durs = [l.get("dur", 0) for l in launches]
        print(f"    Duration: avg={sum(durs)/len(durs)/1e3:.2f}ms, min={min(durs)/1e3:.2f}ms, max={max(durs)/1e3:.2f}ms")

        # Detect decode step kernel counts by sampling a few launches
        sample_indices = [0, len(launches)//4, len(launches)//2, 3*len(launches)//4, len(launches)-1]
        print(f"\n    Sampled launches (kernel count via time-window):")
        gpu_kernels = sorted(
            [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")],
            key=lambda x: x.get("ts", 0)
        )
        for idx in sample_indices:
            if idx >= len(launches):
                continue
            l = launches[idx]
            l_ts = l["ts"]
            l_end = l_ts + l.get("dur", 0)
            matched = sum(1 for k in gpu_kernels if k["ts"] >= l_ts - 100 and k["ts"] <= l_end + 100)
            rel_s = (l_ts - min_ts) / 1e6
            print(f"      Launch[{idx:>4d}] at {rel_s:>8.3f}s  dur={l.get('dur',0)/1e3:>7.2f}ms  ~{matched} kernels")

    # GPU kernel events
    gpu_kernels_all = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    print(f"\n  GPU kernel events: {len(gpu_kernels_all)}")

    # Unique kernel names
    kernel_names = defaultdict(int)
    for e in gpu_kernels_all:
        kernel_names[e.get("name", "")[:80]] += 1
    print(f"  Unique kernel names: {len(kernel_names)}")
    print(f"\n  Top 20 kernels by frequency:")
    for name, cnt in sorted(kernel_names.items(), key=lambda x: -x[1])[:20]:
        print(f"    {cnt:>8d}x  {name}")

    # NVTX / user annotations (layer markers, decode/prefill markers)
    annotations = [e for e in events if e.get("cat") in ("gpu_user_annotation", "user_annotation", "python_function") or (e.get("cat", "").startswith("nvtx"))]
    if annotations:
        ann_names = defaultdict(int)
        for e in annotations:
            ann_names[e.get("name", "")[:60]] += 1
        print(f"\n  Annotations/NVTX: {len(annotations)} events, {len(ann_names)} unique")
        print(f"  Top 20 annotations:")
        for name, cnt in sorted(ann_names.items(), key=lambda x: -x[1])[:20]:
            print(f"    {cnt:>8d}x  {name}")

    # Flow events
    flow_s = sum(1 for e in events if e.get("ph") == "s")
    flow_f = sum(1 for e in events if e.get("ph") == "f")
    print(f"\n  Flow events: {flow_s} starts, {flow_f} ends")

    # Profiler step markers
    steps = [e for e in events if "ProfilerStep" in e.get("name", "") and e.get("ph") == "X"]
    if steps:
        steps.sort(key=lambda x: x.get("ts", 0))
        print(f"\n  ProfilerStep markers: {len(steps)}")
        for s in steps[:5]:
            rel_s = (s["ts"] - min_ts) / 1e6
            dur_s = s.get("dur", 0) / 1e6
            print(f"    {s['name']}: at {rel_s:.3f}s, dur={dur_s:.3f}s")
        if len(steps) > 5:
            print(f"    ... and {len(steps)-5} more")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-decode-step kernel breakdown from CUDA Graph traces"
    )
    parser.add_argument("filepath", help="Path to trace JSON or JSON.GZ file")
    parser.add_argument("--platform", choices=["b200", "mi355x", "auto"], default="auto",
                        help="Platform for kernel name mapping (default: auto-detect)")
    parser.add_argument("--info", action="store_true",
                        help="Print trace metadata only (time span, phases, decode steps, layers)")
    parser.add_argument("--raw", action="store_true",
                        help="Print raw kernel lists instead of logical operator breakdown")
    parser.add_argument("--csv", default=None, help="Output CSV path")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Max decode steps to analyze (default: 20)")
    parser.add_argument("--skip-first", type=int, default=5,
                        help="Skip first N cudaGraphLaunch events (default: 5)")
    parser.add_argument("--show-steps", type=int, default=2,
                        help="Show raw kernels for first N steps (default: 2)")
    parser.add_argument("--gpu-pid", type=int, default=None,
                        help="Filter to specific GPU PID")
    args = parser.parse_args()

    events = load_trace(args.filepath)

    if args.info:
        print_trace_info(events)
        return

    # Auto-detect platform from kernel names
    if args.platform == "auto":
        sample = " ".join(e.get("name", "") for e in events[:5000])
        if "nvjet" in sample or "fmhaSm100" in sample:
            platform = "b200"
        elif "gemm_xdl" in sample or "mla_a8w8" in sample or "rccl" in sample:
            platform = "mi355x"
        else:
            platform = "b200"
        print(f"  Auto-detected platform: {platform}")
    else:
        platform = args.platform

    kernel_map = B200_KERNEL_MAP if platform == "b200" else MI355X_KERNEL_MAP

    # Extract decode steps
    decode_steps = extract_decode_steps(
        events,
        max_steps=args.max_steps,
        skip_first=args.skip_first,
    )

    if not decode_steps:
        print("\nERROR: No decode steps found. Possible reasons:")
        print("  - Trace doesn't contain cudaGraphLaunch events")
        print("  - CUDA Graphs not used (enforce_eager=true)")
        print("  - Trace too short or only contains prefill")
        sys.exit(1)

    # Show raw steps
    if args.raw or args.show_steps:
        n_show = len(decode_steps) if args.raw else min(args.show_steps, len(decode_steps))
        for i in range(n_show):
            launch, kernels = decode_steps[i]
            print_raw_step(launch, kernels, i)

    # Compute and print averaged breakdown
    if not args.raw:
        breakdown = compute_average_breakdown(decode_steps, kernel_map)
        print_breakdown_table(breakdown, len(decode_steps))

        if args.csv:
            write_csv(breakdown, len(decode_steps), args.csv)

    # Summary stats
    step_durations = []
    for launch, kernels in decode_steps:
        if kernels:
            step_dur = sum(k["dur"] for k in kernels)
            step_durations.append(step_dur)

    if step_durations:
        avg_dur = sum(step_durations) / len(step_durations)
        min_dur = min(step_durations)
        max_dur = max(step_durations)
        print(f"\nDecode step duration stats ({len(step_durations)} steps):")
        print(f"  Avg: {avg_dur:.1f}μs ({avg_dur/1000:.2f}ms)")
        print(f"  Min: {min_dur:.1f}μs  Max: {max_dur:.1f}μs")
        if avg_dur > 0:
            print(f"  Range: {max_dur-min_dur:.1f}μs ({100*(max_dur-min_dur)/avg_dur:.1f}% variation)")
        else:
            print(f"  Range: {max_dur-min_dur:.1f}μs")


if __name__ == "__main__":
    main()
