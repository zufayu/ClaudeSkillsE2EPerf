#!/usr/bin/env python3
"""
Extract 10 consecutive stable layers from B200 SGLang trace.

Finds 10 consecutive FMHA-to-FMHA intervals near the average (~280μs),
then for each layer extracts all GPU kernels between:
  START: allreduce_fusion_kernel (lamport)
  END:   vectorized_elementwise_kernel (residual_add)

Outputs per-kernel table with stream, timing, and overlap annotations.

Usage:
    python3 trace_layer_detail.py <trace.json.gz>
"""

import argparse
import gzip
import json
import re
import sys
from statistics import median


def load_trace(filepath):
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rt", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    print(f"Loaded: {len(events)} events")
    return events


# Operator classification matching user's table
OPERATOR_MAP = [
    (r"allreduce_fusion_kernel.*lamport", "EP_AR+residual+RMSNorm(fused)"),
    (r"nvjet_sm100_tst.*splitK_TNT|splitK_TNT", None),  # position-dependent
    (r"splitKreduce.*bf16|splitKreduce.*bfloat|splitKreduce.*Bfloat16", None),  # position-dependent
    (r"FusedAddRMSNorm|RMSNormKernel", "q/k_norm_RMSNorm"),
    (r"_v_bz_TNN|nvjet.*_v_bz_TNN", "q_b_proj_GEMM"),
    (r"_v_bz_TNT|nvjet.*_v_bz_TNT", "uk_gemm(K_expansion)"),
    (r"CatArrayBatchedCopy", "k_concat"),
    (r"RopeQuantizeKernel", "RoPE+KV_cache_write"),
    (r"set_mla_kv_buffer", "set_mla_kv"),
    (r"fmhaSm100|fmhaKernel", "Attention(FMHA)"),
    (r"_h_bz_TNT(?!.*splitK)", "uv_gemm(V_expansion)"),
    (r"_h_bz_splitK_TNT", "o_proj_splitK_GEMM"),
    (r"DeviceGemmFp4GemmSm100|cutlass.*device_kernel.*flashinfer.*gemm", None),  # position-dependent: o_proj or shared
    (r"cvt_fp16_to_fp4", None),  # position-dependent: o_proj_quant or shared_quant
    (r"quantize_with_block_size", "MoE_input_quant(BF16→FP4)"),
    (r"routingMainKernel", "TopK_select"),
    (r"routingIndicesCluster|routingDeepSe", "expert_sort"),
    (r"bmm_E2m1.*E2m1E2m1", "gate_up_GEMM(+SwiGLU)"),
    (r"bmm_Bfloat16.*E2m1|bmm_.*E2m1.*Bfloat", "down_GEMM"),
    (r"finalizeKernelVecLoad", "MoE_finalize+residual"),
    (r"act_and_mul_kernel|silu_and_mul_kernel", "SiLU×Mul"),
    (r"unrolled_elementwise.*direct_copy|elementwise_kernel.*direct_copy", "tensor_copy"),
    (r"vectorized_elementwise_kernel.*CUDAFunctor_add|CUDAFunctorOnSelf_add", "residual_add"),
    (r"allreduce|nccl|ncclDevKernel", "nccl_comm"),
    (r"memcpy|memset", "memop"),
]


def classify_kernel(name, before_fmha, after_moe_start):
    """Classify kernel name to operator, handling position-dependent cases."""
    for pattern, label in OPERATOR_MAP:
        if not re.search(pattern, name, re.IGNORECASE):
            continue
        if label is not None:
            return label
        # Position-dependent cases
        if "splitK_TNT" in pattern:
            return "qkv_a_proj_GEMM" if before_fmha else "router_GEMM"
        if "splitKreduce" in pattern:
            return "qkv_a_splitK_reduce" if before_fmha else "router_splitK_reduce"
        if "DeviceGemmFp4" in pattern or "cutlass" in pattern:
            if before_fmha:
                return "o_proj_GEMM(pre-attn)"  # shouldn't happen
            if after_moe_start:
                return "shared_GEMM(FP4)"
            return "o_proj_GEMM"
        if "cvt_fp16_to_fp4" in pattern:
            if after_moe_start:
                return "shared_quant(BF16→FP4)"
            return "o_proj_quant(BF16→FP4)"
    return f"other:{name[:60]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    args = parser.parse_args()

    events = load_trace(args.filepath)

    # Find GPU PID (rank 0)
    gpu_pids = set()
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu_pids.add(e.get("pid"))
    gpu_pid = min(gpu_pids) if gpu_pids else None
    print(f"GPU PID: {gpu_pid} (from {len(gpu_pids)} PIDs)")

    # Get all GPU kernels for rank 0
    gpu_kernels = []
    for e in events:
        if (e.get("ph") == "X" and e.get("pid") == gpu_pid and
                e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")):
            gpu_kernels.append(e)
    gpu_kernels.sort(key=lambda x: x["ts"])
    print(f"GPU kernels: {len(gpu_kernels)}")

    # Find FMHA events
    fmha_events = [k for k in gpu_kernels if "fmhaSm100" in k.get("name", "")]
    print(f"FMHA events: {len(fmha_events)}")

    # Compute FMHA-to-FMHA intervals
    intervals = []
    for i in range(1, len(fmha_events)):
        dt = fmha_events[i]["ts"] - fmha_events[i - 1]["ts"]
        intervals.append((i - 1, dt))  # (index_of_first_fmha, interval)

    # Filter normal intervals (exclude iteration boundaries)
    med_dt = median([dt for _, dt in intervals])
    normal = [(idx, dt) for idx, dt in intervals if dt < med_dt * 3]
    avg_dt = sum(dt for _, dt in normal) / len(normal)
    print(f"Average FMHA-to-FMHA (normal): {avg_dt:.1f}μs, median: {med_dt:.1f}μs")

    # Find 10 consecutive intervals closest to average
    # Score = sum of |dt - avg| for 10 consecutive intervals
    best_score = float("inf")
    best_start = 0

    # Only search among normal intervals that are actually consecutive in FMHA index
    for scan_start in range(len(normal) - 9):
        # Check if 10 intervals are truly consecutive (FMHA indices are sequential)
        consecutive = True
        for j in range(1, 10):
            if normal[scan_start + j][0] != normal[scan_start + j - 1][0] + 1:
                consecutive = False
                break
        if not consecutive:
            continue

        score = sum(abs(normal[scan_start + j][1] - avg_dt) for j in range(10))
        if score < best_score:
            best_score = score
            best_start = scan_start

    selected_intervals = normal[best_start:best_start + 10]
    print(f"\nSelected 10 consecutive layers starting at FMHA index {selected_intervals[0][0]}")
    print(f"Score (sum |dt-avg|): {best_score:.1f}μs")

    # For each selected layer, extract kernels between allreduce_fusion and vectorized_elementwise
    RE_LAYER_START = re.compile(r"allreduce_fusion_kernel", re.IGNORECASE)
    RE_LAYER_END = re.compile(r"vectorized_elementwise_kernel", re.IGNORECASE)

    for layer_i, (fmha_idx, interval_dt) in enumerate(selected_intervals):
        fmha_start = fmha_events[fmha_idx]
        fmha_next = fmha_events[fmha_idx + 1]

        # Time window: from just before this FMHA to just before next FMHA
        # But we want allreduce_fusion as start, which comes BEFORE FMHA
        # Search backwards from FMHA to find the allreduce_fusion
        window_start = fmha_start["ts"] - 500  # look 500μs before FMHA
        window_end = fmha_next["ts"] + 100  # include a bit after next FMHA

        # Get all kernels in this window
        window_kernels = [k for k in gpu_kernels
                          if k["ts"] >= window_start and k["ts"] < window_end]

        # Find the allreduce_fusion closest before this FMHA
        layer_start_idx = None
        for ki in range(len(window_kernels) - 1, -1, -1):
            if window_kernels[ki]["ts"] >= fmha_start["ts"]:
                continue
            if RE_LAYER_START.search(window_kernels[ki].get("name", "")):
                layer_start_idx = ki
                break

        if layer_start_idx is None:
            # Fallback: use first kernel in window
            layer_start_idx = 0

        # Find the vectorized_elementwise (residual_add) after FMHA but before next FMHA
        layer_end_idx = None
        for ki in range(layer_start_idx + 1, len(window_kernels)):
            wk = window_kernels[ki]
            if wk["ts"] >= fmha_next["ts"]:
                break
            if RE_LAYER_END.search(wk.get("name", "")):
                layer_end_idx = ki

        if layer_end_idx is None:
            layer_end_idx = len(window_kernels) - 1

        # Extract layer kernels
        layer_kernels = window_kernels[layer_start_idx:layer_end_idx + 1]

        if not layer_kernels:
            print(f"\nLayer {layer_i}: NO KERNELS FOUND")
            continue

        base_ts = layer_kernels[0]["ts"]
        layer_wall_end = max(k["ts"] + k.get("dur", 0) for k in layer_kernels)
        layer_walltime = layer_wall_end - base_ts
        kernel_sum = sum(k.get("dur", 0) for k in layer_kernels)

        # Determine position flags for classification
        fmha_rel_ts = fmha_start["ts"] - base_ts
        # Detect MoE start (TopK or expert_sort)
        moe_start_ts = None
        for k in layer_kernels:
            if re.search(r"routingMainKernel|routingIndicesCluster", k.get("name", ""), re.IGNORECASE):
                moe_start_ts = k["ts"] - base_ts
                break

        # Build kernel list with classification
        kernel_rows = []
        for ki, k in enumerate(layer_kernels):
            rel_ts = k["ts"] - base_ts
            before_fmha = (k["ts"] < fmha_start["ts"])
            after_moe = (moe_start_ts is not None and rel_ts >= moe_start_ts)
            op = classify_kernel(k.get("name", ""), before_fmha, after_moe)
            kernel_rows.append({
                "idx": ki,
                "name": k.get("name", ""),
                "op": op,
                "stream": k.get("tid", 0),
                "ts": rel_ts,
                "dur": k.get("dur", 0),
                "end": rel_ts + k.get("dur", 0),
            })

        # Detect overlaps
        for i in range(len(kernel_rows)):
            overlaps = []
            for j in range(len(kernel_rows)):
                if i == j:
                    continue
                # Check if i and j overlap in time
                ov_start = max(kernel_rows[i]["ts"], kernel_rows[j]["ts"])
                ov_end = min(kernel_rows[i]["end"], kernel_rows[j]["end"])
                if ov_end > ov_start + 0.1:  # >0.1μs overlap
                    ov_us = ov_end - ov_start
                    same_stream = kernel_rows[i]["stream"] == kernel_rows[j]["stream"]
                    overlaps.append({
                        "with": j + 1,  # 1-based
                        "us": ov_us,
                        "same_stream": same_stream,
                    })
            kernel_rows[i]["overlaps"] = overlaps

        # Print layer table
        print(f"\n{'='*160}")
        print(f"Layer {layer_i}: FMHA-to-FMHA={interval_dt:.1f}μs | walltime={layer_walltime:.1f}μs | kernel_sum={kernel_sum:.1f}μs | overlap={kernel_sum-layer_walltime:.1f}μs | kernels={len(layer_kernels)}")
        print(f"{'='*160}")
        print(f"{'#':>3} {'Operator':<32} {'Stream':>6} {'Start':>8} {'Dur':>7} {'End':>8} {'Overlap_with':<50}")
        print(f"{'-'*160}")

        for kr in kernel_rows:
            # Format overlap column
            if kr["overlaps"]:
                ov_parts = []
                for ov in kr["overlaps"]:
                    tag = "same" if ov["same_stream"] else "cross"
                    ov_parts.append(f"#{ov['with']}({ov['us']:.1f}μs,{tag})")
                ov_str = " ".join(ov_parts)
            else:
                ov_str = ""

            print(f"{kr['idx']+1:>3} {kr['op']:<32} {kr['stream']:>6} {kr['ts']:>8.1f} {kr['dur']:>7.1f} {kr['end']:>8.1f} {ov_str:<50}")

        # Module summary
        print(f"\n  Module summary:")
        module_map = {}
        for kr in kernel_rows:
            op = kr["op"]
            if op not in module_map:
                module_map[op] = {"count": 0, "sum_us": 0}
            module_map[op]["count"] += 1
            module_map[op]["sum_us"] += kr["dur"]

        for op, stats in module_map.items():
            print(f"    {op:<35} ×{stats['count']}  {stats['sum_us']:>8.1f}μs")
        print(f"    {'TOTAL':<35}      {kernel_sum:>8.1f}μs (wall: {layer_walltime:.1f}μs)")

    # Average across 10 layers
    print(f"\n{'='*160}")
    print(f"10-LAYER SUMMARY")
    print(f"{'='*160}")
    all_dts = [dt for _, dt in selected_intervals]
    print(f"  FMHA-to-FMHA avg: {sum(all_dts)/len(all_dts):.1f}μs")
    print(f"  FMHA-to-FMHA range: {min(all_dts):.1f} - {max(all_dts):.1f}μs")
    print(f"  × 61 layers = {sum(all_dts)/len(all_dts)*61/1000:.2f} ms estimated decode")


if __name__ == "__main__":
    main()
