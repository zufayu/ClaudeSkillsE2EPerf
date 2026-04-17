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
import csv
import gzip
import json
import os
import re
import sys
from collections import Counter
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

# Module classification for each operator
MODULE_MAP = {
    "EP_AR+residual+RMSNorm(fused)": "EP_AR",
    "qkv_a_proj_GEMM": "Attention",
    "qkv_a_splitK_reduce": "Attention",
    "q/k_norm_RMSNorm": "Attention",
    "q_b_proj_GEMM": "Attention",
    "uk_gemm(K_expansion)": "Attention",
    "k_concat": "Attention",
    "RoPE+KV_cache_write": "Attention",
    "set_mla_kv": "Attention",
    "Attention(FMHA)": "Attention",
    "uv_gemm(V_expansion)": "Attention",
    "o_proj_quant(BF16→FP4)": "Proj",
    "o_proj_GEMM": "Proj",
    "o_proj_splitK_GEMM": "Proj",
    "router_GEMM": "MoE_Route",
    "router_splitK_reduce": "MoE_Route",
    "MoE_input_quant(BF16→FP4)": "MoE_Route",
    "tensor_copy": "MoE_Route",
    "TopK_select": "MoE_Route",
    "expert_sort": "MoE_Route",
    "SiLU×Mul": "Shared_Exp",
    "shared_quant(BF16→FP4)": "Shared_Exp",
    "shared_GEMM(FP4)": "Shared_Exp",
    "gate_up_GEMM(+SwiGLU)": "MoE_Expert",
    "down_GEMM": "MoE_Expert",
    "MoE_finalize+residual": "MoE_Expert",
    "residual_add": "Residual",
}


def classify_kernel(name, before_fmha, after_moe_start, after_first_ep_ar):
    """Classify kernel name to operator, handling position-dependent cases.

    Position flags:
        before_fmha: kernel is before FMHA in the layer
        after_moe_start: kernel is after TopK/expert_sort (MoE routing started)
        after_first_ep_ar: kernel is after the first EP_AR that follows FMHA
            (= after o_proj, in the shared expert / MoE region)
    """
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
            if after_first_ep_ar:
                return "shared_GEMM(FP4)"
            return "o_proj_GEMM"
        if "cvt_fp16_to_fp4" in pattern:
            if after_moe_start:
                return "shared_quant(BF16→FP4)"
            if after_first_ep_ar:
                return "shared_quant(BF16→FP4)"
            return "o_proj_quant(BF16→FP4)"
    return f"other:{name[:60]}"


def get_raw_short(name):
    """Extract a short raw kernel name."""
    # Remove common prefixes
    for prefix in ("void ", "sm100_", "sm90_"):
        name = name.replace(prefix, "")
    # Truncate at template params if very long
    if len(name) > 80:
        idx = name.find("<")
        if idx > 0:
            name = name[:idx] + "<...>"
    return name[:80]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    parser.add_argument("--output-dir", default=None, help="Directory to write CSV output")
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
    # fix_torch_trace_pro.py moves PDL-overlapping kernels to _hack/_overlap streams
    # for Perfetto visualization. These are NOT duplicates — each kernel exists on
    # exactly one stream (either original or _hack). We include all of them but
    # normalize the stream ID back to the original stream for classification.
    gpu_kernels = []
    hack_count = 0
    for e in events:
        if (e.get("ph") == "X" and e.get("pid") == gpu_pid and
                e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")):
            tid = str(e.get("tid", ""))
            # Normalize _hack/_overlap stream back to original stream ID
            if "_hack" in tid or "_overlap" in tid:
                hack_count += 1
                # Extract original stream: "23_hack" → 23, "stream 23 23_overlap" → 23
                original = tid.split("_")[0].split()[-1]
                try:
                    e["tid"] = int(original)
                except ValueError:
                    e["tid"] = original
                e["_was_hack"] = True  # mark for overlap detection
            gpu_kernels.append(e)
    gpu_kernels.sort(key=lambda x: x["ts"])
    print(f"GPU kernels: {len(gpu_kernels)} ({hack_count} from _hack/_overlap streams, normalized)")

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

    all_layer_kernels = []  # collect per-kernel data for averaging

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

        # Detect first EP_AR after FMHA (boundary between o_proj and shared expert)
        # On stream 23: FMHA → uv → o_proj_quant → o_proj_GEMM → [EP_AR] → shared expert
        first_ep_ar_after_fmha_ts = None
        for k in layer_kernels:
            if k["ts"] > fmha_start["ts"] and re.search(r"allreduce_fusion_kernel.*lamport", k.get("name", ""), re.IGNORECASE):
                first_ep_ar_after_fmha_ts = k["ts"] - base_ts
                break

        # Build kernel list with classification
        kernel_rows = []
        for ki, k in enumerate(layer_kernels):
            rel_ts = k["ts"] - base_ts
            before_fmha = (k["ts"] < fmha_start["ts"])
            after_moe = (moe_start_ts is not None and rel_ts >= moe_start_ts)
            after_first_ep_ar = (first_ep_ar_after_fmha_ts is not None and rel_ts > first_ep_ar_after_fmha_ts)
            raw_name = k.get("name", "")
            op = classify_kernel(raw_name, before_fmha, after_moe, after_first_ep_ar)
            module = MODULE_MAP.get(op, "Other")
            kernel_rows.append({
                "idx": ki,
                "name": raw_name,
                "raw_short": get_raw_short(raw_name),
                "op": op,
                "module": module,
                "stream": k.get("tid", 0),
                "ts": rel_ts,
                "dur": k.get("dur", 0),
                "end": rel_ts + k.get("dur", 0),
            })

        # Detect overlaps with type classification
        # Types:
        #   pdl:            same-stream overlap (<5μs) — PDL tail/preamble overlap
        #   same_large:     same-stream overlap (>=5μs) — unexpected, likely data issue
        #   cross_parallel: cross-stream overlap (>=3μs) — true dual-stream parallelism
        #   cross_tail:     cross-stream overlap (<3μs) — just edge/boundary overlap
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
                    if same_stream:
                        ov_type = "pdl" if ov_us < 5.0 else "same_large"
                    else:
                        ov_type = "cross_parallel" if ov_us >= 3.0 else "cross_tail"
                    overlaps.append({
                        "with": j + 1,  # 1-based
                        "us": ov_us,
                        "same_stream": same_stream,
                        "type": ov_type,
                    })
            kernel_rows[i]["overlaps"] = overlaps

        # Assign op_key: operator name + occurrence (e.g. "EP_AR+...#1", "EP_AR+...#2")
        op_count = {}
        for kr in kernel_rows:
            op = kr["op"]
            op_count[op] = op_count.get(op, 0) + 1
            kr["op_key"] = f"{op}#{op_count[op]}"

        # Build overlap descriptions using op_key of the overlapping kernel
        for kr in kernel_rows:
            ov_desc_parts = []
            if kr["overlaps"]:
                for ov in kr["overlaps"]:
                    j_idx = ov["with"] - 1  # 0-based
                    j_kr = kernel_rows[j_idx]
                    ov_desc_parts.append(f"{j_kr['op']}({ov['us']:.1f}μs,{ov['type']})")
            kr["ov_desc"] = " | ".join(ov_desc_parts) if ov_desc_parts else ""

        # Print layer table
        print(f"\n{'='*260}")
        print(f"Layer {layer_i}: FMHA-to-FMHA={interval_dt:.1f}μs | walltime={layer_walltime:.1f}μs | kernel_sum={kernel_sum:.1f}μs | overlap={kernel_sum-layer_walltime:.1f}μs | kernels={len(layer_kernels)}")
        print(f"{'='*260}")
        print(f"{'#':>3} {'Module':<12} {'Operator':<32} {'Raw_Kernel':<80} {'Str':>4} {'Start':>8} {'Dur':>7} {'End':>8}  {'Overlap_with':<80}")
        print(f"{'-'*260}")

        for kr in kernel_rows:
            print(f"{kr['idx']+1:>3} {kr['module']:<12} {kr['op']:<32} {kr['raw_short']:<80} {kr['stream']:>4} {kr['ts']:>8.1f} {kr['dur']:>7.1f} {kr['end']:>8.1f}  {kr['ov_desc']:<80}")

        # Collect per-kernel data for averaging, keyed by op_key
        all_layer_kernels.append(kernel_rows)

    # Average across 10 layers — per-kernel table keyed by op_key
    print(f"\n{'='*260}")
    print(f"10-LAYER AVERAGED PER-KERNEL TABLE (ordered by timestamp)")
    print(f"{'='*260}")
    all_dts = [dt for _, dt in selected_intervals]
    all_walls = []
    all_ksums = []

    if all_layer_kernels:
        n_layers = len(all_layer_kernels)

        # Build canonical op_key order from layer 0
        ref_keys = [kr["op_key"] for kr in all_layer_kernels[0]]

        # Collect per op_key: durations, overlap descriptions, and overlap totals
        op_data = {}
        for op_key in ref_keys:
            op_data[op_key] = {"durs": [], "ov_descs": [], "ov_totals": []}

        for lk in all_layer_kernels:
            lk_by_key = {kr["op_key"]: kr for kr in lk}
            for op_key in ref_keys:
                if op_key in lk_by_key:
                    kr = lk_by_key[op_key]
                    op_data[op_key]["durs"].append(kr["dur"])
                    op_data[op_key]["ov_descs"].append(kr.get("ov_desc", ""))
                    # Only count cross_parallel overlap for B200_Overlap_us
                    # PDL and cross_tail are too small / structural to report as "overlap"
                    op_data[op_key]["ov_totals"].append(
                        sum(ov["us"] for ov in kr.get("overlaps", []) if ov.get("type") == "cross_parallel")
                    )

        # Determine most common overlap for each op_key
        def most_common_ov(descs):
            """Get overlap partners that appear in >50% of layers (names only)."""
            partner_counts = Counter()
            for d in descs:
                if not d:
                    continue
                for part in d.split(" | "):
                    paren = part.find("(")
                    if paren > 0:
                        partner = part[:paren]
                        # Extract type tag: pdl, cross_parallel, cross_tail, same_large
                        if "cross_parallel" in part:
                            tag = "cross"
                        elif "cross_tail" in part:
                            tag = "cross_tail"
                        elif "pdl" in part:
                            tag = "pdl"
                        else:
                            tag = "same"
                        partner_counts[(partner, tag)] += 1
            result = []
            for (partner, tag), cnt in partner_counts.most_common():
                if cnt >= len(descs) // 2:
                    result.append(f"{partner}({tag})")
            return " | ".join(result) if result else ""

        def most_common_ov_with_us(descs):
            """Get overlap partners with avg μs, appearing in >50% of layers.
            Only includes cross_parallel overlaps in output."""
            from collections import defaultdict
            partner_us = defaultdict(list)
            for d in descs:
                if not d:
                    continue
                for part in d.split(" | "):
                    # New format: op_name(1.2μs,cross_parallel)
                    m = re.match(r'(.+?)\(([\d.]+)μs,(\w+)\)', part.strip())
                    if m:
                        key = (m.group(1), m.group(3))
                        partner_us[key].append(float(m.group(2)))
            result = []
            for (partner, tag), vals in sorted(partner_us.items(), key=lambda x: -sum(x[1]) / len(x[1])):
                if len(vals) >= len(descs) // 2:
                    avg_us = sum(vals) / len(vals)
                    result.append(f"{partner}({tag}):{avg_us:.1f}μs")
            return " | ".join(result) if result else ""

        print(f"{'#':>3} {'Module':<12} {'Operator':<32} {'Raw_Kernel':<80} {'Str':>4} {'Avg':>7} {'Min':>6} {'Max':>6}  {'Overlap_with':<60}")
        print(f"{'-'*260}")

        total_avg_dur = 0
        for ki, op_key in enumerate(ref_keys):
            durs = op_data[op_key]["durs"]
            avg_dur = sum(durs) / len(durs)
            min_dur = min(durs)
            max_dur = max(durs)
            total_avg_dur += avg_dur

            kr0 = all_layer_kernels[0][ki]
            ov_common = most_common_ov(op_data[op_key]["ov_descs"])
            print(f"{ki+1:>3} {kr0['module']:<12} {kr0['op']:<32} {kr0['raw_short']:<80} {kr0['stream']:>4} {avg_dur:>7.1f} {min_dur:>6.1f} {max_dur:>6.1f}  {ov_common:<60}")

        # Compute wall/sum stats per layer
        for lk in all_layer_kernels:
            ksum = sum(k["dur"] for k in lk)
            wall_end = max(k["end"] for k in lk)
            wall_start = min(k["ts"] for k in lk)
            all_ksums.append(ksum)
            all_walls.append(wall_end - wall_start)

        avg_ksum = sum(all_ksums) / len(all_ksums)
        avg_wall = sum(all_walls) / len(all_walls)
        avg_overlap = avg_ksum - avg_wall

        print(f"{'-'*260}")
        print(f"    {'':12} {'TOTAL':<32} {'':80} {'':>4} {avg_ksum:>7.1f}")
        print(f"\n  Kernel_sum avg: {avg_ksum:.1f}μs | Walltime avg: {avg_wall:.1f}μs | Overlap avg: {avg_overlap:.1f}μs")
        print(f"  FMHA-to-FMHA avg: {sum(all_dts)/len(all_dts):.1f}μs | range: {min(all_dts):.1f} - {max(all_dts):.1f}μs")
        print(f"  × 61 layers: kernel_sum={avg_ksum*61/1000:.2f}ms | walltime={avg_wall*61/1000:.2f}ms")

        # Module subtotals
        print(f"\n  Module subtotals (avg):")
        mod_sums = {}
        for ki, op_key in enumerate(ref_keys):
            avg_d = sum(op_data[op_key]["durs"]) / len(op_data[op_key]["durs"])
            mod = all_layer_kernels[0][ki]["module"]
            mod_sums[mod] = mod_sums.get(mod, 0) + avg_d
        for mod, total in mod_sums.items():
            pct = total / avg_ksum * 100
            print(f"    {mod:<15} {total:>8.1f}μs  ({pct:>5.1f}%)")

        # Write CSV output — kernel map format with overlap and MI355X placeholder columns
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            csv_path = os.path.join(args.output_dir, "layer_kernel_avg.csv")

            # Pass-level grouping: map each operator to a pass
            PASS_MAP = {
                "EP_AR+residual+RMSNorm(fused)": None,  # assigned by occurrence #
                "qkv_a_proj_GEMM": "MHA",
                "qkv_a_splitK_reduce": "MHA",
                "q/k_norm_RMSNorm": "MHA",
                "q_b_proj_GEMM": "MHA",
                "uk_gemm(K_expansion)": "MHA",
                "k_concat": "MHA",
                "RoPE+KV_cache_write": "MHA",
                "set_mla_kv": "MHA",
                "Attention(FMHA)": "MHA",
                "uv_gemm(V_expansion)": "MHA",
                "o_proj_quant(BF16→FP4)": "O_proj",
                "o_proj_GEMM": "O_proj",
                "o_proj_splitK_GEMM": "O_proj",
                "router_GEMM": "MOE",
                "router_splitK_reduce": "MOE",
                "Moe_Expert_quant(BF16→FP4)": "MOE",
                "MoE_input_quant(BF16→FP4)": "MOE",
                "tensor_copy": "MOE",
                "TopK_select": "MOE",
                "expert_sort": "MOE",
                "SiLU×Mul": "MOE",
                "shared_quant(BF16→FP4)": "MOE",
                "shared_GEMM(FP4)": "MOE",
                "gate_up_GEMM(+SwiGLU)": "MOE",
                "down_GEMM": "MOE",
                "MoE_finalize+residual": "MOE",
                "residual_add": "MOE",
            }

            # Collect rows and compute pass sums
            pass_sums = {}
            csv_rows = []

            for ki, op_key in enumerate(ref_keys):
                durs = op_data[op_key]["durs"]
                avg_dur = sum(durs) / len(durs)
                ov_tots = op_data[op_key]["ov_totals"]
                avg_ov = sum(ov_tots) / len(ov_tots) if ov_tots else 0
                kr0 = all_layer_kernels[0][ki]
                ov_with = most_common_ov_with_us(op_data[op_key]["ov_descs"])

                # Determine pass
                op = kr0["op"]
                pass_name = PASS_MAP.get(op, "MOE")
                if pass_name is None:
                    # EP_AR: #1 → EP_AR_before_MHA, #2 → EP_AR_before_MOE
                    if op_key.endswith("#1"):
                        pass_name = "EP_AR_before_MHA"
                    else:
                        pass_name = "EP_AR_before_MOE"
                # o_proj_GEMM #2 goes to O_proj pass
                if op == "o_proj_GEMM" or op == "o_proj_quant(BF16→FP4)":
                    pass_name = "O_proj"

                pass_sums[pass_name] = pass_sums.get(pass_name, 0) + avg_dur

                csv_rows.append([
                    ki + 1,
                    kr0["module"],
                    kr0["op"],
                    kr0["raw_short"],
                    kr0["stream"],
                    f"{avg_dur:.1f}",
                    f"{avg_ov:.1f}" if avg_ov > 0.05 else "0",
                    ov_with,
                    "",  # MI355X_Module placeholder
                    "",  # MI355X_Kernel placeholder
                    "",  # MI355X_Avg_us placeholder
                    "",  # Notes placeholder
                ])

            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["#", "B200_Module", "B200_Operator", "B200_Raw_Kernel", "B200_Stream",
                             "B200_Avg_us", "B200_Overlap_us", "B200_Overlap_With",
                             "MI355X_Module", "MI355X_Kernel", "MI355X_Avg_us", "Notes"])
                for row in csv_rows:
                    w.writerow(row)
                w.writerow([])
                w.writerow(["", "", "B200 TOTAL (kernel_sum)", "", "", f"{avg_ksum:.1f}", "", "", "", "MI355X TOTAL", "", ""])
                w.writerow(["", "", "B200 Walltime", "", "", f"{avg_wall:.1f}", "", "", "", "", "", ""])
                w.writerow(["", "", "B200 Overlap", "", "", f"{avg_overlap:.1f}", "", "", "", "", "", ""])
                w.writerow([])
                # Pass-level summary
                w.writerow(["", "", "PASS", "", "B200", "MI355X", "gap", "", "NV_Kernels", "AMD_Kernels", "", ""])
                pass_order = ["MOE", "MHA", "O_proj", "EP_AR_before_MHA", "EP_AR_before_MOE"]
                for pname in pass_order:
                    b200_val = pass_sums.get(pname, 0)
                    w.writerow(["", "", pname, "", f"{b200_val:.1f}", "", "", "", "", "", "", ""])
            print(f"\n  CSV written: {csv_path}")

    # === allreduce_fusion_kernel distribution analysis ===
    print(f"\n{'='*120}")
    print(f"ALLREDUCE_FUSION_KERNEL DISTRIBUTION ANALYSIS")
    print(f"{'='*120}")

    ar_events = [k for k in gpu_kernels if re.search(r"allreduce_fusion_kernel", k.get("name", ""), re.IGNORECASE)]
    print(f"Total allreduce_fusion_kernel events: {len(ar_events)}")

    if ar_events:
        ar_durs = [k["dur"] for k in ar_events]
        ar_avg = sum(ar_durs) / len(ar_durs)
        ar_med = median(ar_durs)
        ar_min = min(ar_durs)
        ar_max = max(ar_durs)
        print(f"  avg={ar_avg:.1f}μs  median={ar_med:.1f}μs  min={ar_min:.1f}μs  max={ar_max:.1f}μs")

        # Histogram with 2μs bins
        bins = {}
        for d in ar_durs:
            b = int(d // 2) * 2
            bins[b] = bins.get(b, 0) + 1
        print(f"\n  Duration histogram (2μs bins):")
        for b in sorted(bins.keys()):
            bar = "#" * min(bins[b], 80)
            print(f"    {b:>5}-{b+2:<5}μs: {bins[b]:>5}  {bar}")

        # Check alternating pattern: look at consecutive pairs
        # Each layer has 2 allreduce_fusion calls (#1 and #14 in the 29-kernel sequence)
        # #1 is the "big" one (start of layer, overlaps with qkv_a_proj)
        # #14 is the "small" one (between o_proj and MoE, overlaps with router_GEMM)
        # Classify: <15μs = "small", >=15μs = "big"
        threshold = 15.0
        small_durs = [d for d in ar_durs if d < threshold]
        big_durs = [d for d in ar_durs if d >= threshold]
        print(f"\n  Bimodal split at {threshold}μs:")
        print(f"    Small (<{threshold}μs): n={len(small_durs)}, avg={sum(small_durs)/len(small_durs):.1f}μs" if small_durs else f"    Small: n=0")
        print(f"    Big   (>={threshold}μs): n={len(big_durs)}, avg={sum(big_durs)/len(big_durs):.1f}μs" if big_durs else f"    Big: n=0")

        # Show pattern of consecutive events (first 40)
        print(f"\n  Consecutive pattern (first 60 events, S=small B=big):")
        pattern_str = ""
        for i, d in enumerate(ar_durs[:60]):
            pattern_str += "B" if d >= threshold else "S"
            if (i + 1) % 20 == 0:
                print(f"    [{i-19:>4}-{i:>4}]: {pattern_str}")
                pattern_str = ""
        if pattern_str:
            start_i = (len(ar_durs[:60]) // 20) * 20
            print(f"    [{start_i:>4}-{start_i+len(pattern_str)-1:>4}]: {pattern_str}")

        # Odd vs even index analysis (each layer has 2 allreduce_fusion calls)
        # Separate stable-state events (exclude prefill outliers >100μs)
        stable_ar = [d for d in ar_durs if d < 100]
        odd_durs = [stable_ar[i] for i in range(0, len(stable_ar), 2)]  # 1st occurrence per layer
        even_durs = [stable_ar[i] for i in range(1, len(stable_ar), 2)]  # 2nd occurrence per layer
        print(f"\n  Odd/Even split (stable <100μs, n={len(stable_ar)}):")
        if odd_durs:
            print(f"    Odd  (1st per layer, #1):  n={len(odd_durs):>5}  avg={sum(odd_durs)/len(odd_durs):>6.1f}μs  median={median(odd_durs):>6.1f}μs  min={min(odd_durs):>5.1f}  max={max(odd_durs):>5.1f}")
        if even_durs:
            print(f"    Even (2nd per layer, #14): n={len(even_durs):>5}  avg={sum(even_durs)/len(even_durs):>6.1f}μs  median={median(even_durs):>6.1f}μs  min={min(even_durs):>5.1f}  max={max(even_durs):>5.1f}")

        # Show consecutive pairs (1st,2nd) for first 20 pairs in stable state
        print(f"\n  First 20 consecutive pairs (1st, 2nd):")
        for i in range(0, min(40, len(stable_ar)), 2):
            d1 = stable_ar[i]
            d2 = stable_ar[i + 1] if i + 1 < len(stable_ar) else 0
            print(f"    pair {i//2:>3}: 1st={d1:>6.1f}μs  2nd={d2:>6.1f}μs")

    if args.output_dir and ar_events:
        csv_ar = os.path.join(args.output_dir, "allreduce_fusion_dist.csv")
        with open(csv_ar, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "timestamp_us", "duration_us", "type"])
            for i, k in enumerate(ar_events):
                tag = "big" if k["dur"] >= threshold else "small"
                w.writerow([i, f"{k['ts']:.1f}", f"{k['dur']:.1f}", tag])
        print(f"\n  Allreduce CSV written: {csv_ar}")


if __name__ == "__main__":
    main()
