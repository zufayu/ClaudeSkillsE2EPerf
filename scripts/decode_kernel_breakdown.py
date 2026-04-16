#!/usr/bin/env python3
"""
Extract per-layer kernel breakdown from ATOM trace (ROCm 7.2.2+ compatible).

Replaces parse_trace.py's parse_decode() which requires norm modules.
Uses decode[bs=N] markers and reduce_scatter as layer boundaries.
Output: xlsx matching old parse_trace.py format (2 sheets, module-grouped).

Usage:
    python3 scripts/decode_kernel_breakdown.py <trace.json.gz> \
        --target-bs 64 --layers 10-40
"""

import argparse
import collections
import gzip
import json
import os
import re
import sys


# ============================================================================
# Layer structure for DeepSeek-R1 (per transformer layer, execution order):
#
#   input_layernorm:    reduce_scatter + local_device_load_rmsnorm
#   per_token_quant:    dynamic_per_token_scaled_quant  (NEW in rocm722)
#   gemm_a8w8:          gemm_xdl (qkv_a proj)
#   q_k_norm:           fused_qk_rmsnorm_group_quant   (NEW: fused norm+quant)
#   q_proj_and_k_up:    FlatmmKernel/Cijk (q_b) + batched_gemm (k_up)
#   rope_and_kv_cache:  fuse_qk_rope_concat
#   mla_decode:         mla_a8w8 + kn_mla_reduce
#   v_up_proj_o_proj:   batched_gemm (v_up) + gemm_xdl (o_proj)
#   post_attn_layernorm: reduce_scatter + local_device_load_rmsnorm
#   gemm_a16w16:        bf16gemm (router)
#   mxfp4_moe:          grouped_topk + MoeSorting + quant + moe_mxgemm ×2
# ============================================================================

# Positional classifier: maps (kernel_pattern, occurrence_index) → module
# For kernels that appear multiple times per layer, the index disambiguates.
KERNEL_MODULE_RULES = [
    # (pattern, module_for_1st_occurrence, module_for_2nd_occurrence, ...)
    ("reduce_scatter_cross_device", ["input_layernorm", "post_attn_layernorm"]),
    ("local_device_load_rmsnorm", ["input_layernorm", "post_attn_layernorm"]),
    ("dynamic_per_token_scaled_quant", ["per_token_quant_hip"]),
    ("add_rmsnorm_quant", ["hipLaunchKernel"]),
    ("fused_qk_rmsnorm_group_quant", ["q_proj_and_k_up_proj"]),
    ("gemm_xdl_cshuffle_v3_multi_d_b_preshuffle", ["gemm_a8w8_bpreshuffle", "v_up_proj_and_o_proj"]),
    ("FlatmmKernel", ["q_proj_and_k_up_proj"]),
    ("Cijk_", ["q_proj_and_k_up_proj", "v_up_proj_and_o_proj"]),
    ("batched_gemm_a8w8", ["q_proj_and_k_up_proj", "v_up_proj_and_o_proj"]),
    ("bf16gemm", ["gemm_a16w16"]),
    ("fuse_qk_rope_concat", ["rope_and_kv_cache"]),
    ("mla_a8w8", ["mla_decode"]),
    ("kn_mla_reduce", ["mla_decode"]),
    ("triton_poi_fused", ["triton_poi"]),
    ("grouped_topk_opt_sort", ["mxfp4_moe"]),
    ("MoeSortingMultiPhase", ["mxfp4_moe"]),
    ("mxfp4_quant_moe_sort", ["mxfp4_moe"]),
    ("kernel_moe_mxgemm", ["mxfp4_moe"]),
]


def classify_kernel_positional(kernel_name, occurrence_counters):
    """Classify kernel by name pattern + occurrence count within the layer."""
    for pattern, modules in KERNEL_MODULE_RULES:
        if pattern in kernel_name:
            idx = occurrence_counters.get(pattern, 0)
            occurrence_counters[pattern] = idx + 1
            return modules[min(idx, len(modules) - 1)]
    return "other"


def load_trace(path):
    print(f"Loading {path} ({os.path.getsize(path)/1e6:.0f}MB)...")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    evts = data.get("traceEvents", [])
    print(f"  {len(evts)} events")
    return evts


def select_decode(evts, target_bs, skip_ratio=0.5):
    decodes = [
        e for e in evts
        if e.get("name", "").startswith("decode[")
        and e.get("ph") == "X"
        and f"bs={target_bs}" in e.get("name", "")
    ]
    if not decodes:
        print(f"ERROR: no decode events with bs={target_bs}")
        return None
    idx = int(len(decodes) * skip_ratio)
    idx = min(idx, len(decodes) - 1)
    d = decodes[idx]
    dur_ms = d.get("dur", 0) / 1000
    print(f"  decode events (bs={target_bs}): {len(decodes)}")
    print(f"  Selected #{idx}/{len(decodes)} (skip_ratio={skip_ratio})")
    print(f"  {d['name']} ts={d['ts']:.0f} dur={dur_ms:.2f}ms")
    return d


def extract_kernels_in_window(evts, ts_start, ts_end):
    return sorted(
        [e for e in evts if e.get("cat") == "kernel" and e.get("ph") == "X"
         and e["ts"] >= ts_start and e["ts"] <= ts_end],
        key=lambda e: e["ts"]
    )


def split_layers(kernels):
    """Split kernels into layers using reduce_scatter as boundary.

    Auto-detects whether the first reduce_scatter is pre-attn or post-attn
    by checking what follows (router/topk → post-attn, quant/gemm → pre-attn).
    """
    # Find all reduce_scatter positions
    rs_positions = [i for i, k in enumerate(kernels)
                    if "reduce_scatter_cross_device" in k.get("name", "")]

    if len(rs_positions) < 2:
        return [kernels]

    # Determine if first RS is pre-attn or post-attn by looking at next non-RS kernel
    first_rs = rs_positions[0]
    # Look at kernels after first RS+rmsnorm pair (skip RS and rmsnorm)
    look_ahead = min(first_rs + 3, len(kernels) - 1)
    next_kernel = kernels[look_ahead].get("name", "")
    # If followed by router (bf16gemm) or topk → first RS is post-attn (MoE phase)
    first_is_post_attn = any(p in next_kernel for p in ["bf16gemm", "grouped_topk", "moe"])
    # Pre-attn RS index parity: if first is post-attn, pre-attn starts at even; else odd
    pre_attn_parity = 0 if first_is_post_attn else 1  # 0=even, 1=odd

    layers = []
    current = []
    rs_count = 0
    for k in kernels:
        if "reduce_scatter_cross_device" in k.get("name", ""):
            rs_count += 1
            # Pre-attn RS = layer start
            if rs_count % 2 == pre_attn_parity and current:
                layers.append(current)
                current = []
        current.append(k)
    if current:
        layers.append(current)
    return layers


def classify_layer(layer_kernels):
    """Classify each kernel in a layer using positional rules. Returns [(module, kernel_name, dur_us)]."""
    counters = {}
    result = []
    for k in layer_kernels:
        name = k.get("name", "unknown")
        module = classify_kernel_positional(name, counters)
        result.append((module, name, k.get("dur", 0)))
    return result


def main():
    parser = argparse.ArgumentParser(description="Decode kernel breakdown from ATOM trace")
    parser.add_argument("trace", help="Path to trace JSON or JSON.GZ")
    parser.add_argument("--target-bs", type=int, default=64)
    parser.add_argument("--skip-ratio", type=float, default=0.5)
    parser.add_argument("--layers", type=str, default="10-40")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        m = re.search(r'_(c\d+)[_.]', os.path.basename(args.trace))
        suffix = m.group(1) if m else f"c{args.target_bs}"
        args.output = f"decode_breakdown_{suffix}.xlsx"

    layer_start, layer_end = map(int, args.layers.split("-"))
    evts = load_trace(args.trace)

    decode = select_decode(evts, args.target_bs, args.skip_ratio)
    if decode is None:
        sys.exit(1)

    ts0, dur = decode["ts"], decode.get("dur", 0)
    dur_ms = dur / 1000
    print(f"\n  Decode walltime: {dur_ms:.2f}ms")

    kernels = extract_kernels_in_window(evts, ts0, ts0 + dur)
    print(f"  Kernels in window: {len(kernels)}")

    layers = split_layers(kernels)
    print(f"  Layers detected: {len(layers)}")

    if len(layers) <= layer_end:
        layer_end = len(layers) - 1
        print(f"  Adjusted range: {layer_start}-{layer_end}")

    selected = layers[layer_start:layer_end + 1]
    n_layers = len(selected)

    # Classify each layer and compute per-kernel averages
    # Use representative layer (middle of selected range) for single-layer column
    rep_idx = n_layers // 2
    rep_layer = classify_layer(selected[rep_idx])

    # Compute multi-layer averages: aggregate by (position, kernel_name)
    all_classified = [classify_layer(l) for l in selected]
    # Average duration per position across layers
    max_pos = max(len(cl) for cl in all_classified)
    avg_durs = []
    for pos in range(max_pos):
        durs = [cl[pos][2] for cl in all_classified if pos < len(cl)]
        avg_durs.append(sum(durs) / len(durs) if durs else 0)

    total_rep = sum(d for _, _, d in rep_layer)
    total_avg = sum(avg_durs[:len(rep_layer)])

    print(f"\n{'='*60}")
    print(f"LAYER {layer_start}-{layer_end} ANALYSIS ({n_layers} layers)")
    print(f"{'='*60}")
    print(f"  Representative layer: {layer_start + rep_idx} ({len(rep_layer)} kernels)")
    print(f"  Single-layer total: {total_rep:.1f}us ({total_rep/1000:.2f}ms)")
    print(f"  Multi-layer avg total: {total_avg:.1f}us ({total_avg/1000:.2f}ms)")
    print(f"  Est decode (x61): {total_avg * 61 / 1000:.1f}ms (actual: {dur_ms:.2f}ms)")

    # Print layer structure
    print(f"\n{'#':<3} {'Module':<40} {'Kernel':<55} {'dur(us)':>8} {'avg':>8}")
    print("-" * 120)
    cur_mod = ""
    for i, (mod, kname, d) in enumerate(rep_layer):
        avg = avg_durs[i] if i < len(avg_durs) else 0
        show_mod = mod if mod != cur_mod else ""
        if mod != cur_mod:
            cur_mod = mod
        print(f"  {i:<3} {show_mod:<40} {kname[:53]:<55} {d:>7.1f} {avg:>7.1f}")

    # Write xlsx
    from openpyxl import Workbook
    wb = Workbook()

    # Sheet 1: "decode" — sequential, module-grouped
    ws = wb.active
    ws.title = "decode"
    ws.append(["cpu_module", "gpu_kernel", "duration_us", "pct%",
               "sum per module", "module_pct%", "avg duration_us", "avg sum per module"])

    # Group consecutive kernels by module
    groups = []
    for i, (mod, kname, d) in enumerate(rep_layer):
        avg = avg_durs[i] if i < len(avg_durs) else 0
        if not groups or groups[-1][0] != mod:
            groups.append((mod, []))
        groups[-1][1].append((kname, d, avg))

    for mod, kernel_list in groups:
        mod_sum = sum(d for _, d, _ in kernel_list)
        mod_avg_sum = sum(a for _, _, a in kernel_list)
        mod_pct = mod_sum / total_rep * 100 if total_rep else 0
        first = True
        for kname, d, avg in kernel_list:
            pct = d / total_rep * 100 if total_rep else 0
            ws.append([
                mod if first else "",
                kname,
                round(d, 3),
                round(pct, 1),
                round(mod_sum, 3) if first else "",
                round(mod_pct, 1) if first else "",
                round(avg, 3),
                round(mod_avg_sum, 3) if first else "",
            ])
            first = False

    ws.append(["TOTAL", "", round(total_rep, 3), "100", "", "100",
               round(total_avg, 3), ""])

    # Sheet 2: "kernel_summary" — aggregated by kernel name, sorted by avg duration
    ws2 = wb.create_sheet("kernel_summary")
    ws2.append(["gpu_kernel", "calls", "total_duration_us", "avg_duration_us", "pct%"])
    by_name = collections.defaultdict(lambda: {"count": 0, "dur": 0})
    for mod, kname, d in rep_layer:
        by_name[kname]["count"] += 1
        by_name[kname]["dur"] += d
    for kname, v in sorted(by_name.items(), key=lambda x: -x[1]["dur"]):
        pct = v["dur"] / total_rep * 100 if total_rep else 0
        avg_d = v["dur"] / v["count"] if v["count"] else 0
        ws2.append([kname, v["count"], round(v["dur"], 1), round(avg_d, 1), round(pct, 1)])

    print(f"\nWriting {args.output}...")
    wb.save(args.output)
    print(f"Done. {len(rep_layer)} kernel rows, {len(groups)} module groups.")


if __name__ == "__main__":
    main()
