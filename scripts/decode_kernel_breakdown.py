#!/usr/bin/env python3
"""
Extract per-layer kernel breakdown from ATOM trace.

Replaces parse_trace.py's parse_decode() which requires norm modules.
Uses decode[bs=N] markers and module user_annotations to identify layers.

Usage:
    python3 scripts/decode_kernel_breakdown.py <trace.json.gz> \
        --target-bs 64 --layers 10-40 --output decode_breakdown_c64.csv
"""

import argparse
import collections
import gzip
import json
import os
import re
import sys


def load_trace(path):
    print(f"Loading {path} ({os.path.getsize(path)/1e6:.0f}MB)...")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    evts = data.get("traceEvents", [])
    print(f"  {len(evts)} events")
    return evts


def select_decode(evts, target_bs, skip_ratio=0.5):
    """Select a steady-state decode event at given skip_ratio position."""
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
    """Get all GPU kernels within a time window."""
    kernels = [
        e for e in evts
        if e.get("cat") == "kernel"
        and e.get("ph") == "X"
        and e["ts"] >= ts_start
        and e["ts"] <= ts_end
    ]
    return sorted(kernels, key=lambda e: e["ts"])


def classify_kernel(name):
    """Map kernel name to cpu_module (matching old parse_trace.py format)."""
    patterns = [
        ("reduce_scatter_cross_device", "input_layernorm"),
        ("local_device_load_rmsnorm", "input_layernorm"),
        ("add_rmsnorm_quant", "hipLaunchKernel"),
        ("dynamic_per_token_scaled_quant", "per_token_quant_hip"),
        ("fused_qk_rmsnorm_group_quant", "q_proj_and_k_up_proj"),
        ("FlatmmKernel", "q_proj_and_k_up_proj"),
        ("Cijk_BBS", "q_proj_and_k_up_proj"),
        ("bf16gemm", "gemm_a16w16"),
        ("fuse_qk_rope_concat", "rope_and_kv_cache"),
        ("mla_a8w8", "mla_decode"),
        ("kn_mla_reduce", "mla_decode"),
        ("batched_gemm_a8w8", "v_up_proj_and_o_proj"),
        ("kernel_moe_mxgemm", "mxfp4_moe"),
        ("mxfp4_quant_moe_sort", "mxfp4_moe"),
        ("MoeSortingMultiPhase", "mxfp4_moe"),
        ("grouped_topk_opt_sort", "mxfp4_moe"),
        ("triton_poi_fused", "triton_poi"),
        ("gemm_xdl_cshuffle_v3_multi_d_b_preshuffle", "gemm_a8w8_bpreshuffle"),
    ]
    for pattern, module in patterns:
        if pattern in name:
            return module
    return "other"


def identify_layers_by_kernel(kernels):
    """Split kernels into layers using reduce_scatter as boundary.

    DeepSeek-R1 layer structure:
      reduce_scatter (pre-attn) → rmsnorm → qkv → rope → mla → o_proj →
      reduce_scatter (post-attn) → rmsnorm → router → mxfp4_moe →
      [next layer reduce_scatter]

    The first reduce_scatter in each pair (odd count) marks layer start.
    """
    layers = []
    current_layer = []
    rs_count = 0

    for k in kernels:
        name = k.get("name", "")
        if "reduce_scatter_cross_device" in name:
            rs_count += 1
            if rs_count % 2 == 1 and current_layer:  # odd = pre-attn = layer start
                layers.append(current_layer)
                current_layer = []
        current_layer.append(k)

    if current_layer:
        layers.append(current_layer)

    return layers


def main():
    parser = argparse.ArgumentParser(description="Decode kernel breakdown from ATOM trace")
    parser.add_argument("trace", help="Path to trace JSON or JSON.GZ")
    parser.add_argument("--target-bs", type=int, default=64)
    parser.add_argument("--skip-ratio", type=float, default=0.5)
    parser.add_argument("--layers", type=str, default="10-40", help="Layer range (e.g., 10-40)")
    parser.add_argument("--output", type=str, default=None, help="Output xlsx path")
    args = parser.parse_args()

    # Auto-detect output filename
    if args.output is None:
        m = re.search(r'_(c\d+)[_.]', os.path.basename(args.trace))
        suffix = m.group(1) if m else f"c{args.target_bs}"
        args.output = f"decode_breakdown_{suffix}.xlsx"

    layer_start, layer_end = map(int, args.layers.split("-"))

    evts = load_trace(args.trace)

    # Select decode event
    decode = select_decode(evts, args.target_bs, args.skip_ratio)
    if decode is None:
        sys.exit(1)

    ts0 = decode["ts"]
    dur = decode.get("dur", 0)
    ts1 = ts0 + dur
    dur_ms = dur / 1000

    # Sanity check: decode duration should be reasonable
    print(f"\n  Decode walltime: {dur_ms:.2f}ms")
    if dur_ms < 5 or dur_ms > 100:
        print(f"  WARNING: unusual decode duration {dur_ms:.2f}ms")

    # Extract kernels in decode window
    kernels = extract_kernels_in_window(evts, ts0, ts1)
    print(f"  Kernels in window: {len(kernels)}")

    # Identify layers using reduce_scatter boundaries
    layers = identify_layers_by_kernel(kernels)
    print(f"  Layers detected: {len(layers)}")

    if len(layers) < layer_end:
        print(f"  WARNING: only {len(layers)} layers, adjusting to {layer_start}-{len(layers)-1}")
        layer_end = len(layers) - 1

    # Kernel name distribution (all layers)
    all_names = collections.Counter(k.get("name", "")[:80] for k in kernels)
    print(f"\n  Kernel name distribution (top 10):")
    for name, cnt in all_names.most_common(10):
        print(f"    {cnt:4d}x {name[:70]}")

    # Per-layer analysis for selected range
    selected_layers = layers[layer_start:layer_end + 1]
    n_layers = len(selected_layers)
    print(f"\n{'='*60}")
    print(f"LAYER {layer_start}-{layer_end} ANALYSIS ({n_layers} layers)")
    print(f"{'='*60}")

    # Aggregate kernels across selected layers
    by_name = collections.defaultdict(lambda: {"count": 0, "dur": 0})
    for layer_kernels in selected_layers:
        for k in layer_kernels:
            nm = k.get("name", "unknown")
            by_name[nm]["count"] += 1
            by_name[nm]["dur"] += k.get("dur", 0)

    total_dur = sum(v["dur"] for v in by_name.values())
    per_layer_dur = total_dur / n_layers if n_layers else 0
    print(f"\nTotal kernel time (layers {layer_start}-{layer_end}): {total_dur/1000:.1f}ms")
    print(f"Per-layer average: {per_layer_dur:.1f}us ({per_layer_dur/1000:.2f}ms)")
    print(f"Estimated decode (x61): {per_layer_dur * 61 / 1000:.1f}ms (actual: {dur_ms:.2f}ms)")

    # Detailed kernel list sorted by duration
    print(f"\n{'Kernel':<75} {'Avg(us)':>8} {'Count':>6} {'%':>6}")
    print("-" * 100)
    flat = []
    for k_name, v in by_name.items():
        avg = v["dur"] / n_layers if n_layers else 0
        pct = v["dur"] / total_dur * 100 if total_dur else 0
        flat.append((k_name, avg, v["count"] / n_layers, pct))
    flat.sort(key=lambda x: -x[1])
    for k_name, avg, cnt, pct in flat[:25]:
        print(f"  {k_name[:73]:<73} {avg:>7.1f} {cnt:>5.1f}x {pct:>5.1f}%")

    # Classify kernels into modules
    classified = []
    for k_name, avg, cnt, pct in flat:
        module = classify_kernel(k_name)
        classified.append((module, k_name, avg, cnt, pct))

    # Group by module (preserve order of first appearance)
    module_order = []
    module_kernels = collections.defaultdict(list)
    for module, k_name, avg, cnt, pct in classified:
        if module not in module_order:
            module_order.append(module)
        module_kernels[module].append((k_name, avg, cnt, pct))

    # Write xlsx (matching old parse_trace.py format)
    from openpyxl import Workbook
    wb = Workbook()

    # Sheet 1: "decode" — grouped by module
    ws = wb.active
    ws.title = "decode"
    ws.append(["cpu_module", "gpu_kernel", "duration_us", "pct%",
               "sum per module", "module_pct%", "avg duration_us", "avg sum per module"])

    for module in module_order:
        kernels_list = module_kernels[module]
        mod_sum = sum(k[1] for k in kernels_list)
        mod_pct = mod_sum / per_layer_dur * 100 if per_layer_dur else 0
        first = True
        for k_name, avg, cnt, pct in kernels_list:
            row = [
                module if first else "",
                k_name,
                round(avg, 3),
                round(pct, 1),
                round(mod_sum, 3) if first else "",
                round(mod_pct, 1) if first else "",
                round(avg, 3),
                round(mod_sum, 3) if first else "",
            ]
            ws.append(row)
            first = False

    ws.append(["TOTAL", "", round(per_layer_dur, 3), "100", "", "100",
               round(per_layer_dur, 3), ""])

    # Sheet 2: "kernel_summary" — flat sorted by duration
    ws2 = wb.create_sheet("kernel_summary")
    ws2.append(["gpu_kernel", "calls", "total_duration_us", "avg_duration_us", "pct%"])
    for k_name, avg, cnt, pct in flat:
        ws2.append([k_name, round(cnt, 1), round(avg * cnt, 1), round(avg, 1), round(pct, 1)])

    print(f"\nWriting {args.output}...")
    wb.save(args.output)
    print(f"Done. {len(flat)} kernels, {len(module_order)} modules.")


if __name__ == "__main__":
    main()
