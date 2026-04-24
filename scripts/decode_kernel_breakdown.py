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

# Import shared trace utilities (R5)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_utils import load_trace_events  # noqa: E402


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
    ("gemm_xdl_cshuffle_v3_multi_d_b_preshuffle", ["gemm_a8w8_bpreshuffle", "q_proj_and_k_up_proj"]),
    ("FlatmmKernel", ["v_up_proj_and_o_proj"]),
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
    """Delegates to trace_utils.load_trace_events."""
    return load_trace_events(path)


def select_decodes(evts, target_bs, skip_warmup=5, max_steps=20):
    """Return a list of decode events to aggregate over.

    Skips the first `skip_warmup` decodes (cold caches / JIT) and caps at
    `max_steps`. Replaces the old `select_decode` (single-decode picker) so
    we can aggregate ~600 (step,layer) samples per operator instead of ~30.
    Convention matches B300/B200 trace_layer_detail.py for cross-platform
    apples-to-apples comparison.
    """
    # cat='gpu_user_annotation' filter is critical: without it we also catch
    # capture-graph replay events that look like decode[bs=N] but are sub-ms
    # (graph capture / MTP draft). With it, we only get the real decode
    # iterations (typically 10-30ms each on MI355X).
    decodes = sorted(
        [
            e for e in evts
            if e.get("name", "").startswith("decode[")
            and e.get("ph") == "X"
            and e.get("cat") == "gpu_user_annotation"
            and f"bs={target_bs}" in e.get("name", "")
        ],
        key=lambda x: x["ts"],
    )
    if not decodes:
        print(f"ERROR: no decode events with bs={target_bs} (cat=gpu_user_annotation)")
        return []
    if len(decodes) <= skip_warmup:
        print(f"WARN: only {len(decodes)} decodes — can't skip {skip_warmup} warmup; using all")
        return decodes
    selected = decodes[skip_warmup : skip_warmup + max_steps] if max_steps > 0 \
        else decodes[skip_warmup:]
    print(f"  decode events (bs={target_bs}): {len(decodes)} total")
    print(f"  Selected {len(selected)} steady-state decodes "
          f"(skipped first {skip_warmup} warmup, cap {max_steps})")
    return selected


def extract_kernels_in_window(evts, ts_start, ts_end):
    return sorted(
        [e for e in evts if e.get("cat") == "kernel" and e.get("ph") == "X"
         and e["ts"] >= ts_start and e["ts"] <= ts_end],
        key=lambda e: e["ts"]
    )


def split_layers(kernels):
    """Split kernels into layers.

    Layer structure (old trace, rocm711):
      input_layernorm(RS+norm) → qkv_a → q/k_norm → q_b → k_up → rope →
      mla → mla_reduce → v_up → o_proj → post_attn(RS+norm) → router → MoE

    Layer structure (new trace, rocm722, norms fused):
      RS+norm → per_token_quant → qkv_a → fused_qk_rmsnorm_quant → q_b → k_up →
      rope → mla → mla_reduce → v_up → o_proj → quant → RS+norm → router → MoE

    Use mla_a8w8 (MLA attention, exactly 1 per layer) as anchor.
    Layer boundary = midpoint between consecutive mla_a8w8 occurrences,
    which falls in the MoE → next pre-attn transition.
    """
    # Find all MLA attention positions
    mla_positions = [i for i, k in enumerate(kernels)
                     if "mla_a8w8" in k.get("name", "")]

    if len(mla_positions) < 2:
        return [kernels]

    # For each pair of consecutive MLA positions, find the layer boundary between them.
    # The boundary is the reduce_scatter that comes AFTER the MoE phase.
    # Strategy: search backwards from each MLA to find the nearest pre-attn RS.
    # Pre-attn RS is the one followed by quant/gemm_xdl (not router/topk).
    boundaries = []
    for mla_pos in mla_positions[1:]:
        # Search backwards from MLA for the nearest reduce_scatter
        for j in range(mla_pos - 1, -1, -1):
            if "reduce_scatter_cross_device" in kernels[j].get("name", ""):
                # This RS is pre-attn if it's followed by quant or gemm (not router)
                # Look 2-3 ahead
                for look in range(j+1, min(j+4, len(kernels))):
                    n = kernels[look].get("name", "")
                    if "dynamic_per_token_scaled_quant" in n or "gemm_xdl" in n:
                        boundaries.append(j)
                        break
                    if "bf16gemm" in n or "grouped_topk" in n:
                        break  # This is post-attn RS, keep searching
                break

    # Split at boundaries
    layers = []
    prev = 0
    for b in sorted(set(boundaries)):
        if b > prev:
            layers.append(kernels[prev:b])
        prev = b
    if prev < len(kernels):
        layers.append(kernels[prev:])
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
    parser.add_argument("--skip-warmup", type=int, default=5,
                        help="Drop first N decode steps (cold caches / JIT). "
                             "Matches B300 trace_layer_detail convention.")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Aggregate over N steady-state decodes (default 20 → "
                             "~20 × 30 layers = 600 samples per operator). 0 = all available.")
    parser.add_argument("--skip-ratio", type=float, default=None,
                        help="DEPRECATED. Old single-decode picker. Ignored when "
                             "--skip-warmup / --max-steps are used.")
    parser.add_argument("--layers", type=str, default="10-40")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        m = re.search(r'_(c\d+)[_.]', os.path.basename(args.trace))
        suffix = m.group(1) if m else f"c{args.target_bs}"
        args.output = f"decode_breakdown_{suffix}.xlsx"

    layer_start, layer_end = map(int, args.layers.split("-"))
    evts = load_trace(args.trace)

    decodes = select_decodes(evts, args.target_bs, args.skip_warmup, args.max_steps)
    if not decodes:
        sys.exit(1)

    # Pool classified layers across all selected decodes.
    # Each decode contributes ~30 layers (layer_start..layer_end); with 20 decodes
    # we get ~600 layer samples → per-operator stats much tighter.
    all_classified = []  # list of [(module, kname, dur), ...] per layer
    total_dur_ms = 0.0
    for decode in decodes:
        ts0, dur = decode["ts"], decode.get("dur", 0)
        total_dur_ms += dur / 1000
        kernels = extract_kernels_in_window(evts, ts0, ts0 + dur)
        layers = split_layers(kernels)
        # Skip partial first layer (may start mid-layer in post-attn phase)
        if len(layers) > 1:
            first_names = [k.get("name", "") for k in layers[0][:5]]
            if any(any(p in n for p in ["bf16gemm", "grouped_topk", "kernel_moe"])
                   for n in first_names):
                layers = layers[1:]
        # Trim to requested layer range (per-decode), respecting available layers
        decode_layer_end = min(layer_end, len(layers) - 1)
        selected = layers[layer_start:decode_layer_end + 1]
        for l in selected:
            all_classified.append(classify_layer(l))

    n_layers_total = len(all_classified)
    n_decodes = len(decodes)
    avg_dur_ms = total_dur_ms / n_decodes if n_decodes else 0
    print(f"\n  Aggregated {n_layers_total} layer samples across {n_decodes} decodes")
    print(f"  Avg decode walltime: {avg_dur_ms:.2f}ms")

    if not all_classified:
        print("ERROR: no layers extracted — check --target-bs and --layers")
        sys.exit(1)

    # Representative layer (for xlsx Sheet 1 single-layer column display only —
    # the per-position aggregate stats below are what actually matter).
    rep_layer = all_classified[len(all_classified) // 2]

    # Per-position aggregates (mean, median, p95) across all (decode × layer) samples
    max_pos = max(len(cl) for cl in all_classified)
    avg_durs = []
    med_durs = []
    p95_durs = []
    for pos in range(max_pos):
        durs = sorted(cl[pos][2] for cl in all_classified if pos < len(cl))
        if not durs:
            avg_durs.append(0); med_durs.append(0); p95_durs.append(0)
            continue
        avg_durs.append(sum(durs) / len(durs))
        med_durs.append(durs[len(durs) // 2])
        p95_idx = max(0, int(len(durs) * 0.95) - 1)
        p95_durs.append(durs[p95_idx])

    total_rep = sum(d for _, _, d in rep_layer)
    total_avg = sum(avg_durs[:len(rep_layer)])
    n_layers = n_layers_total  # for back-compat below

    print(f"\n{'='*60}")
    print(f"LAYER {layer_start}-{layer_end} ANALYSIS "
          f"({n_layers_total} layer samples × {n_decodes} decodes)")
    print(f"{'='*60}")
    print(f"  Representative layer: {len(rep_layer)} kernels")
    print(f"  Representative-layer total: {total_rep:.1f}us ({total_rep/1000:.2f}ms)")
    print(f"  Pooled-mean per-layer total: {total_avg:.1f}us ({total_avg/1000:.2f}ms)")
    print(f"  Est decode (x61): {total_avg * 61 / 1000:.1f}ms (actual avg: {avg_dur_ms:.2f}ms)")

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
               "sum per module", "module_pct%",
               "avg_us", "median_us", "p95_us", "avg sum per module",
               "n_samples"])

    # Group consecutive kernels by module
    groups = []
    for i, (mod, kname, d) in enumerate(rep_layer):
        avg = avg_durs[i] if i < len(avg_durs) else 0
        med = med_durs[i] if i < len(med_durs) else 0
        p95 = p95_durs[i] if i < len(p95_durs) else 0
        if not groups or groups[-1][0] != mod:
            groups.append((mod, []))
        groups[-1][1].append((kname, d, avg, med, p95))

    for mod, kernel_list in groups:
        mod_sum = sum(d for _, d, _, _, _ in kernel_list)
        mod_avg_sum = sum(a for _, _, a, _, _ in kernel_list)
        mod_pct = mod_sum / total_rep * 100 if total_rep else 0
        first = True
        for kname, d, avg, med, p95 in kernel_list:
            pct = d / total_rep * 100 if total_rep else 0
            ws.append([
                mod if first else "",
                kname,
                round(d, 3),
                round(pct, 1),
                round(mod_sum, 3) if first else "",
                round(mod_pct, 1) if first else "",
                round(avg, 3),
                round(med, 3),
                round(p95, 3),
                round(mod_avg_sum, 3) if first else "",
                n_layers_total,
            ])
            first = False

    ws.append(["TOTAL", "", round(total_rep, 3), "100", "", "100",
               round(total_avg, 3), "", "", "", n_layers_total])

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
