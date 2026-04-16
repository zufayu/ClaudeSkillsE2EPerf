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
import csv
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


def extract_modules_in_window(evts, ts_start, ts_end):
    """Get all user_annotation module markers within a time window."""
    modules = [
        e for e in evts
        if e.get("cat") == "user_annotation"
        and e.get("ph") == "X"
        and e["ts"] >= ts_start
        and e["ts"] <= ts_end
        and not e.get("name", "").startswith("decode[")
        and not e.get("name", "").startswith("prefill[")
    ]
    return sorted(modules, key=lambda e: e["ts"])


def identify_layers(modules):
    """Group modules into layers using q_proj_and_k_up_proj as layer start."""
    layers = []
    current_layer = []
    layer_marker = "q_proj_and_k_up_proj"

    for m in modules:
        name = m.get("name", "")
        if name == layer_marker and current_layer:
            layers.append(current_layer)
            current_layer = []
        current_layer.append(m)

    if current_layer:
        layers.append(current_layer)

    return layers


def assign_kernels_to_modules(kernels, modules):
    """Assign each kernel to its enclosing module annotation."""
    result = []
    for k in kernels:
        k_ts = k["ts"]
        k_end = k_ts + k.get("dur", 0)
        enclosing = None
        for m in modules:
            m_ts = m["ts"]
            m_end = m_ts + m.get("dur", 0)
            if m_ts <= k_ts and k_end <= m_end + 1:  # +1us tolerance
                enclosing = m
                break
        result.append({
            "kernel_name": k.get("name", "unknown"),
            "module": enclosing.get("name", "unassigned") if enclosing else "unassigned",
            "dur_us": k.get("dur", 0),
            "ts": k_ts,
        })
    return result


def main():
    parser = argparse.ArgumentParser(description="Decode kernel breakdown from ATOM trace")
    parser.add_argument("trace", help="Path to trace JSON or JSON.GZ")
    parser.add_argument("--target-bs", type=int, default=64)
    parser.add_argument("--skip-ratio", type=float, default=0.5)
    parser.add_argument("--layers", type=str, default="10-40", help="Layer range (e.g., 10-40)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    # Auto-detect output filename
    if args.output is None:
        m = re.search(r'_(c\d+)[_.]', os.path.basename(args.trace))
        suffix = m.group(1) if m else f"c{args.target_bs}"
        args.output = f"decode_breakdown_{suffix}.csv"

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

    # Extract kernels and modules
    kernels = extract_kernels_in_window(evts, ts0, ts1)
    modules = extract_modules_in_window(evts, ts0, ts1)
    print(f"  Kernels in window: {len(kernels)}")
    print(f"  Module markers in window: {len(modules)}")

    # Identify layers
    layers = identify_layers(modules)
    print(f"  Layers detected: {len(layers)}")

    if len(layers) < layer_end:
        print(f"  WARNING: only {len(layers)} layers, adjusting range to {layer_start}-{len(layers)-1}")
        layer_end = len(layers) - 1

    # Module distribution
    mod_names = [m.get("name", "") for m in modules]
    mod_counts = collections.Counter(mod_names)
    print(f"\n  Module distribution:")
    for name, cnt in mod_counts.most_common():
        print(f"    {cnt:4d}x {name}")

    # Per-layer analysis for selected range
    selected_layers = layers[layer_start:layer_end + 1]
    n_layers = len(selected_layers)
    print(f"\n{'='*60}")
    print(f"LAYER {layer_start}-{layer_end} ANALYSIS ({n_layers} layers)")
    print(f"{'='*60}")

    # For each selected layer, find kernels within its time range
    all_layer_kernels = []
    for i, layer_modules in enumerate(selected_layers):
        layer_ts_start = min(m["ts"] for m in layer_modules)
        layer_ts_end = max(m["ts"] + m.get("dur", 0) for m in layer_modules)

        layer_kernels = [
            k for k in kernels
            if k["ts"] >= layer_ts_start and k["ts"] <= layer_ts_end
        ]
        assigned = assign_kernels_to_modules(layer_kernels, layer_modules)
        all_layer_kernels.extend(assigned)

    # Aggregate by module and kernel name
    by_module = collections.defaultdict(lambda: collections.defaultdict(lambda: {"count": 0, "dur": 0}))
    for k in all_layer_kernels:
        by_module[k["module"]][k["kernel_name"]]["count"] += k["count"] if "count" in k else 1
        by_module[k["module"]][k["kernel_name"]]["dur"] += k["dur_us"]

    # Print per-module summary
    total_dur = sum(k["dur_us"] for k in all_layer_kernels)
    per_layer_dur = total_dur / n_layers if n_layers else 0
    print(f"\nTotal kernel time (layers {layer_start}-{layer_end}): {total_dur/1000:.1f}ms")
    print(f"Per-layer average: {per_layer_dur:.1f}us ({per_layer_dur/1000:.2f}ms)")
    print(f"Estimated decode (×61): {per_layer_dur * 61 / 1000:.1f}ms (actual: {dur_ms:.2f}ms)")

    print(f"\n{'Module':<30} {'Avg/layer(us)':>12} {'%':>6} {'Kernels':>8}")
    print("-" * 60)
    module_summary = {}
    for mod_name in ["q_proj_and_k_up_proj", "rope_and_kv_cache", "mla_decode",
                     "v_up_proj_and_o_proj", "mxfp4_moe", "unassigned"]:
        if mod_name not in by_module and mod_name != "unassigned":
            continue
        kernels_in_mod = by_module.get(mod_name, {})
        mod_dur = sum(v["dur"] for v in kernels_in_mod.values())
        mod_avg = mod_dur / n_layers if n_layers else 0
        mod_pct = mod_dur / total_dur * 100 if total_dur else 0
        n_kernels = sum(v["count"] for v in kernels_in_mod.values())
        avg_kernels = n_kernels / n_layers if n_layers else 0
        module_summary[mod_name] = {"avg_us": mod_avg, "pct": mod_pct}
        print(f"  {mod_name:<28} {mod_avg:>10.1f} {mod_pct:>6.1f}% {avg_kernels:>7.1f}")

    # Detailed kernel list
    print(f"\n{'Kernel':<70} {'Module':<25} {'Avg(us)':>8} {'Count':>6} {'%':>6}")
    print("-" * 120)
    flat = []
    for mod_name, kernels_dict in by_module.items():
        for k_name, v in kernels_dict.items():
            avg = v["dur"] / n_layers if n_layers else 0
            pct = v["dur"] / total_dur * 100 if total_dur else 0
            flat.append((k_name, mod_name, avg, v["count"] / n_layers, pct))
    flat.sort(key=lambda x: -x[2])
    for k_name, mod_name, avg, cnt, pct in flat[:30]:
        print(f"  {k_name[:68]:<68} {mod_name:<25} {avg:>7.1f} {cnt:>5.1f}x {pct:>5.1f}%")

    # Write CSV
    print(f"\nWriting {args.output}...")
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel_name", "module", "avg_us", "count_per_layer", "pct"])
        for k_name, mod_name, avg, cnt, pct in flat:
            writer.writerow([k_name, mod_name, f"{avg:.1f}", f"{cnt:.1f}", f"{pct:.1f}"])
    print(f"Done. {len(flat)} kernels written.")


if __name__ == "__main__":
    main()
