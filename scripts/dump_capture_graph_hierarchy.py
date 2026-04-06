#!/usr/bin/env python3
"""
Dump capture_graph module hierarchy from ATOM/vLLM trace files.

Diagnostic tool to understand how EP changes the capture_graph module tree.
Used to debug parse_trace.py failures on EP>1 configurations.

Usage:
    python3 scripts/dump_capture_graph_hierarchy.py <trace.json.gz> [--max-depth 4]

Looks for a companion capture trace (same dir, *capture*.json.gz) automatically.
"""

import argparse
import gzip
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional


def load_trace(path: str) -> Dict[str, Any]:
    """Load a JSON or JSON.GZ trace file."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def find_capture_trace(run_trace_path: str) -> Optional[str]:
    """Find the capture graph trace file alongside the run trace."""
    directory = os.path.dirname(run_trace_path) or "."
    basename = os.path.basename(run_trace_path)

    # Try ATOM convention: same dir, filename contains "capture"
    for fn in sorted(os.listdir(directory)):
        if fn == basename:
            continue
        if "capture" in fn.lower() and fn.endswith((".json.gz", ".json")):
            return os.path.join(directory, fn)

    # Try parse_trace.py convention
    try:
        sys.path.insert(0, "/app/ATOM/tools")
        import parse_trace
        return parse_trace.find_capture_graph_trace_path(run_trace_path)
    except Exception:
        pass

    return None


def get_children(events: List[Dict], parent: Dict) -> List[Dict]:
    """Get direct children of a parent event (X events nested within it)."""
    p_start = parent.get("ts", 0)
    p_end = p_start + parent.get("dur", 0)
    p_tid = parent.get("tid")

    # Find all events strictly within parent
    candidates = []
    for e in events:
        if e.get("ph") != "X":
            continue
        if e.get("tid") != p_tid:
            continue
        e_start = e.get("ts", 0)
        e_end = e_start + e.get("dur", 0)
        if e_start >= p_start and e_end <= p_end and e is not parent:
            candidates.append(e)

    if not candidates:
        return []

    # Sort by start time
    candidates.sort(key=lambda x: x.get("ts", 0))

    # Filter to direct children (not grandchildren)
    direct = []
    covered_end = 0
    for c in candidates:
        c_start = c.get("ts", 0)
        if c_start >= covered_end:
            direct.append(c)
            covered_end = c_start + c.get("dur", 0)

    return direct


def has_kernel_launch(events: List[Dict], event: Dict) -> bool:
    """Check if event contains any cuda_runtime kernel launches."""
    e_start = event.get("ts", 0)
    e_end = e_start + event.get("dur", 0)
    e_tid = event.get("tid")
    for e in events:
        if e.get("tid") != e_tid:
            continue
        if e.get("cat") != "cuda_runtime":
            continue
        name = e.get("name", "")
        if "launch" in name.lower() or "LaunchKernel" in name:
            ts = e.get("ts", 0)
            if e_start <= ts <= e_end:
                return True
    return False


def count_kernel_launches(events: List[Dict], event: Dict) -> int:
    """Count kernel launches within an event."""
    e_start = event.get("ts", 0)
    e_end = e_start + event.get("dur", 0)
    e_tid = event.get("tid")
    count = 0
    for e in events:
        if e.get("tid") != e_tid:
            continue
        if e.get("cat") != "cuda_runtime":
            continue
        name = e.get("name", "")
        if "launch" in name.lower() or "LaunchKernel" in name:
            ts = e.get("ts", 0)
            if e_start <= ts <= e_end:
                count += 1
    return count


def is_norm_name(name: str) -> bool:
    """Check if name is a normalization layer."""
    lower = name.lower()
    return any(k in lower for k in ("layernorm", "rmsnorm", "rmsnorm_quant"))


def dump_hierarchy(events: List[Dict], parent: Dict, depth: int, max_depth: int,
                   prefix: str = "", show_kernels: bool = False) -> List[str]:
    """Recursively dump the module hierarchy."""
    lines = []
    children = get_children(events, parent)

    for i, child in enumerate(children):
        is_last = (i == len(children) - 1)
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        name = child.get("name", "<unknown>")
        dur_us = child.get("dur", 0)
        cat = child.get("cat", "")
        n_kernels = count_kernel_launches(events, child)
        norm_marker = " [NORM]" if is_norm_name(name) else ""

        info = f"{prefix}{connector}{name} (dur={dur_us:.0f}µs, cat={cat}, kernels={n_kernels}){norm_marker}"
        lines.append(info)

        if depth < max_depth:
            lines.extend(dump_hierarchy(events, child, depth + 1, max_depth,
                                       child_prefix, show_kernels))

    return lines


def main():
    parser = argparse.ArgumentParser(description="Dump capture_graph module hierarchy")
    parser.add_argument("filepath", help="Path to run trace .json.gz")
    parser.add_argument("--capture-trace", default=None, help="Path to capture trace (auto-detected if omitted)")
    parser.add_argument("--max-depth", type=int, default=4, help="Max tree depth (default: 4)")
    parser.add_argument("--target-bs", type=int, default=None, help="Target batch size for capture_graph (default: largest)")
    args = parser.parse_args()

    # Load run trace
    print(f"Loading run trace: {args.filepath}")
    run_trace = load_trace(args.filepath)
    run_events = run_trace.get("traceEvents", [])
    print(f"  {len(run_events)} events")

    # Find decode events
    decodes = sorted(
        [e for e in run_events
         if e.get("name", "").startswith("decode[")
         and e.get("ph") == "X"
         and e.get("cat") == "gpu_user_annotation"],
        key=lambda x: x["ts"]
    )
    bs_counts = {}
    for d in decodes:
        m = re.search(r"bs=(\d+)", d.get("name", ""))
        if m:
            bs = int(m.group(1))
            bs_counts[bs] = bs_counts.get(bs, 0) + 1
    print(f"  Decode events: {len(decodes)}")
    print(f"  BS distribution: {dict(sorted(bs_counts.items()))}")

    if decodes:
        # Show decode durations for most frequent BS
        freq_bs = max(bs_counts, key=lambda b: bs_counts[b]) if bs_counts else None
        if freq_bs:
            bs_decodes = [d for d in decodes if f"bs={freq_bs}" in d.get("name", "")]
            durs_ms = [d.get("dur", 0) / 1000 for d in bs_decodes]
            print(f"  Decode bs={freq_bs}: count={len(durs_ms)}, min={min(durs_ms):.2f}ms, max={max(durs_ms):.2f}ms, median={sorted(durs_ms)[len(durs_ms)//2]:.2f}ms")

    # Count GPU kernels in first decode
    if decodes:
        first_d = decodes[0]
        d_start = first_d["ts"]
        d_end = d_start + first_d.get("dur", 0)
        gpu_kernels = [e for e in run_events if e.get("cat") == "kernel" and d_start <= e["ts"] <= d_end]
        print(f"  GPU kernels in first decode: {len(gpu_kernels)}")
        if gpu_kernels:
            total_dur_us = sum(k.get("dur", 0) for k in gpu_kernels)
            print(f"  Total GPU kernel time in first decode: {total_dur_us:.0f}µs ({total_dur_us/1000:.2f}ms)")

    # Load capture trace
    capture_path = args.capture_trace or find_capture_trace(args.filepath)
    if capture_path is None:
        print("\nERROR: No capture trace found. Specify with --capture-trace")
        sys.exit(1)

    print(f"\nLoading capture trace: {capture_path}")
    capture_trace = load_trace(capture_path)
    capture_events = capture_trace.get("traceEvents", [])
    print(f"  {len(capture_events)} events")

    # Find all capture_graph events
    cg_events = sorted(
        [e for e in capture_events
         if e.get("name", "").startswith("capture_graph") and e.get("ph") == "X"],
        key=lambda x: x.get("ts", 0)
    )
    print(f"\n  capture_graph events found: {len(cg_events)}")
    for cg in cg_events:
        print(f"    {cg.get('name')} (dur={cg.get('dur', 0):.0f}µs, tid={cg.get('tid')})")

    # Select target capture_graph
    target_bs = args.target_bs
    if target_bs is not None:
        target_name = f"capture_graph_bs_{target_bs}"
        selected = [e for e in cg_events if e.get("name") == target_name]
    else:
        # Use largest BS
        bs_cgs = []
        for e in cg_events:
            m = re.match(r"capture_graph_bs_(\d+)", e.get("name", ""))
            if m:
                bs_cgs.append((int(m.group(1)), e))
        if bs_cgs:
            max_bs = max(b for b, _ in bs_cgs)
            selected = [e for b, e in bs_cgs if b == max_bs]
            target_bs = max_bs
        else:
            selected = cg_events[:1]

    if not selected:
        print("ERROR: No matching capture_graph found")
        sys.exit(1)

    cg = selected[0]
    print(f"\n{'='*80}")
    print(f"CAPTURE GRAPH HIERARCHY: {cg.get('name')}")
    print(f"{'='*80}")

    # Get events within this capture_graph
    cg_start = cg["ts"]
    cg_end = cg_start + cg.get("dur", 0)
    cg_scope_events = [
        e for e in capture_events
        if e.get("ph") == "X"
        and e.get("ts", 0) >= cg_start
        and e.get("ts", 0) + e.get("dur", 0) <= cg_end
    ]
    print(f"Events within capture_graph: {len(cg_scope_events)}")

    # Dump hierarchy
    lines = dump_hierarchy(cg_scope_events, cg, 0, args.max_depth)
    for line in lines:
        print(line)

    # Summary: module list as parse_trace would see it
    print(f"\n{'='*80}")
    print("MODULE LIST (as parse_trace.py would build it)")
    print(f"{'='*80}")

    direct_children = get_children(cg_scope_events, cg)
    print(f"Direct children of {cg.get('name')}: {len(direct_children)}")

    all_modules = []
    for child in direct_children:
        child_name = child.get("name", "")
        sub_children = get_children(cg_scope_events, child)
        sub_kernel_children = [sc for sc in sub_children if count_kernel_launches(cg_scope_events, sc) > 0]

        modules = sub_kernel_children if sub_kernel_children else [child]
        for mod in modules:
            mod_name = mod.get("name", "<unknown>")
            n_kernels = count_kernel_launches(cg_scope_events, mod)
            norm = " [NORM]" if is_norm_name(mod_name) else ""
            all_modules.append((mod_name, n_kernels, norm))

    print(f"\nTotal modules: {len(all_modules)}")
    for i, (name, nk, norm) in enumerate(all_modules):
        print(f"  [{i:3d}] {name} (kernels={nk}){norm}")

    # Count norms
    norm_indices = [i for i, (name, _, _) in enumerate(all_modules) if is_norm_name(name)]
    print(f"\nNorm module indices: {norm_indices}")
    print(f"Norm count: {len(norm_indices)}")
    if norm_indices:
        layers = (len(norm_indices) - 1) // 2
        print(f"Implied layers (norms/2 - final): {layers}")

    print("\nDone.")


if __name__ == "__main__":
    main()
