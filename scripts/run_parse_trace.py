#!/usr/bin/env python3
"""
Wrapper around ATOM's parse_trace.py with --target-bs and multi-EP support.

ATOM's parse_trace.py assumes all transformer layer modules are direct children
of a single CompiledFxGraph block inside capture_graph. With EP>1, torch.compile
splits the model into multiple CompiledFxGraph blocks (one per few layers), and
parse_trace only sees modules from the first block → incomplete breakdown.

This wrapper fixes both issues:
  1. Target-BS: selects a steady-state decode event instead of the first one
  2. Multi-EP: flattens the capture_graph hierarchy so all modules from all
     CompiledFxGraph blocks appear as direct children of capture_graph

Usage:
    python3 scripts/run_parse_trace.py <trace.json.gz> [--layer N] [--target-bs N]

    # Auto-select most frequent bs (steady state):
    python3 scripts/run_parse_trace.py trace.json.gz --layer 40

    # Explicit bs=64:
    python3 scripts/run_parse_trace.py trace.json.gz --layer 40 --target-bs 64

Output:
    decode_breakdown_c64.xlsx   (kernel breakdown at target bs, suffix auto-detected from filename)
    prefill_breakdown_c64.xlsx  (prefill breakdown, unchanged from upstream)

Requires ATOM's tools/ on sys.path. Set ATOM_TOOLS env var if not at default.
"""

import argparse
import copy
import re
import sys
import os

# Add ATOM tools to path
_SEARCH_PATHS = [
    os.environ.get("ATOM_TOOLS", ""),
    "/app/ATOM/tools",
    os.path.expanduser("~/ATOM/tools"),
    "/home/kqian/ATOM/tools",
]
for p in _SEARCH_PATHS:
    if p and os.path.isfile(os.path.join(p, "parse_trace.py")):
        sys.path.insert(0, p)
        break

try:
    import parse_trace
except ImportError:
    print("ERROR: Cannot import parse_trace. Set ATOM_TOOLS=/path/to/ATOM/tools")
    sys.exit(1)


def select_decode_bs(events, target_bs=None, skip_ratio=0.5):
    """Find target bs and a steady-state decode event at that bs.

    Picks the decode event at skip_ratio position (by timestamp) among all
    events with the target batch size, so the system is well into steady
    state — graph replay is warm, caches are hot, batch is full.
    """
    decodes = sorted(
        [
            e for e in events
            if e.get("name", "").startswith("decode[")
            and e.get("ph") == "X"
            and e.get("cat") == "gpu_user_annotation"
        ],
        key=lambda x: x["ts"],
    )
    if not decodes:
        return None, None

    bs_counts = {}
    decodes_by_bs = {}
    for d in decodes:
        m = re.search(r"bs=(\d+)", d.get("name", ""))
        if m:
            bs = int(m.group(1))
            bs_counts[bs] = bs_counts.get(bs, 0) + 1
            decodes_by_bs.setdefault(bs, []).append(d)

    print(f"Decode bs distribution: {dict(sorted(bs_counts.items()))}")

    if target_bs is None:
        target_bs = max(bs_counts, key=lambda b: bs_counts[b])
        print(f"Auto-selected target_bs={target_bs} ({bs_counts[target_bs]} events)")
    else:
        print(f"--target-bs={target_bs} ({bs_counts.get(target_bs, 0)} events)")

    candidates = decodes_by_bs.get(target_bs)
    if not candidates:
        return None, target_bs

    # Pick event at skip_ratio position for steady state
    idx = int(len(candidates) * skip_ratio)
    idx = min(idx, len(candidates) - 1)
    selected = candidates[idx]
    dur_ms = selected.get("dur", 0) / 1000
    print(
        f"Picked decode #{idx}/{len(candidates)} "
        f"(skip_ratio={skip_ratio}, dur={dur_ms:.2f}ms)"
    )
    return selected, target_bs


def detect_multi_compiled_graph(capture_events, cg_event):
    """Detect if capture_graph contains multiple CompiledFxGraph blocks.

    Returns list of CompiledFxGraph events if multi-EP, empty list if standard.
    """
    cg_start = cg_event["ts"]
    cg_end = cg_start + cg_event.get("dur", 0)

    compiled_graphs = []
    for e in capture_events:
        if e.get("ph") != "X":
            continue
        name = e.get("name", "")
        if "CompiledFxGraph" not in name:
            continue
        e_start = e.get("ts", 0)
        e_end = e_start + e.get("dur", 0)
        if e_start >= cg_start and e_end <= cg_end:
            compiled_graphs.append(e)

    return compiled_graphs


def flatten_capture_events_for_multi_ep(capture_events, cg_event, compiled_graphs):
    """Flatten multi-CompiledFxGraph hierarchy for parse_trace compatibility.

    parse_trace expects: capture_graph → module → kernel_launch (2 levels)
    Multi-EP has:       capture_graph → CompiledFxGraph → module → kernel_launch (3 levels)

    Fix: remove CompiledFxGraph wrapper events from capture_events, so their
    children become direct children of capture_graph. We do this by shrinking
    each CompiledFxGraph event's duration to 0 (making it invisible to
    EventIndex.get_direct_children which checks time containment).
    """
    # Build set of CompiledFxGraph event IDs for fast lookup
    cg_ids = set(id(e) for e in compiled_graphs)

    flattened = []
    for e in capture_events:
        if id(e) in cg_ids:
            # Shrink CompiledFxGraph to 0 duration so its children
            # "escape" and become direct children of capture_graph
            e_copy = dict(e)
            e_copy["dur"] = 0
            flattened.append(e_copy)
        else:
            flattened.append(e)

    return flattened


def find_capture_graph_event(capture_events, target_bs):
    """Find the capture_graph_bs_N event matching target_bs."""
    target_name = f"capture_graph_bs_{target_bs}"

    # Exact match
    matches = [
        e for e in capture_events
        if e.get("name") == target_name and e.get("ph") == "X"
        and e.get("dur", 0) > 100  # filter out the tiny duplicate events
    ]
    if matches:
        return matches[0]

    # Fallback: largest bs smaller than target
    bs_events = []
    for e in capture_events:
        if e.get("ph") != "X":
            continue
        m = re.match(r"^capture_graph_bs_(\d+)$", e.get("name", ""))
        if m and e.get("dur", 0) > 100:
            bs_events.append((int(m.group(1)), e))

    if bs_events:
        # Try closest smaller
        smaller = [(b, e) for b, e in bs_events if b <= target_bs]
        if smaller:
            best = max(smaller, key=lambda x: x[0])
            return best[1]
        # Any
        return bs_events[0][1]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="parse_trace.py wrapper with --target-bs and multi-EP support."
    )
    parser.add_argument("filepath", help="Path to trace JSON or JSON.GZ file")
    parser.add_argument("--layer", type=int, default=3, help="Target layer index")
    parser.add_argument(
        "--target-bs", type=int, default=None,
        help="Decode batch size (default: most frequent)",
    )
    parser.add_argument(
        "--skip-ratio", type=float, default=0.5,
        help="Position within target-bs decodes to pick (0.0=first, 0.5=median, 0.9=late). Default: 0.5",
    )
    parser.add_argument(
        "--capture-trace", type=str, default=None,
        help="Explicit path to capture_graph trace file (bypasses auto-detection)",
    )
    parser.add_argument(
        "--suffix", type=str, default=None,
        help="Suffix for output files (e.g., 'c64'). Auto-detected from filename if not set.",
    )
    args = parser.parse_args()

    filepath = args.filepath

    # Auto-detect suffix from filename (e.g., ..._c64_full.log → c64)
    if args.suffix is None:
        m = re.search(r'_(c\d+)[_.]', os.path.basename(filepath))
        args.suffix = m.group(1) if m else None

    print(f"Loading run trace: {filepath}")
    trace = parse_trace.load_trace(filepath)
    events = trace.get("traceEvents", [])
    print(f"Loaded {len(events)} events")

    if args.capture_trace:
        capture_trace_path = args.capture_trace
        print(f"Using explicit capture trace: {capture_trace_path}")
    else:
        capture_trace_path = parse_trace.find_capture_graph_trace_path(filepath)
    if capture_trace_path is None:
        print("Warning: no capture trace found, using run trace for hierarchy")
        capture_events = events
    else:
        print(f"Loading capture trace: {capture_trace_path}")
        capture_events = parse_trace.load_trace(capture_trace_path).get(
            "traceEvents", []
        )
        print(f"Loaded {len(capture_events)} capture events")

    # --- Prefill: unchanged ---
    print("\n" + "=" * 60)
    print("PREFILL ANALYSIS")
    print("=" * 60)
    pfx = f"_{args.suffix}" if args.suffix else ""
    parse_trace.parse_prefill(events, f"prefill_breakdown{pfx}.xlsx", target_layer=args.layer)

    # --- Decode: select target bs ---
    print("\n" + "=" * 60)
    print("DECODE ANALYSIS (with --target-bs and multi-EP support)")
    print("=" * 60)

    target_decode, actual_bs = select_decode_bs(events, args.target_bs, args.skip_ratio)
    if target_decode is None:
        print("ERROR: no decode events at target bs")
        sys.exit(1)

    dur_ms = target_decode.get("dur", 0) / 1000
    print(f"Selected: {target_decode.get('name')} (ts={target_decode['ts']:.0f}, dur={dur_ms:.2f}ms)")

    # --- Multi-EP detection and flattening ---
    cg_event = find_capture_graph_event(capture_events, actual_bs)
    if cg_event is None:
        print("WARNING: No capture_graph event found, proceeding without multi-EP fix")
        final_capture_events = capture_events
    else:
        print(f"Using capture_graph: {cg_event.get('name')} (dur={cg_event.get('dur',0)}µs)")
        compiled_graphs = detect_multi_compiled_graph(capture_events, cg_event)

        if len(compiled_graphs) > 1:
            print(f"\n*** MULTI-EP DETECTED: {len(compiled_graphs)} CompiledFxGraph blocks ***")
            for i, cg in enumerate(compiled_graphs):
                print(f"  [{i}] {cg.get('name', '')[:60]}... (dur={cg.get('dur',0)}µs)")
            print("Flattening hierarchy for parse_trace compatibility...")
            final_capture_events = flatten_capture_events_for_multi_ep(
                capture_events, cg_event, compiled_graphs
            )
            print(f"Flattened: {len(capture_events)} → {len(final_capture_events)} events")
        else:
            print("Standard single-graph hierarchy (EP=1 or single CompiledFxGraph)")
            final_capture_events = capture_events

    # Remove all OTHER decode gpu_user_annotation events from the event list
    # so parse_trace.parse_decode sees only our selected one as "first".
    filtered_events = [
        e for e in events
        if not (
            e.get("name", "").startswith("decode[")
            and e.get("ph") == "X"
            and e.get("cat") == "gpu_user_annotation"
            and e is not target_decode
        )
    ]
    print(f"Filtered events: {len(events)} -> {len(filtered_events)} (removed {len(events) - len(filtered_events)} other decode events)")

    parse_trace.parse_decode(
        filtered_events, final_capture_events, f"decode_breakdown{pfx}.xlsx", target_layer=args.layer
    )


if __name__ == "__main__":
    main()
