#!/usr/bin/env python3
"""
Wrapper around ATOM's parse_trace.py with --target-bs support.

ATOM's parse_trace.py always uses the first decode event (often bs=1 after
prefill ramp-up, or an early bs=64 before steady state). This wrapper calls
parse_trace for prefill unchanged, then for decode it selects a mid-run
decode event at the target batch size (default: most frequent bs) so the
kernel timings reflect steady-state behavior with warm caches and graphs.

Usage:
    python3 scripts/run_parse_trace.py <trace.json.gz> [--layer N] [--target-bs N]

    # Auto-select most frequent bs (steady state):
    python3 scripts/run_parse_trace.py trace.json.gz --layer 40

    # Explicit bs=64:
    python3 scripts/run_parse_trace.py trace.json.gz --layer 40 --target-bs 64

Output:
    decode_breakdown.xlsx   (kernel breakdown at target bs)
    prefill_breakdown.xlsx  (prefill breakdown, unchanged from upstream)

Requires ATOM's tools/ on sys.path. Set ATOM_TOOLS env var if not at default.
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="parse_trace.py wrapper with --target-bs for decode analysis."
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
    args = parser.parse_args()

    filepath = args.filepath

    print(f"Loading run trace: {filepath}")
    trace = parse_trace.load_trace(filepath)
    events = trace.get("traceEvents", [])
    print(f"Loaded {len(events)} events")

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
    parse_trace.parse_prefill(events, "prefill_breakdown.xlsx", target_layer=args.layer)

    # --- Decode: select target bs ---
    print("\n" + "=" * 60)
    print("DECODE ANALYSIS (with --target-bs)")
    print("=" * 60)

    target_decode, actual_bs = select_decode_bs(events, args.target_bs, args.skip_ratio)
    if target_decode is None:
        print("ERROR: no decode events at target bs")
        sys.exit(1)

    # Trick: temporarily set target decode's timestamp to be the earliest
    # among all decode events so parse_trace.parse_decode picks it as "first".
    all_decodes = [
        e for e in events
        if e.get("name", "").startswith("decode[")
        and e.get("ph") == "X"
        and e.get("cat") == "gpu_user_annotation"
    ]
    min_ts = min(d["ts"] for d in all_decodes)
    saved_ts = target_decode["ts"]
    target_decode["ts"] = min_ts - 1

    print(f"Selected: {target_decode.get('name')} (original ts={saved_ts:.0f})")

    parse_trace.parse_decode(
        events, capture_events, "decode_breakdown.xlsx", target_layer=args.layer
    )

    # Restore
    target_decode["ts"] = saved_ts


if __name__ == "__main__":
    main()
