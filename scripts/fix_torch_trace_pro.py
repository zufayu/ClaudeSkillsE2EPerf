#!/usr/bin/env python3
"""
Fix overlapping CUDA kernel events in torch profiler traces.

Moves overlapping kernels on the same (pid, tid) to separate "_overlap" streams
so they can be visualized without stacking in chrome://tracing.

Usage:
    python3 fix_torch_trace_pro.py <trace.json.gz> [--output PATH]
    # Default output: <trace>.fix.json.gz (alongside input)
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict


def process_events(events):
    last_end = defaultdict(lambda: -1)
    modified = 0

    for e in events:
        if e.get("ph") == "X" and "registers per thread" in e.get("args", {}):
            pid = e["pid"]
            tid = e["tid"]
            ts = e["ts"]
            dur = e["dur"]

            key = (pid, tid)
            if ts < last_end[key]:
                e["tid"] = f"stream {tid} {tid}_overlap"
                modified += 1

            last_end[(pid, e["tid"])] = ts + dur

    print(f"Total events: {len(events)}")
    print(f"Fixed overlapping events: {modified}")
    return events


def main():
    parser = argparse.ArgumentParser(
        description="Fix overlapping CUDA kernel events in torch profiler traces."
    )
    parser.add_argument(
        "filepath",
        help="Path to trace JSON.GZ file from torch profiler",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path (default: replace .json.gz with .fix.json.gz alongside input)",
    )
    args = parser.parse_args()

    filename = args.filepath
    filename_out = args.output or filename.replace(".json.gz", ".fix.json.gz")

    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        trace = json.load(f)

    trace["traceEvents"] = process_events(trace.get("traceEvents", []))

    with gzip.open(filename_out, 'wt', encoding='utf-8') as f:
        json.dump(trace, f)

    print("Done!")
    print(f"Output: {filename_out}")


if __name__ == "__main__":
    main()
