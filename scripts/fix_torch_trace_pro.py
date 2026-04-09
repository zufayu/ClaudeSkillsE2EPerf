#!/usr/bin/env python3
"""
Fix overlapping CUDA kernel events in torch profiler traces.

Moves overlapping kernels on the same (pid, tid) to separate "_overlap" streams
so they can be visualized without stacking in chrome://tracing.

Usage:
    python3 fix_torch_trace_pro.py <trace.json.gz>
    # Outputs: <trace>.fix.json.gz
"""

import gzip
import json
from collections import defaultdict
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_torch_trace_pro.py <trace.json.gz>")
        return

    filename = sys.argv[1]
    filename_out = filename.replace(".json.gz", ".fix.json.gz")

    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        trace = json.load(f)

    trace["traceEvents"] = process_events(trace.get("traceEvents", []))

    with gzip.open(filename_out, 'wt', encoding='utf-8') as f:
        json.dump(trace, f)

    print("Done!")
    print(f"Output: {filename_out}")


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


if __name__ == "__main__":
    main()
