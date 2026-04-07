#!/usr/bin/env python3
"""
Fix multi-stream kernel overlap visualization in Perfetto / chrome://tracing.

SGLang uses PDL (Programmatic Dependent Launch) with parallel kernels on
different CUDA streams (e.g. compute stream + communication stream for
lamport allreduce). Perfetto cannot properly display overlapping events
on the same TID — they get hidden/collapsed.

This script detects temporally overlapping GPU kernel events on the same
(pid, tid) and assigns conflicting kernels to new synthetic TIDs
(tid + "_s2", "_s3", etc.) so Perfetto renders them on separate rows.

Usage:
    python3 serialize_trace.py trace.json.gz
    # Output: trace_serialized.json.gz

    python3 serialize_trace.py trace.json.gz -o fixed_trace.json.gz
    # Output: fixed_trace.json.gz

Based on: https://github.com/ROCm/oss-dashboard/blob/main/utils/perfetto_serialize.py
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict


def serialize_events(events):
    """Fix overlapping kernel events by splitting them to separate TIDs."""
    total = len(events)
    last_end = defaultdict(lambda: -1)
    modified = 0
    suffix_idx = 2

    for e in events:
        if e.get("ph") != "X":
            continue
        args = e.get("args", {})
        if "registers per thread" not in args:
            continue

        pid = e["pid"]
        tid = e["tid"]
        ts = e.get("ts", 0)
        dur = e.get("dur", 0)
        key = (pid, tid)

        while ts < last_end[key]:
            e["tid"] = f"{tid}_s{suffix_idx}"
            key = (pid, e["tid"])
            suffix_idx += 1
            modified += 1

        last_end[key] = ts + dur

    print(f"  Events: {total}, overlaps fixed: {modified}")
    return events


def main():
    parser = argparse.ArgumentParser(
        description="Fix multi-stream kernel overlap for Perfetto visualization"
    )
    parser.add_argument("input", help="Input trace file (.json.gz)")
    parser.add_argument("-o", "--output", help="Output file (default: <input>_serialized.json.gz)")
    args = parser.parse_args()

    output = args.output or args.input.replace(".json.gz", "_serialized.json.gz")
    if output == args.input:
        output = args.input + "_serialized.gz"

    print(f"Reading: {args.input}")
    with gzip.open(args.input, "rt", encoding="utf-8") as f:
        trace = json.loads(f.read())

    events = trace.get("traceEvents", [])
    print(f"Processing {len(events)} events...")
    trace["traceEvents"] = serialize_events(events)

    print(f"Writing: {output}")
    with gzip.open(output, "wt", encoding="utf-8") as f:
        json.dump(trace, f)

    print("Done.")


if __name__ == "__main__":
    main()
