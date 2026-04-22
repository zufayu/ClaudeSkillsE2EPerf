#!/usr/bin/env python3
"""
Shared utilities for Chrome trace JSON analysis.

Eliminates 6+ duplicate load_trace/find_gpu_kernels definitions across
analysis scripts. Import from here instead of copy-pasting.

Usage:
    from trace_utils import load_trace, find_gpu_kernels

    events = load_trace("trace.json.gz")
    kernels = find_gpu_kernels(events)
"""

import gzip
import json
import os
import sys


def load_trace(filepath):
    """Load Chrome trace JSON (optionally gzipped).

    Handles both formats:
      - {"traceEvents": [...]}  (standard Chrome trace)
      - [...]                    (bare event list)

    Returns (events_list, raw_data_dict).
    """
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"Loading {filepath} ({size_mb:.0f}MB)...")

    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rt", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        events = data
    else:
        events = data.get("traceEvents", [])

    print(f"  {len(events)} events loaded")
    return events, data


def load_trace_events(filepath):
    """Load trace and return just the events list (convenience wrapper)."""
    events, _ = load_trace(filepath)
    return events


def find_gpu_kernels(events, gpu_pid=None):
    """Extract GPU kernel events for rank 0 (smallest PID), sorted by ts.

    Args:
        events: List of Chrome trace events
        gpu_pid: Force a specific PID (default: auto-detect smallest = rank 0)

    Returns list of kernel events sorted by timestamp.
    """
    kernels = []
    pids = set()

    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append(e)
            pids.add(e.get("pid"))

    if gpu_pid is not None:
        kernels = [k for k in kernels if k.get("pid") == gpu_pid]
    elif len(pids) > 1:
        # Multi-rank trace: use smallest PID (rank 0)
        min_pid = min(pids)
        kernels = [k for k in kernels if k.get("pid") == min_pid]
        print(f"  Multiple GPU PIDs ({len(pids)}), using PID {min_pid} (rank 0)")

    kernels.sort(key=lambda x: x.get("ts", 0))
    print(f"  {len(kernels)} GPU kernels (rank 0)")
    return kernels


def find_cpu_events(events, name_pattern=None, cat_filter=None):
    """Extract CPU-side events, optionally filtered by name regex or category.

    Args:
        events: List of Chrome trace events
        name_pattern: Regex to match event name (optional)
        cat_filter: Set/list of categories to include (optional)

    Returns list of matching events sorted by timestamp.
    """
    import re
    result = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue  # skip GPU events
        if cat_filter and cat not in cat_filter:
            continue
        if name_pattern and not re.search(name_pattern, e.get("name", ""), re.IGNORECASE):
            continue
        result.append(e)

    result.sort(key=lambda x: x.get("ts", 0))
    return result


def get_trace_time_span(events):
    """Get (start_ts, end_ts, duration_s) for all X-type events."""
    ts_list = []
    end_list = []
    for e in events:
        if e.get("ph") == "X" and "ts" in e:
            ts_list.append(e["ts"])
            end_list.append(e["ts"] + e.get("dur", 0))

    if not ts_list:
        return 0, 0, 0

    start = min(ts_list)
    end = max(end_list)
    return start, end, (end - start) / 1e6  # duration in seconds


def get_kernel_streams(kernels):
    """Get set of (pid, tid) stream identifiers from kernel events."""
    return {(k.get("pid"), k.get("tid")) for k in kernels}


def filter_decode_kernels(kernels, decode_markers):
    """Filter kernels that fall within decode marker time ranges.

    Args:
        kernels: sorted GPU kernel events
        decode_markers: list of (start_ts, end_ts) tuples for decode windows

    Returns kernels that overlap with any decode window.
    """
    result = []
    marker_idx = 0
    for k in kernels:
        ts = k.get("ts", 0)
        dur = k.get("dur", 0)
        k_end = ts + dur

        # Advance marker index
        while marker_idx < len(decode_markers) and decode_markers[marker_idx][1] < ts:
            marker_idx += 1

        if marker_idx >= len(decode_markers):
            break

        m_start, m_end = decode_markers[marker_idx]
        if ts >= m_start and k_end <= m_end:
            result.append(k)

    return result
