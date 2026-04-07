#!/usr/bin/env python3
"""
Measure per-layer time using consecutive FMHA events as layer boundaries.

Two consecutive fmhaSm100fKernel_Qkv events define one layer's time span.
Take N stable-state layers, compute average, multiply by 61 for full decode estimate.

Usage:
    python3 trace_fmha_layer_time.py <trace.json.gz>
    python3 trace_fmha_layer_time.py <trace.json.gz> --layers 2000
"""

import argparse
import gzip
import json
import sys
from statistics import median, stdev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    parser.add_argument("--layers", type=int, default=2000, help="Number of stable layers to use")
    parser.add_argument("--model-layers", type=int, default=61, help="Total model layers")
    args = parser.parse_args()

    # Load trace
    print(f"Loading: {args.filepath}")
    opener = gzip.open if args.filepath.endswith(".gz") else open
    with opener(args.filepath, "rt", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    print(f"Total events: {len(events)}")

    # Trace time span
    all_ts = [e["ts"] for e in events if "ts" in e and e.get("ph") == "X"]
    all_ends = [e["ts"] + e.get("dur", 0) for e in events if "ts" in e and e.get("ph") == "X"]
    if all_ts:
        trace_start = min(all_ts)
        trace_end = max(all_ends)
        trace_span = trace_end - trace_start
        print(f"Trace span: {trace_span/1e6:.3f}s  [{trace_start/1e6:.3f}s, {trace_end/1e6:.3f}s]")

    # Find GPU kernel events, identify PIDs
    gpu_pids = set()
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu_pids.add(e.get("pid"))

    if len(gpu_pids) > 1:
        gpu_pid = min(gpu_pids)
        print(f"Multiple GPU PIDs: {sorted(gpu_pids)}, using {gpu_pid} (rank 0)")
    elif gpu_pids:
        gpu_pid = gpu_pids.pop()
        print(f"GPU PID: {gpu_pid}")
    else:
        print("ERROR: No GPU kernel events found")
        sys.exit(1)

    # Find ALL fmhaSm100fKernel events on rank 0
    fmha_events = []
    for e in events:
        if (e.get("ph") == "X" and
            e.get("pid") == gpu_pid and
            "fmhaSm100" in e.get("name", "")):
            fmha_events.append(e)

    fmha_events.sort(key=lambda x: x["ts"])
    print(f"\nFMHA events (fmhaSm100*): {len(fmha_events)}")

    if len(fmha_events) < 10:
        print("ERROR: Too few FMHA events")
        sys.exit(1)

    # Show first/last few
    print(f"\nFirst 5 FMHA events:")
    for i, e in enumerate(fmha_events[:5]):
        rel = (e["ts"] - fmha_events[0]["ts"]) / 1e6
        print(f"  [{i:>5d}] ts={e['ts']}  rel={rel:.6f}s  dur={e['dur']:.1f}μs  tid={e['tid']}")

    print(f"\nLast 5 FMHA events:")
    for i in range(max(0, len(fmha_events)-5), len(fmha_events)):
        e = fmha_events[i]
        rel = (e["ts"] - fmha_events[0]["ts"]) / 1e6
        print(f"  [{i:>5d}] ts={e['ts']}  rel={rel:.6f}s  dur={e['dur']:.1f}μs  tid={e['tid']}")

    # Compute inter-FMHA intervals (start-to-start)
    intervals = []
    for i in range(1, len(fmha_events)):
        dt = fmha_events[i]["ts"] - fmha_events[i-1]["ts"]
        intervals.append(dt)

    print(f"\nTotal inter-FMHA intervals: {len(intervals)}")
    print(f"All intervals stats: avg={sum(intervals)/len(intervals):.1f}μs  median={median(intervals):.1f}μs  min={min(intervals):.1f}μs  max={max(intervals):.1f}μs  std={stdev(intervals):.1f}μs")

    # Histogram: detect decode iterations (large gaps between prefill/decode)
    # and layer intervals (small, consistent gaps within decode)
    sorted_intervals = sorted(intervals)
    p10 = sorted_intervals[len(sorted_intervals)//10]
    p50 = sorted_intervals[len(sorted_intervals)//2]
    p90 = sorted_intervals[9*len(sorted_intervals)//10]
    p99 = sorted_intervals[99*len(sorted_intervals)//100]
    print(f"Percentiles: p10={p10:.1f}  p50={p50:.1f}  p90={p90:.1f}  p99={p99:.1f}")

    # Detect large gaps (iteration boundaries: >5x median)
    large_gap_threshold = p50 * 5
    large_gaps = [(i, dt) for i, dt in enumerate(intervals) if dt > large_gap_threshold]
    print(f"\nLarge gaps (>{large_gap_threshold:.0f}μs, likely iteration boundaries): {len(large_gaps)}")
    for idx, dt in large_gaps[:10]:
        print(f"  interval[{idx}]: {dt:.1f}μs ({dt/1000:.2f}ms) — between FMHA[{idx}] and FMHA[{idx+1}]")
    if len(large_gaps) > 10:
        print(f"  ... and {len(large_gaps)-10} more")

    # Estimate layers per decode iteration
    if large_gaps:
        gap_positions = [g[0] for g in large_gaps]
        # First segment: 0 to first gap
        segment_lengths = [gap_positions[0]]
        for i in range(1, len(gap_positions)):
            segment_lengths.append(gap_positions[i] - gap_positions[i-1])
        # Last segment
        segment_lengths.append(len(intervals) - gap_positions[-1])
        print(f"\nFMHA events per decode iteration (segments between large gaps):")
        print(f"  Segments: {len(segment_lengths)}")
        if segment_lengths:
            print(f"  Avg: {sum(segment_lengths)/len(segment_lengths):.1f}  Min: {min(segment_lengths)}  Max: {max(segment_lengths)}")
            # Show first few
            for i, sl in enumerate(segment_lengths[:10]):
                print(f"  Iteration {i}: {sl} FMHA events (layers)")
            if len(segment_lengths) > 10:
                print(f"  ... and {len(segment_lengths)-10} more")

    # Select stable-state layers: exclude large gaps, take middle portion
    normal_intervals = [(i, dt) for i, dt in enumerate(intervals) if dt <= large_gap_threshold]
    print(f"\nNormal intervals (layer-to-layer): {len(normal_intervals)}")

    if len(normal_intervals) < args.layers:
        print(f"WARNING: Only {len(normal_intervals)} normal intervals, using all")
        selected = [dt for _, dt in normal_intervals]
    else:
        # Skip first 10% as warmup, take from middle
        skip = len(normal_intervals) // 10
        selected = [dt for _, dt in normal_intervals[skip:skip + args.layers]]
        print(f"Selected {len(selected)} stable layers (skipping first {skip})")

    avg_layer = sum(selected) / len(selected)
    med_layer = median(selected)
    std_layer = stdev(selected) if len(selected) > 1 else 0
    min_layer = min(selected)
    max_layer = max(selected)

    print(f"\n{'='*60}")
    print(f"STABLE-STATE PER-LAYER TIME (FMHA-to-FMHA)")
    print(f"{'='*60}")
    print(f"  Samples:  {len(selected)}")
    print(f"  Average:  {avg_layer:.1f} μs ({avg_layer/1000:.3f} ms)")
    print(f"  Median:   {med_layer:.1f} μs ({med_layer/1000:.3f} ms)")
    print(f"  Std:      {std_layer:.1f} μs")
    print(f"  Min:      {min_layer:.1f} μs  Max: {max_layer:.1f} μs")
    print(f"  Range:    {max_layer-min_layer:.1f} μs ({(max_layer-min_layer)/avg_layer*100:.1f}%)")

    print(f"\n  === FULL DECODE ESTIMATE ===")
    est_decode = avg_layer * args.model_layers
    print(f"  {avg_layer:.1f} μs × {args.model_layers} layers = {est_decode:.1f} μs = {est_decode/1000:.2f} ms")
    print(f"  (median: {med_layer:.1f} × {args.model_layers} = {med_layer*args.model_layers/1000:.2f} ms)")

    # Compare with cudaGraphLaunch-based decode time
    launches = [e for e in events if "cudaGraphLaunch" in e.get("name", "") and e.get("ph") == "X"]
    launches.sort(key=lambda x: x["ts"])
    if launches:
        launch_durs = [l["dur"] for l in launches]
        launch_gaps = [launches[i+1]["ts"] - launches[i]["ts"] for i in range(len(launches)-1)]
        avg_launch_gap = sum(launch_gaps) / len(launch_gaps) if launch_gaps else 0
        print(f"\n  === DECODE ITERATION REFERENCE ===")
        print(f"  cudaGraphLaunch events: {len(launches)}")
        print(f"  Avg inter-launch gap: {avg_launch_gap:.1f} μs ({avg_launch_gap/1000:.2f} ms)")
        print(f"  Avg launch dur: {sum(launch_durs)/len(launch_durs):.1f} μs ({sum(launch_durs)/len(launch_durs)/1000:.2f} ms)")

        # Wall-clock decode step time from GPU kernels
        gpu_kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel" and e.get("pid") == gpu_pid]
        gpu_kernels.sort(key=lambda x: x["ts"])
        if gpu_kernels and len(launches) > 15:
            # Use middle launches for stable state
            mid = len(launches) // 2
            sample_launches = launches[mid-5:mid+5]
            step_walls = []
            for li in range(len(sample_launches)-1):
                l_start = sample_launches[li]["ts"]
                l_end = sample_launches[li+1]["ts"]
                step_kernels = [k for k in gpu_kernels if k["ts"] >= l_start and k["ts"] < l_end]
                if step_kernels:
                    wall_start = min(k["ts"] for k in step_kernels)
                    wall_end = max(k["ts"] + k["dur"] for k in step_kernels)
                    step_walls.append(wall_end - wall_start)
            if step_walls:
                avg_step_wall = sum(step_walls) / len(step_walls)
                print(f"  Avg GPU wall-clock per decode step (mid-trace sample): {avg_step_wall:.1f} μs ({avg_step_wall/1000:.2f} ms)")
                print(f"  Ratio (estimate/actual): {est_decode/avg_step_wall:.3f}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
