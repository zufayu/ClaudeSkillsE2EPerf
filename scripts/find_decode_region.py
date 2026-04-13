#!/usr/bin/env python3
"""
Find steady-state decode region from nsys trace.

Strategy:
  1. Count total inference kernels → take middle
  2. Detect repeating pattern at middle → kernels_per_decode
  3. Validate: sum(one decode iteration duration) ≈ TPOT
  4. Output --launch-skip for ncu

Model-agnostic, concurrency-agnostic. Works because:
  - Beginning = warmup/prefill/ramp-up (unstable)
  - Middle = steady-state decode (what we want)
  - End = ramp-down (unstable)

Scalable: for large traces (c=64 serving → millions of kernels),
uses COUNT + LIMIT/OFFSET to only load a window around the middle.

Usage:
    python3 scripts/find_decode_region.py \\
        --nsys-rep trace.nsys-rep \\
        [--tpot-ms 5.2] \\
        [--kernel-regex "nvjet|fmha|..."] \\
        [--launch-count 50] \\
        [--sample-window 10000]
"""

import argparse
import os
import re
import sqlite3
import subprocess
import sys


DEFAULT_KERNEL_REGEX = (
    "nvjet|fmha|cutlass|flash_attn|kernel_mha|allreduce|"
    "reduce_scatter|all_gather|nccl|deep_gemm"
)

SAMPLE_THRESHOLD = 50000  # above this, use windowed sampling


def ensure_sqlite(nsys_rep):
    """Export nsys-rep to sqlite if needed."""
    db = nsys_rep.replace(".nsys-rep", ".sqlite")
    if not os.path.exists(db):
        print(f"  Exporting nsys to sqlite...")
        subprocess.run(
            ["nsys", "stats", nsys_rep, "--report", "cuda_gpu_kern_sum",
             "--format", "csv", "-o", "/dev/null"],
            capture_output=True, timeout=600
        )
    if not os.path.exists(db):
        print(f"ERROR: could not create {db}")
        sys.exit(1)
    return db


def _build_where_clause(kernel_regex):
    """Build SQL WHERE clause from kernel regex."""
    parts = kernel_regex.split("|")
    return " OR ".join([f"s.value LIKE '%{p}%'" for p in parts])


def count_kernels(db_path, kernel_regex):
    """Count total matching kernels without loading them."""
    conn = sqlite3.connect(db_path)
    where = _build_where_clause(kernel_regex)
    cur = conn.execute(f"""
        SELECT COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        WHERE {where}
    """)
    n = cur.fetchone()[0]
    conn.close()
    return n


def extract_kernels(db_path, kernel_regex, limit=None, offset=0):
    """Extract inference kernels ordered by launch time.

    Args:
        limit: max rows to fetch (None = all)
        offset: skip first N rows
    """
    conn = sqlite3.connect(db_path)
    where = _build_where_clause(kernel_regex)

    sql = f"""
        SELECT s.value, k.start, k.end, (k.end - k.start) as dur
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        WHERE {where}
        ORDER BY k.start
    """
    if limit is not None:
        sql += f" LIMIT {limit} OFFSET {offset}"

    cur = conn.execute(sql)
    rows = cur.fetchall()
    conn.close()
    return rows


def short_name(name):
    name = re.sub(r'<[^>]*>', '', name)
    name = re.sub(r'\([^)]*\)', '', name)
    parts = name.split('::')
    return parts[-1].strip().replace('void ', '')


def detect_pattern_at(names, pos, max_plen=20, min_repeats=3):
    """Detect smallest repeating kernel pattern (= one transformer layer)."""
    for plen in range(3, max_plen + 1):
        if pos + plen * min_repeats > len(names):
            continue
        pattern = names[pos:pos + plen]
        repeats = 0
        for i in range(pos, len(names) - plen + 1, plen):
            if names[i:i + plen] == pattern:
                repeats += 1
            else:
                break
        if repeats >= min_repeats:
            return pattern, plen, repeats
    return None, 0, 0


def detect_decode_pass_length(names, pos, layer_plen, layer_repeats):
    """Detect full decode pass = N_layers × kernels_per_layer.

    One decode pass processes all transformer layers. The layer pattern
    repeats N_layers times within one pass, then the next pass starts
    with the same pattern again.

    Returns kernels_per_decode_pass and number of full passes found.
    """
    if layer_plen == 0:
        return 0, 0

    # The layer pattern repeats layer_repeats times continuously.
    # But we need to find where one full decode pass ends and the next begins.
    # Try multiples of layer_plen as candidate decode pass lengths.
    best_pass_len = layer_plen * layer_repeats  # fallback: all repeats = 1 pass
    best_passes = 1

    for n_layers in range(layer_repeats, 0, -1):
        pass_len = layer_plen * n_layers
        if pos + pass_len * 2 > len(names):
            continue
        # Check if this pass_len repeats
        pass_block = names[pos:pos + pass_len]
        passes = 0
        for i in range(pos, len(names) - pass_len + 1, pass_len):
            if names[i:i + pass_len] == pass_block:
                passes += 1
            else:
                break
        if passes >= 2:
            best_pass_len = pass_len
            best_passes = passes
            break

    return best_pass_len, best_passes


def main():
    p = argparse.ArgumentParser(description="Find steady-state decode region")
    p.add_argument("--nsys-rep", required=True, help="Path to .nsys-rep file")
    p.add_argument("--tpot-ms", type=float, default=None,
                    help="Expected TPOT in ms (for validation)")
    p.add_argument("--kernel-regex", default=DEFAULT_KERNEL_REGEX)
    p.add_argument("--launch-count", type=int, default=50,
                    help="Desired number of kernels to capture")
    p.add_argument("--sample-window", type=int, default=0,
                    help="Window size for large traces (0=auto)")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    args = p.parse_args()

    db = ensure_sqlite(args.nsys_rep)

    # Step 0: count total kernels (cheap query, no data transfer)
    N = count_kernels(db, args.kernel_regex)
    print(f"Total inference kernels: {N}")

    if N == 0:
        print("ERROR: no inference kernels found")
        sys.exit(1)

    # Determine if we need windowed sampling
    global_offset = 0  # offset into the full kernel list
    if N > SAMPLE_THRESHOLD:
        window = args.sample_window if args.sample_window > 0 else max(10000, args.launch_count * 20)
        window = min(window, N)  # can't be larger than total
        global_offset = max(0, N // 2 - window // 2)
        print(f"Large trace detected ({N} kernels > {SAMPLE_THRESHOLD})")
        print(f"  Sampling window: [{global_offset}..{global_offset + window}] ({window} kernels)")
        rows = extract_kernels(db, args.kernel_regex, limit=window, offset=global_offset)
    else:
        rows = extract_kernels(db, args.kernel_regex)

    local_N = len(rows)
    names = [short_name(r[0]) for r in rows]
    durs_ns = [r[3] for r in rows]

    # Step 1: take middle of the LOCAL window
    mid = local_N // 2
    print(f"Middle index: {mid} (global: {mid + global_offset})")

    # Step 2: detect layer pattern at middle
    best_pattern = None
    best_plen = 0
    best_repeats = 0
    best_start = mid

    for offset in range(20):
        pos = mid - offset
        if pos < 0:
            continue
        pattern, plen, repeats = detect_pattern_at(names, pos)
        if repeats > best_repeats or (repeats == best_repeats and plen < best_plen):
            best_pattern = pattern
            best_plen = plen
            best_repeats = repeats
            best_start = pos

    if best_pattern is None:
        print("WARNING: no repeating pattern found at middle")
        print("  Falling back to launch-skip = mid")
        launch_skip = mid + global_offset
        kernels_per_layer = 0
        kernels_per_decode = 0
        n_layers = 0
    else:
        kernels_per_layer = best_plen
        print(f"Layer pattern: {kernels_per_layer} kernels/layer, "
              f"repeats {best_repeats}x at index {best_start}")
        for i, k in enumerate(best_pattern):
            print(f"  [{i}] {k}")

        # Detect full decode pass (all layers)
        decode_pass_len, n_decode_passes = detect_decode_pass_length(
            names, best_start, kernels_per_layer, best_repeats)
        n_layers = decode_pass_len // kernels_per_layer
        kernels_per_decode = decode_pass_len
        print(f"\nDecode pass: {n_layers} layers × {kernels_per_layer} kernels "
              f"= {kernels_per_decode} kernels/pass, {n_decode_passes} passes found")

        # Step 2b: use duration to find STEADY-STATE DECODE
        # Key insight: prefill kernels are much longer than decode kernels
        # (prefill processes ISL tokens, decode processes 1 token)
        # Also: ramp-up decode kernels get longer as BS increases
        # Steady-state decode = consistent short duration
        print(f"\nDuration analysis (per {kernels_per_decode}-kernel pass):")
        pass_durations = []
        for i in range(0, local_N - kernels_per_decode + 1, kernels_per_decode):
            dur = sum(durs_ns[i:i + kernels_per_decode]) / 1e6
            pass_durations.append((i, dur))

        if len(pass_durations) >= 3:
            # Find the median duration (robust to outliers from prefill/ramp)
            sorted_durs = sorted(d for _, d in pass_durations)
            median_dur = sorted_durs[len(sorted_durs) // 2]

            # Steady-state decode = passes with duration close to median
            # (prefill passes are >> median, warmup passes may be < median)
            tolerance = 0.5  # within 50% of median
            steady_passes = [(idx, dur) for idx, dur in pass_durations
                             if abs(dur - median_dur) / median_dur < tolerance]

            if steady_passes:
                steady_start = steady_passes[0][0]
                steady_end = steady_passes[-1][0] + kernels_per_decode
                n_steady = len(steady_passes)
                print(f"  Total passes: {len(pass_durations)}")
                print(f"  Median duration: {median_dur:.3f} ms")
                print(f"  Steady-state passes: {n_steady} "
                      f"(index {steady_start}..{steady_end})")

                # Show duration distribution
                short_passes = sum(1 for _, d in pass_durations if d < median_dur * 0.5)
                long_passes = sum(1 for _, d in pass_durations if d > median_dur * 1.5)
                print(f"  Short (<0.5x median, warmup): {short_passes}")
                print(f"  Normal (steady decode): {n_steady}")
                print(f"  Long (>1.5x median, prefill): {long_passes}")

                # Center capture in steady-state region
                steady_mid_idx = len(steady_passes) // 2
                passes_to_capture = max(1, args.launch_count // kernels_per_decode)
                center_pass_idx = steady_mid_idx
                start_pass_idx = max(0, center_pass_idx - passes_to_capture // 2)
                launch_skip = steady_passes[start_pass_idx][0] + global_offset

                print(f"\n  Targeting middle of steady-state region:")
                print(f"    Pass duration at target: {steady_passes[start_pass_idx][1]:.3f} ms")
            else:
                print(f"  WARNING: no steady-state passes found")
                launch_skip = best_start + global_offset
        else:
            # Not enough passes, fall back to pattern-based approach
            pattern_start = best_start
            passes_to_mid = (mid - pattern_start) // kernels_per_decode
            passes_to_capture = max(1, args.launch_count // kernels_per_decode)
            start_pass = max(0, passes_to_mid - passes_to_capture // 2)
            launch_skip = pattern_start + start_pass * kernels_per_decode + global_offset

    # Step 3: validate against TPOT
    decode_dur_ms = 0
    if kernels_per_decode > 0:
        # Sum duration of one full decode pass at the middle
        # Use local indices for array access
        local_skip = launch_skip - global_offset
        iter_start = local_skip + (args.launch_count // 2)
        # Align to decode pass boundary
        iter_start = local_skip + ((iter_start - local_skip) // kernels_per_decode) * kernels_per_decode
        if iter_start + kernels_per_decode <= local_N:
            decode_dur_ns = sum(durs_ns[iter_start:iter_start + kernels_per_decode])
            decode_dur_ms = decode_dur_ns / 1e6
            print(f"\nDuration of 1 full decode pass (at index {iter_start + global_offset}):")
            print(f"  {n_layers} layers × {kernels_per_layer} kernels = {kernels_per_decode} kernels")
            print(f"  Kernel time: {decode_dur_ms:.3f} ms")

            if args.tpot_ms:
                ratio = decode_dur_ms / args.tpot_ms
                status = "OK" if 0.5 < ratio < 1.5 else "MISMATCH"
                print(f"  Expected TPOT: {args.tpot_ms:.3f} ms")
                print(f"  Ratio (kernel/TPOT): {ratio:.2f}  [{status}]")
                if status == "OK":
                    print(f"  VALIDATED: captured region is steady-state decode")
                elif ratio < 0.5:
                    print(f"  WARNING: kernel time << TPOT, may have missed kernels")
                elif ratio > 1.5:
                    print(f"  WARNING: kernel time >> TPOT, may have captured prefill")
            else:
                print(f"  (pass --tpot-ms to validate against benchmark TPOT)")

    # Ensure launch_count doesn't exceed available kernels
    actual_count = min(args.launch_count, N - launch_skip)

    print(f"\n{'='*60}")
    print(f"  RESULT")
    print(f"{'='*60}")
    print(f"  --launch-skip  {launch_skip}")
    print(f"  --launch-count {actual_count}")
    if kernels_per_decode > 0:
        print(f"  kernels/layer:  {kernels_per_layer}")
        print(f"  layers/pass:    {n_layers}")
        print(f"  kernels/decode: {kernels_per_decode}")
        print(f"  decode passes:  {actual_count // kernels_per_decode}")
    if decode_dur_ms > 0:
        print(f"  decode duration: {decode_dur_ms:.3f} ms (kernel time only)")
    print(f"  region: [{launch_skip} .. {launch_skip + actual_count}] of {N}")
    print(f"{'='*60}")

    if args.json:
        import json
        result = {
            "launch_skip": launch_skip,
            "launch_count": actual_count,
            "total_kernels": N,
            "kernels_per_layer": kernels_per_layer,
            "n_layers": n_layers,
            "kernels_per_decode": kernels_per_decode,
            "pattern": best_pattern,
            "decode_duration_ms": decode_dur_ms,
        }
        print(json.dumps(result))


if __name__ == "__main__":
    main()
