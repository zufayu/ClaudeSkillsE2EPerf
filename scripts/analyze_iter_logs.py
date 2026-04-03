#!/usr/bin/env python3
"""Analyze server iter logs for prefill/decode interleaving statistics."""

import re
import sys
import os
from collections import Counter


def analyze_log(log_path):
    """Parse a server log and return prefill/decode/mixed iteration stats."""
    prefill_only = 0
    decode_only = 0
    mixed = 0
    empty = 0
    prefill_tokens_list = []
    gen_tokens_list = []
    step_times_prefill = []
    step_times_decode = []
    step_times_mixed = []

    with open(log_path) as f:
        for line in f:
            m = re.search(
                r"num_ctx_requests': (\d+), 'num_ctx_tokens': (\d+), 'num_generation_tokens': (\d+)",
                line,
            )
            if not m:
                continue
            ctx_req = int(m.group(1))
            ctx_tok = int(m.group(2))
            gen_tok = int(m.group(3))

            tm = re.search(r"host_step_time = ([\d.]+)", line)
            step_ms = float(tm.group(1)) if tm else 0

            if ctx_req > 0 and gen_tok > 0:
                mixed += 1
                step_times_mixed.append(step_ms)
            elif ctx_req > 0:
                prefill_only += 1
                prefill_tokens_list.append(ctx_tok)
                step_times_prefill.append(step_ms)
            elif gen_tok > 0:
                decode_only += 1
                gen_tokens_list.append(gen_tok)
                step_times_decode.append(step_ms)
            else:
                empty += 1

    total = prefill_only + decode_only + mixed + empty
    if total == 0:
        print("  No iter data found")
        return

    print(f"  Total iters:    {total}")
    print(f"  Pure prefill:   {prefill_only} ({prefill_only/total*100:.1f}%)")
    print(f"  Pure decode:    {decode_only} ({decode_only/total*100:.1f}%)")
    print(f"  Mixed (overlap):{mixed} ({mixed/total*100:.1f}%)")
    print(f"  Empty:          {empty}")

    if step_times_decode:
        avg_decode = sum(step_times_decode) / len(step_times_decode)
        steady = [t for t in step_times_decode if t < 100]
        avg_steady = sum(steady) / len(steady) if steady else 0
        print(f"  Decode step (all): {avg_decode:.2f}ms avg")
        print(
            f"  Decode step (steady <100ms): {avg_steady:.2f}ms avg (n={len(steady)})"
        )

    if step_times_prefill:
        warm = [t for t in step_times_prefill if t < 500]
        if warm:
            print(
                f"  Prefill step (warm <500ms): {sum(warm)/len(warm):.2f}ms avg (n={len(warm)})"
            )

    if step_times_mixed:
        warm = [t for t in step_times_mixed if t < 500]
        if warm:
            print(
                f"  Mixed step (warm <500ms): {sum(warm)/len(warm):.2f}ms avg (n={len(warm)})"
            )

    if gen_tokens_list:
        gc = Counter(gen_tokens_list)
        top3 = gc.most_common(3)
        print(f"  Gen tokens distribution: {top3}")
    print()


def main():
    search_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print("=" * 60)
    print("  Prefill Interleaving Analysis from Server Iter Logs")
    print("=" * 60)
    print()

    found = False
    for dirpath, dirnames, filenames in sorted(os.walk(search_dir)):
        dirname = os.path.basename(dirpath)
        if not dirname.startswith("results_b200"):
            continue
        for fname in sorted(filenames):
            if not fname.startswith("server_") or not fname.endswith(".log"):
                continue
            if "trimmed" in fname:
                continue
            log_path = os.path.join(dirpath, fname)
            lines = sum(1 for _ in open(log_path))
            if lines < 100:
                continue
            basename = fname.replace(".log", "")
            print(f"=== {dirname} / {basename} ({lines} lines) ===")
            analyze_log(log_path)
            found = True

    if not found:
        print("No server logs with >100 lines found.")


if __name__ == "__main__":
    main()
