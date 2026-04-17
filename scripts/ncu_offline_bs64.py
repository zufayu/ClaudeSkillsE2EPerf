#!/usr/bin/env python3
"""Offline bs=64 inference for NCU profiling.

Wraps with: ncu --profile-from-start off -k "fmhaSm100f" --launch-count 10 ...

Uses sglang.Engine offline API to load model once, warmup, then
profile decode phase with cudaProfilerStart/Stop.
"""

import argparse
import torch
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2, help="warmup rounds")
    args = parser.parse_args()

    import sglang as sgl

    print(f"Loading model: {args.model}, TP={args.tp}")
    engine = sgl.Engine(
        model_path=args.model,
        tp_size=args.tp,
        mem_fraction_static=0.85,
        chunked_prefill_size=16384,
        kv_cache_dtype="fp8_e4m3",
    )

    # Build prompts
    prompt = "The quick brown fox " * (args.input_len // 5)
    prompts = [prompt] * args.bs

    # Warmup
    print(f"Warmup: {args.warmup} rounds of bs={args.bs}...")
    for i in range(args.warmup):
        t0 = time.time()
        engine.generate(
            prompts,
            sampling_params={"max_new_tokens": args.output_len, "temperature": 0},
        )
        print(f"  warmup {i+1}: {time.time()-t0:.1f}s")

    torch.cuda.synchronize()
    print("Warmup done. Starting profiled inference...")

    # Profile
    torch.cuda.cudart().cudaProfilerStart()
    print("cudaProfilerStart() called")

    t0 = time.time()
    engine.generate(
        prompts,
        sampling_params={"max_new_tokens": args.output_len, "temperature": 0},
    )
    elapsed = time.time() - t0
    print(f"Profiled inference: {elapsed:.1f}s, bs={args.bs}, output_len={args.output_len}")

    torch.cuda.cudart().cudaProfilerStop()
    print("cudaProfilerStop() called")

    engine.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
