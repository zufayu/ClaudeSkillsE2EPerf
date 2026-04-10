#!/usr/bin/env python3
"""
Offline inference script for ncu profiling.

Loads model via offline engine (SGLang or TRT-LLM), warms up to
steady state, then uses cudaProfilerStart/Stop to precisely capture
one decode iteration for ncu analysis.

Usage with ncu:
    ncu --profile-from-start off --set full --graph-profiling node \
        --target-processes all -o ncu_report \
        python scripts/ncu_infer.py \
        --backend sglang \
        --model /path/to/model --tp 8 --ep 8 \
        --quantization modelopt_fp4 \
        --warmup-prompts 5 --isl 1024 --osl 64

Then open ncu_report.ncu-rep in Nsight Compute GUI.
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Offline inference for ncu profiling")
    parser.add_argument("--backend", default="sglang", choices=["sglang", "trtllm"],
                        help="Inference backend (default: sglang)")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--ep", type=int, default=4, help="Expert parallel size")
    parser.add_argument("--quantization", default=None, help="Quantization (e.g. modelopt_fp4)")
    parser.add_argument("--warmup-prompts", type=int, default=5, help="Number of warmup prompts")
    parser.add_argument("--isl", type=int, default=1024, help="Input sequence length")
    parser.add_argument("--osl", type=int, default=64, help="Output sequence length (keep short for ncu)")
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--chunked-prefill-size", type=int, default=16384)
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=256)
    parser.add_argument("--max-running-requests", type=int, default=256)
    args = parser.parse_args()

    import torch

    # Build prompt of target ISL length (approximate with repeated text)
    base_text = "Explain the architecture of modern large language models in detail. "
    prompt_text = (base_text * (args.isl // 10 + 1))[:args.isl * 4]  # ~4 chars/token

    if args.backend == "sglang":
        engine, generate_fn, shutdown_fn = _init_sglang(args, prompt_text)
    elif args.backend == "trtllm":
        engine, generate_fn, shutdown_fn = _init_trtllm(args, prompt_text)

    # Warmup: run several prompts to reach steady state (CUDA graph capture + warm caches)
    print(f"\nWarmup: {args.warmup_prompts} prompts (ISL~{args.isl}, OSL={args.osl})...")
    for i in range(args.warmup_prompts):
        n_tokens = generate_fn()
        print(f"  warmup {i+1}/{args.warmup_prompts}: {n_tokens} tokens")

    torch.cuda.synchronize()

    # Profile: one inference pass
    print(f"\n=== cudaProfilerStart === (ncu capture begins)")
    torch.cuda.cudart().cudaProfilerStart()

    n_tokens = generate_fn()

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"=== cudaProfilerStop === (ncu capture ends)")
    print(f"  Profiled: {n_tokens} tokens generated")

    shutdown_fn()
    print("Done.")


def _init_sglang(args, prompt_text):
    """Initialize SGLang offline engine."""
    import sglang as sgl

    sampling_params = {"max_new_tokens": args.osl, "temperature": 0}

    engine_kwargs = {
        "model_path": args.model,
        "tp_size": args.tp,
        "dp_size": 1,
        "mem_fraction_static": args.mem_fraction_static,
        "chunked_prefill_size": args.chunked_prefill_size,
        "kv_cache_dtype": args.kv_cache_dtype,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "max_running_requests": args.max_running_requests,
        "trust_remote_code": True,
        "log_level": "warning",
    }
    if args.ep > 1:
        engine_kwargs["ep_size"] = args.ep
    if args.quantization:
        engine_kwargs["quantization"] = args.quantization

    print(f"Loading SGLang model: {args.model}")
    print(f"  TP={args.tp} EP={args.ep} quant={args.quantization}")
    t0 = time.time()
    engine = sgl.Engine(**engine_kwargs)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    def generate_fn():
        out = engine.generate(prompt_text, sampling_params)
        return len(out.get("text", "").split())

    return engine, generate_fn, engine.shutdown


def _init_trtllm(args, prompt_text):
    """Initialize TRT-LLM offline engine."""
    from tensorrt_llm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(max_tokens=args.osl, temperature=0)

    print(f"Loading TRT-LLM model: {args.model}")
    print(f"  TP={args.tp} quant={args.quantization}")

    def generate_fn():
        outputs = llm.generate([prompt_text], sampling_params=sampling_params)
        return sum(len(o.outputs[0].token_ids) for o in outputs)

    def shutdown_fn():
        pass  # TRT-LLM LLM cleans up on del

    return llm, generate_fn, shutdown_fn


if __name__ == "__main__":
    main()
