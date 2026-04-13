#!/usr/bin/env python3
"""
Inference script for ncu/nsys profiling.

Supports two modes:
  - offline: Load model via offline engine, run warmup + inference (BS=1)
  - serve:   Launch server, warmup with concurrent requests, run benchmark

Usage with ncu (offline):
    ncu --set full --graph-profiling node --target-processes all \
        -o ncu_report \
        python scripts/ncu_infer.py \
        --backend sglang --mode offline \
        --model /path/to/model --tp 8 --ep 8 \
        --warmup-prompts 5 --isl 1024 --osl 64

Usage with ncu (serve, concurrent):
    ncu --set full --graph-profiling node --target-processes all \
        -o ncu_report \
        python scripts/ncu_infer.py \
        --backend sglang --mode serve \
        --model /path/to/model --tp 8 --ep 8 \
        --concurrency 64 --scenario chat \
        --warmup-prompts 128 --num-prompts 640
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Inference for ncu/nsys profiling")
    parser.add_argument("--backend", default="sglang", choices=["sglang", "trtllm"])
    parser.add_argument("--mode", default="offline", choices=["offline", "serve"])
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--warmup-prompts", type=int, default=5)
    parser.add_argument("--isl", type=int, default=1024)
    parser.add_argument("--osl", type=int, default=64)
    # Serve mode args
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--scenario", default="chat", choices=["chat", "reasoning", "summarize"])
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--num-prompts", type=int, default=0, help="Benchmark prompts (default: concurrency*10)")
    parser.add_argument("--bench-only", action="store_true", help="Serve mode: skip server launch, assume server is already running")
    parser.add_argument("--skip-warmup", action="store_true", help="Serve mode: skip warmup phase")
    # SGLang server params
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--chunked-prefill-size", type=int, default=16384)
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=256)
    parser.add_argument("--max-running-requests", type=int, default=256)
    args = parser.parse_args()

    if args.mode == "offline":
        run_offline(args)
    elif args.mode == "serve":
        run_serve(args)


def run_offline(args):
    """Offline mode: load model, warmup, inference (original behavior)."""
    import torch

    base_text = "Explain the architecture of modern large language models in detail. "
    prompt_text = (base_text * (args.isl // 10 + 1))[:args.isl * 4]

    if args.backend == "sglang":
        engine, generate_fn, shutdown_fn = _init_sglang(args, prompt_text)
    elif args.backend == "trtllm":
        engine, generate_fn, shutdown_fn = _init_trtllm(args, prompt_text)

    print(f"\nWarmup: {args.warmup_prompts} prompts (ISL~{args.isl}, OSL={args.osl})...")
    for i in range(args.warmup_prompts):
        n_tokens = generate_fn()
        print(f"  warmup {i+1}/{args.warmup_prompts}: {n_tokens} tokens")

    torch.cuda.synchronize()

    print(f"\n=== Profiled inference pass ===")
    n_tokens = generate_fn()
    torch.cuda.synchronize()
    print(f"  Profiled: {n_tokens} tokens generated")

    shutdown_fn()
    print("Done.")


def run_serve(args):
    """Serve mode: launch server, warmup + benchmark with concurrent requests.

    With --bench-only: assume server is already running, skip launch/shutdown.
    This is critical for ncu profiling — ncu wraps only the benchmark client
    while the server runs outside ncu (avoiding massive startup overhead).

    With --skip-warmup: skip warmup phase (server already warmed up externally).
    """
    scenario_map = {
        "chat": (1024, 1024),
        "reasoning": (1024, 8192),
        "summarize": (8192, 1024),
    }
    isl, osl = scenario_map[args.scenario]
    num_prompts = args.num_prompts if args.num_prompts > 0 else args.concurrency * 10
    warmup_prompts = args.warmup_prompts if args.warmup_prompts > 0 else args.concurrency * 2

    print(f"=== Serve mode: {args.backend} ===")
    print(f"  Scenario: {args.scenario} (ISL={isl}, OSL={osl})")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Bench-only: {args.bench_only}")
    print(f"  Skip-warmup: {args.skip_warmup}")
    print(f"  Warmup: {warmup_prompts} prompts")
    print(f"  Benchmark: {num_prompts} prompts")

    server_proc = None

    if args.bench_only:
        # Server already running externally — just verify it's healthy
        print(f"\n=== bench-only mode: checking server on port {args.port} ===")
        _wait_for_server(args.port, timeout=60)
    else:
        # Launch server (may be under ncu — slow!)
        server_proc = _launch_server(args)
        _wait_for_server(args.port, timeout=1200)

    try:
        if not args.skip_warmup:
            print(f"\n=== Warmup ({warmup_prompts} prompts, c={args.concurrency}) ===")
            _run_benchmark_serving(args, isl, osl, warmup_prompts, tag="warmup")

        # Benchmark (this is what ncu captures during steady state)
        print(f"\n=== Benchmark ({num_prompts} prompts, c={args.concurrency}) ===")
        _run_benchmark_serving(args, isl, osl, num_prompts, tag="profiled")

        print("\n=== Benchmark complete ===")
    finally:
        if server_proc is not None:
            # Kill server only if we launched it
            if server_proc.poll() is None:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server_proc.kill()
                    server_proc.wait()
            subprocess.run(["pkill", "-f", "sglang.launch_server"], capture_output=True)
            subprocess.run(["pkill", "-f", "trtllm-serve"], capture_output=True)
            time.sleep(3)
        else:
            print("  (bench-only mode: server left running)")

    print("Done.")


def _launch_server(args):
    """Launch SGLang or TRT-LLM server as subprocess."""
    if args.backend == "sglang":
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", args.model,
            "--host", "0.0.0.0",
            "--port", str(args.port),
            "--trust-remote-code",
            "--tensor-parallel-size", str(args.tp),
            "--data-parallel-size", "1",
            "--cuda-graph-max-bs", str(args.cuda_graph_max_bs),
            "--max-running-requests", str(args.max_running_requests),
            "--mem-fraction-static", str(args.mem_fraction_static),
            "--kv-cache-dtype", args.kv_cache_dtype,
            "--chunked-prefill-size", str(args.chunked_prefill_size),
            "--ep-size", str(args.ep),
            "--enable-flashinfer-allreduce-fusion",
            "--enable-symm-mem",
            "--disable-radix-cache",
            "--attention-backend", "trtllm_mla",
            "--moe-runner-backend", "flashinfer_trtllm",
            "--stream-interval", "10",
        ]
        if args.quantization:
            cmd += ["--quantization", args.quantization]
    elif args.backend == "trtllm":
        cmd = [
            "trtllm-serve", args.model,
            "--port", str(args.port),
            "--trust_remote_code",
            "--backend", "pytorch",
            "--tp_size", str(args.tp),
            "--ep_size", str(args.ep),
        ]

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"

    print(f"Launching server: {' '.join(cmd[:6])}...")
    server_proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return server_proc


def _wait_for_server(port, timeout=1200):
    """Wait for server to be ready via health endpoint.

    Under ncu/nsys profiling, server startup takes 10-20x longer
    due to instrumentation overhead. Default timeout is 1200s (20min)
    but can be overridden via NCU_SERVER_TIMEOUT env var.
    """
    import urllib.request
    import urllib.error

    timeout = int(os.environ.get("NCU_SERVER_TIMEOUT", timeout))
    url = f"http://localhost:{port}/health"
    start = time.time()
    last_print = 0

    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                elapsed = time.time() - start
                print(f"  Server ready in {elapsed:.0f}s")
                return
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass

        elapsed = time.time() - start
        if elapsed - last_print >= 30:
            print(f"  Waiting for server... ({elapsed:.0f}s)")
            last_print = elapsed
        time.sleep(5)

    raise TimeoutError(f"Server not ready after {timeout}s")


def _run_benchmark_serving(args, isl, osl, num_prompts, tag="bench"):
    """Run benchmark_serving.py as subprocess."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Use benchmark_serving from the repo (InferenceX version) or sglang built-in
    bench_script = os.path.join(script_dir, "benchmark_serving.py")
    use_sglang_builtin = not os.path.exists(bench_script)
    if use_sglang_builtin:
        bench_script = "-m"  # will use sys.executable -m sglang.bench_serving

    if use_sglang_builtin:
        # SGLang built-in bench_serving uses different arg names
        cmd = [
            sys.executable, "-m", "sglang.bench_serving",
            "--model", args.model,
            "--port", str(args.port),
            "--backend", "vllm" if args.backend == "sglang" else "openai",
            "--dataset-name", "random",
            "--random-input-len", str(isl),
            "--random-output-len", str(osl),
            "--random-range-ratio", "0.8",
            "--num-prompts", str(num_prompts),
            "--max-concurrency", str(args.concurrency),
            "--warmup-requests", "0",
            "--output-file", f"/tmp/ncu_{tag}.jsonl",
        ]
    else:
        # InferenceX benchmark_serving.py
        cmd = [
            sys.executable, bench_script,
            "--model", args.model,
            "--port", str(args.port),
            "--backend", "vllm" if args.backend == "sglang" else "openai",
            "--input-len", str(isl),
            "--output-len", str(osl),
            "--random-range-ratio", "0.8",
            "--num-prompts", str(num_prompts),
            "--max-concurrency", str(args.concurrency),
            "--num-warmups", "0",
            "--result-filename", f"ncu_{tag}",
            "--result-dir", "/tmp",
        ]

    print(f"  Running: {os.path.basename(bench_script)} --num-prompts {num_prompts} --max-concurrency {args.concurrency}")
    result = subprocess.run(cmd, capture_output=False, timeout=3600)
    if result.returncode != 0:
        print(f"  WARNING: benchmark_serving exited with rc={result.returncode}")


# ======================== Offline backends ====================================

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
        pass

    return llm, generate_fn, shutdown_fn


if __name__ == "__main__":
    main()
