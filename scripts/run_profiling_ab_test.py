#!/usr/bin/env python3
"""
MI355X Profiling A/B Test — measures profiling overhead with different settings.

Phases:
  baseline:         No profiler, pure benchmark (throughput reference)
  with-stack-true:  PyTorch profiler with_stack=True (ATOM default)
  with-stack-false: PyTorch profiler with_stack=False (reduced overhead)

Each phase:
  1. Kill residual processes
  2. Start ATOM server (with/without profiling flags)
  3. Warmup (CONC*2 prompts)
  4. Run benchmark (CONC*10 prompts, save result)
  5. Stop server

Usage:
  python3 scripts/run_profiling_ab_test.py \
    --model /path/to/model --phase baseline --tp 8 --ep \
    --concurrency 64 --scenario chat --result-dir ./results/ab_test
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="MI355X Profiling A/B Test")
    p.add_argument("--model", required=True)
    p.add_argument("--phase", required=True, choices=["baseline", "with-stack-true", "with-stack-false"])
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--ep", action="store_true")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--scenario", default="chat", choices=["chat", "reasoning", "summarize"])
    p.add_argument("--result-dir", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    return p.parse_args()


SCENARIOS = {
    "chat": (1024, 1024),
    "reasoning": (1024, 8192),
    "summarize": (8192, 1024),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cleanup(port):
    """Kill anything on the port and GPU."""
    subprocess.run(f"fuser -k -9 {port}/tcp", shell=True, capture_output=True)
    time.sleep(2)
    result = subprocess.run("fuser /dev/kfd", shell=True, capture_output=True, text=True)
    pids = result.stdout.strip()
    if pids:
        log(f"Killing GPU processes: {pids}")
        subprocess.run(f"kill -9 {pids}", shell=True, capture_output=True)
        time.sleep(2)
    subprocess.run("rm -f /dev/shm/aiter_*", shell=True, capture_output=True)
    subprocess.run("rm -rf /tmp/trace", shell=True, capture_output=True)


def start_server(args, phase):
    """Start ATOM server with appropriate profiling settings."""
    cmd = [
        "python3", "-m", "atom.entrypoints.openai_server",
        "--model", args.model,
        "--server-port", str(args.port),
        "--tensor-parallel-size", str(args.tp),
        "--max-num-seqs", "512",
        "--gpu-memory-utilization", str(args.gpu_mem_util),
        "--kv_cache_dtype", "fp8",
    ]
    if args.ep:
        cmd.append("--enable-expert-parallel")

    env = os.environ.copy()

    if phase == "baseline":
        # No profiling flags
        pass
    elif phase == "with-stack-true":
        cmd.extend(["--torch-profiler-dir", "/tmp/trace", "--mark-trace"])
        # with_stack=True is ATOM default, no env var needed
    elif phase == "with-stack-false":
        cmd.extend(["--torch-profiler-dir", "/tmp/trace", "--mark-trace"])
        env["PYTORCH_PROFILER_WITH_STACK"] = "0"

    log_file = Path(args.result_dir) / f"server_{phase}.log"
    log(f"Starting server (phase={phase})...")
    log(f"  cmd: {' '.join(cmd)}")
    if phase == "with-stack-false":
        log("  env: PYTORCH_PROFILER_WITH_STACK=0")

    f = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return proc, f


def wait_for_server(port, timeout=600):
    """Wait for server /health endpoint."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://0.0.0.0:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(5)
    return False


def get_served_model(port):
    """Get the served model name from /v1/models."""
    import urllib.request
    resp = urllib.request.urlopen(f"http://0.0.0.0:{port}/v1/models")
    data = json.loads(resp.read())
    return data["data"][0]["id"]


def run_benchmark(args, phase, served_model, is_warmup=False):
    """Run benchmark_serving."""
    isl, osl = SCENARIOS[args.scenario]
    if is_warmup:
        num_prompts = args.concurrency * 2
        tag = "warmup"
    else:
        num_prompts = args.concurrency * 10
        tag = phase

    cmd = [
        "python3", "-u", "-m", "atom.benchmarks.benchmark_serving",
        "--model", served_model,
        "--backend", "vllm",
        "--base-url", f"http://0.0.0.0:{args.port}",
        "--dataset-name", "random",
        "--random-input-len", str(isl),
        "--random-output-len", str(osl),
        "--random-range-ratio", "0.8",
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(args.concurrency),
        "--request-rate", "inf",
        "--ignore-eos",
    ]

    if not is_warmup:
        result_file = f"result_{tag}.json"
        cmd.extend([
            "--save-result",
            "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--result-dir", args.result_dir,
            "--result-filename", result_file,
        ])

    log_path = Path(args.result_dir) / f"bench_{tag}.log"
    log(f"Running {'warmup' if is_warmup else 'benchmark'} ({num_prompts} prompts)...")

    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        log(f"WARNING: benchmark exited with rc={result.returncode}")

    return result.returncode


def start_stop_profile(port, action):
    """Start or stop the profiler via HTTP."""
    import urllib.request
    try:
        urllib.request.urlopen(
            urllib.request.Request(f"http://0.0.0.0:{port}/{action}_profile", method="POST"),
            timeout=10
        )
        log(f"Profiler {action}ed")
    except Exception as e:
        log(f"WARNING: {action}_profile failed: {e}")


def stop_server(proc, log_fh):
    """Stop server gracefully."""
    if proc and proc.poll() is None:
        log("Stopping server...")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if log_fh:
        log_fh.close()


def run_phase(args, phase):
    """Run one phase of the A/B test."""
    log(f"{'='*60}")
    log(f"  PHASE: {phase}")
    log(f"{'='*60}")

    cleanup(args.port)
    proc, log_fh = start_server(args, phase)

    try:
        if not wait_for_server(args.port):
            log("ERROR: Server failed to start")
            return None
        log("Server ready")

        served_model = get_served_model(args.port)
        log(f"Served model: {served_model}")

        # Warmup
        run_benchmark(args, phase, served_model, is_warmup=True)
        log("Warmup done")

        # For profiling phases, start profiler before benchmark
        if phase != "baseline":
            start_stop_profile(args.port, "start")

        # Main benchmark
        run_benchmark(args, phase, served_model, is_warmup=False)

        # For profiling phases, stop profiler after benchmark
        if phase != "baseline":
            start_stop_profile(args.port, "stop")
            # Wait for trace flush
            log("Waiting for trace flush (30s)...")
            time.sleep(30)

        # Read result
        result_file = Path(args.result_dir) / f"result_{phase}.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            out_tput = data.get("output_throughput", 0)
            tpot = data.get("tpot_p50", data.get("median_tpot_ms", 0))
            ttft = data.get("ttft_p50", data.get("median_ttft_ms", 0))
            log(f"Result: Output Tput={out_tput:.1f}, TPOT={tpot:.1f}ms, TTFT={ttft:.1f}ms")
            return {"output_throughput": out_tput, "tpot_p50": tpot, "ttft_p50": ttft}
        else:
            log("WARNING: No result file")
            return None
    finally:
        stop_server(proc, log_fh)
        cleanup(args.port)


def generate_comparison(args, results):
    """Generate comparison markdown."""
    md_path = Path(args.result_dir) / "ab_comparison.md"
    with open(md_path, "w") as f:
        f.write("# MI355X Profiling A/B Test\n\n")
        f.write(f"Config: EP8 TP8, {args.scenario} c={args.concurrency}\n\n")
        f.write("| Phase | Output Tput | TPOT p50 (ms) | TTFT p50 (ms) | Tput OH% | TPOT OH% |\n")
        f.write("|-------|------------|---------------|---------------|----------|----------|\n")

        baseline = results.get("baseline")
        for phase in ["baseline", "with-stack-true", "with-stack-false"]:
            r = results.get(phase)
            if not r:
                f.write(f"| {phase} | FAILED | - | - | - | - |\n")
                continue
            if baseline and phase != "baseline":
                tput_oh = (r["output_throughput"] / baseline["output_throughput"] - 1) * 100
                tpot_oh = (r["tpot_p50"] / baseline["tpot_p50"] - 1) * 100
                f.write(f"| {phase} | {r['output_throughput']:.1f} | {r['tpot_p50']:.1f} | {r['ttft_p50']:.1f} | {tput_oh:+.1f}% | {tpot_oh:+.1f}% |\n")
            else:
                f.write(f"| {phase} | {r['output_throughput']:.1f} | {r['tpot_p50']:.1f} | {r['ttft_p50']:.1f} | — | — |\n")

    log(f"Comparison written to: {md_path}")
    with open(md_path) as f:
        print(f.read())


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    result = run_phase(args, args.phase)

    # If all 3 phases have results, generate comparison
    results = {}
    for phase in ["baseline", "with-stack-true", "with-stack-false"]:
        rfile = Path(args.result_dir) / f"result_{phase}.json"
        if rfile.exists():
            with open(rfile) as f:
                data = json.load(f)
            results[phase] = {
                "output_throughput": data.get("output_throughput", 0),
                "tpot_p50": data.get("tpot_p50", data.get("median_tpot_ms", 0)),
                "ttft_p50": data.get("ttft_p50", data.get("median_ttft_ms", 0)),
            }

    if len(results) == 3:
        generate_comparison(args, results)

    log("Phase complete.")


if __name__ == "__main__":
    main()
