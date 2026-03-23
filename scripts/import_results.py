#!/usr/bin/env python3
"""Import benchmark results from results_*/ directories into unified run JSON format.

Usage:
    python scripts/import_results.py \
        --results-dir ./results_b200 \
        --platform "8×B200" \
        --framework "TRT-LLM 1.2.0rc4" \
        --quantization NVFP4

    # Compare same platform under different environments:
    python scripts/import_results.py \
        --results-dir ./results_b200_docker_a \
        --platform "8×B200" --framework "TRT-LLM" --quantization FP8 \
        --env-tag "docker-v1"

    python scripts/import_results.py \
        --results-dir ./results_b200_docker_b \
        --platform "8×B200" --framework "TRT-LLM" --quantization FP8 \
        --env-tag "docker-v2"

    # This produces two series in the dashboard:
    #   8×B200 DeepSeek-R1-0528 FP8 (TRT-LLM) [docker-v1]
    #   8×B200 DeepSeek-R1-0528 FP8 (TRT-LLM) [docker-v2]
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime


def parse_result_filename(filename):
    """Parse result_fp4_throughput_chat_ep1_c64.json into components."""
    base = os.path.basename(filename).replace("result_", "").replace(".json", "")
    parts = base.split("_")

    info = {
        "quant": parts[0] if len(parts) > 0 else None,
        "config": parts[1] if len(parts) > 1 else None,
        "scenario": parts[2] if len(parts) > 2 else None,
        "ep_size": None,
        "conc": None,
        "dp_attention": "dp" in parts,
    }

    for p in parts:
        if p.startswith("ep"):
            try:
                info["ep_size"] = int(p.replace("ep", ""))
            except ValueError:
                pass
        if p.startswith("c") and p[1:].isdigit():
            info["conc"] = int(p[1:])

    return info


SCENARIO_MAP = {
    "chat": (1024, 1024),
    "reasoning": (1024, 8192),
    "summarize": (8192, 1024),
}


def import_dar_results(results_dir):
    """Scan dar_*.json files and extract DAR (Draft Acceptance Rate) metrics.

    Returns dict mapping scenario -> {dar_avg, dar_p50, dar_p90, dar_p99,
    acceptance_len_avg, acceptance_len_p50}.
    """
    dar_files = sorted(glob.glob(os.path.join(results_dir, "dar_*.json")))
    dar_by_scenario = {}

    for f in dar_files:
        # Parse scenario from filename: dar_report_fp8_mtp3_ep1_chat.json -> chat
        base = os.path.basename(f).replace(".json", "")
        parts = base.split("_")
        # Scenario is always the last part (filenames vary in length)
        if len(parts) < 4:
            continue
        scenario = parts[-1]

        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, IOError) as e:
            print(f"WARN: Skipping DAR file {f}: {e}", file=sys.stderr)
            continue

        ds = data.get("decoding_stats", {})
        dar = ds.get("draft_acceptance_rate_percentiles", {})
        acc_len = ds.get("acceptance_length_percentiles", {})

        if not dar and not acc_len:
            print(f"WARN: No decoding_stats in {f}", file=sys.stderr)
            continue

        dar_metrics = {}
        if dar:
            dar_metrics["dar_avg"] = dar.get("average")
            dar_metrics["dar_p50"] = dar.get("p50")
            dar_metrics["dar_p90"] = dar.get("p90")
            dar_metrics["dar_p99"] = dar.get("p99")
        if acc_len:
            dar_metrics["acceptance_len_avg"] = acc_len.get("average")
            dar_metrics["acceptance_len_p50"] = acc_len.get("p50")

        # Remove None values
        dar_metrics = {k: v for k, v in dar_metrics.items() if v is not None}

        if dar_metrics:
            dar_by_scenario[scenario] = dar_metrics
            print(f"  DAR [{scenario}]: {dar_metrics}")

    return dar_by_scenario


def extract_metrics(data, file_info):
    """Extract metrics from a benchmark result JSON into unified format."""
    scenario = file_info.get("scenario", "unknown")
    isl, osl = SCENARIO_MAP.get(scenario, (0, 0))

    # Extract server-side config from metadata (injected via --metadata)
    SERVER_CONFIG_KEYS = [
        # Common
        "max_model_len",
        "kv_cache_dtype",
        "tensor_parallel_size",
        "mtp_layers",
        "random_range_ratio",
        # MI355X / ATOM (vLLM)
        "gpu_memory_utilization",
        "enforce_eager",
        "max_num_seqs",
        # B200 / TRT-LLM
        "kv_cache_free_mem_fraction",
        "ep_size",
        "dp_attention",
        "moe_backend",
        "piecewise_cuda_graphs",
    ]
    server_config = {k: data[k] for k in SERVER_CONFIG_KEYS if data.get(k) is not None}

    result = {
        "isl": isl,
        "osl": osl,
        "conc": file_info.get("conc", data.get("max_concurrency", 0)),
        "scenario": scenario,
        "config": file_info.get("config", "unknown"),
        "ep_size": file_info.get("ep_size", 1),
        "dp_attention": file_info.get("dp_attention", False),
        # Throughput
        "output_tps": data.get("output_throughput", 0),
        "total_tps": data.get("total_token_throughput", 0),
        "request_tps": data.get("request_throughput", 0),
        # Latency p50
        "tpot_p50": data.get("median_tpot_ms", 0),
        "ttft_p50": data.get("median_ttft_ms", 0),
        "itl_p50": data.get("median_itl_ms", 0),
        "e2el_p50": data.get("median_e2el_ms", 0),
        # Latency p99
        "tpot_p99": data.get("p99_tpot_ms", 0),
        "ttft_p99": data.get("p99_ttft_ms", 0),
        "itl_p99": data.get("p99_itl_ms", 0),
        "e2el_p99": data.get("p99_e2el_ms", 0),
        # Latency mean
        "tpot_mean": data.get("mean_tpot_ms", 0),
        "ttft_mean": data.get("mean_ttft_ms", 0),
        "e2el_mean": data.get("mean_e2el_ms", 0),
        # Meta
        "num_prompts": data.get("num_prompts", 0),
        "completed": data.get("completed", 0),
        "duration": data.get("duration", 0),
        # Reproduce info (injected by sa_bench_b200.sh)
        "server_cmd": data.get("server_cmd", ""),
        "benchmark_cmd": data.get("benchmark_cmd", ""),
        "config_yaml": data.get("config_yaml", ""),
    }

    if server_config:
        result["server_config"] = server_config

    return result


def main():
    parser = argparse.ArgumentParser(description="Import benchmark results to unified run format")
    parser.add_argument("--results-dir", required=True, help="Directory containing result_*.json files")
    parser.add_argument("--platform", required=True, help='Platform name, e.g. "8×B200"')
    parser.add_argument("--framework", required=True, help='Framework and version, e.g. "TRT-LLM 1.2.0rc4"')
    parser.add_argument("--quantization", required=True, help='Quantization type, e.g. "NVFP4", "FP8"')
    parser.add_argument("--model", default="DeepSeek-R1-0528", help="Model name")
    parser.add_argument("--gpu-count", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--tag", default=None, help="Optional config tag filter (e.g. fp4-throughput)")
    parser.add_argument("--env-tag", required=True,
                        help='Environment tag to distinguish runs on the same platform '
                             '(e.g. "mtp3-ep1", "mtp0-ep1"). '
                             'Appears in series_key for side-by-side comparison.')
    parser.add_argument("--output", default=None, help="Output JSON path (auto-generated if not set)")
    parser.add_argument("--source", default="manual", help='Data source: "manual" or "ci"')
    args = parser.parse_args()

    # Normalize platform name: 8xB200 → 8×B200, 8xMI355X → 8×MI355X
    args.platform = re.sub(r'(\d+)x', lambda m: m.group(1) + '×', args.platform)

    result_files = sorted(glob.glob(os.path.join(args.results_dir, "result_*.json")))
    if not result_files:
        print(f"ERROR: No result_*.json files found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    dates = []
    docker_images = set()

    for f in result_files:
        file_info = parse_result_filename(f)

        # Filter by tag if specified
        if args.tag:
            tag_parts = args.tag.split("-")
            if len(tag_parts) >= 2:
                if file_info["quant"] != tag_parts[0] or file_info["config"] != tag_parts[1]:
                    continue

        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, IOError) as e:
            print(f"WARN: Skipping {f}: {e}", file=sys.stderr)
            continue

        metrics = extract_metrics(data, file_info)
        results.append(metrics)

        # Collect docker image from result if present
        di = data.get("docker_image", "")
        if di:
            docker_images.add(di)

        # Try to get date from the data
        date_str = data.get("date", "")
        if date_str:
            dates.append(date_str[:8])  # e.g. "20260316"

    if not results:
        print("ERROR: No results extracted", file=sys.stderr)
        sys.exit(1)

    # Import DAR results if any dar_*.json files exist
    dar_by_scenario = import_dar_results(args.results_dir)
    if dar_by_scenario:
        merged = 0
        for r in results:
            scenario = r.get("scenario")
            if scenario in dar_by_scenario:
                r.update(dar_by_scenario[scenario])
                merged += 1
        print(f"  Merged DAR data into {merged} result entries "
              f"({len(dar_by_scenario)} scenarios)")

    # Determine date
    run_date = max(dates) if dates else datetime.now().strftime("%Y%m%d")
    iso_date = f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}"

    # Build run ID
    platform_short = args.platform.replace("×", "x").replace(" ", "").lower()
    quant_short = args.quantization.lower()
    env_suffix = f"-{args.env_tag}" if args.env_tag else ""
    run_id = f"{platform_short}-{quant_short}-{run_date}{env_suffix}"

    run = {
        "run_id": run_id,
        "platform": args.platform,
        "framework": args.framework,
        "model": args.model,
        "quantization": args.quantization,
        "gpu_count": args.gpu_count,
        "source": args.source,
        "date": iso_date,
        "docker_image": list(docker_images)[0] if len(docker_images) == 1 else ", ".join(sorted(docker_images)),
        "results": results,
    }
    if args.env_tag:
        run["env_tag"] = args.env_tag

    # Output path
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join("runs", f"{run_id}.json")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(run, f, indent=2)

    print(f"Imported {len(results)} data points → {out_path}")
    print(f"  Platform:     {args.platform}")
    print(f"  Framework:    {args.framework}")
    print(f"  Quantization: {args.quantization}")
    print(f"  Date:         {iso_date}")
    if args.env_tag:
        print(f"  Env Tag:      {args.env_tag}")

    # Print summary
    scenarios = set(r["scenario"] for r in results)
    concs = sorted(set(r["conc"] for r in results))
    print(f"  Scenarios:    {', '.join(sorted(scenarios))}")
    print(f"  Concurrency:  {concs}")


if __name__ == "__main__":
    main()
