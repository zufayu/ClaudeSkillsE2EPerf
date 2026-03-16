#!/usr/bin/env python3
"""Import benchmark results from results_*/ directories into unified run JSON format.

Usage:
    python scripts/import_results.py \
        --results-dir ./results_b200 \
        --platform "8×B200" \
        --framework "TRT-LLM 1.2.0rc4" \
        --quantization NVFP4 \
        --tag "fp4-throughput" \
        --output runs/b200-fp4-trtllm-20260316.json

    # Auto-detect tag and date from filenames:
    python scripts/import_results.py \
        --results-dir ./results_b200 \
        --platform "8×B200" \
        --framework "TRT-LLM 1.2.0rc4" \
        --quantization NVFP4
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


def extract_metrics(data, file_info):
    """Extract metrics from a benchmark result JSON into unified format."""
    scenario = file_info.get("scenario", "unknown")
    isl, osl = SCENARIO_MAP.get(scenario, (0, 0))

    return {
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
    }


def main():
    parser = argparse.ArgumentParser(description="Import benchmark results to unified run format")
    parser.add_argument("--results-dir", required=True, help="Directory containing result_*.json files")
    parser.add_argument("--platform", required=True, help='Platform name, e.g. "8×B200"')
    parser.add_argument("--framework", required=True, help='Framework and version, e.g. "TRT-LLM 1.2.0rc4"')
    parser.add_argument("--quantization", required=True, help='Quantization type, e.g. "NVFP4", "FP8"')
    parser.add_argument("--model", default="DeepSeek-R1-0528", help="Model name")
    parser.add_argument("--gpu-count", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--tag", default=None, help="Optional config tag filter (e.g. fp4-throughput)")
    parser.add_argument("--output", default=None, help="Output JSON path (auto-generated if not set)")
    parser.add_argument("--source", default="manual", help='Data source: "manual" or "ci"')
    args = parser.parse_args()

    result_files = sorted(glob.glob(os.path.join(args.results_dir, "result_*.json")))
    if not result_files:
        print(f"ERROR: No result_*.json files found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    dates = []

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

        # Try to get date from the data
        date_str = data.get("date", "")
        if date_str:
            dates.append(date_str[:8])  # e.g. "20260316"

    if not results:
        print("ERROR: No results extracted", file=sys.stderr)
        sys.exit(1)

    # Determine date
    run_date = max(dates) if dates else datetime.now().strftime("%Y%m%d")
    iso_date = f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}"

    # Build run ID
    platform_short = args.platform.replace("×", "x").replace(" ", "").lower()
    quant_short = args.quantization.lower()
    run_id = f"{platform_short}-{quant_short}-{run_date}"

    run = {
        "run_id": run_id,
        "platform": args.platform,
        "framework": args.framework,
        "model": args.model,
        "quantization": args.quantization,
        "gpu_count": args.gpu_count,
        "source": args.source,
        "date": iso_date,
        "results": results,
    }

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

    # Print summary
    scenarios = set(r["scenario"] for r in results)
    concs = sorted(set(r["conc"] for r in results))
    print(f"  Scenarios:    {', '.join(sorted(scenarios))}")
    print(f"  Concurrency:  {concs}")


if __name__ == "__main__":
    main()
