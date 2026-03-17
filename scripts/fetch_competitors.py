#!/usr/bin/env python3
"""Fetch competitor benchmark data and convert to unified run format.

Currently supports:
  - ATOM (ROCm) MI355X: https://rocm.github.io/ATOM/benchmark-dashboard/data.js

Usage:
    python scripts/fetch_competitors.py
    python scripts/fetch_competitors.py --output-dir runs/
    python scripts/fetch_competitors.py --source atom --dry-run
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

try:
    from urllib.request import urlopen, Request
    from urllib.error import URLError
except ImportError:
    print("ERROR: urllib required", file=sys.stderr)
    sys.exit(1)


ATOM_DATA_URL = "https://rocm.github.io/ATOM/benchmark-dashboard/data.js"


def fetch_atom_data(url=ATOM_DATA_URL):
    """Fetch and parse ATOM's data.js from GitHub Pages."""
    req = Request(url, headers={"User-Agent": "ClaudeSkillsE2EPerf/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
    except (URLError, OSError) as e:
        print(f"ERROR: Failed to fetch {url}: {e}", file=sys.stderr)
        sys.exit(1)

    # Strip "window.BENCHMARK_DATA = " prefix
    json_str = content.replace("window.BENCHMARK_DATA = ", "", 1).rstrip().rstrip(";")
    return json.loads(json_str)


def parse_atom_bench_name(name):
    """Parse ATOM bench name like 'DeepSeek-R1-0528 1024/1024 c=128 throughput (tok/s)'."""
    m = re.match(r"^(\S+)\s+(\d+)/(\d+)\s+c=(\d+)\s+(.+?)\s*\((.+)\)$", name)
    if m:
        return {
            "model": m.group(1),
            "isl": int(m.group(2)),
            "osl": int(m.group(3)),
            "conc": int(m.group(4)),
            "metric": m.group(5).strip(),
            "unit": m.group(6),
        }
    # _gpu_count special case
    m = re.match(r"^(\S+)\s+(\d+)/(\d+)\s+c=(\d+)\s+(_gpu_count)$", name)
    if m:
        return {
            "model": m.group(1),
            "isl": int(m.group(2)),
            "osl": int(m.group(3)),
            "conc": int(m.group(4)),
            "metric": "_gpu_count",
            "unit": "",
        }
    return None


METRIC_MAP = {
    "throughput": "output_tps",
    "Total Tput": "total_tps",
    "TPOT": "tpot_p50",
    "TTFT": "ttft_p50",
    "ITL": "itl_p50",
    "E2EL": "e2el_p50",
    "_gpu_count": "_gpu_count",
}

ISL_OSL_TO_SCENARIO = {
    (1024, 1024): "chat",
    (1024, 8192): "reasoning",
    (8192, 1024): "summarize",
}


def convert_atom_to_runs(atom_data):
    """Convert ATOM's latest benchmark entry to unified run format.

    Returns a dict of {model_key: run_dict}.
    """
    entries = atom_data.get("entries", {}).get("Benchmark", [])
    if not entries:
        print("ERROR: No benchmark entries found in ATOM data", file=sys.stderr)
        sys.exit(1)

    latest = entries[-1]
    commit = latest.get("commit", {})
    date_ts = latest.get("date", 0)
    date_str = datetime.fromtimestamp(date_ts / 1000).strftime("%Y-%m-%d") if date_ts else "unknown"

    # Group benches by model
    model_benches = {}
    for bench in latest.get("benches", []):
        parsed = parse_atom_bench_name(bench["name"])
        if not parsed:
            continue

        model = parsed["model"]
        if model not in model_benches:
            model_benches[model] = {}

        key = (parsed["isl"], parsed["osl"], parsed["conc"])
        if key not in model_benches[model]:
            model_benches[model][key] = {}

        metric_key = METRIC_MAP.get(parsed["metric"])
        if metric_key:
            model_benches[model][key][metric_key] = bench["value"]

    # Build run objects
    runs = {}
    for model, bench_points in model_benches.items():
        # Determine if MTP variant
        is_mtp = "-mtp" in model.lower()
        mtp_tag = ""
        if is_mtp:
            m = re.search(r"-mtp(\d+)", model.lower())
            mtp_tag = f"-mtp{m.group(1)}" if m else "-mtp"

        model_clean = re.sub(r"-mtp\d*", "", model)
        # ATOM CI uses FP8 block-scale quantization for all DeepSeek-R1 runs
        # (model config.json contains quantization_config with quant_method=fp8)
        # Only the model name "GLM-5-FP8" explicitly has FP8 suffix;
        # DeepSeek-R1-0528 is also FP8 but without the suffix.
        if "FP8" in model:
            quant = "FP8"
            model_clean = model_clean.replace("-FP8", "")
        elif "DeepSeek-R1" in model:
            quant = "FP8"
        else:
            quant = "BF16"

        results = []
        gpu_count = 8
        for (isl, osl, conc), metrics in sorted(bench_points.items()):
            if "_gpu_count" in metrics:
                gpu_count = int(metrics["_gpu_count"])

            scenario = ISL_OSL_TO_SCENARIO.get((isl, osl), f"{isl}/{osl}")

            result = {
                "isl": isl,
                "osl": osl,
                "conc": conc,
                "scenario": scenario,
                "config": f"{'mtp3-' if is_mtp else ''}throughput",
                "ep_size": 0,
                "dp_attention": False,
                "output_tps": metrics.get("output_tps", 0),
                "total_tps": metrics.get("total_tps", 0),
                "request_tps": 0,
                "tpot_p50": metrics.get("tpot_p50", 0),
                "ttft_p50": metrics.get("ttft_p50", 0),
                "itl_p50": metrics.get("itl_p50", 0),
                "e2el_p50": metrics.get("e2el_p50", 0),
            }
            results.append(result)

        run_key = f"atom-mi355x-{model_clean.lower()}{mtp_tag}"
        runs[run_key] = {
            "run_id": run_key,
            "platform": f"{gpu_count}×MI355X",
            "framework": f"ATOM ({commit.get('id', 'unknown')[:8]})",
            "model": model_clean,
            "quantization": quant + (f" +MTP3" if is_mtp else ""),
            "gpu_count": gpu_count,
            "source": "ci-nightly",
            "date": date_str,
            "commit": commit.get("id", ""),
            "commit_url": commit.get("url", ""),
            "results": results,
        }

    return runs


def main():
    parser = argparse.ArgumentParser(description="Fetch competitor benchmark data")
    parser.add_argument("--source", default="atom", choices=["atom", "all"], help="Which competitor to fetch")
    parser.add_argument("--output-dir", default="runs", help="Output directory for run JSONs")
    parser.add_argument("--dry-run", action="store_true", help="Print data without saving")
    parser.add_argument("--url", default=None, help="Override data URL")
    args = parser.parse_args()

    if args.source in ("atom", "all"):
        print(f"Fetching ATOM data from {args.url or ATOM_DATA_URL} ...")
        atom_data = fetch_atom_data(args.url or ATOM_DATA_URL)
        runs = convert_atom_to_runs(atom_data)

        for run_key, run in runs.items():
            if args.dry_run:
                print(f"\n=== {run_key} ===")
                print(f"  Platform:     {run['platform']}")
                print(f"  Framework:    {run['framework']}")
                print(f"  Model:        {run['model']}")
                print(f"  Quantization: {run['quantization']}")
                print(f"  Date:         {run['date']}")
                print(f"  Data points:  {len(run['results'])}")
                for r in run["results"][:3]:
                    print(f"    {r['scenario']} c={r['conc']}: {r['output_tps']:.1f} tok/s, TPOT={r['tpot_p50']:.1f}ms")
                if len(run["results"]) > 3:
                    print(f"    ... and {len(run['results']) - 3} more")
            else:
                out_path = os.path.join(args.output_dir, f"{run_key}.json")
                os.makedirs(args.output_dir, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(run, f, indent=2)
                print(f"  Saved {len(run['results'])} points → {out_path}")
                print(f"    {run['platform']} | {run['model']} {run['quantization']} | {run['date']}")


if __name__ == "__main__":
    main()
