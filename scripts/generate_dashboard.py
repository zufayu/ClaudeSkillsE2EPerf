#!/usr/bin/env python3
"""Generate docs/data.js from all run JSONs in runs/ directory.

Usage:
    python scripts/generate_dashboard.py
    python scripts/generate_dashboard.py --runs-dir runs/ --output docs/data.js
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime


def load_runs(runs_dir):
    """Load all run JSON files from the runs directory."""
    run_files = sorted(glob.glob(os.path.join(runs_dir, "*.json")))
    runs = []
    for f in run_files:
        try:
            with open(f) as fh:
                run = json.load(fh)
                run["_file"] = os.path.basename(f)
                runs.append(run)
        except (json.JSONDecodeError, IOError) as e:
            print(f"WARN: Skipping {f}: {e}", file=sys.stderr)
    return runs


def build_series_key(run):
    """Build a display key for a run, used as series name in charts.

    Same platform+model+quantization from different sources (ci-nightly vs manual)
    should produce the same series_key so they merge into one series.
    """
    platform = run.get("platform", "unknown")
    model = run.get("model", "")
    quant = run.get("quantization", "")
    framework_short = run.get("framework", "").split(" ")[0]
    return f"{platform} {model} {quant} ({framework_short})"


def deduplicate_runs(runs):
    """When multiple runs share the same series_key, merge their results.

    For the same (isl, osl, conc, config) data point across runs,
    keep the one from the newer run (by date). The merged run preserves
    source info from all contributors.
    """
    from collections import defaultdict

    by_series = defaultdict(list)
    for run in runs:
        sk = build_series_key(run)
        by_series[sk].append(run)

    merged_runs = []
    for sk, group in by_series.items():
        if len(group) == 1:
            merged_runs.append(group[0])
            continue

        # Sort by date ascending (latest last = highest priority)
        # Within same date, manual > ci-nightly (manual is more trusted)
        source_priority = {"manual": 1, "ci-nightly": 0}
        group.sort(key=lambda r: (r.get("date", ""),
                                  source_priority.get(r.get("source", ""), 0)))

        # Merge results: latest wins per (isl, osl, conc, config)
        result_map = {}
        sources = []
        for run in group:
            src = run.get("source", "unknown")
            sources.append(src)
            for r in run.get("results", []):
                key = (r.get("isl"), r.get("osl"), r.get("conc"),
                       r.get("config", ""))
                result_map[key] = {**r, "_source": src}

        latest = group[-1]
        merged = {
            "run_id": latest.get("run_id", ""),
            "platform": latest.get("platform", ""),
            "framework": latest.get("framework", ""),
            "model": latest.get("model", ""),
            "quantization": latest.get("quantization", ""),
            "gpu_count": latest.get("gpu_count", 8),
            "source": latest.get("source", ""),
            "sources": sorted(set(sources)),
            "date": latest.get("date", ""),
            "commit": latest.get("commit", ""),
            "commit_url": latest.get("commit_url", ""),
            "results": sorted(result_map.values(),
                              key=lambda r: (r.get("isl", 0), r.get("osl", 0),
                                             r.get("conc", 0))),
        }
        merged_runs.append(merged)
        print(f"  Merged {len(group)} runs for '{sk}': "
              f"{len(merged['results'])} unique points "
              f"(sources: {', '.join(sorted(set(sources)))})")

    return merged_runs


def generate_data_js(runs):
    """Generate the data.js content for the dashboard."""
    dashboard_data = {
        "generated_at": datetime.now().isoformat(),
        "runs": [],
    }

    # Collect all unique filter values
    all_platforms = set()
    all_scenarios = set()
    all_concs = set()
    all_models = set()

    for run in runs:
        series_key = build_series_key(run)

        # Ensure total_tps exists on each result
        results = run.get("results", [])
        for r in results:
            if not r.get("total_tps"):
                output_tps = r.get("output_tps", 0)
                input_tps = r.get("input_tps", 0)
                r["total_tps"] = output_tps + input_tps if input_tps else output_tps

        run_entry = {
            "run_id": run.get("run_id", ""),
            "series_key": series_key,
            "platform": run.get("platform", ""),
            "framework": run.get("framework", ""),
            "model": run.get("model", ""),
            "quantization": run.get("quantization", ""),
            "gpu_count": run.get("gpu_count", 8),
            "source": run.get("source", ""),
            "sources": run.get("sources", [run.get("source", "")]),
            "date": run.get("date", ""),
            "commit": run.get("commit", ""),
            "commit_url": run.get("commit_url", ""),
            "results": results,
        }

        all_platforms.add(run.get("platform", ""))
        all_models.add(run.get("model", ""))
        for r in run.get("results", []):
            all_scenarios.add(f"{r.get('isl', 0)}/{r.get('osl', 0)}")
            all_concs.add(r.get("conc", 0))

        dashboard_data["runs"].append(run_entry)

    dashboard_data["filters"] = {
        "platforms": sorted(all_platforms),
        "scenarios": sorted(all_scenarios),
        "concurrencies": sorted(all_concs),
        "models": sorted(all_models),
    }

    return dashboard_data


def main():
    parser = argparse.ArgumentParser(description="Generate dashboard data from run JSONs")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run JSON files")
    parser.add_argument("--output", default="docs/data.js", help="Output data.js path")
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"ERROR: No run files found in {args.runs_dir}/", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(runs)} run files:")
    for run in runs:
        n = len(run.get("results", []))
        src = run.get("source", "?")
        print(f"  {run.get('run_id', '?'):40s}  {n:3d} points  src={src}  ({run.get('date', '?')})")

    # Deduplicate: same series_key merges, latest date wins per data point
    runs = deduplicate_runs(runs)

    print(f"\nAfter dedup: {len(runs)} series")

    data = generate_data_js(runs)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    js_content = f"window.DASHBOARD_DATA = {json.dumps(data, indent=2)};\n"
    with open(args.output, "w") as f:
        f.write(js_content)

    total_points = sum(len(r.get("results", [])) for r in runs)
    print(f"\nGenerated {args.output} ({total_points} total data points)")


if __name__ == "__main__":
    main()
