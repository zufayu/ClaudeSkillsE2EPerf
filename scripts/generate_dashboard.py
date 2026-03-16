#!/usr/bin/env python3
"""Generate dashboard/data.js from all run JSONs in runs/ directory.

Usage:
    python scripts/generate_dashboard.py
    python scripts/generate_dashboard.py --runs-dir runs/ --output dashboard/data.js
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
    """Build a display key for a run, used as series name in charts."""
    platform = run.get("platform", "unknown")
    quant = run.get("quantization", "")
    framework_short = run.get("framework", "").split(" ")[0]
    return f"{platform} {quant} ({framework_short})"


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

        run_entry = {
            "run_id": run.get("run_id", ""),
            "series_key": series_key,
            "platform": run.get("platform", ""),
            "framework": run.get("framework", ""),
            "model": run.get("model", ""),
            "quantization": run.get("quantization", ""),
            "gpu_count": run.get("gpu_count", 8),
            "source": run.get("source", ""),
            "date": run.get("date", ""),
            "commit": run.get("commit", ""),
            "commit_url": run.get("commit_url", ""),
            "results": run.get("results", []),
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
    parser.add_argument("--output", default="dashboard/data.js", help="Output data.js path")
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"ERROR: No run files found in {args.runs_dir}/", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(runs)} runs:")
    for run in runs:
        n = len(run.get("results", []))
        print(f"  {run.get('run_id', '?'):40s}  {n:3d} points  ({run.get('date', '?')})")

    data = generate_data_js(runs)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    js_content = f"window.DASHBOARD_DATA = {json.dumps(data, indent=2)};\n"
    with open(args.output, "w") as f:
        f.write(js_content)

    total_points = sum(len(r.get("results", [])) for r in runs)
    print(f"\nGenerated {args.output} ({total_points} total data points)")


if __name__ == "__main__":
    main()
