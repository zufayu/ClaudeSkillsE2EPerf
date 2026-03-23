#!/usr/bin/env python3
"""Fetch competitor benchmark data and convert to unified run format.

Currently supports:
  - ATOM (ROCm) MI355X: https://rocm.github.io/ATOM/benchmark-dashboard/data.js

Usage:
    python scripts/fetch_competitors.py
    python scripts/fetch_competitors.py --output-dir runs/
    python scripts/fetch_competitors.py --source atom --dry-run

Environment variables:
    GITHUB_TOKEN    GitHub PAT with repo scope + SSO authorized for ROCm org.
                    Required for fetching MTP acceptance rate from CI logs.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

try:
    from urllib.request import urlopen, Request, HTTPRedirectHandler, build_opener
    from urllib.error import URLError, HTTPError
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


ATOM_REPO = "ROCm/ATOM"
ATOM_BENCHMARK_WORKFLOW_ID = 226260402


class _StripAuthRedirectHandler(HTTPRedirectHandler):
    """Follow redirects but strip Authorization header (Azure Blob rejects it)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return Request(newurl, headers={"User-Agent": "ClaudeSkillsE2EPerf/1.0"})


def _gh_api(url, token):
    """Make an authenticated GitHub API request, returns parsed JSON or raw bytes."""
    req = Request(url, headers={
        "User-Agent": "ClaudeSkillsE2EPerf/1.0",
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    })
    opener = build_opener(_StripAuthRedirectHandler)
    resp = opener.open(req, timeout=60)
    ct = resp.headers.get("Content-Type", "")
    data = resp.read()
    if "json" in ct:
        return json.loads(data)
    return data.decode("utf-8", errors="replace")


def _parse_mtp_stats(log_text):
    """Extract final cumulative MTP Stats from a CI job log.

    Returns dict with dar_avg, avg_toks_fwd, accepted_distribution or None.
    """
    # Find all cumulative (non-Interval) MTP Stats lines
    cumulative_lines = [
        line for line in log_text.split("\n")
        if "[MTP Stats" in line and "Interval" not in line
    ]
    if not cumulative_lines:
        return None

    last = cumulative_lines[-1]

    # Parse: Average toks/fwd: 2.68, Accepted/Total Draft tokens: 893967/1596000,
    #        Acceptance rate: 56.01%, Accepted tokens distribution: {0: '19.45%', ...}
    ar_match = re.search(r"Acceptance rate:\s*([\d.]+)%", last)
    toks_match = re.search(r"Average toks/fwd:\s*([\d.]+)", last)
    draft_match = re.search(r"Accepted/Total Draft tokens:\s*(\d+)/(\d+)", last)
    dist_match = re.search(r"Accepted tokens distribution:\s*\{(.+?)\}", last)

    if not ar_match:
        return None

    result = {
        "dar_avg": round(float(ar_match.group(1)) / 100, 4),  # Convert % to fraction
    }
    if toks_match:
        result["avg_toks_fwd"] = round(float(toks_match.group(1)), 2)
    if draft_match:
        result["dar_accepted_tokens"] = int(draft_match.group(1))
        result["dar_total_draft_tokens"] = int(draft_match.group(2))
    if dist_match:
        # Parse {0: '19.45%', 1: '27.48%', ...}
        dist = {}
        for m in re.finditer(r"(\d+):\s*'([\d.]+)%'", dist_match.group(1)):
            dist[int(m.group(1))] = round(float(m.group(2)) / 100, 4)
        result["dar_distribution"] = dist

    result["dar_source"] = "atom-ci-log"
    return result


def fetch_atom_ar(commit_sha, token=None):
    """Fetch acceptance rate from ATOM CI logs for MTP models.

    Finds the benchmark workflow run matching commit_sha, downloads MTP job
    logs, and parses [MTP Stats] for acceptance rate.

    Returns dict mapping (model_name, isl, osl, conc) -> ar_dict.
    """
    token = token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("  SKIP: GITHUB_TOKEN not set, cannot fetch AR from CI logs",
              file=sys.stderr)
        return {}

    # Find workflow run by commit SHA
    runs_url = (f"https://api.github.com/repos/{ATOM_REPO}/actions/workflows/"
                f"{ATOM_BENCHMARK_WORKFLOW_ID}/runs?head_sha={commit_sha}&per_page=5")
    try:
        runs_data = _gh_api(runs_url, token)
    except (HTTPError, URLError) as e:
        print(f"  WARN: Failed to query CI runs: {e}", file=sys.stderr)
        return {}

    # Find a successful run
    run_id = None
    for r in runs_data.get("workflow_runs", []):
        if r.get("conclusion") == "success":
            run_id = r["id"]
            break

    if not run_id:
        # Fallback: find latest successful run (commit may not match exactly)
        fallback_url = (f"https://api.github.com/repos/{ATOM_REPO}/actions/workflows/"
                        f"{ATOM_BENCHMARK_WORKFLOW_ID}/runs?status=completed&per_page=10")
        try:
            fallback_data = _gh_api(fallback_url, token)
            for r in fallback_data.get("workflow_runs", []):
                if r.get("conclusion") == "success":
                    run_id = r["id"]
                    print(f"  NOTE: No run for commit {commit_sha[:8]}, "
                          f"using latest successful run {run_id}")
                    break
        except (HTTPError, URLError):
            pass

    if not run_id:
        print(f"  WARN: No successful benchmark run found", file=sys.stderr)
        return {}

    # Get jobs for this run
    jobs_url = (f"https://api.github.com/repos/{ATOM_REPO}/actions/runs/"
                f"{run_id}/jobs?per_page=100")
    try:
        jobs_data = _gh_api(jobs_url, token)
    except (HTTPError, URLError) as e:
        print(f"  WARN: Failed to list jobs: {e}", file=sys.stderr)
        return {}

    # Find MTP jobs and download their logs
    ar_map = {}
    job_pattern = re.compile(
        r"^(.+?)\s+\(isl=(\d+)\s+osl=(\d+)\s+c=(\d+)\)$"
    )

    for job in jobs_data.get("jobs", []):
        name = job["name"]
        if "MTP" not in name and "mtp" not in name:
            continue
        if job.get("conclusion") != "success":
            continue

        m = job_pattern.match(name)
        if not m:
            continue

        model_name = m.group(1).strip()
        isl, osl, conc = int(m.group(2)), int(m.group(3)), int(m.group(4))

        # Download job log
        log_url = (f"https://api.github.com/repos/{ATOM_REPO}/actions/jobs/"
                   f"{job['id']}/logs")
        try:
            log_text = _gh_api(log_url, token)
        except (HTTPError, URLError) as e:
            print(f"  WARN: Failed to download log for {name}: {e}",
                  file=sys.stderr)
            continue

        stats = _parse_mtp_stats(log_text)
        if stats:
            ar_map[(model_name, isl, osl, conc)] = stats
            print(f"  AR [{model_name} {isl}/{osl} c={conc}]: "
                  f"{stats['dar_avg']:.4f} ({stats['dar_avg']*100:.1f}%)")
        else:
            print(f"  WARN: No MTP Stats in log for {name}", file=sys.stderr)

    return ar_map


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
        mtp_env_tag = f"mtp{m.group(1)}" if (is_mtp and m) else "mtp0"
        runs[run_key] = {
            "run_id": run_key,
            "platform": f"{gpu_count}×MI355X",
            "framework": f"ATOM ({commit.get('id', 'unknown')[:8]})",
            "model": model_clean,
            "quantization": quant,
            "env_tag": mtp_env_tag,
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

        # Only keep DeepSeek-R1-0528 base model (skip GLM-5, gpt-oss, MXFP4 variants)
        KEEP_MODELS = {"DeepSeek-R1-0528"}
        runs = {k: v for k, v in runs.items() if v["model"] in KEEP_MODELS}

        # Fetch acceptance rate from CI logs for MTP runs
        mtp_runs = {k: v for k, v in runs.items() if "-mtp" in k}
        if mtp_runs:
            # Use commit SHA from any MTP run (they share the same CI run)
            commit_sha = next(iter(mtp_runs.values())).get("commit", "")
            if commit_sha:
                print(f"\nFetching AR from CI logs (commit {commit_sha[:8]})...")
                ar_map = fetch_atom_ar(commit_sha)

                # Inject AR into MTP results
                if ar_map:
                    for run_key, run in mtp_runs.items():
                        # Find matching AR: prefer exact model match
                        run_model = run.get("model", "")
                        matching_ar = None
                        for (model_name, isl, osl, conc), ar in ar_map.items():
                            # Match "DeepSeek-R1-0528 MTP3" to run model "DeepSeek-R1-0528"
                            if run_model in model_name and "MXFP4" not in model_name:
                                matching_ar = ar
                                break
                        if not matching_ar:
                            # Fallback: use any available AR
                            matching_ar = next(iter(ar_map.values()), None)

                        if matching_ar:
                            # AR is model-intrinsic, apply to all results
                            for result in run["results"]:
                                result.update(matching_ar)
                            print(f"  Applied AR={matching_ar['dar_avg']:.4f} to all "
                                  f"{len(run['results'])} results for {run_key}")

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
                    ar_str = f", AR={r['dar_avg']:.1%}" if "dar_avg" in r else ""
                    print(f"    {r['scenario']} c={r['conc']}: {r['output_tps']:.1f} tok/s, TPOT={r['tpot_p50']:.1f}ms{ar_str}")
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
