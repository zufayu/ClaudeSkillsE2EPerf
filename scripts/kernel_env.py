#!/usr/bin/env python3
"""Environment preflight for kernel-level profiling in ClaudeSkillsE2EPerf.

Detects platform (NVIDIA/AMD), GPU inventory, profiling tool availability,
and selects idle GPUs for trace capture.

Usage:
    python3 scripts/kernel_env.py detect
    python3 scripts/kernel_env.py idle-gpus --count 1
    python3 scripts/kernel_env.py check-tools
    python3 scripts/kernel_env.py suggest --scenario chat --concurrency 64
"""

import argparse
import json
import os
import subprocess
import sys


# ---- Theoretical bandwidth (GB/s) for known GPUs ----
GPU_BANDWIDTH = {
    "B200": 8000,       # B200 SXM: ~8 TB/s HBM3e
    "B100": 8000,
    "H200": 4800,       # H200 SXM: ~4.8 TB/s HBM3e
    "H100": 3350,       # H100 SXM: ~3.35 TB/s HBM3
    "H20":  4000,       # H20: ~4 TB/s HBM3e
    "A100": 2000,       # A100 80GB SXM: ~2 TB/s HBM2e
    "L40S": 864,
    "T4":   320,
    "MI355X": 8000,     # MI355X: ~8 TB/s HBM3e
    "MI325X": 5300,     # MI325X: ~5.3 TB/s HBM3e
    "MI308X": 5300,     # MI308X: ~5.3 TB/s HBM3e (same die as MI300X)
    "MI300X": 5300,     # MI300X: ~5.3 TB/s HBM3
    "MI250X": 3276,
}


def _run(cmd, check=False):
    """Run a command and return (returncode, stdout)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return r.returncode, r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 1, ""


def detect_platform():
    """Detect GPU platform. Returns 'nvidia', 'amd', or 'none'."""
    rc, _ = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if rc == 0:
        return "nvidia"
    rc, _ = _run(["rocm-smi", "--showproductname"])
    if rc == 0:
        return "amd"
    return "none"


def get_nvidia_gpus():
    """Get NVIDIA GPU inventory."""
    rc, out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,compute_cap",
        "--format=csv,noheader,nounits"
    ])
    if rc != 0:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        name = parts[1]
        bw = 0
        for key, val in GPU_BANDWIDTH.items():
            if key in name:
                bw = val
                break
        gpus.append({
            "index": int(parts[0]),
            "name": name,
            "memory_total_mib": int(parts[2]),
            "memory_used_mib": int(parts[3]),
            "utilization_pct": int(parts[4]),
            "compute_cap": parts[5],
            "theoretical_bw_gbps": bw,
        })
    return gpus


def get_amd_gpus():
    """Get AMD GPU inventory."""
    rc, out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"])
    if rc != 0:
        return []
    gpus = []
    # Fallback: parse rocm-smi JSON
    rc2, out2 = _run(["rocm-smi", "--showproductname", "--json"])
    if rc2 == 0:
        try:
            data = json.loads(out2)
            for card_key, info in data.items():
                idx = int(card_key.replace("card", "")) if card_key.startswith("card") else len(gpus)
                name = info.get("Card Series", info.get("Card series", "Unknown"))
                bw = 0
                for key, val in GPU_BANDWIDTH.items():
                    if key.lower() in name.lower():
                        bw = val
                        break
                gpus.append({
                    "index": idx,
                    "name": name,
                    "memory_total_mib": 0,
                    "memory_used_mib": 0,
                    "utilization_pct": 0,
                    "theoretical_bw_gbps": bw,
                })
        except (json.JSONDecodeError, AttributeError):
            pass
    return gpus


def get_gpus(platform=None):
    """Get GPU inventory for detected platform."""
    platform = platform or detect_platform()
    if platform == "nvidia":
        return get_nvidia_gpus()
    elif platform == "amd":
        return get_amd_gpus()
    return []


def get_idle_gpus(count=1, max_mem_mib=100, max_util_pct=10):
    """Select idle GPUs. Returns list of GPU indices."""
    platform = detect_platform()
    gpus = get_gpus(platform)
    if not gpus:
        raise RuntimeError("No GPUs detected")

    if platform == "nvidia":
        # Also check for compute processes
        rc, out = _run([
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader"
        ])
        busy_indices = set()
        if rc == 0 and out:
            # Map UUIDs to indices
            rc2, out2 = _run([
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader"
            ])
            if rc2 == 0:
                uuid_to_idx = {}
                for line in out2.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        uuid_to_idx[parts[1]] = int(parts[0])
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if parts[0] in uuid_to_idx:
                        busy_indices.add(uuid_to_idx[parts[0]])

        idle = [
            g["index"] for g in gpus
            if g["index"] not in busy_indices
            and g["memory_used_mib"] <= max_mem_mib
            and g["utilization_pct"] <= max_util_pct
        ]
    else:
        # AMD: check /dev/kfd users
        rc, out = _run(["fuser", "/dev/kfd"])
        has_users = rc == 0 and out.strip()
        idle = [g["index"] for g in gpus] if not has_users else []

    if len(idle) < count:
        raise RuntimeError(
            f"Not enough idle GPUs: need {count}, found {len(idle)} idle "
            f"out of {len(gpus)} total"
        )
    return idle[:count]


def check_tools():
    """Check profiling tool availability."""
    tools = {
        "nvidia": [
            ("nsys", "nsys --version"),
            ("ncu", "ncu --version"),
            ("nvidia-smi", "nvidia-smi --query-gpu=name --format=csv,noheader"),
        ],
        "amd": [
            ("rocm-smi", "rocm-smi --version"),
            ("rocprof", "rocprof --version"),
            ("omniperf", "omniperf --version"),
        ],
        "common": [
            ("python3", "python3 --version"),
            ("sqlite3", "sqlite3 --version"),
        ],
    }

    platform = detect_platform()
    results = {}

    for category in ["common", platform]:
        if category not in tools:
            continue
        for name, cmd in tools[category]:
            rc, out = _run(cmd.split())
            version = out.split("\n")[0][:60] if rc == 0 else "NOT FOUND"
            status = "OK" if rc == 0 else "MISSING"
            results[name] = {"status": status, "version": version}

    return results


def suggest_commands(platform, scenario="chat", concurrency=64, quant="fp8", config="throughput"):
    """Suggest trace collection and analysis commands."""
    isl_osl = {"chat": "1K/1K", "reasoning": "1K/8K", "summarize": "8K/1K"}
    label = isl_osl.get(scenario, scenario)

    lines = [
        f"# Suggested profiling workflow for {platform.upper()} — {scenario} ({label}) c={concurrency}",
        "",
    ]

    if platform == "nvidia":
        lines.extend([
            "# Level 1: Collect nsys trace",
            f"bash scripts/collect_nsys_trace.sh \\",
            f"  --model /path/to/DeepSeek-R1-0528 \\",
            f"  --mode bench --scenario {scenario} --concurrency {concurrency} \\",
            f"  --quant {quant} --config {config} --iter-range 100-150",
            "",
            "# Level 1: Analyze category breakdown",
            "bash scripts/analyze_nsys_trace.sh --trace traces/nsys_*.nsys-rep --top 30",
            "",
            "# Level 2: Compare before/after",
            "python3 scripts/compare_traces.py \\",
            "  --baseline traces/baseline.sqlite --current traces/new.sqlite --md",
            "",
            "# Level 3: ncu deep kernel analysis (targeted)",
            "bash scripts/ncu_kernel_analysis.sh \\",
            f"  --model /path/to/DeepSeek-R1-0528 \\",
            f"  --mode targeted --kernel-name 'nvjet' \\",
            f"  --scenario {scenario} --concurrency {concurrency}",
            "",
            "# Level 3: ncu discovery (top bottleneck kernels)",
            "bash scripts/ncu_kernel_analysis.sh \\",
            f"  --model /path/to/DeepSeek-R1-0528 \\",
            f"  --mode discovery --scenario {scenario} --concurrency {concurrency}",
        ])
    elif platform == "amd":
        lines.extend([
            "# Level 1: Collect Kineto trace",
            f"bash scripts/collect_atom_trace.sh \\",
            f"  --model /path/to/DeepSeek-R1-0528 \\",
            f"  --scenario {scenario} --concurrency {concurrency} \\",
            f"  --result-dir ./results_mi355x_trace",
            "",
            "# Level 1: Analyze prefill impact on TPOT",
            "python3 scripts/analyze_prefill_impact.py traces/*.json.gz",
            "",
            "# Level 2: Compare before/after",
            "python3 scripts/compare_traces.py \\",
            "  --baseline results_old/decode_walltime_*.csv \\",
            "  --current results_new/decode_walltime_*.csv --md",
        ])

    return "\n".join(lines)


def cmd_detect(args):
    platform = detect_platform()
    gpus = get_gpus(platform)
    print(f"Platform: {platform}")
    print(f"GPUs:     {len(gpus)}")
    for g in gpus:
        bw = f", theoretical BW: {g['theoretical_bw_gbps']} GB/s" if g.get("theoretical_bw_gbps") else ""
        mem = f", mem: {g['memory_used_mib']}/{g['memory_total_mib']} MiB" if g.get("memory_total_mib") else ""
        print(f"  [{g['index']}] {g['name']}{mem}{bw}")


def cmd_idle_gpus(args):
    try:
        idle = get_idle_gpus(count=args.count)
        print(",".join(str(i) for i in idle))
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_check_tools(args):
    results = check_tools()
    for name, info in results.items():
        status = "OK" if info["status"] == "OK" else "MISS"
        print(f"  [{status:4s}] {name:<12s} {info['version']}")


def cmd_suggest(args):
    platform = detect_platform()
    if platform == "none":
        print("ERROR: No GPU platform detected", file=sys.stderr)
        sys.exit(1)
    print(suggest_commands(platform, args.scenario, args.concurrency, args.quant, args.config))


def main():
    parser = argparse.ArgumentParser(description="Kernel profiling environment preflight")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("detect", help="Detect platform and GPU inventory")

    idle_p = sub.add_parser("idle-gpus", help="Print comma-separated idle GPU indices")
    idle_p.add_argument("--count", type=int, default=1, help="Number of idle GPUs needed")

    sub.add_parser("check-tools", help="Check profiling tool availability")

    suggest_p = sub.add_parser("suggest", help="Suggest profiling commands")
    suggest_p.add_argument("--scenario", default="chat", choices=["chat", "reasoning", "summarize"])
    suggest_p.add_argument("--concurrency", type=int, default=64)
    suggest_p.add_argument("--quant", default="fp8", choices=["fp4", "fp8"])
    suggest_p.add_argument("--config", default="throughput", choices=["throughput", "latency"])

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {"detect": cmd_detect, "idle-gpus": cmd_idle_gpus,
     "check-tools": cmd_check_tools, "suggest": cmd_suggest}[args.command](args)


if __name__ == "__main__":
    main()
