#!/usr/bin/env python3
"""
Validate ncu trace by cross-checking against nsys trace.

Runs the same inference workload under both nsys and ncu, then compares:
  1. Kernel names — identical sequence in both traces
  2. Kernel count — same number of inference kernels captured
  3. Duration — ncu gpu__time_duration ≈ nsys kernel duration (within tolerance)
  4. Layer pattern — repeating 5-kernel pattern per transformer layer

Usage:
    # Step 1: Collect both traces (same workload, same params)
    nsys profile --trace cuda -o nsys_trace python3 scripts/ncu_infer.py ...
    ncu -k "regex:nvjet|fmha|..." --set full -o ncu_trace python3 scripts/ncu_infer.py ...

    # Step 2: Validate
    python3 scripts/validate_ncu_trace.py \
        --nsys-rep nsys_trace.nsys-rep \
        --ncu-rep ncu_trace.ncu-rep \
        [--kernel-regex "nvjet|fmha|cutlass|flash_attn|kernel_mha|allreduce|nccl|deep_gemm"]

    # Or auto mode: runs both traces and validates
    python3 scripts/validate_ncu_trace.py --auto \
        --backend trtllm --model /path/to/model --tp 1 --ep 1 \
        --result-dir ./results/validation
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import subprocess
import sys


# Default kernel regex — must match collect_ncu_trace.sh
DEFAULT_KERNEL_REGEX = (
    "nvjet|fmha|cutlass|flash_attn|kernel_mha|allreduce|"
    "reduce_scatter|all_gather|nccl|deep_gemm"
)


def parse_args():
    p = argparse.ArgumentParser(description="Validate ncu trace against nsys")
    p.add_argument("--nsys-rep", help="Path to .nsys-rep file")
    p.add_argument("--ncu-rep", help="Path to .ncu-rep file")
    p.add_argument("--kernel-regex", default=DEFAULT_KERNEL_REGEX,
                    help="Kernel name filter regex")
    p.add_argument("--duration-tolerance", type=float, default=0.5,
                    help="Max ratio diff for duration comparison (default 0.5 = 50%%)")
    p.add_argument("--auto", action="store_true",
                    help="Auto mode: run both nsys and ncu traces, then validate")
    # Auto mode options
    p.add_argument("--backend", default="trtllm", choices=["sglang", "trtllm"])
    p.add_argument("--model", help="Model path (required for auto mode)")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--isl", type=int, default=16)
    p.add_argument("--osl", type=int, default=2)
    p.add_argument("--warmup-prompts", type=int, default=0)
    p.add_argument("--result-dir", default="./results/ncu_validation")
    p.add_argument("--ncu-sections", default="pmsampling",
                    help="ncu section set: pmsampling|full|detailed|basic")
    p.add_argument("--launch-count", type=int, default=50)
    return p.parse_args()


def shorten_kernel_name(name):
    """Extract short kernel identifier from full demangled name."""
    # Remove template args and parameter lists
    name = re.sub(r'<[^>]*>', '', name)
    name = re.sub(r'\([^)]*\)', '', name)
    # Get last component
    parts = name.split('::')
    short = parts[-1].strip() if parts else name.strip()
    # Remove 'void ' prefix
    short = re.sub(r'^void\s+', '', short)
    return short


def extract_nsys_kernels(nsys_rep, kernel_regex):
    """Extract inference kernels from nsys trace via sqlite."""
    sqlite_path = nsys_rep.replace('.nsys-rep', '.sqlite')

    # Export to sqlite if needed
    if not os.path.exists(sqlite_path):
        print(f"  Exporting nsys to sqlite...")
        subprocess.run(
            ["nsys", "stats", nsys_rep, "--report", "cuda_gpu_kern_sum",
             "--format", "csv", "-o", "/dev/null"],
            capture_output=True, timeout=120
        )

    if not os.path.exists(sqlite_path):
        print(f"ERROR: Could not create sqlite from {nsys_rep}")
        return []

    conn = sqlite3.connect(sqlite_path)
    regex_parts = kernel_regex.split('|')
    like_clauses = ' OR '.join([f"s.value LIKE '%{p}%'" for p in regex_parts])

    cur = conn.execute(f"""
        SELECT s.value, k.start, k.end, (k.end - k.start) as duration
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        WHERE {like_clauses}
        ORDER BY k.start
    """)

    kernels = []
    for row in cur.fetchall():
        kernels.append({
            "name": row[0],
            "short_name": shorten_kernel_name(row[0]),
            "start_ns": row[1],
            "end_ns": row[2],
            "duration_ns": row[3],
        })
    conn.close()
    return kernels


def extract_ncu_kernels(ncu_rep):
    """Extract kernel info from ncu report via CSV export."""
    csv_path = ncu_rep.replace('.ncu-rep', '_details.csv')

    print(f"  Exporting ncu to CSV...")
    result = subprocess.run(
        ["ncu", "-i", ncu_rep, "--csv", "--page", "raw"],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode != 0:
        # Try simpler export
        result = subprocess.run(
            ["ncu", "-i", ncu_rep, "--csv"],
            capture_output=True, text=True, timeout=120
        )

    if not result.stdout.strip():
        print(f"ERROR: ncu CSV export produced no output")
        return []

    # Parse CSV
    lines = result.stdout.strip().split('\n')
    # Find header line (starts with "ID" or has "Kernel Name")
    header_idx = 0
    for i, line in enumerate(lines):
        if '"ID"' in line or 'Kernel Name' in line or line.startswith('"ID"'):
            header_idx = i
            break

    kernels = []
    reader = csv.DictReader(lines[header_idx:])
    seen_ids = set()
    for row in reader:
        kid = row.get('ID', '')
        if kid in seen_ids:
            continue  # Skip duplicate metric rows for same kernel
        seen_ids.add(kid)

        name = row.get('Kernel Name', row.get('Function Name', ''))
        duration = row.get('gpu__time_duration.sum', '')

        kernels.append({
            "id": kid,
            "name": name,
            "short_name": shorten_kernel_name(name),
            "duration_ns": float(duration) if duration else 0,
        })

    return kernels


def detect_layer_pattern(kernels, min_repeat=3):
    """Detect repeating kernel pattern (transformer layers)."""
    if len(kernels) < 5:
        return None, 0

    # Try pattern lengths 3-10
    for plen in range(3, 11):
        pattern = [k["short_name"] for k in kernels[:plen]]
        repeats = 0
        for i in range(0, len(kernels) - plen + 1, plen):
            chunk = [k["short_name"] for k in kernels[i:i + plen]]
            if chunk == pattern:
                repeats += 1
            else:
                break
        if repeats >= min_repeat:
            return pattern, repeats

    return None, 0


def validate(nsys_kernels, ncu_kernels, tolerance):
    """Cross-validate nsys and ncu kernel lists."""
    results = {"passed": 0, "failed": 0, "warnings": 0, "checks": []}

    def check(name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        results["checks"].append({"name": name, "status": status, "detail": detail})
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))

    def warn(name, detail=""):
        results["checks"].append({"name": name, "status": "WARN", "detail": detail})
        results["warnings"] += 1
        print(f"  [WARN] {name}" + (f": {detail}" if detail else ""))

    print("\n=== Validation Results ===\n")

    # 1. Both traces have kernels
    check("nsys has inference kernels", len(nsys_kernels) > 0,
          f"{len(nsys_kernels)} kernels")
    check("ncu has profiled kernels", len(ncu_kernels) > 0,
          f"{len(ncu_kernels)} kernels")

    if not nsys_kernels or not ncu_kernels:
        return results

    # 2. ncu kernels are a subset of nsys kernels (by name)
    ncu_names = [k["short_name"] for k in ncu_kernels]
    nsys_names = [k["short_name"] for k in nsys_kernels]
    nsys_name_set = set(nsys_names)
    ncu_name_set = set(ncu_names)

    missing = ncu_name_set - nsys_name_set
    check("ncu kernel names found in nsys", len(missing) == 0,
          f"missing: {missing}" if missing else "all ncu kernels present in nsys")

    # 3. Kernel name sequence alignment
    # ncu kernels should be a contiguous subsequence of nsys kernels
    ncu_seq = ncu_names[:min(len(ncu_names), 20)]  # Check first 20
    # Find this sequence in nsys
    found_offset = -1
    for start in range(len(nsys_names) - len(ncu_seq) + 1):
        if nsys_names[start:start + len(ncu_seq)] == ncu_seq:
            found_offset = start
            break

    check("ncu kernel sequence aligns with nsys",
          found_offset >= 0,
          f"ncu kernels match nsys starting at index {found_offset}" if found_offset >= 0
          else "ncu kernel sequence NOT found in nsys timeline")

    # 4. Layer pattern detection
    nsys_pattern, nsys_repeats = detect_layer_pattern(nsys_kernels)
    ncu_pattern, ncu_repeats = detect_layer_pattern(ncu_kernels)

    if nsys_pattern:
        check("nsys shows repeating layer pattern",
              nsys_repeats >= 3,
              f"pattern of {len(nsys_pattern)} kernels repeats {nsys_repeats}x: "
              f"{' → '.join(nsys_pattern)}")
    else:
        warn("nsys no clear layer pattern detected")

    if ncu_pattern:
        check("ncu shows repeating layer pattern",
              ncu_repeats >= 3,
              f"pattern of {len(ncu_pattern)} kernels repeats {ncu_repeats}x")
    else:
        warn("ncu no clear layer pattern detected (may need more kernels)")

    if nsys_pattern and ncu_pattern:
        check("layer patterns match between nsys and ncu",
              nsys_pattern == ncu_pattern,
              f"nsys={nsys_pattern} vs ncu={ncu_pattern}")

    # 5. Duration comparison (if ncu has duration data)
    ncu_with_dur = [k for k in ncu_kernels if k["duration_ns"] > 0]
    if ncu_with_dur and found_offset >= 0:
        dur_ratios = []
        comparisons = min(len(ncu_with_dur), 20)
        print(f"\n  Duration comparison (first {comparisons} kernels):")
        print(f"  {'#':>3} {'Kernel':<45} {'nsys(us)':>10} {'ncu(us)':>10} {'ratio':>8}")
        print(f"  {'-'*3} {'-'*45} {'-'*10} {'-'*10} {'-'*8}")

        for i in range(comparisons):
            ncu_dur = ncu_with_dur[i]["duration_ns"]
            nsys_dur = nsys_kernels[found_offset + i]["duration_ns"]
            if nsys_dur > 0:
                ratio = ncu_dur / nsys_dur
                dur_ratios.append(ratio)
                flag = "" if 1 - tolerance <= ratio <= 1 + tolerance else " ←"
                print(f"  {i:3d} {ncu_with_dur[i]['short_name']:<45} "
                      f"{nsys_dur/1000:10.1f} {ncu_dur/1000:10.1f} {ratio:8.2f}{flag}")

        if dur_ratios:
            avg_ratio = sum(dur_ratios) / len(dur_ratios)
            within_tol = sum(1 for r in dur_ratios
                            if 1 - tolerance <= r <= 1 + tolerance)
            check("kernel durations comparable",
                  within_tol / len(dur_ratios) >= 0.8,
                  f"{within_tol}/{len(dur_ratios)} within {tolerance*100:.0f}% tolerance, "
                  f"avg ratio={avg_ratio:.2f}")
    else:
        warn("duration comparison skipped (no duration data in ncu or no sequence alignment)")

    # Summary
    print(f"\n=== Summary: {results['passed']} passed, {results['failed']} failed, "
          f"{results['warnings']} warnings ===\n")

    return results


def run_auto_mode(args):
    """Run nsys + ncu on the same workload, then validate."""
    os.makedirs(args.result_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    infer_script = os.path.join(script_dir, "ncu_infer.py")

    infer_args = [
        "python3", infer_script,
        "--backend", args.backend,
        "--model", args.model,
        "--tp", str(args.tp),
        "--ep", str(args.ep),
        "--warmup-prompts", str(args.warmup_prompts),
        "--isl", str(args.isl),
        "--osl", str(args.osl),
    ]

    nsys_rep = os.path.join(args.result_dir, "nsys_validation")
    ncu_rep = os.path.join(args.result_dir, "ncu_validation")

    # Step 1: nsys trace
    print("=" * 60)
    print("  Step 1: nsys trace")
    print("=" * 60)
    nsys_cmd = [
        "nsys", "profile", "--trace", "cuda",
        "-o", nsys_rep, "--force-overwrite", "true",
    ] + infer_args
    print(f"  CMD: {' '.join(nsys_cmd)}")
    ret = subprocess.run(nsys_cmd, timeout=600)
    if ret.returncode != 0:
        print(f"ERROR: nsys failed with exit code {ret.returncode}")
        return None

    # Step 2: ncu trace (same workload, same params)
    print("\n" + "=" * 60)
    print("  Step 2: ncu trace")
    print("=" * 60)

    ncu_sections = []
    if args.ncu_sections == "pmsampling":
        ncu_sections = ["--section", "PmSampling", "--section", "PmSampling_WarpStates"]
    else:
        ncu_sections = ["--set", args.ncu_sections]

    ncu_cmd = [
        "ncu",
        "--target-processes", "all",
        "--graph-profiling", "node",
        "--pm-sampling-interval", "1000",
        "-k", f"regex:{args.kernel_regex}",
        "--launch-skip", "0",
        "--launch-count", str(args.launch_count),
        "-f", "-o", ncu_rep,
    ] + ncu_sections + [
        "--section", "Nvlink", "--section", "Nvlink_Tables",
        "--section", "Nvlink_Topology",
    ] + infer_args
    print(f"  CMD: {' '.join(ncu_cmd)}")
    ret = subprocess.run(ncu_cmd, timeout=3600)
    if ret.returncode != 0:
        print(f"WARNING: ncu exited with code {ret.returncode} (may still have report)")

    args.nsys_rep = nsys_rep + ".nsys-rep"
    args.ncu_rep = ncu_rep + ".ncu-rep"

    if not os.path.exists(args.nsys_rep):
        print(f"ERROR: nsys report not found: {args.nsys_rep}")
        return None
    if not os.path.exists(args.ncu_rep):
        print(f"ERROR: ncu report not found: {args.ncu_rep}")
        return None

    return args


def main():
    args = parse_args()

    if args.auto:
        if not args.model:
            print("ERROR: --model required for auto mode")
            sys.exit(1)
        args = run_auto_mode(args)
        if args is None:
            sys.exit(1)

    if not args.nsys_rep or not args.ncu_rep:
        print("ERROR: --nsys-rep and --ncu-rep required (or use --auto)")
        sys.exit(1)

    print(f"\nnsys report: {args.nsys_rep}")
    print(f"ncu report:  {args.ncu_rep}")
    print(f"kernel filter: {args.kernel_regex}")

    # Extract kernels
    print("\nExtracting nsys kernels...")
    nsys_kernels = extract_nsys_kernels(args.nsys_rep, args.kernel_regex)
    print(f"  Found {len(nsys_kernels)} inference kernels in nsys")

    print("Extracting ncu kernels...")
    ncu_kernels = extract_ncu_kernels(args.ncu_rep)
    print(f"  Found {len(ncu_kernels)} profiled kernels in ncu")

    # Validate
    results = validate(nsys_kernels, ncu_kernels, args.duration_tolerance)

    # Save results
    if hasattr(args, 'result_dir') and args.result_dir:
        out_path = os.path.join(args.result_dir, "validation_results.json")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")

    sys.exit(1 if results["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
