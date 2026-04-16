#!/usr/bin/env python3
"""aiter Tune Advisor — data-driven kernel tunability analysis for MI355X.

Reads ATOM profiling output (decode_breakdown.xlsx) or a manual kernel list,
classifies kernels into tune families, computes GEMM shapes from model config,
checks existing tuned configs in aiter, and generates tune input CSVs.

Runs inside the MI355X container (needs `import aiter` and access to aiter source).

Usage:
    # From xlsx (primary)
    python3 scripts/aiter_tune_advisor.py \\
        --xlsx results/.../decode_breakdown.xlsx --tp 8 --bs 64 \\
        --output-dir ./results/mi355x_aiter_gemm_tune

    # From manual kernel list (fallback)
    python3 scripts/aiter_tune_advisor.py \\
        --kernels "bf16gemm_fp32bf16_tn_32x64_splitk_clean" \\
                  "Cijk_Alik_Bljk_BBS_BH_..._ISA950" \\
                  "batched_gemm_a8w8_..._M16_N128_K128" \\
        --tp 8 --bs 64 --output-dir ./results/mi355x_aiter_gemm_tune
"""

import argparse
import csv
import glob
import json
import os
import sys

# ── DeepSeek R1 671B architecture constants ──────────────────────
DSR1_CONFIG = {
    "hidden_size": 7168,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "num_attention_heads": 128,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
}

# ── Tune family definitions ──────────────────────────────────────
TUNE_FAMILIES = {
    "fp8_dense": {
        "patterns": ["Cijk_", "gemm_xdl_preshuffle"],
        "tune_script": "csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py",
        "config_env": "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
        "input_cols": ["M", "N", "K", "q_dtype_w"],
    },
    "fp8_batched": {
        "patterns": ["batched_gemm_a8w8"],
        "tune_script": "csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py",
        "config_env": "AITER_CONFIG_A8W8_BATCHED_GEMM",
        "input_cols": ["B", "M", "N", "K"],
    },
}

NOT_TUNABLE = {
    "bf16_asm": {"patterns": ["bf16gemm"], "reason": "aiter hand-written ASM, already tuned for gfx950"},
    "moe_ck": {"patterns": ["kernel_moe_mxgemm"], "reason": "CK grouped GEMM, requires CK instance changes"},
    "mla_asm": {"patterns": ["mla_a8w8"], "reason": "aiter ASM persistent kernel, no tune script"},
    "mla_reduce": {"patterns": ["kn_mla_reduce"], "reason": "aiter CUDA kernel, no tune script"},
    "norm_fused": {"patterns": ["rmsnorm", "add_rmsnorm_quant", "local_device_load_rmsnorm"], "reason": "fused norm kernel, not a GEMM"},
    "comm": {"patterns": ["reduce_scatter", "local_device_load", "allreduce"], "reason": "communication kernel"},
    "rope": {"patterns": ["fuse_qk_rope", "rope_concat"], "reason": "RoPE + KV cache kernel"},
    "sort": {"patterns": ["MoeSorting", "grouped_topk"], "reason": "MoE routing/sorting kernel"},
    "quant": {"patterns": ["fused_dynamic_mxfp4_quant", "per_token_quant", "dynamic_per_token_scaled_quant"], "reason": "quantization kernel"},
    "triton_misc": {"patterns": ["triton_poi_fused"], "reason": "Triton JIT memory/transpose op"},
}

# ── Module → operator mapping (xlsx module name → logical operator) ──
# Modules that map to multiple operators are listed in execution order.
MODULE_TO_OPS = {
    "gemm_a16w16": ["qkv_a", "router"],
    "gemm_a8w8_bpreshuffle": ["qkv_a"],
    "per_token_quant_hip": ["input_quant"],
    "hipLaunchKernel": ["q_a_norm", "kv_a_norm"],
    "q_proj_and_k_up_proj": ["q_b_proj", "k_up"],
    "v_up_proj_and_o_proj": ["uv", "o_proj"],
    "rope_and_kv_cache": ["rope_cache"],
    "mla_decode": ["mla", "mla_reduce"],
    "input_layernorm": ["pre_attn_comm"],
    "post_attn_layernorm": ["pre_moe_comm"],
    "mxfp4_moe": ["moe_sort", "moe_quant", "moe_gate_up", "moe_down"],
}


def compute_shapes(tp, bs):
    """Compute GEMM shapes from DeepSeek R1 config + TP size."""
    heads_per_gpu = DSR1_CONFIG["num_attention_heads"] // tp
    nope = DSR1_CONFIG["qk_nope_head_dim"]
    rope = DSR1_CONFIG["qk_rope_head_dim"]
    v_head = DSR1_CONFIG["v_head_dim"]
    q_lora = DSR1_CONFIG["q_lora_rank"]
    kv_lora = DSR1_CONFIG["kv_lora_rank"]
    hidden = DSR1_CONFIG["hidden_size"]

    return {
        "qkv_a": {"type": "dense", "M": bs, "K": hidden, "N": q_lora + kv_lora + rope},
        "q_b_proj": {"type": "dense", "M": bs, "K": q_lora, "N": heads_per_gpu * (nope + rope)},
        "k_up": {"type": "batched", "B": heads_per_gpu, "M": bs, "K": kv_lora, "N": nope},
        "uv": {"type": "batched", "B": heads_per_gpu, "M": bs, "K": kv_lora, "N": v_head},
        "o_proj": {"type": "dense", "M": bs, "K": heads_per_gpu * v_head, "N": hidden},
        "router": {"type": "dense", "M": bs, "K": hidden, "N": DSR1_CONFIG["n_routed_experts"]},
    }


def classify_kernel(kernel_name):
    """Classify a kernel name into a tune family or not-tunable category."""
    kn_lower = kernel_name.lower()

    # Check tunable families first
    for family_name, family_info in TUNE_FAMILIES.items():
        for pattern in family_info["patterns"]:
            if pattern.lower() in kn_lower:
                return family_name, True, family_info

    # Check not-tunable families
    for family_name, family_info in NOT_TUNABLE.items():
        for pattern in family_info["patterns"]:
            if pattern.lower() in kn_lower:
                return family_name, False, family_info

    return "unknown", False, {"reason": "unrecognized kernel"}


def map_kernel_to_operator(kernel_name, module_name, position, all_kernels):
    """Map a kernel to a logical operator based on module + kernel name + position."""
    kn_lower = kernel_name.lower()

    # Direct kernel-based mapping (more specific than module)
    if "bf16gemm" in kn_lower:
        # Disambiguate qkv_a vs router by position:
        # qkv_a appears early in the layer, router appears after post-attn norm
        # Use a simple heuristic: if it's the first bf16gemm, it's qkv_a
        bf16_positions = [i for i, k in enumerate(all_kernels)
                          if "bf16gemm" in k.lower()]
        if position == bf16_positions[0] if bf16_positions else -1:
            return "qkv_a"
        elif len(bf16_positions) >= 2 and position == bf16_positions[1]:
            return "q_b_proj"
        elif len(bf16_positions) >= 3 and position == bf16_positions[2]:
            return "o_proj"
        elif len(bf16_positions) >= 4 and position == bf16_positions[3]:
            return "router"
        return "qkv_a"  # default

    if "batched_gemm_a8w8" in kn_lower:
        batched_positions = [i for i, k in enumerate(all_kernels)
                             if "batched_gemm_a8w8" in k.lower()]
        if position == batched_positions[0] if batched_positions else -1:
            return "k_up"
        elif len(batched_positions) >= 2 and position == batched_positions[1]:
            return "uv"
        return "k_up"

    if "cijk_" in kn_lower or "gemm_xdl_preshuffle" in kn_lower:
        cijk_positions = [i for i, k in enumerate(all_kernels)
                          if "cijk_" in k.lower() or "gemm_xdl_preshuffle" in k.lower()]
        if len(cijk_positions) == 1:
            # Single FP8 dense GEMM — likely qkv_a in FP8 model
            return "qkv_a"
        elif len(cijk_positions) == 2:
            # Two FP8 dense GEMMs — q_b_proj and o_proj
            if position == cijk_positions[0]:
                return "q_b_proj"
            return "o_proj"
        elif len(cijk_positions) >= 3:
            # Three FP8 dense GEMMs — qkv_a, q_b_proj, o_proj
            if position == cijk_positions[0]:
                return "qkv_a"
            elif position == cijk_positions[1]:
                return "q_b_proj"
            return "o_proj"
        return "q_b_proj"

    if "mla_a8w8" in kn_lower:
        return "mla"
    if "kn_mla_reduce" in kn_lower:
        return "mla_reduce"
    if "kernel_moe_mxgemm" in kn_lower:
        if "mulabscaleshuffled" in kn_lower and "expertweight" not in kn_lower:
            return "moe_gate_up"
        return "moe_down"
    if "add_rmsnorm_quant" in kn_lower:
        rmsnorm_positions = [i for i, k in enumerate(all_kernels)
                             if "add_rmsnorm_quant" in k.lower()]
        if position == rmsnorm_positions[0] if rmsnorm_positions else -1:
            return "q_a_norm"
        return "kv_a_norm"
    if "reduce_scatter" in kn_lower:
        return "tp_reduce_scatter"
    if "local_device_load_rmsnorm" in kn_lower:
        return "tp_load_rmsnorm"
    if "fuse_qk_rope" in kn_lower:
        return "rope_cache"
    if "grouped_topk" in kn_lower:
        return "moe_topk"
    if "moesorting" in kn_lower:
        return "moe_sort"
    if "fused_dynamic_mxfp4_quant" in kn_lower:
        return "moe_quant_sort"
    if "dynamic_per_token_scaled_quant" in kn_lower:
        return "input_quant"
    if "triton_poi_fused" in kn_lower:
        return "triton_transpose"

    return "unknown"


def parse_xlsx(xlsx_path):
    """Parse ATOM decode_breakdown.xlsx. Returns list of (module, kernel, avg_us)."""
    try:
        import openpyxl
    except ImportError:
        print("ERROR: openpyxl required. pip install openpyxl", file=sys.stderr)
        return None

    if not os.path.exists(xlsx_path):
        print(f"WARNING: xlsx not found: {xlsx_path}", file=sys.stderr)
        return None

    try:
        wb = openpyxl.load_workbook(xlsx_path, read_only=True)
        ws = wb.active
        all_rows = list(ws.iter_rows(values_only=True))
        wb.close()
    except Exception as e:
        print(f"WARNING: failed to parse xlsx: {e}", file=sys.stderr)
        return None

    # header: cpu_module, gpu_kernel, duration_us, pct%, sum_per_module, module_pct%, avg_duration_us, ...
    results = []
    cur_module = ""
    for r in all_rows[1:]:
        module = str(r[0] or "").strip()
        kernel = str(r[1] or "").strip()
        avg_us = r[6]  # avg_duration_us

        if module == "TOTAL" or not kernel or avg_us is None:
            continue

        if module:
            cur_module = module

        try:
            avg_us_f = float(avg_us)
        except (ValueError, TypeError):
            continue

        results.append({
            "module": cur_module,
            "kernel": kernel,
            "avg_us": avg_us_f,
        })

    return results


def parse_manual_kernels(kernel_names):
    """Parse manual kernel name list. Returns list of (module, kernel, avg_us=0)."""
    return [{"module": "", "kernel": k.strip(), "avg_us": 0.0} for k in kernel_names if k.strip()]


def detect_aiter_repo():
    """Auto-detect aiter repo root via import."""
    try:
        import aiter
        return os.path.dirname(aiter.__path__[0])
    except ImportError:
        return None


def detect_arch():
    """Detect GPU architecture string."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.gcnArchName if hasattr(props, "gcnArchName") else "unknown"
    except Exception:
        pass

    # Fallback: check rocm-smi
    try:
        import subprocess
        out = subprocess.check_output(["rocm-smi", "--showproductname"], text=True, timeout=5)
        if "MI355X" in out or "MI350" in out:
            return "gfx950"
    except Exception:
        pass

    return "unknown"


def check_tune_status(aiter_repo, family_name, shape):
    """Check if a shape has been tuned in aiter's config dir."""
    if family_name not in TUNE_FAMILIES:
        return "not_applicable"

    family = TUNE_FAMILIES[family_name]
    tune_script = os.path.join(aiter_repo, family["tune_script"])

    if not os.path.exists(tune_script):
        return "no_script"

    # Look for existing tuned config CSVs in the same directory
    config_dir = os.path.dirname(tune_script)
    csv_files = glob.glob(os.path.join(config_dir, "*.csv"))

    for csv_file in csv_files:
        try:
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if family_name == "fp8_dense":
                        if (row.get("M") == str(shape.get("M")) and
                                row.get("N") == str(shape.get("N")) and
                                row.get("K") == str(shape.get("K"))):
                            return "tuned"
                    elif family_name == "fp8_batched":
                        if (row.get("B") == str(shape.get("B")) and
                                row.get("M") == str(shape.get("M")) and
                                row.get("N") == str(shape.get("N")) and
                                row.get("K") == str(shape.get("K"))):
                            return "tuned"
        except Exception:
            continue

    return "not_tuned"


def shorten_kernel(kernel_name, max_len=40):
    """Shorten kernel name for display."""
    if len(kernel_name) <= max_len:
        return kernel_name
    # Keep prefix and important parts
    if kernel_name.startswith("Cijk_"):
        # Extract tile info: MT64x32x128, SK3, ISA950
        parts = []
        for seg in kernel_name.split("_"):
            if seg.startswith("MT") or seg.startswith("SK") or seg.startswith("ISA"):
                parts.append(seg)
        return "Cijk_..." + "_".join(parts) if parts else kernel_name[:max_len]
    if "batched_gemm_a8w8" in kernel_name:
        # Extract block sizes
        parts = []
        for seg in kernel_name.split("_"):
            if seg.startswith("M") or seg.startswith("N") or seg.startswith("K"):
                if any(c.isdigit() for c in seg):
                    parts.append(seg)
        return "batched_gemm_a8w8_" + "_".join(parts[:3]) if parts else kernel_name[:max_len]
    return kernel_name[:max_len]


def run_advisor(kernel_entries, tp, bs, aiter_repo, arch):
    """Run the full advisor analysis."""
    shapes = compute_shapes(tp, bs)
    all_kernel_names = [e["kernel"] for e in kernel_entries]

    operators = []
    for i, entry in enumerate(kernel_entries):
        kernel = entry["kernel"]
        module = entry["module"]
        avg_us = entry["avg_us"]

        # Step 2: classify
        family, tunable, family_info = classify_kernel(kernel)

        # Map to logical operator
        op_name = map_kernel_to_operator(kernel, module, i, all_kernel_names)

        op_entry = {
            "operator": op_name,
            "module": module,
            "kernel": kernel,
            "family": family,
            "tunable": tunable,
            "avg_us": avg_us,
        }

        if not tunable:
            op_entry["reason"] = family_info.get("reason", "not tunable")
        else:
            # Step 3: get shape
            shape = shapes.get(op_name)
            if shape:
                op_entry["shape"] = {k: v for k, v in shape.items() if k != "type"}
            else:
                op_entry["shape"] = {}

            # Step 4: check tune status
            if aiter_repo and shape:
                tune_status = check_tune_status(aiter_repo, family, shape)
            else:
                tune_status = "unknown"

            op_entry["tune_status"] = tune_status
            op_entry["tune_script"] = family_info.get("tune_script", "")
            op_entry["config_env"] = family_info.get("config_env", "")

        operators.append(op_entry)

    return operators


def generate_tune_inputs(operators, output_dir, fp8_dtype="torch.float8_e4m3fnuz"):
    """Generate tune input CSVs for un-tuned operators."""
    ptpc_shapes = []
    batched_shapes = []

    for op in operators:
        if not op.get("tunable") or op.get("tune_status") != "not_tuned":
            continue
        shape = op.get("shape", {})
        if not shape:
            continue

        if op["family"] == "fp8_dense":
            entry = {"M": shape["M"], "N": shape["N"], "K": shape["K"], "q_dtype_w": fp8_dtype}
            if entry not in ptpc_shapes:
                ptpc_shapes.append(entry)
        elif op["family"] == "fp8_batched":
            entry = {"B": shape["B"], "M": shape["M"], "N": shape["N"], "K": shape["K"]}
            if entry not in batched_shapes:
                batched_shapes.append(entry)

    generated = []

    if ptpc_shapes:
        path = os.path.join(output_dir, "untuned_ptpc.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["M", "N", "K", "q_dtype_w"])
            w.writeheader()
            w.writerows(ptpc_shapes)
        generated.append(("untuned_ptpc.csv", len(ptpc_shapes)))

    if batched_shapes:
        path = os.path.join(output_dir, "untuned_batched.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["B", "M", "N", "K"])
            w.writeheader()
            w.writerows(batched_shapes)
        generated.append(("untuned_batched.csv", len(batched_shapes)))

    return generated


def print_summary(operators, config, generated):
    """Print human-readable summary."""
    print()
    print("=" * 80)
    print(f"  aiter Tune Advisor")
    print(f"  Source: {config['source']} | TP={config['tp']} BS={config['bs']} | Arch={config['arch']}")
    print("=" * 80)
    print()

    # Header
    fmt = " {:<16} {:<40} {:<14} {:<8} {:<12} {:<22} {:>6}"
    print(fmt.format("Operator", "Kernel", "Family", "Tunable", "Status", "Shape", "μs"))
    print(" " + "-" * 122)

    for op in operators:
        kernel_short = shorten_kernel(op["kernel"])
        tunable_str = "YES" if op["tunable"] else "No"
        status = op.get("tune_status", op.get("reason", "—"))
        if len(status) > 12:
            status = status[:11] + "…"

        shape = op.get("shape", {})
        if shape:
            if "B" in shape:
                shape_str = f"B={shape['B']} M={shape['M']} K={shape['K']} N={shape['N']}"
            else:
                shape_str = f"M={shape['M']} K={shape['K']} N={shape['N']}"
        else:
            shape_str = "—"

        avg_us = f"{op['avg_us']:.1f}" if op["avg_us"] > 0 else "—"
        print(fmt.format(op["operator"], kernel_short, op["family"], tunable_str, status, shape_str, avg_us))

    # Summary
    total = len(operators)
    tunable = sum(1 for o in operators if o.get("tunable"))
    needs_tune = sum(1 for o in operators if o.get("tune_status") == "not_tuned")
    already_tuned = sum(1 for o in operators if o.get("tune_status") == "tuned")
    not_tunable = total - tunable

    print()
    print(f" Summary: {total} operators total, {tunable} tunable, "
          f"{already_tuned} already tuned, {needs_tune} need tuning, {not_tunable} not tunable")

    if generated:
        print(f" Generated: {', '.join(f'{name} ({n} shapes)' for name, n in generated)}")
    elif needs_tune == 0 and tunable > 0:
        print(" All tunable ops already tuned!")
    elif tunable == 0:
        print(" No tunable GEMM kernels found (model may use BF16 attention weights)")

    print()


def main():
    parser = argparse.ArgumentParser(description="aiter Tune Advisor")
    parser.add_argument("--xlsx", help="Path to decode_breakdown*.xlsx (primary input)")
    parser.add_argument("--kernels", nargs="+", help="Manual kernel name list (fallback)")
    parser.add_argument("--tp", type=int, required=True, choices=[1, 2, 4, 8], help="TP size")
    parser.add_argument("--bs", type=int, default=64, help="Batch size (M dimension)")
    parser.add_argument("--aiter-repo", help="aiter repo root (auto-detect if omitted)")
    parser.add_argument("--output-dir", default=".", help="Output directory for CSVs and report")
    parser.add_argument("--fp8-dtype", default="torch.float8_e4m3fnuz",
                        help="FP8 dtype string for tune CSV (default: torch.float8_e4m3fnuz)")
    args = parser.parse_args()

    # Detect aiter repo
    aiter_repo = args.aiter_repo or detect_aiter_repo()
    if aiter_repo:
        print(f"aiter repo: {aiter_repo}")
    else:
        print("WARNING: aiter repo not found, tune status check skipped", file=sys.stderr)

    # Detect arch
    arch = detect_arch()
    print(f"Architecture: {arch}")

    # Parse input
    kernel_entries = None
    source = "unknown"

    if args.xlsx:
        kernel_entries = parse_xlsx(args.xlsx)
        if kernel_entries:
            source = os.path.basename(args.xlsx)
        else:
            print(f"WARNING: xlsx parse failed, trying --kernels fallback", file=sys.stderr)

    if kernel_entries is None and args.kernels:
        kernel_entries = parse_manual_kernels(args.kernels)
        source = "manual"

    if kernel_entries is None:
        print("ERROR: no input provided. Use --xlsx or --kernels", file=sys.stderr)
        sys.exit(1)

    if not kernel_entries:
        print("ERROR: no kernels found in input", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(kernel_entries)} kernels from {source}")

    # Run advisor
    operators = run_advisor(kernel_entries, args.tp, args.bs, aiter_repo, arch)

    # Generate tune inputs
    os.makedirs(args.output_dir, exist_ok=True)
    generated = generate_tune_inputs(operators, args.output_dir, args.fp8_dtype)

    # Build config for report
    config = {
        "tp": args.tp,
        "bs": args.bs,
        "model": "DeepSeek-R1-671B",
        "arch": arch,
        "source": source,
        "aiter_repo": aiter_repo or "",
    }

    # Print summary
    print_summary(operators, config, generated)

    # Save JSON report
    total = len(operators)
    tunable = sum(1 for o in operators if o.get("tunable"))
    needs_tune = sum(1 for o in operators if o.get("tune_status") == "not_tuned")
    already_tuned = sum(1 for o in operators if o.get("tune_status") == "tuned")

    report = {
        "config": config,
        "operators": operators,
        "summary": {
            "total_operators": total,
            "tunable": tunable,
            "already_tuned": already_tuned,
            "needs_tune": needs_tune,
            "not_tunable": total - tunable,
        },
    }

    report_path = os.path.join(args.output_dir, "advisor_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
