#!/usr/bin/env python3
"""
Generate B200 vs MI355X kernel map CSV from:
  - B200 layer_kernel_avg.csv (from trace_layer_detail.py)
  - MI355X decode_breakdown.xlsx (from run_parse_trace.py)

Output: kernel_map CSV with strict timeline alignment, overlap, and PASS grouping.

Rules:
  1. No missing operators — every B200 and MI355X kernel appears
  2. Original kernel names preserved exactly
  3. Data sources noted in footer
  4. Sum verification: total == sum of individual values

Usage:
    python3 generate_kernel_map.py \\
        --b200-csv layer_kernel_avg.csv \\
        --mi355x-xlsx decode_breakdown.xlsx \\
        --output b200_vs_mi355x_kernel_map.csv
"""

import argparse
import csv
import os
import sys

try:
    import openpyxl
except ImportError:
    print("ERROR: openpyxl required. pip install openpyxl", file=sys.stderr)
    sys.exit(1)


# ── B200 operator → PASS classification ──
PASS_MAP = {
    "EP_AR+residual+RMSNorm(fused)": None,  # position-dependent: #1→EP_AR_before_MHA, #2→EP_AR_before_MOE
    "qkv_a_proj_GEMM": "MHA",
    "qkv_a_splitK_reduce": "MHA",
    "q/k_norm_RMSNorm": "MHA",
    "q_b_proj_GEMM": "MHA",
    "uk_gemm(K_expansion)": "MHA",
    "k_concat": "MHA",
    "RoPE+KV_cache_write": "MHA",
    "set_mla_kv": "MHA",
    "Attention(FMHA)": "MHA",
    "uv_gemm(V_expansion)": "MHA",
    "q_b_proj_GEMM #2": "MHA",
    "o_proj_quant(BF16→FP4)": "O_proj",
    "o_proj_GEMM": "O_proj",
    "o_proj_GEMM #1": "O_proj",
    "o_proj_GEMM #2": "O_proj",
    "o_proj_quant(BF16→FP4) #1": "O_proj",
    "o_proj_quant(BF16→FP4) #2": "O_proj",
    "Moe_Expert_quant(BF16→FP4)": "MOE",
    "router_GEMM": "MOE",
    "router_splitK_reduce": "MOE",
    "MoE_input_quant(BF16→FP4)": "MOE",
    "tensor_copy": "MOE",
    "TopK_select": "MOE",
    "expert_sort": "MOE",
    "SiLU×Mul": "MOE",
    "shared_quant(BF16→FP4)": "MOE",
    "shared_GEMM(FP4)": "MOE",
    "gate_up_GEMM(+SwiGLU)": "MOE",
    "down_GEMM": "MOE",
    "MoE_finalize+residual": "MOE",
    "MoE_finalize": "MOE",
    "residual_add": "MOE",
}

# ── MI355X module → PASS classification ──
# Used to classify MI355X kernels independently of which B200 row they're mapped to.
# This is needed because B200's cross-stream overlap can place o_proj_GEMM after
# router_GEMM, causing MI355X MoE kernels to be mapped to a B200 O_proj row.
MI355X_MODULE_PASS = {
    "input_layernorm": "EP_AR_before_MHA",
    "gemm_a16w16": None,  # position-dependent (MHA or MOE)
    "hipLaunchKernel": "MHA",
    "q_proj_and_k_up_proj": "MHA",
    "rope_and_kv_cache": "MHA",
    "mla_decode": "MHA",
    "v_up_proj_and_o_proj": "MHA",
    "triton_poi_fused_as_strided_clone_copy__0": "MHA",
    "post_attn_layernorm": "EP_AR_before_MOE",
    "triton_poi_fused_as_strided_clone_1": "EP_AR_before_MOE",
    "triton_poi": "EP_AR_before_MOE",
    "rocm_aiter_biased_grouped_topk_impl": "MOE",
    "mxfp4_moe": "MOE",
}


def get_pass(operator, occurrence_num):
    """Get PASS name for a B200 operator.

    Operators from trace_layer_detail.py may have 'other:' prefix with full
    kernel name (e.g. 'other:void router_gemm_kernel_float_output<...').
    We pattern-match these to the correct PASS.
    """
    p = PASS_MAP.get(operator)
    if p is not None:
        return p
    # EP_AR is position-dependent
    if "EP_AR" in operator:
        return "EP_AR_before_MHA" if occurrence_num == 1 else "EP_AR_before_MOE"
    # Pattern-match 'other:' prefixed operators by kernel name keywords
    op_lower = operator.lower()
    if "router_gemm" in op_lower:
        return "MOE"
    if "finalizekernel" in op_lower or "finalize" in op_lower:
        return "MOE"
    if "routingmainkernel" in op_lower or "routingindices" in op_lower:
        return "MOE"
    if "bmm_" in op_lower and ("e2m1" in op_lower or "bfloat16" in op_lower):
        return "MOE"
    if "act_and_mul" in op_lower:
        return "MOE"
    if "fused_a_gemm" in op_lower or "shared" in op_lower:
        return "MOE"
    # other/unknown → other
    return "other"


def get_mi355x_pass(module, b200_pass):
    """Get PASS for MI355X kernel, using its module name when available.

    Falls back to b200_pass when the MI355X module is unknown or position-dependent.
    """
    if not module:
        return b200_pass
    p = MI355X_MODULE_PASS.get(module)
    if p is not None:
        return p
    # For position-dependent modules (gemm_a16w16), use B200 PASS
    return b200_pass


def parse_b200_csv(path):
    """Parse B200 layer_kernel_avg.csv. Returns list of dicts."""
    rows = []
    totals = {}
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Detect column positions by name — B200 columns only (exclude MI355X)
        col_map = {}
        for i, h in enumerate(header):
            h_raw = h.strip()
            h_lower = h_raw.lower()
            if "mi355x" in h_lower:
                continue  # skip MI355X columns
            if "operator" in h_lower:
                col_map["operator"] = i
            elif "raw_kernel" in h_lower:
                col_map["raw_kernel"] = i
            elif "stream" in h_lower:
                col_map["stream"] = i
            elif "avg_us" in h_lower and "overlap" not in h_lower:
                col_map["avg_us"] = i
            elif "overlap_us" in h_lower:
                col_map["overlap_us"] = i
            elif "overlap_with" in h_lower:
                col_map["overlap_with"] = i
            elif "module" in h_lower:
                col_map["module"] = i
            elif h_raw == "#":
                col_map["num"] = i

        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue
            # Summary rows
            joined = ",".join(row)
            if "TOTAL" in joined or "Walltime" in joined or "Overlap" in joined or "PASS" in joined or "Silicon" in joined:
                # Parse totals
                for i, c in enumerate(row):
                    c = c.strip()
                    if "kernel_sum" in c or "TOTAL" in c:
                        for j in range(i + 1, len(row)):
                            try:
                                totals["kernel_sum"] = float(row[j].strip())
                                break
                            except (ValueError, IndexError):
                                continue
                    if "Walltime" in c:
                        for j in range(i + 1, len(row)):
                            try:
                                totals["walltime"] = float(row[j].strip())
                                break
                            except (ValueError, IndexError):
                                continue
                    if c == "Overlap" or "B200 Overlap" in c:
                        for j in range(i + 1, len(row)):
                            try:
                                totals["overlap"] = float(row[j].strip())
                                break
                            except (ValueError, IndexError):
                                continue
                continue

            operator = row[col_map["operator"]].strip() if "operator" in col_map else ""
            if not operator:
                continue

            try:
                avg_us = float(row[col_map["avg_us"]].strip()) if "avg_us" in col_map else 0
            except (ValueError, IndexError):
                continue

            overlap_us = 0
            if "overlap_us" in col_map:
                try:
                    overlap_us = float(row[col_map["overlap_us"]].strip())
                except (ValueError, IndexError):
                    pass

            rows.append({
                "operator": operator,
                "raw_kernel": row[col_map.get("raw_kernel", 0)].strip() if "raw_kernel" in col_map else "",
                "stream": row[col_map.get("stream", 0)].strip() if "stream" in col_map else "",
                "avg_us": avg_us,
                "overlap_us": overlap_us,
                "overlap_with": row[col_map.get("overlap_with", 0)].strip() if "overlap_with" in col_map else "",
                "module": row[col_map.get("module", 0)].strip() if "module" in col_map else "",
            })

    return rows, totals


def parse_mi355x_xlsx(path):
    """Parse MI355X decode_breakdown.xlsx. Returns list of dicts."""
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    all_rows = list(ws.iter_rows(values_only=True))
    wb.close()

    # header: cpu_module, gpu_kernel, duration_us, pct%, sum_per_module, module_pct%, avg_duration_us, avg_sum_per_module
    rows = []
    total_avg = None
    for r in all_rows[1:]:
        module = r[0] or ""
        kernel = r[1] or ""
        avg_us = r[6]  # avg duration_us

        if str(module).strip() == "TOTAL":
            total_avg = float(avg_us) if avg_us else None
            continue

        if avg_us is None:
            continue

        rows.append({
            "module": str(module).strip(),
            "kernel": str(kernel).strip(),
            "avg_us": float(avg_us),
        })

    # Fill in parent_module: the module each row actually belongs to
    # (inheriting from previous row when module is empty)
    cur_mod = ""
    for r in rows:
        if r["module"]:
            cur_mod = r["module"]
        r["parent_module"] = cur_mod

    return rows, total_avg


def generate_map(b200_rows, b200_totals, mi355x_rows, mi355x_total, b200_csv_path, mi355x_xlsx_path, output_path):
    """Generate kernel map CSV.

    The map is a left-right table:
    - Left side: ALL B200 operators (in order, no omissions)
    - Right side: ALL MI355X operators (in order, no omissions)
    - Timeline alignment: MI355X rows placed next to the B200 row they correspond to
    - B200-only rows: MI355X columns blank
    - MI355X-only rows: added as continuation rows (B200 columns blank except for alignment)
    """

    # ── B200-only operator patterns (no MI355X equivalent) ──
    B200_ONLY_KEYWORDS = [
        "set_mla_kv", "splitK_reduce", "MoE_input_quant", "tensor_copy",
        "MoE_finalize", "residual_add", "o_proj_quant", "shared_quant",
        "Moe_Expert_quant", "fused_a_gemm_kernel",
    ]

    def is_b200_only(operator):
        for kw in B200_ONLY_KEYWORDS:
            if kw in operator:
                return True
        return False

    # ── Pre-count MI355X kernels per module for continuation logic ──
    # Only allow continuation (grouping sub-kernels under one B200 row)
    # for small modules (≤2 kernels). Large modules like
    # rocm_aiter_biased_grouped_topk_impl (7 kernels spanning MOE routing+compute)
    # must be mapped individually.
    mi355x_module_sizes = {}
    cur_mod = None
    for mi in mi355x_rows:
        mod = mi["module"] if mi["module"] else cur_mod
        if mod:
            mi355x_module_sizes[mod] = mi355x_module_sizes.get(mod, 0) + 1
            cur_mod = mod

    # ── Build output rows ──
    output_rows = []
    mi355x_idx = 0
    ep_ar_count = 0

    # Track PASS sums
    pass_b200 = {}
    pass_mi355x = {}
    pass_nv_kernels = {}
    pass_amd_kernels = {}

    b200_sum_check = 0.0
    mi355x_sum_check = 0.0

    for b200 in b200_rows:
        op = b200["operator"]

        # Track EP_AR occurrence for PASS classification
        if "EP_AR" in op and "#" not in op:
            ep_ar_count += 1
            pass_name = get_pass(op, ep_ar_count)
        elif "#2" in op and "EP_AR" in op:
            pass_name = "EP_AR_before_MOE"
        elif "#1" in op and "EP_AR" in op:
            pass_name = "EP_AR_before_MHA"
        else:
            pass_name = get_pass(op, 0)

        # Accumulate B200 pass
        pass_b200[pass_name] = pass_b200.get(pass_name, 0) + b200["avg_us"]
        b200_sum_check += b200["avg_us"]

        # Track NV kernels per pass
        if pass_name not in pass_nv_kernels:
            pass_nv_kernels[pass_name] = []
        pass_nv_kernels[pass_name].append(b200["raw_kernel"])

        if is_b200_only(op):
            # B200-only row — no MI355X
            output_rows.append({
                "b200_operator": op,
                "b200_raw_kernel": b200["raw_kernel"],
                "b200_stream": b200["stream"],
                "b200_avg_us": b200["avg_us"],
                "b200_overlap_us": b200["overlap_us"],
                "b200_overlap_with": b200["overlap_with"],
                "mi355x_module": "",
                "mi355x_kernel": "",
                "mi355x_avg_us": "",
                "notes": "B200 only",
                "pass": pass_name,
            })
        else:
            # Map next MI355X row(s) to this B200 row
            if mi355x_idx < len(mi355x_rows):
                mi = mi355x_rows[mi355x_idx]
                mi355x_idx += 1

                mi355x_sum_check += mi["avg_us"]
                # Classify MI355X kernel by its own module, not the B200 row
                mi_pass = get_mi355x_pass(mi["parent_module"], pass_name)
                if mi_pass not in pass_mi355x:
                    pass_mi355x[mi_pass] = 0
                pass_mi355x[mi_pass] += mi["avg_us"]
                if mi_pass not in pass_amd_kernels:
                    pass_amd_kernels[mi_pass] = []
                pass_amd_kernels[mi_pass].append(mi["kernel"])

                output_rows.append({
                    "b200_operator": op,
                    "b200_raw_kernel": b200["raw_kernel"],
                    "b200_stream": b200["stream"],
                    "b200_avg_us": b200["avg_us"],
                    "b200_overlap_us": b200["overlap_us"],
                    "b200_overlap_with": b200["overlap_with"],
                    "mi355x_module": mi["module"],
                    "mi355x_kernel": mi["kernel"],
                    "mi355x_avg_us": mi["avg_us"],
                    "notes": "",
                    "pass": pass_name,
                })

                # Check if MI355X module has more kernels (continuation rows)
                # Only allow continuation for small modules (≤2 kernels).
                # Large modules (e.g. rocm_aiter_biased_grouped_topk_impl with 7 kernels)
                # must be mapped 1:1 to individual B200 rows.
                while mi355x_idx < len(mi355x_rows):
                    next_mi = mi355x_rows[mi355x_idx]
                    # Use parent_module to determine the true module for empty-module rows
                    parent_mod = next_mi["parent_module"]
                    module_size = mi355x_module_sizes.get(parent_mod, 1)
                    # Continuation: same parent module as current, or module is empty (sub-kernel)
                    # BUT only if the module has ≤2 kernels total
                    if module_size <= 2 and (next_mi["module"] == "" or next_mi["module"] == mi["module"]):
                        mi355x_idx += 1
                        mi355x_sum_check += next_mi["avg_us"]
                        cont_mi_pass = get_mi355x_pass(next_mi["parent_module"], pass_name)
                        if cont_mi_pass not in pass_mi355x:
                            pass_mi355x[cont_mi_pass] = 0
                        pass_mi355x[cont_mi_pass] += next_mi["avg_us"]
                        if cont_mi_pass not in pass_amd_kernels:
                            pass_amd_kernels[cont_mi_pass] = []
                        pass_amd_kernels[cont_mi_pass].append(next_mi["kernel"])
                        output_rows.append({
                            "b200_operator": "",
                            "b200_raw_kernel": "",
                            "b200_stream": "",
                            "b200_avg_us": "",
                            "b200_overlap_us": "",
                            "b200_overlap_with": "",
                            "mi355x_module": next_mi["module"],
                            "mi355x_kernel": next_mi["kernel"],
                            "mi355x_avg_us": next_mi["avg_us"],
                            "notes": f"({mi['module'] or mi['parent_module']} cont.)",
                            "pass": pass_name,
                        })
                    else:
                        break
            else:
                # No more MI355X rows
                output_rows.append({
                    "b200_operator": op,
                    "b200_raw_kernel": b200["raw_kernel"],
                    "b200_stream": b200["stream"],
                    "b200_avg_us": b200["avg_us"],
                    "b200_overlap_us": b200["overlap_us"],
                    "b200_overlap_with": b200["overlap_with"],
                    "mi355x_module": "",
                    "mi355x_kernel": "",
                    "mi355x_avg_us": "",
                    "notes": "",
                    "pass": pass_name,
                })

    # ── Remaining MI355X rows (not mapped to any B200 row) ──
    unmapped_mi355x = []
    while mi355x_idx < len(mi355x_rows):
        mi = mi355x_rows[mi355x_idx]
        mi355x_idx += 1
        mi355x_sum_check += mi["avg_us"]
        unmapped_mi355x.append(mi)
        output_rows.append({
            "b200_operator": "",
            "b200_raw_kernel": "",
            "b200_stream": "",
            "b200_avg_us": "",
            "b200_overlap_us": "",
            "b200_overlap_with": "",
            "mi355x_module": mi["module"],
            "mi355x_kernel": mi["kernel"],
            "mi355x_avg_us": mi["avg_us"],
            "notes": "MI355X only (unmapped)",
            "pass": "unmapped",
        })

    # ── Checksum verification ──
    b200_expected = b200_totals.get("kernel_sum")
    mi355x_expected = mi355x_total

    checksum_notes = []
    if b200_expected is not None:
        diff = abs(b200_sum_check - b200_expected)
        if diff > 0.5:
            checksum_notes.append(f"WARNING: B200 sum mismatch: computed={b200_sum_check:.2f} expected={b200_expected:.1f} diff={diff:.2f}")
        else:
            checksum_notes.append(f"OK: B200 sum={b200_sum_check:.2f} matches expected={b200_expected:.1f}")

    if mi355x_expected is not None:
        diff = abs(mi355x_sum_check - mi355x_expected)
        if diff > 0.2:
            checksum_notes.append(f"WARNING: MI355X sum mismatch: computed={mi355x_sum_check:.2f} expected={mi355x_expected:.2f} diff={diff:.2f}")
        else:
            checksum_notes.append(f"OK: MI355X sum={mi355x_sum_check:.2f} matches expected={mi355x_expected:.2f}")

    if unmapped_mi355x:
        checksum_notes.append(f"WARNING: {len(unmapped_mi355x)} MI355X operators unmapped")

    # ── Write CSV ──
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)

        # Header
        w.writerow(["B200_Operator", "B200_Raw_Kernel", "B200_Stream", "B200_Avg_us",
                     "B200_Overlap_us", "B200_Overlap_With",
                     "MI355X_Module", "MI355X_Kernel", "MI355X_Avg_us", "Notes"])

        # Data rows
        for r in output_rows:
            w.writerow([
                r["b200_operator"], r["b200_raw_kernel"], r["b200_stream"],
                f'{r["b200_avg_us"]:.1f}' if isinstance(r["b200_avg_us"], (int, float)) and r["b200_avg_us"] != "" else r["b200_avg_us"],
                f'{r["b200_overlap_us"]:.1f}' if isinstance(r["b200_overlap_us"], (int, float)) and r["b200_overlap_us"] != "" else r["b200_overlap_us"],
                r["b200_overlap_with"],
                r["mi355x_module"], r["mi355x_kernel"],
                f'{r["mi355x_avg_us"]:.2f}' if isinstance(r["mi355x_avg_us"], (int, float)) and r["mi355x_avg_us"] != "" else r["mi355x_avg_us"],
                r["notes"],
            ])

        # Blank line
        w.writerow([])

        # Totals
        w.writerow(["B200 TOTAL (kernel_sum)", "", "", f"{b200_sum_check:.1f}", "", "",
                     "MI355X TOTAL", "", f"{mi355x_sum_check:.2f}", ""])
        if b200_totals.get("walltime"):
            w.writerow(["B200 Walltime", "", "", f'{b200_totals["walltime"]:.1f}', "", "", "", "", "", ""])
        if b200_totals.get("overlap"):
            overlap = b200_totals["overlap"]
            pct = overlap / b200_sum_check * 100 if b200_sum_check > 0 else 0
            w.writerow(["B200 Overlap", "", "", f"{overlap:.1f}", "", "", "", "", "",
                         f"{overlap:.1f}us overlap = {pct:.1f}% of kernel sum"])

        # Blank line
        w.writerow([])

        # PASS summary
        w.writerow(["", "Silicon NV", "Silicon AMD", "", "", "", "", "", "", ""])
        w.writerow(["PASS", "B200", "MI355X", "gap", "NV_Kernels", "AMD_Kernels", "", "", "", ""])

        pass_order = ["EP_AR_before_MHA", "MHA", "O_proj", "EP_AR_before_MOE", "MOE", "other"]
        for pname in pass_order:
            b200_val = pass_b200.get(pname, 0)
            mi355x_val = pass_mi355x.get(pname, 0)
            gap = mi355x_val - b200_val
            nv_k = "\n".join(pass_nv_kernels.get(pname, []))
            amd_k = "\n".join(pass_amd_kernels.get(pname, []))
            if b200_val > 0 or mi355x_val > 0:
                w.writerow([pname, f"{b200_val:.1f}", f"{mi355x_val:.2f}",
                            f"{gap:+.1f}", nv_k, amd_k, "", "", "", ""])

        # Blank line
        w.writerow([])

        # Footer: data sources + checksum
        w.writerow(["# Data Sources:", "", "", "", "", "", "", "", "", ""])
        w.writerow([f"#   B200: {os.path.abspath(b200_csv_path)}", "", "", "", "", "", "", "", "", ""])
        w.writerow([f"#   MI355X: {os.path.abspath(mi355x_xlsx_path)}", "", "", "", "", "", "", "", "", ""])
        w.writerow(["# Checksum:", "", "", "", "", "", "", "", "", ""])
        for note in checksum_notes:
            w.writerow([f"#   {note}", "", "", "", "", "", "", "", "", ""])

    print(f"Written: {output_path}")
    print(f"  B200 operators: {len(b200_rows)}, MI355X operators: {len(mi355x_rows)}")
    print(f"  Output rows: {len(output_rows)}")
    for note in checksum_notes:
        print(f"  {note}")


def main():
    parser = argparse.ArgumentParser(description="Generate B200 vs MI355X kernel map")
    parser.add_argument("--b200-csv", required=True, help="B200 layer_kernel_avg.csv")
    parser.add_argument("--mi355x-xlsx", required=True, help="MI355X decode_breakdown.xlsx")
    parser.add_argument("--output", default="b200_vs_mi355x_kernel_map.csv", help="Output CSV path")
    args = parser.parse_args()

    b200_rows, b200_totals = parse_b200_csv(args.b200_csv)
    mi355x_rows, mi355x_total = parse_mi355x_xlsx(args.mi355x_xlsx)

    print(f"B200: {len(b200_rows)} operators, totals={b200_totals}")
    print(f"MI355X: {len(mi355x_rows)} operators, total_avg={mi355x_total}")

    generate_map(b200_rows, b200_totals, mi355x_rows, mi355x_total,
                 args.b200_csv, args.mi355x_xlsx, args.output)


if __name__ == "__main__":
    main()
