#!/usr/bin/env python3
"""
Extract per-decode-step kernel breakdown from Torch Profiler traces
containing CUDA Graph captures.

Works with SGLang and any framework that uses CUDA Graphs during decode.
The trace viewer shows cudaGraphLaunch as a single block on the GPU timeline,
but the underlying trace JSON contains flow events (ph="s"/"f") linking each
launch to the individual GPU kernels executed inside the graph.

This script:
  1. Finds all cudaGraphLaunch events on the CPU timeline
  2. Follows flow events to map each launch → list of GPU kernels
  3. Falls back to time-window matching if flow events are absent
  4. Maps kernel names to logical operators (configurable per platform)
  5. Outputs per-decode-step and averaged kernel breakdown tables

Usage:
    # Analyze SGLang B200 trace:
    python3 scripts/extract_cuda_graph_kernels.py trace.json.gz --platform b200

    # Analyze ATOM MI355X trace:
    python3 scripts/extract_cuda_graph_kernels.py trace.json.gz --platform mi355x

    # Just dump raw kernels per step (no logical mapping):
    python3 scripts/extract_cuda_graph_kernels.py trace.json.gz --raw

    # Output CSV:
    python3 scripts/extract_cuda_graph_kernels.py trace.json.gz --csv breakdown.csv
"""

import argparse
import csv
import gzip
import json
import re
import sys
from collections import defaultdict, OrderedDict

# =============================================================================
# Kernel name → logical operator mapping
# =============================================================================

# B200 SGLang FP4 kernel mapping
# Kernel names vary between nsys (short) and torch profiler (full C++ mangled).
# Regexes must match both forms.
B200_KERNEL_MAP = OrderedDict([
    # Pre-attention: MoE finalize + residual + lamport allreduce
    (r"finalizeKernelVecLoad|moefinalize", "pre_attn: MoE_finalize+residual"),
    (r"vectorized_elementwise_kernel.*CUDAFunctor_add|elementwise.*add", "pre_attn: residual_add"),
    # Communication: lamport allreduce fusion (includes RMSNorm)
    (r"allreduce_fusion_kernel.*lamport|moefinalize_lamport", "comm: lamport_AR+RMSNorm"),
    # QKV projection
    (r"splitK_TNT|nvjet_splitK_TNT", "qkv_proj: qkv_a_proj_GEMM"),
    (r"splitKreduce.*bf16|splitKreduce.*bfloat|splitKreduce.*Bfloat16", "qkv_proj: qkv_a_splitK_reduce"),
    (r"RMSNormKernel", "qkv_proj: q/k_norm_RMSNorm"),
    (r"_v_bz_TNN|nvjet_tst_TNN", "qkv_proj: q_b_proj_GEMM"),
    (r"_v_bz_TNT|nvjet_sm100_tst_128x64.*TNT", "qkv_proj: uk_gemm"),
    (r"CatArrayBatchedCopy", "qkv_proj: k_concat"),
    # RoPE + Attention
    (r"RopeQuantizeKernel|applyMLARopeAndAssignQKV", "rope_attn: RoPE+KV_write"),
    (r"fmhaSm100|fmhaKernel", "rope_attn: Attention_FMHA"),
    (r"set_mla_kv_buffer", "rope_attn: set_mla_kv"),
    # Output projection
    (r"_h_bz_TNT(?!.*splitK)|nvjet_tst_TNT", "out_proj: uv_gemm"),
    (r"_h_bz_splitK_TNT", "out_proj: o_proj_splitK_GEMM"),
    (r"nvjet_ootst_FP4|DeviceGemmFp4GemmSm100", "out_proj/shared: FP4_GEMM"),
    (r"quantize_with_block_size", "quant: FP4_blockwise_quant"),
    # Post-attention communication
    (r"userbuffers_rmsnorm", "post_attn: TP_AR+RMSNorm"),
    (r"userbuffers_allgather", "post_attn: EP_allgather"),
    # Router
    (r"nvjet_tss_splitK|splitK.*router", "router: router_GEMM"),
    (r"splitKreduce.*fp32|splitKreduce.*float32|splitKreduce.*Fp32", "router: router_splitK_reduce"),
    (r"routingMainKernel", "router: TopK_select"),
    (r"routingIndicesCluster", "router: expert_sort"),
    # MoE expert
    (r"bmm_E2m1.*[Ss]wi[Gg]lu|bmm_E2m1.*E2m1E2m1", "moe: gate_up_GEMM"),
    (r"bmm_Bfloat16|bmm_.*E2m1.*Bfloat", "moe: down_GEMM"),
    # Shared expert
    (r"act_and_mul_kernel|silu_and_mul_kernel", "shared: SiLU_mul"),
    # Elementwise
    (r"cvt_fp16_to_fp4|cvt_fp4", "quant: FP4_convert"),
    # Communication (catch-all after specific lamport pattern)
    (r"allreduce|reduce_scatter|all_gather|nccl", "comm: allreduce/other"),
    # Memory
    (r"memcpy|memset", "mem: copy/set"),
    # Copy kernels
    (r"unrolled_elementwise_kernel.*direct_copy|direct_copy_kernel", "mem: tensor_copy"),
])

# MI355X ATOM MXFP4 kernel mapping
MI355X_KERNEL_MAP = OrderedDict([
    (r"reduce_scatter", "pre_attn_comm: reduce_scatter"),
    (r"rmsnorm|rms_norm", "norm: RMSNorm"),
    (r"dynamic_per_token_scaled_quant", "quant: per_token_quant"),
    (r"fused_rms_fp8_group_quant", "quant: fused_rms_fp8_group_quant"),
    (r"gemm_xdl_preshuffle", "gemm: preshuffle_GEMM"),
    (r"batched_gemm_a8w8", "gemm: batched_a8w8_GEMM"),
    (r"fuse_qk_rope_concat_and_cache_mla", "rope_attn: RoPE+KV_write"),
    (r"mla_a8w8_qh16", "rope_attn: Attention_MLA"),
    (r"kn_mla_reduce", "rope_attn: MLA_reduce"),
    (r"bf16gemm_splitk", "router: router_GEMM"),
    (r"grouped_topk_opt_sort", "router: TopK_select"),
    (r"MoeSorting", "router: MoE_sort"),
    (r"fused_mxfp4_quant_moe_sort", "moe: fused_quant_sort"),
    (r"kernel_moe_mxgemm", "moe: MoE_GEMM"),
    (r"allreduce|all_gather|rccl", "comm: allreduce/other"),
    (r"memcpy|memset", "mem: copy/set"),
])


def classify_kernel(name, kernel_map):
    """Map a kernel name to its logical operator using regex patterns."""
    for pattern, label in kernel_map.items():
        if re.search(pattern, name, re.IGNORECASE):
            return label
    return f"other: {name[:50]}"


# =============================================================================
# Per-layer analysis: split decode step into transformer layers
# =============================================================================

# Module definitions for per-layer analysis.
# Each layer's kernels are assigned to modules based on execution position
# relative to FMHA (attention) anchor kernel.
# Module order within one transformer layer (DeepSeek-R1 MLA decode):
#   1. comm_norm: AllReduce+RMSNorm (pre-attention)
#   2. qkv_proj: q_a_proj, splitK_reduce, RMSNorm, q_b_proj, uk_gemm
#   3. rope_attn: RoPE+KV_write, set_mla_kv, FMHA (anchor)
#   4. out_proj: uv_gemm, FP4_GEMM(out), FP4_convert(out)
#   5. comm_norm: AllReduce+RMSNorm (post-attention) + residual
#   6. router: router_GEMM(splitK), TopK, expert_sort
#   7. moe_expert: FP4_blockwise_quant, gate_up_GEMM(fused SwiGlu), down_GEMM, finalize
#   8. shared_expert: FP4_GEMM(shared), FP4_convert(shared), SiLU_mul

# Kernel classifiers for layer splitting (regex → short tag)
LAYER_KERNEL_TAGS = OrderedDict([
    (r"fmhaSm100|fmhaKernel", "FMHA"),
    (r"allreduce_fusion_kernel.*lamport", "LAMPORT"),
    (r"splitK_TNT", "SPLITK_TNT"),
    (r"splitKreduce.*bf16|splitKreduce.*bfloat|splitKreduce.*Bfloat16", "SPLITK_REDUCE"),
    (r"RMSNormKernel|FusedAddRMSNorm", "RMSNORM"),
    (r"_v_bz_TNN|nvjet_tst_TNN", "Q_B_PROJ"),
    (r"_v_bz_TNT|nvjet_sm100_tst_128x64.*TNT|nvjet_sm100_tst_256x64.*TNT", "UK_GEMM"),
    (r"RopeQuantizeKernel", "ROPE"),
    (r"set_mla_kv_buffer", "SET_MLA_KV"),
    (r"_h_bz_TNT(?!.*splitK)", "UV_GEMM"),
    (r"DeviceGemmFp4GemmSm100", "FP4_GEMM"),
    (r"cvt_fp16_to_fp4", "FP4_CONVERT"),
    (r"vectorized_elementwise_kernel.*CUDAFunctor_add|CUDAFunctorOnSelf_add", "RESIDUAL_ADD"),
    (r"routingMainKernel", "TOPK"),
    (r"routingIndicesCluster", "EXPERT_SORT"),
    (r"quantize_with_block_size", "FP4_BLOCK_QUANT"),
    (r"bmm_E2m1.*E2m1E2m1", "MOE_GATE_UP"),
    (r"bmm_Bfloat16.*E2m1|bmm_.*E2m1.*Bfloat", "MOE_DOWN"),
    (r"finalizeKernelVecLoad", "MOE_FINALIZE"),
    (r"act_and_mul_kernel|silu_and_mul_kernel", "SILU_MUL"),
    (r"allreduce|nccl", "NCCL"),
    (r"memcpy|memset", "MEMOP"),
    (r"elementwise_kernel.*direct_copy|unrolled_elementwise.*direct_copy", "TENSOR_COPY"),
])


def tag_kernel(name):
    """Assign a short tag to a kernel based on its name."""
    for pattern, tag in LAYER_KERNEL_TAGS.items():
        if re.search(pattern, name, re.IGNORECASE):
            return tag
    return "OTHER"


def split_step_into_layers(kernels):
    """Split a decode step's kernels into per-layer groups using FMHA as anchor.

    FMHA appears exactly once per transformer layer, so we use it as the
    layer boundary marker. Kernels between consecutive FMHAs belong to
    the same layer (with the FMHA itself marking the attention point).

    Returns list of layer dicts, each with:
      - layer_idx: 0-based layer number
      - kernels: list of kernel dicts with added 'tag' field
      - fmha_pos: index of FMHA within the layer's kernel list
    """
    # Tag all kernels
    tagged = []
    for k in kernels:
        t = dict(k)
        t["tag"] = tag_kernel(k["name"])
        tagged.append(t)

    # Find FMHA positions
    fmha_indices = [i for i, k in enumerate(tagged) if k["tag"] == "FMHA"]

    if not fmha_indices:
        return []

    # Build layers: each layer spans from one "pre-FMHA boundary" to the next.
    # Strategy: FMHA is in the middle of each layer. The layer boundary is
    # roughly halfway between consecutive FMHAs' surrounding LAMPORT kernels.
    # Simpler approach: find the LAMPORT (pre-attn) kernel before each FMHA
    # and use that as the layer start.
    layers = []
    for li, fmha_idx in enumerate(fmha_indices):
        # Find the pre-attention LAMPORT before this FMHA
        # Scan backwards from FMHA to find the nearest LAMPORT
        layer_start = 0
        for j in range(fmha_idx - 1, -1, -1):
            if tagged[j]["tag"] == "LAMPORT":
                layer_start = j
                break
            # Stop if we hit a previous FMHA (shouldn't happen)
            if tagged[j]["tag"] == "FMHA":
                layer_start = j + 1
                break

        # If this isn't the first layer, don't overlap with previous layer
        if li > 0:
            prev_fmha = fmha_indices[li - 1]
            # Layer boundary is after the previous layer's MOE_FINALIZE
            # Find it by scanning forward from prev FMHA
            boundary = prev_fmha + 1
            for j in range(prev_fmha + 1, fmha_idx):
                if tagged[j]["tag"] == "MOE_FINALIZE":
                    boundary = j + 1
                    break
                if tagged[j]["tag"] == "LAMPORT":
                    boundary = j
                    break
            layer_start = max(layer_start, boundary)

        # Layer end: just before next layer's start, or end of kernels
        if li + 1 < len(fmha_indices):
            next_fmha = fmha_indices[li + 1]
            # Find the LAMPORT before next FMHA
            layer_end = next_fmha
            for j in range(next_fmha - 1, fmha_idx, -1):
                if tagged[j]["tag"] == "LAMPORT":
                    layer_end = j
                    break
            # Also check for MOE_FINALIZE as boundary
            for j in range(fmha_idx + 1, next_fmha):
                if tagged[j]["tag"] == "MOE_FINALIZE":
                    layer_end = j + 1
                    # Include kernels after MOE_FINALIZE until next LAMPORT
                    for jj in range(j + 1, next_fmha):
                        if tagged[jj]["tag"] == "LAMPORT":
                            layer_end = jj
                            break
                        layer_end = jj + 1
                    break
        else:
            layer_end = len(tagged)

        layer_kernels = tagged[layer_start:layer_end]
        fmha_pos = fmha_idx - layer_start

        layers.append({
            "layer_idx": li,
            "kernels": layer_kernels,
            "fmha_pos": fmha_pos,
            "n_kernels": len(layer_kernels),
        })

    return layers


# Module assignment based on position within layer
# Position phases relative to FMHA:
#   BEFORE FMHA: comm_norm(pre) → qkv_proj → rope → [FMHA]
#   AFTER FMHA:  out_proj → comm_norm(post) → router → moe_expert/shared_expert

B200_MODULE_MAP = {
    # Before FMHA
    "LAMPORT":        lambda before: "comm_norm" if before else "comm_norm",
    "SPLITK_TNT":     lambda before: "qkv_proj" if before else "router",
    "SPLITK_REDUCE":  lambda before: "qkv_proj" if before else "router",
    "RMSNORM":        lambda before: "qkv_proj" if before else "residual_norm",
    "Q_B_PROJ":       lambda before: "qkv_proj",
    "UK_GEMM":        lambda before: "qkv_proj",
    "ROPE":           lambda before: "rope_attn",
    "SET_MLA_KV":     lambda before: "rope_attn",
    "FMHA":           lambda before: "rope_attn",
    # After FMHA
    "UV_GEMM":        lambda before: "out_proj",
    "FP4_GEMM":       lambda before: "out_proj" if before else None,  # resolved by position
    "FP4_CONVERT":    lambda before: "out_proj" if before else None,  # resolved by position
    "RESIDUAL_ADD":   lambda before: "residual_mem",
    "TOPK":           lambda before: "router",
    "EXPERT_SORT":    lambda before: "router",
    "FP4_BLOCK_QUANT": lambda before: "moe_expert",
    "MOE_GATE_UP":    lambda before: "moe_expert",
    "MOE_DOWN":       lambda before: "moe_expert",
    "MOE_FINALIZE":   lambda before: "moe_expert",
    "SILU_MUL":       lambda before: "shared_expert",
    "NCCL":           lambda before: "comm_norm",
    "MEMOP":          lambda before: "residual_mem",
    "TENSOR_COPY":    lambda before: "residual_mem",
    "OTHER":          lambda before: "other",
}


def assign_modules_to_layer(layer):
    """Assign each kernel in a layer to a module based on position.

    Uses FMHA position as the dividing point. Kernels before FMHA are
    in the attention path; kernels after are in the MoE/output path.

    For FP4_GEMM and FP4_CONVERT which appear in both out_proj and shared_expert:
    - First occurrence(s) after FMHA → out_proj
    - Later occurrences (after MoE kernels start) → shared_expert
    """
    kernels = layer["kernels"]
    fmha_pos = layer["fmha_pos"]

    # Track phases after FMHA for FP4_GEMM/FP4_CONVERT assignment
    moe_started = False  # True after we see TOPK/EXPERT_SORT
    first_fp4_gemm_after_fmha = True

    for i, k in enumerate(kernels):
        before_fmha = (i < fmha_pos)
        tag = k["tag"]

        # Detect MoE phase start
        if tag in ("TOPK", "EXPERT_SORT", "FP4_BLOCK_QUANT", "MOE_GATE_UP"):
            moe_started = True

        # Special handling for FP4_GEMM/FP4_CONVERT after FMHA
        if tag in ("FP4_GEMM", "FP4_CONVERT") and not before_fmha:
            if not moe_started and first_fp4_gemm_after_fmha:
                k["module"] = "out_proj"
                if tag == "FP4_GEMM":
                    first_fp4_gemm_after_fmha = False
            else:
                k["module"] = "shared_expert"
            continue

        # Special handling for SPLITK_TNT after FMHA (router GEMM)
        if tag == "SPLITK_TNT" and not before_fmha:
            k["module"] = "router"
            continue
        if tag == "SPLITK_REDUCE" and not before_fmha:
            k["module"] = "router"
            continue

        # Default assignment
        mapper = B200_MODULE_MAP.get(tag)
        if mapper:
            k["module"] = mapper(before_fmha)
        else:
            k["module"] = "other"

    return kernels


def analyze_per_layer(decode_steps, layer_start=10, layer_end=40, kernel_map=None):
    """Per-layer analysis across multiple decode steps.

    For each decode step:
      1. Split into layers using FMHA anchor
      2. Assign modules based on position
      3. Select layers [layer_start, layer_end)

    Returns per-module breakdown averaged over selected layers.
    """
    # Module order for display
    MODULE_ORDER = [
        "comm_norm", "qkv_proj", "rope_attn", "out_proj",
        "shared_expert", "router", "moe_expert",
        "residual_mem", "residual_norm", "other"
    ]

    # Collect per-layer per-module stats across all steps
    # module → operator → {count, total_us, kernel_names}
    all_module_stats = defaultdict(lambda: defaultdict(lambda: {
        "count": 0, "total_us": 0.0, "kernel_names": set()
    }))
    n_layers_total = 0
    layer_counts_per_step = []
    layer_totals_per_step = []  # per-layer total μs for validation

    for step_idx, (launch, kernels) in enumerate(decode_steps):
        layers = split_step_into_layers(kernels)
        layer_counts_per_step.append(len(layers))

        if len(layers) < layer_end:
            # Not enough layers; use what we have
            selected = layers[layer_start:] if len(layers) > layer_start else layers
        else:
            selected = layers[layer_start:layer_end]

        for layer in selected:
            assign_modules_to_layer(layer)
            layer_total = 0.0
            for k in layer["kernels"]:
                module = k.get("module", "other")
                op_label = classify_kernel(k["name"], kernel_map) if kernel_map else k["tag"]
                stats = all_module_stats[module][op_label]
                stats["count"] += 1
                stats["total_us"] += k["dur"]
                stats["kernel_names"].add(k["name"])
                layer_total += k["dur"]
            layer_totals_per_step.append(layer_total)
            n_layers_total += 1

    if n_layers_total == 0:
        print("ERROR: No layers found in selected range")
        return [], 0

    # Build result table
    result = []
    for module in MODULE_ORDER:
        if module not in all_module_stats:
            continue
        ops = all_module_stats[module]
        for op_label, stats in sorted(ops.items(), key=lambda x: -x[1]["total_us"]):
            avg_us = stats["total_us"] / n_layers_total
            avg_count = stats["count"] / n_layers_total
            result.append({
                "module": module,
                "operator": op_label,
                "avg_us": avg_us,
                "avg_count": avg_count,
                "total_us": stats["total_us"],
                "n_layers": n_layers_total,
                "kernel_names": sorted(stats["kernel_names"]),
            })

    # Add modules not in MODULE_ORDER
    for module in all_module_stats:
        if module not in MODULE_ORDER:
            ops = all_module_stats[module]
            for op_label, stats in sorted(ops.items(), key=lambda x: -x[1]["total_us"]):
                avg_us = stats["total_us"] / n_layers_total
                avg_count = stats["count"] / n_layers_total
                result.append({
                    "module": module,
                    "operator": op_label,
                    "avg_us": avg_us,
                    "avg_count": avg_count,
                    "total_us": stats["total_us"],
                    "n_layers": n_layers_total,
                    "kernel_names": sorted(stats["kernel_names"]),
                })

    # Print summary
    avg_layer_count = sum(layer_counts_per_step) / len(layer_counts_per_step)
    avg_layer_us = sum(layer_totals_per_step) / len(layer_totals_per_step)
    print(f"\n  Layers detected per step: avg={avg_layer_count:.1f} (range: {min(layer_counts_per_step)}-{max(layer_counts_per_step)})")
    print(f"  Selected layers: {layer_start}-{layer_end} ({n_layers_total} total across {len(decode_steps)} steps)")
    print(f"  Per-layer kernel sum avg: {avg_layer_us:.1f}μs ({avg_layer_us/1000:.3f}ms)")
    print(f"  Estimated full model: {avg_layer_us:.1f} × {int(avg_layer_count)} = {avg_layer_us * avg_layer_count / 1000:.2f}ms")

    return result, n_layers_total


def print_per_layer_table(result, n_layers):
    """Print per-module per-layer breakdown table."""
    total_us = sum(r["avg_us"] for r in result)

    print(f"\n{'='*120}")
    print(f"Per-Module Kernel Breakdown (per-layer avg, {n_layers} layers, layer total={total_us:.1f}μs)")
    print(f"{'='*120}")
    print(f"{'Module':<16} | {'#':>2} | {'Operator':<42} | {'Avg(μs)':>8} | {'Pct%':>5} | {'Cnt':>4} | {'Kernel(s)':<30}")
    print("-" * 120)

    current_module = None
    op_idx = 0
    module_subtotals = defaultdict(float)

    for r in result:
        module_subtotals[r["module"]] += r["avg_us"]

    for r in result:
        op_idx += 1
        pct = 100 * r["avg_us"] / total_us if total_us > 0 else 0
        knames = ", ".join(r["kernel_names"][:1])
        if len(knames) > 30:
            knames = knames[:27] + "..."

        # Module header
        if r["module"] != current_module:
            if current_module is not None:
                sub = module_subtotals[current_module]
                sub_pct = 100 * sub / total_us if total_us > 0 else 0
                print(f"{'':>16}   {'':>2}   {'Subtotal':<42}   {sub:>8.1f}   {sub_pct:>5.1f}")
                print("-" * 120)
            current_module = r["module"]

        print(f"{r['module'] if r['module'] != current_module or op_idx == 1 or result[op_idx-2]['module'] != current_module else '':.<16} | {op_idx:>2} | {r['operator']:<42} | {r['avg_us']:>8.1f} | {pct:>5.1f} | {r['avg_count']:>4.1f} | {knames:<30}")

    # Last module subtotal
    if current_module is not None:
        sub = module_subtotals[current_module]
        sub_pct = 100 * sub / total_us if total_us > 0 else 0
        print(f"{'':>16}   {'':>2}   {'Subtotal':<42}   {sub:>8.1f}   {sub_pct:>5.1f}")

    print(f"{'='*120}")
    print(f"{'TOTAL':<16}   {'':>2}   {'':42}   {total_us:>8.1f}   100.0")
    print(f"{'='*120}")


def write_per_layer_csv(result, n_layers, filepath):
    """Write per-layer breakdown to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "module", "operator", "avg_us", "pct", "avg_count",
            "total_us", "n_layers", "kernel_names"
        ])
        writer.writeheader()
        total_us = sum(r["avg_us"] for r in result)
        for r in result:
            pct = 100 * r["avg_us"] / total_us if total_us > 0 else 0
            writer.writerow({
                "module": r["module"],
                "operator": r["operator"],
                "avg_us": f"{r['avg_us']:.2f}",
                "pct": f"{pct:.1f}",
                "avg_count": f"{r['avg_count']:.2f}",
                "total_us": f"{r['total_us']:.1f}",
                "n_layers": n_layers,
                "kernel_names": "; ".join(r["kernel_names"]),
            })
    print(f"\nPer-layer CSV written to: {filepath}")


# =============================================================================
# Trace parsing
# =============================================================================

def load_trace(filepath):
    """Load Chrome trace JSON (optionally gzipped)."""
    print(f"Loading: {filepath}")
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rt", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    print(f"  Total events: {len(events)}")
    return events


def find_cuda_graph_launches(events):
    """Find all cudaGraphLaunch events on the CPU timeline."""
    launches = []
    for e in events:
        name = e.get("name", "")
        if "cudaGraphLaunch" in name and e.get("ph") == "X":
            launches.append(e)
    launches.sort(key=lambda x: x.get("ts", 0))
    print(f"  cudaGraphLaunch events: {len(launches)}")
    return launches


def find_gpu_kernels(events, gpu_pid=None):
    """Find all GPU kernel events."""
    kernels = []
    pids = set()
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append(e)
            pids.add(e.get("pid"))

    if gpu_pid is not None:
        kernels = [k for k in kernels if k.get("pid") == gpu_pid]
    elif len(pids) > 1:
        # Use rank 0 (smallest PID)
        min_pid = min(pids)
        kernels = [k for k in kernels if k.get("pid") == min_pid]
        print(f"  Multiple GPU PIDs, using {min_pid} (rank 0)")

    kernels.sort(key=lambda x: x.get("ts", 0))
    print(f"  GPU kernel events: {len(kernels)}")
    return kernels


def build_flow_map(events):
    """Build flow event map: flow_id → (source_event, target_event).

    Flow events in Chrome trace format:
      ph="s" (start): origin of the flow arrow
      ph="f" (finish): destination of the flow arrow
      id: unique flow identifier

    In Torch Profiler traces, flows connect cudaGraphLaunch (CPU) to
    individual GPU kernels inside the graph.
    """
    flow_starts = {}  # id → event
    flow_ends = defaultdict(list)  # id → [events]

    for e in events:
        ph = e.get("ph")
        if ph == "s":
            fid = e.get("id")
            if fid is not None:
                flow_starts[fid] = e
        elif ph == "f":
            fid = e.get("id")
            if fid is not None:
                flow_ends[fid].append(e)

    print(f"  Flow events: {len(flow_starts)} starts, {sum(len(v) for v in flow_ends.values())} ends")
    return flow_starts, flow_ends


def match_kernels_by_flow(launch, all_events, flow_starts, flow_ends):
    """Find GPU kernels linked to a cudaGraphLaunch via flow events.

    Strategy: find flow events that originate from the same (pid,tid)
    and overlap temporally with the launch event.
    """
    launch_pid = launch.get("pid")
    launch_tid = launch.get("tid")
    launch_ts = launch.get("ts", 0)
    launch_dur = launch.get("dur", 0)
    launch_end = launch_ts + launch_dur

    matched_kernels = []

    for fid, start_evt in flow_starts.items():
        # Flow start should be from the same thread as the launch
        if start_evt.get("pid") != launch_pid or start_evt.get("tid") != launch_tid:
            continue
        # Flow start should be within the launch's time window
        flow_ts = start_evt.get("ts", 0)
        if flow_ts < launch_ts or flow_ts > launch_end:
            continue
        # Follow to the flow end(s) — these point to GPU kernel events
        # Filter: only accept actual GPU kernel/memcpy/memset events,
        # not ac2g (async cuda graph correlation) or other non-kernel events
        for end_evt in flow_ends.get(fid, []):
            end_cat = end_evt.get("cat", "")
            if end_cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
                continue
            matched_kernels.append({
                "name": end_evt.get("name", ""),
                "ts": end_evt.get("ts", 0),
                "dur": end_evt.get("dur", 0),
                "tid": end_evt.get("tid"),
                "pid": end_evt.get("pid"),
                "cat": end_cat,
            })

    matched_kernels.sort(key=lambda x: x["ts"])
    return matched_kernels


def match_kernels_by_time(launch, gpu_kernels):
    """Fallback: match GPU kernels that execute during a cudaGraphLaunch window.

    When flow events are absent, we use temporal overlap:
    a GPU kernel belongs to this launch if it starts within the launch's
    duration window (with a small margin).
    """
    launch_ts = launch.get("ts", 0)
    launch_dur = launch.get("dur", 0)
    launch_end = launch_ts + launch_dur
    margin = 100  # μs margin for clock skew between CPU/GPU

    matched = []
    for k in gpu_kernels:
        k_ts = k.get("ts", 0)
        if k_ts >= launch_ts - margin and k_ts <= launch_end + margin:
            matched.append({
                "name": k.get("name", ""),
                "ts": k_ts,
                "dur": k.get("dur", 0),
                "tid": k.get("tid"),
                "pid": k.get("pid"),
                "cat": k.get("cat", ""),
            })

    matched.sort(key=lambda x: x["ts"])
    return matched


def match_kernels_by_interval(launches, gpu_kernels, idx):
    """Match GPU kernels in the interval [launch[idx].ts, launch[idx+1].ts).

    Each cudaGraphLaunch is one decode iteration. GPU kernels execute
    asynchronously well past the CPU launch duration, so we use the
    inter-launch interval (not launch.dur) as the kernel ownership window.
    """
    start_ts = launches[idx].get("ts", 0)
    if idx + 1 < len(launches):
        end_ts = launches[idx + 1].get("ts", 0)
    else:
        # Last launch: use start + 2 * median gap as window
        if idx > 0:
            gaps = [launches[j+1]["ts"] - launches[j]["ts"] for j in range(max(0, idx-5), idx)]
            median_gap = sorted(gaps)[len(gaps)//2] if gaps else 20000
            end_ts = start_ts + 2 * median_gap
        else:
            end_ts = start_ts + 20000  # 20ms default

    matched = []
    for k in gpu_kernels:
        k_ts = k.get("ts", 0)
        if k_ts >= start_ts and k_ts < end_ts:
            matched.append({
                "name": k.get("name", ""),
                "ts": k_ts,
                "dur": k.get("dur", 0),
                "tid": k.get("tid"),
                "pid": k.get("pid"),
                "cat": k.get("cat", ""),
            })

    matched.sort(key=lambda x: x["ts"])
    return matched


def extract_decode_steps(events, max_steps=None, skip_first=5):
    """Extract kernel lists for each decode step (cudaGraphLaunch).

    Each cudaGraphLaunch = one full decode iteration. GPU kernels are
    matched using inter-launch intervals (not flow events or launch dur),
    because GPU execution extends well past the CPU launch call.

    Args:
        events: raw trace events
        max_steps: limit number of steps to process
        skip_first: skip first N launches (warmup/prefill)

    Returns:
        List of (launch_event, [kernel_dicts]) tuples
    """
    launches = find_cuda_graph_launches(events)
    gpu_kernels = find_gpu_kernels(events)

    # Skip first few launches (often prefill or ramp-up)
    if skip_first and len(launches) > skip_first:
        launches = launches[skip_first:]
        print(f"  Skipped first {skip_first} launches, using {len(launches)} remaining")

    if max_steps:
        launches = launches[:max_steps]

    decode_steps = []

    for i, launch in enumerate(launches):
        kernels = match_kernels_by_interval(launches, gpu_kernels, i)

        if kernels:
            decode_steps.append((launch, kernels))

        if i == 0:
            print(f"  Kernel matching method: inter-launch interval")
            if kernels:
                span = kernels[-1]['ts'] - kernels[0]['ts']
                print(f"  First decode step: {len(kernels)} kernels, span={span:.0f}μs ({span/1000:.2f}ms)")
            else:
                print(f"  First decode step: 0 kernels")

    print(f"  Total decode steps extracted: {len(decode_steps)}")
    return decode_steps


# =============================================================================
# Analysis and output
# =============================================================================

def compute_step_breakdown(kernels, kernel_map):
    """Compute per-logical-operator timing for one decode step."""
    ops = OrderedDict()
    for k in kernels:
        label = classify_kernel(k["name"], kernel_map)
        if label not in ops:
            ops[label] = {"count": 0, "total_us": 0, "kernels": []}
        ops[label]["count"] += 1
        ops[label]["total_us"] += k["dur"]
        ops[label]["kernels"].append(k["name"])
    return ops


def compute_average_breakdown(decode_steps, kernel_map):
    """Average kernel breakdown across multiple decode steps."""
    all_ops = defaultdict(lambda: {"count": 0, "total_us": 0, "n_steps": 0, "kernel_names": set()})

    for launch, kernels in decode_steps:
        step_ops = compute_step_breakdown(kernels, kernel_map)
        for label, stats in step_ops.items():
            all_ops[label]["count"] += stats["count"]
            all_ops[label]["total_us"] += stats["total_us"]
            all_ops[label]["n_steps"] += 1
            for kn in stats["kernels"]:
                all_ops[label]["kernel_names"].add(kn)

    n_steps = len(decode_steps)
    result = []
    for label, stats in all_ops.items():
        avg_us = stats["total_us"] / n_steps if n_steps > 0 else 0
        avg_count = stats["count"] / n_steps if n_steps > 0 else 0
        result.append({
            "operator": label,
            "avg_us": avg_us,
            "avg_count": avg_count,
            "total_us": stats["total_us"],
            "n_steps_present": stats["n_steps"],
            "kernel_names": sorted(stats["kernel_names"]),
        })

    result.sort(key=lambda x: -x["avg_us"])
    return result


def print_breakdown_table(breakdown, n_steps):
    """Print formatted breakdown table."""
    total_us = sum(r["avg_us"] for r in breakdown)

    print(f"\n{'='*110}")
    print(f"Per-Decode-Step Kernel Breakdown (averaged over {n_steps} steps, total={total_us:.1f}μs)")
    print(f"{'='*110}")
    print(f"{'#':>3} | {'Logical Operator':<45} | {'Avg(μs)':>8} | {'Pct%':>5} | {'Count':>5} | {'Kernel(s)':<40}")
    print("-" * 110)

    for i, r in enumerate(breakdown):
        pct = 100 * r["avg_us"] / total_us if total_us > 0 else 0
        # Show first kernel name (truncated)
        knames = ", ".join(r["kernel_names"][:2])
        if len(knames) > 40:
            knames = knames[:37] + "..."
        print(f"{i+1:>3} | {r['operator']:<45} | {r['avg_us']:>8.1f} | {pct:>5.1f} | {r['avg_count']:>5.1f} | {knames:<40}")

    print(f"\n  Total per-step: {total_us:.1f}μs ({total_us/1000:.2f}ms)")


def print_raw_step(launch, kernels, step_idx):
    """Print raw kernel list for one decode step."""
    launch_dur = launch.get("dur", 0)
    print(f"\n--- Decode Step {step_idx} (cudaGraphLaunch dur={launch_dur:.0f}μs, {len(kernels)} kernels) ---")
    print(f"{'#':>3} | {'Kernel Name':<60} | {'ts':>12} | {'dur(μs)':>8} | {'stream':>8}")
    print("-" * 100)
    base_ts = kernels[0]["ts"] if kernels else 0
    for i, k in enumerate(kernels):
        name = k["name"][:60]
        rel_ts = k["ts"] - base_ts
        print(f"{i+1:>3} | {name:<60} | {rel_ts:>12.0f} | {k['dur']:>8.1f} | {k['tid']}")


def write_csv(breakdown, n_steps, filepath):
    """Write breakdown to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "operator", "avg_us", "pct", "avg_count",
            "total_us", "n_steps_present", "kernel_names"
        ])
        writer.writeheader()
        total_us = sum(r["avg_us"] for r in breakdown)
        for i, r in enumerate(breakdown):
            pct = 100 * r["avg_us"] / total_us if total_us > 0 else 0
            writer.writerow({
                "rank": i + 1,
                "operator": r["operator"],
                "avg_us": f"{r['avg_us']:.2f}",
                "pct": f"{pct:.1f}",
                "avg_count": f"{r['avg_count']:.1f}",
                "total_us": f"{r['total_us']:.1f}",
                "n_steps_present": r["n_steps_present"],
                "kernel_names": "; ".join(r["kernel_names"]),
            })
    print(f"\nCSV written to: {filepath}")


# =============================================================================
# Main
# =============================================================================

def print_trace_info(events):
    """Print trace metadata: time span, phases, decode steps, layers, kernel counts."""
    # Time span
    all_ts = [e.get("ts", 0) for e in events if e.get("ts")]
    all_dur = [e.get("dur", 0) for e in events if e.get("dur")]
    if not all_ts:
        print("  No timestamped events found.")
        return

    min_ts = min(all_ts)
    max_ts = max(all_ts)
    # Account for duration of last events
    max_end = max(e.get("ts", 0) + e.get("dur", 0) for e in events if e.get("ts"))
    span_us = max_end - min_ts
    span_s = span_us / 1e6

    print(f"\n{'='*80}")
    print(f"TRACE INFO")
    print(f"{'='*80}")
    print(f"  Total events:    {len(events)}")
    print(f"  Time span:       {span_s:.2f}s ({span_us/1e6:.2f}s)")
    print(f"  Time range:      [{min_ts/1e6:.3f}s, {max_end/1e6:.3f}s]")

    # Categorize events
    cats = defaultdict(int)
    phs = defaultdict(int)
    for e in events:
        cats[e.get("cat", "")] += 1
        phs[e.get("ph", "")] += 1

    print(f"\n  Event categories:")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1])[:15]:
        print(f"    {cat or '(none)':<35s} {cnt:>8d}")

    # PIDs (processes/GPUs)
    pids = defaultdict(set)
    for e in events:
        pid = e.get("pid")
        cat = e.get("cat", "")
        if pid is not None:
            pids[pid].add(cat)
    print(f"\n  PIDs ({len(pids)}):")
    for pid in sorted(pids, key=lambda x: str(x)):
        cats_str = ", ".join(sorted(c for c in pids[pid] if c)[:5])
        print(f"    PID {pid}: {cats_str}")

    # cudaGraphLaunch events
    launches = [e for e in events if "cudaGraphLaunch" in e.get("name", "") and e.get("ph") == "X"]
    launches.sort(key=lambda x: x.get("ts", 0))
    print(f"\n  cudaGraphLaunch events: {len(launches)}")
    if launches:
        first_ts = (launches[0]["ts"] - min_ts) / 1e6
        last_ts = (launches[-1]["ts"] - min_ts) / 1e6
        print(f"    First at: {first_ts:.3f}s, Last at: {last_ts:.3f}s")
        durs = [l.get("dur", 0) for l in launches]
        print(f"    Duration: avg={sum(durs)/len(durs)/1e3:.2f}ms, min={min(durs)/1e3:.2f}ms, max={max(durs)/1e3:.2f}ms")

        # Detect decode step kernel counts by sampling a few launches
        sample_indices = [0, len(launches)//4, len(launches)//2, 3*len(launches)//4, len(launches)-1]
        print(f"\n    Sampled launches (kernel count via time-window):")
        gpu_kernels = sorted(
            [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")],
            key=lambda x: x.get("ts", 0)
        )
        for idx in sample_indices:
            if idx >= len(launches):
                continue
            l = launches[idx]
            l_ts = l["ts"]
            l_end = l_ts + l.get("dur", 0)
            matched = sum(1 for k in gpu_kernels if k["ts"] >= l_ts - 100 and k["ts"] <= l_end + 100)
            rel_s = (l_ts - min_ts) / 1e6
            print(f"      Launch[{idx:>4d}] at {rel_s:>8.3f}s  dur={l.get('dur',0)/1e3:>7.2f}ms  ~{matched} kernels")

    # GPU kernel events
    gpu_kernels_all = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    print(f"\n  GPU kernel events: {len(gpu_kernels_all)}")

    # Unique kernel names
    kernel_names = defaultdict(int)
    for e in gpu_kernels_all:
        kernel_names[e.get("name", "")[:80]] += 1
    print(f"  Unique kernel names: {len(kernel_names)}")
    print(f"\n  Top 20 kernels by frequency:")
    for name, cnt in sorted(kernel_names.items(), key=lambda x: -x[1])[:20]:
        print(f"    {cnt:>8d}x  {name}")

    # NVTX / user annotations (layer markers, decode/prefill markers)
    annotations = [e for e in events if e.get("cat") in ("gpu_user_annotation", "user_annotation", "python_function") or (e.get("cat", "").startswith("nvtx"))]
    if annotations:
        ann_names = defaultdict(int)
        for e in annotations:
            ann_names[e.get("name", "")[:60]] += 1
        print(f"\n  Annotations/NVTX: {len(annotations)} events, {len(ann_names)} unique")
        print(f"  Top 20 annotations:")
        for name, cnt in sorted(ann_names.items(), key=lambda x: -x[1])[:20]:
            print(f"    {cnt:>8d}x  {name}")

    # Flow events
    flow_s = sum(1 for e in events if e.get("ph") == "s")
    flow_f = sum(1 for e in events if e.get("ph") == "f")
    print(f"\n  Flow events: {flow_s} starts, {flow_f} ends")

    # Profiler step markers
    steps = [e for e in events if "ProfilerStep" in e.get("name", "") and e.get("ph") == "X"]
    if steps:
        steps.sort(key=lambda x: x.get("ts", 0))
        print(f"\n  ProfilerStep markers: {len(steps)}")
        for s in steps[:5]:
            rel_s = (s["ts"] - min_ts) / 1e6
            dur_s = s.get("dur", 0) / 1e6
            print(f"    {s['name']}: at {rel_s:.3f}s, dur={dur_s:.3f}s")
        if len(steps) > 5:
            print(f"    ... and {len(steps)-5} more")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-decode-step kernel breakdown from CUDA Graph traces"
    )
    parser.add_argument("filepath", help="Path to trace JSON or JSON.GZ file")
    parser.add_argument("--platform", choices=["b200", "mi355x", "auto"], default="auto",
                        help="Platform for kernel name mapping (default: auto-detect)")
    parser.add_argument("--info", action="store_true",
                        help="Print trace metadata only (time span, phases, decode steps, layers)")
    parser.add_argument("--raw", action="store_true",
                        help="Print raw kernel lists instead of logical operator breakdown")
    parser.add_argument("--csv", default=None, help="Output CSV path")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Max decode steps to analyze (default: 20)")
    parser.add_argument("--skip-first", type=int, default=5,
                        help="Skip first N cudaGraphLaunch events (default: 5)")
    parser.add_argument("--show-steps", type=int, default=2,
                        help="Show raw kernels for first N steps (default: 2)")
    parser.add_argument("--gpu-pid", type=int, default=None,
                        help="Filter to specific GPU PID")
    parser.add_argument("--per-layer", action="store_true",
                        help="Per-layer analysis: split decode steps into layers using FMHA anchor")
    parser.add_argument("--layer-range", default="10-40",
                        help="Layer range for per-layer analysis (default: 10-40)")
    parser.add_argument("--per-layer-csv", default=None,
                        help="Output per-layer CSV path")
    args = parser.parse_args()

    events = load_trace(args.filepath)

    if args.info:
        print_trace_info(events)
        return

    # Auto-detect platform from kernel names
    if args.platform == "auto":
        sample = " ".join(e.get("name", "") for e in events[:5000])
        if "nvjet" in sample or "fmhaSm100" in sample:
            platform = "b200"
        elif "gemm_xdl" in sample or "mla_a8w8" in sample or "rccl" in sample:
            platform = "mi355x"
        else:
            platform = "b200"
        print(f"  Auto-detected platform: {platform}")
    else:
        platform = args.platform

    kernel_map = B200_KERNEL_MAP if platform == "b200" else MI355X_KERNEL_MAP

    # Extract decode steps
    decode_steps = extract_decode_steps(
        events,
        max_steps=args.max_steps,
        skip_first=args.skip_first,
    )

    if not decode_steps:
        print("\nERROR: No decode steps found. Possible reasons:")
        print("  - Trace doesn't contain cudaGraphLaunch events")
        print("  - CUDA Graphs not used (enforce_eager=true)")
        print("  - Trace too short or only contains prefill")
        sys.exit(1)

    # Show raw steps
    if args.raw or args.show_steps:
        n_show = len(decode_steps) if args.raw else min(args.show_steps, len(decode_steps))
        for i in range(n_show):
            launch, kernels = decode_steps[i]
            print_raw_step(launch, kernels, i)

    # Compute and print averaged breakdown
    if not args.raw:
        breakdown = compute_average_breakdown(decode_steps, kernel_map)
        print_breakdown_table(breakdown, len(decode_steps))

        if args.csv:
            write_csv(breakdown, len(decode_steps), args.csv)

    # Per-layer analysis
    if args.per_layer:
        layer_range = args.layer_range.split("-")
        layer_start = int(layer_range[0])
        layer_end = int(layer_range[1]) if len(layer_range) > 1 else layer_start + 30

        per_layer_result, n_layers = analyze_per_layer(
            decode_steps, layer_start=layer_start, layer_end=layer_end,
            kernel_map=kernel_map
        )
        if per_layer_result:
            print_per_layer_table(per_layer_result, n_layers)
            if args.per_layer_csv:
                write_per_layer_csv(per_layer_result, n_layers, args.per_layer_csv)

    # Summary stats
    step_durations_sum = []  # sum of kernel durations (overcounts overlapping streams)
    step_durations_wall = []  # wall-clock: span from first kernel to last kernel end
    for launch, kernels in decode_steps:
        if kernels:
            step_durations_sum.append(sum(k["dur"] for k in kernels))
            wall_start = min(k["ts"] for k in kernels)
            wall_end = max(k["ts"] + k["dur"] for k in kernels)
            step_durations_wall.append(wall_end - wall_start)

    if step_durations_wall:
        avg_wall = sum(step_durations_wall) / len(step_durations_wall)
        min_wall = min(step_durations_wall)
        max_wall = max(step_durations_wall)
        avg_sum = sum(step_durations_sum) / len(step_durations_sum)
        print(f"\nDecode step duration stats ({len(step_durations_wall)} steps):")
        print(f"  Wall-clock avg: {avg_wall:.1f}μs ({avg_wall/1000:.2f}ms)")
        print(f"  Wall-clock min: {min_wall:.1f}μs ({min_wall/1000:.2f}ms)  max: {max_wall:.1f}μs ({max_wall/1000:.2f}ms)")
        print(f"  Kernel sum avg: {avg_sum:.1f}μs ({avg_sum/1000:.2f}ms) (includes stream overlap)")
        if avg_wall > 0:
            print(f"  Range: {max_wall-min_wall:.1f}μs ({100*(max_wall-min_wall)/avg_wall:.1f}% variation)")
            print(f"  Overlap ratio: {avg_sum/avg_wall:.2f}x (sum/wall, >1.0 means multi-stream overlap)")


if __name__ == "__main__":
    main()
