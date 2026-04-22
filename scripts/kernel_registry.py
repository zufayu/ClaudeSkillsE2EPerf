#!/usr/bin/env python3
"""
Central kernel classification registry for ClaudeSkillsE2EPerf.

Single source of truth for kernel name → operator → module → category mapping.
All analysis scripts should import from here instead of defining their own maps.

Design (Hermes COMMAND_REGISTRY pattern):
  - Define once, derive everywhere
  - Change one mapping → all 6+ analysis scripts auto-update
  - Supports positional disambiguation for ambiguous kernels

Evidence: 8 fix commits for kernel classification errors (1a5636a, 8be04ce,
  07bfac1, 9ee8d43, d23af92, 6913b4c, e123ad7, 1516e0a)

Usage:
    from kernel_registry import classify_kernel, get_operator_map
    from kernel_registry import get_category, get_module
    from kernel_registry import CATEGORIES

    # Simple classification
    op = classify_kernel("fmhaSm100fKernel_Qkv_fp16", platform="b200")
    # → "Attention_FMHA"

    # Get full map for a platform
    op_map = get_operator_map("b200")

    # Higher-level lookups
    cat = get_category("Attention_FMHA")   # → "Attention"
    mod = get_module("Attention_FMHA")     # → "MHA"
"""

import re
from collections import OrderedDict

# =============================================================================
# Level 3 → Level 2: Operator → Module/PASS
# =============================================================================

# Module grouping for DeepSeek-R1 transformer layer (execution order)
OPERATOR_TO_MODULE = {
    # Pre-attention communication
    "EP_AR+residual+RMSNorm(fused)":    None,  # position-dependent: see below
    "EP_AR_before_MHA":                 "Comm",
    "EP_AR_before_MOE":                 "Comm",
    "residual_add":                     "Comm",
    "TP_AR+RMSNorm":                    "Comm",
    "EP_allgather":                     "Comm",
    "lamport_AR+RMSNorm":               "Comm",
    "reduce_scatter":                   "Comm",
    "allreduce/other":                  "Comm",
    "nccl_comm":                        "Comm",

    # MHA (Multi-Head Latent Attention)
    "qkv_a_proj_GEMM":                  "MHA",
    "qkv_a_splitK_reduce":              "MHA",
    "q/k_norm_RMSNorm":                 "MHA",
    "q_b_proj_GEMM":                    "MHA",
    "uk_gemm":                          "MHA",
    "k_concat":                         "MHA",
    "RoPE+KV_write":                    "MHA",
    "set_mla_kv":                       "MHA",
    "Attention_FMHA":                   "MHA",
    "Attention_MLA":                    "MHA",
    "MLA_reduce":                       "MHA",
    "uv_gemm":                          "MHA",
    "per_token_quant":                  "MHA",
    "fused_qk_rmsnorm_group_quant":     "MHA",
    "preshuffle_GEMM":                  "MHA",
    "batched_a8w8_GEMM":               "MHA",

    # Output projection
    "o_proj_splitK_GEMM":               "O_proj",
    "o_proj_GEMM":                      "O_proj",
    "FP4_GEMM":                         None,  # position-dependent: o_proj or shared
    "o_proj_quant":                     "O_proj",

    # Router
    "router_GEMM":                      "Router",
    "router_splitK_reduce":             "Router",
    "TopK_select":                      "Router",
    "expert_sort":                      "Router",
    "MoE_sort":                         "Router",

    # MoE Expert
    "MoE_input_quant":                  "MoE",
    "fused_quant_sort":                 "MoE",
    "gate_up_GEMM":                     "MoE",
    "down_GEMM":                        "MoE",
    "MoE_GEMM":                         "MoE",
    "MoE_finalize+residual":            "MoE",
    "MoE_finalize":                     "MoE",

    # Shared expert
    "SiLU_mul":                         "Shared",
    "shared_GEMM":                      "Shared",
    "shared_quant":                     "Shared",

    # Legacy GEMM (cuBLAS path — typically router or shared expert)
    "cuBLAS_GEMM_legacy":               "GEMM",

    # Quantization
    "FP4_blockwise_quant":              "Quant",
    "FP4_convert":                      "Quant",

    # Norm (standalone)
    "RMSNorm":                          "Norm",
    "fused_rms_fp8_group_quant":        "Norm",

    # Memory
    "memop":                            "Mem",
    "tensor_copy":                      "Mem",
}

# =============================================================================
# Level 2 → Level 1: Module → Category
# =============================================================================

MODULE_TO_CATEGORY = {
    "Comm":     "Communication",
    "MHA":      "Attention",
    "O_proj":   "GEMM/Projection",
    "GEMM":     "GEMM/Projection",
    "Router":   "MoE/Expert",
    "MoE":      "MoE/Expert",
    "Shared":   "MoE/Expert",
    "Quant":    "Quantization",
    "Norm":     "Normalization",
    "Mem":      "Memory",
}

# Top-level categories for high-level breakdown
CATEGORIES = list(dict.fromkeys(MODULE_TO_CATEGORY.values()))

# =============================================================================
# Level 4 → Level 3: Kernel regex → Operator (per platform)
# =============================================================================

# B200 SGLang/TRT-LLM FP4 kernel mapping
# Covers both nsys (short names) and torch profiler (full C++ mangled names)
B200_OPERATOR_MAP = OrderedDict([
    # Pre-attention: MoE finalize + residual + lamport allreduce
    (r"finalizeKernelVecLoad|moefinalize(?!.*lamport)", "MoE_finalize+residual"),
    (r"vectorized_elementwise_kernel.*CUDAFunctor_add|CUDAFunctorOnSelf_add|elementwise.*add", "residual_add"),
    (r"allreduce_fusion_kernel.*lamport|moefinalize_lamport", "lamport_AR+RMSNorm"),
    # QKV projection
    (r"splitK_TNT|nvjet_splitK_TNT|nvjet_sm100_tst.*splitK_TNT", None),  # position-dependent: qkv_a or router
    (r"splitKreduce.*bf16|splitKreduce.*bfloat|splitKreduce.*Bfloat16", None),  # position-dependent
    (r"FusedAddRMSNorm|RMSNormKernel", "q/k_norm_RMSNorm"),
    (r"_v_bz_TNN|nvjet.*_v_bz_TNN|nvjet_tst_TNN", "q_b_proj_GEMM"),
    (r"_v_bz_TNT|nvjet_sm100_tst_128x64.*TNT", "uk_gemm"),
    (r"CatArrayBatchedCopy", "k_concat"),
    # RoPE + Attention
    (r"RopeQuantizeKernel|applyMLARopeAndAssignQKV", "RoPE+KV_write"),
    (r"set_mla_kv_buffer", "set_mla_kv"),
    (r"fmhaSm100|fmhaKernel|flashinfer.*fmha", "Attention_FMHA"),
    # Output projection
    (r"_h_bz_TNT(?!.*splitK)|nvjet_tst_TNT", "uv_gemm"),
    (r"_h_bz_splitK_TNT", "o_proj_splitK_GEMM"),
    (r"nvjet_ootst_FP4|DeviceGemmFp4GemmSm100|cutlass.*device_kernel.*flashinfer.*gemm", None),  # position-dependent: o_proj or shared
    (r"quantize_with_block_size", "FP4_blockwise_quant"),
    # Post-attention communication
    (r"userbuffers_rmsnorm", "TP_AR+RMSNorm"),
    (r"userbuffers_allgather", "EP_allgather"),
    # Router (must come AFTER qkv splitK patterns)
    (r"nvjet_tss_splitK|splitK.*router", "router_GEMM"),
    (r"splitKreduce.*fp32|splitKreduce.*float32|splitKreduce.*Fp32", "router_splitK_reduce"),
    (r"routingMainKernel", "TopK_select"),
    (r"routingIndicesCluster|routingDeepSe", "expert_sort"),
    # MoE expert
    (r"bmm_E2m1.*[Ss]wi[Gg]lu|bmm_E2m1.*E2m1E2m1", "gate_up_GEMM"),
    (r"bmm_Bfloat16|bmm_.*E2m1.*Bfloat", "down_GEMM"),
    # Shared expert (and fused MoE activation variants)
    (r"act_and_mul_kernel|silu_and_mul_kernel|fused_silu_and_mul.*kernel", "SiLU_mul"),
    # Legacy cuBLAS GEMMs (e.g. ampere_h884gemm, cublasLt)
    # Catch-all for non-nvjet/non-cutlass GEMM kernels — typically used in
    # router or shared-expert paths where TensorCore-centric backend isn't selected.
    (r"ampere_.*gemm|h884gemm|h1688gemm|cublasLt|cublas_lt", "cuBLAS_GEMM_legacy"),
    # Elementwise
    (r"cvt_fp16_to_fp4|cvt_fp4", "FP4_convert"),
    # Communication (catch-all)
    (r"allreduce|reduce_scatter|all_gather|nccl|ncclDevKernel", "allreduce/other"),
    # Memory
    (r"memcpy|memset", "memop"),
    (r"unrolled_elementwise.*direct_copy|direct_copy_kernel|elementwise_kernel.*direct_copy", "tensor_copy"),
])

# MI355X ATOM MXFP4 kernel mapping (ROCm 7.2.2+)
MI355X_OPERATOR_MAP = OrderedDict([
    # Communication
    (r"reduce_scatter_cross_device|reduce_scatter", "reduce_scatter"),
    (r"local_device_load_rmsnorm", "lamport_AR+RMSNorm"),
    # Normalization
    (r"fused_qk_rmsnorm_group_quant", "fused_qk_rmsnorm_group_quant"),
    (r"add_rmsnorm_quant|rmsnorm|rms_norm", "RMSNorm"),
    # Quantization
    (r"dynamic_per_token_scaled_quant", "per_token_quant"),
    (r"fused_rms_fp8_group_quant", "fused_rms_fp8_group_quant"),
    # GEMM
    (r"gemm_xdl.*preshuffle", "preshuffle_GEMM"),
    (r"FlatmmKernel", "preshuffle_GEMM"),
    (r"batched_gemm_a8w8", "batched_a8w8_GEMM"),
    (r"Cijk_", "preshuffle_GEMM"),
    # RoPE + Attention
    (r"fuse_qk_rope_concat_and_cache_mla|fuse_qk_rope_concat", "RoPE+KV_write"),
    (r"mla_a8w8_qh16", "Attention_MLA"),
    (r"kn_mla_reduce", "MLA_reduce"),
    # Router
    (r"bf16gemm_splitk|bf16gemm", "router_GEMM"),
    (r"grouped_topk_opt_sort", "TopK_select"),
    (r"MoeSorting", "MoE_sort"),
    # MoE
    (r"fused_mxfp4_quant_moe_sort|mxfp4_quant_moe_sort", "fused_quant_sort"),
    (r"kernel_moe_mxgemm", "MoE_GEMM"),
    # Communication (catch-all)
    (r"allreduce|all_gather|rccl", "allreduce/other"),
    # Memory
    (r"memcpy|memset", "memop"),
])

# MI355X positional disambiguation rules
# Some kernels appear multiple times per layer — occurrence index determines the module
MI355X_POSITIONAL_RULES = [
    # (pattern, [module_for_1st, module_for_2nd, ...])
    ("reduce_scatter_cross_device", ["input_layernorm", "post_attn_layernorm"]),
    ("local_device_load_rmsnorm", ["input_layernorm", "post_attn_layernorm"]),
    ("gemm_xdl_cshuffle_v3_multi_d_b_preshuffle", ["gemm_a8w8_bpreshuffle", "q_proj_and_k_up_proj"]),
    ("FlatmmKernel", ["v_up_proj_and_o_proj"]),
    ("Cijk_", ["q_proj_and_k_up_proj", "v_up_proj_and_o_proj"]),
    ("batched_gemm_a8w8", ["q_proj_and_k_up_proj", "v_up_proj_and_o_proj"]),
]

# B200 positional disambiguation
B200_POSITIONAL_RULES = [
    # FP4_GEMM appears twice: first = o_proj, second = shared expert
    ("DeviceGemmFp4GemmSm100|cutlass.*flashinfer.*gemm|nvjet_ootst_FP4", ["o_proj_GEMM", "shared_GEMM"]),
    # FP4 convert: first = o_proj quant, second = shared quant
    ("cvt_fp16_to_fp4", ["o_proj_quant", "shared_quant"]),
    # splitK: first = qkv_a, later = router
    ("splitK_TNT|nvjet_splitK_TNT", ["qkv_a_proj_GEMM", "router_GEMM"]),
    ("splitKreduce.*bf16|splitKreduce.*bfloat", ["qkv_a_splitK_reduce", "router_splitK_reduce"]),
]

# =============================================================================
# High-level category classifier (for parse_torch_trace.py compatibility)
# =============================================================================

# Simple keyword-based category classification (no regex, fast)
CATEGORY_KEYWORDS = OrderedDict([
    ("Attention", ["fmha", "flash_fwd", "flash_bwd", "flash_attn", "mha_",
                   "merge_attn", "concat_and_cast_mha", "set_mla_kv", "mla_a8w8"]),
    # NOTE: 'nvjet_sm100' was previously here but is too broad — nvjet is a generic
    # NVIDIA GEMM library prefix (sm100/sm90 = arch). MoE-specific kernels are
    # caught by 'expert', 'routing', 'swiglu', 'topk', etc. below.
    ("MoE/Expert", ["moe", "expert", "routing", "swiglu", "swig",
                    "expandInput", "doActivation",
                    "topk", "buildExpert", "Dispatch", "Combine"]),
    ("Communication", ["allreduce", "reduce_scatter", "all_gather", "allgather",
                       "nccl", "rccl", "ncclkernel", "device_load", "device_store",
                       "userbuffers"]),
    ("GEMM/MatMul", ["gemm", "gemv", "cutlass", "cublas", "matmul", "nvjet",
                     "splitkreduce", "cijk_", "bf16gemm", "bmm"]),
    ("Normalization", ["layernorm", "rmsnorm", "batchnorm", "groupnorm"]),
    ("Activation/Elementwise", ["silu", "gelu", "relu", "elementwise",
                                "add_kernel", "mul_kernel", "act_and_mul"]),
    ("Quantization", ["quant", "dequant", "cvt_fp16_to_fp4", "cvt_fp4",
                      "fp4", "fp8", "mxfp"]),
    ("Memory", ["memcpy", "memset", "copy", "transpose"]),
    ("Embedding/RoPE", ["embedding", "rotary", "rope"]),
    ("Sampling", ["sample", "argmax", "topk_sampling", "topp"]),
])


# =============================================================================
# API Functions
# =============================================================================

def get_operator_map(platform):
    """Get the kernel regex → operator mapping for a platform.

    Returns OrderedDict of (regex_pattern, operator_name_or_None).
    None means position-dependent — use classify_by_position() for those.
    """
    platform = platform.lower()
    if platform in ("b200", "b300", "h200", "h20"):
        return B200_OPERATOR_MAP
    elif platform in ("mi355x", "mi325x", "mi300x"):
        return MI355X_OPERATOR_MAP
    else:
        raise ValueError(f"Unknown platform: {platform}")


def get_positional_rules(platform):
    """Get positional disambiguation rules for a platform."""
    platform = platform.lower()
    if platform in ("b200", "b300", "h200", "h20"):
        return B200_POSITIONAL_RULES
    elif platform in ("mi355x", "mi325x", "mi300x"):
        return MI355X_POSITIONAL_RULES
    else:
        return []


def classify_kernel(name, platform="b200"):
    """Classify a kernel name to its operator.

    Returns operator name string, or None if the kernel is position-dependent.
    For position-dependent kernels, use classify_by_position().
    """
    op_map = get_operator_map(platform)
    for pattern, operator in op_map.items():
        if re.search(pattern, name, re.IGNORECASE):
            return operator
    return f"other: {name[:60]}"


def classify_by_position(name, occurrence_index, platform="b200"):
    """Classify a position-dependent kernel by its occurrence index within a layer.

    Args:
        name: GPU kernel name
        occurrence_index: 0-based index of this kernel's occurrence within the layer
        platform: Hardware platform

    Returns operator name, or None if not a positional kernel.
    """
    rules = get_positional_rules(platform)
    for pattern, modules in rules:
        if re.search(pattern, name, re.IGNORECASE):
            idx = min(occurrence_index, len(modules) - 1)
            return modules[idx]
    return None


def classify_category(name):
    """High-level category classification (Attention, MoE, GEMM, etc.).

    Compatible with parse_torch_trace.py's classify_kernel().
    Uses simple keyword matching — fast, no regex.
    """
    n = name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k in n for k in keywords):
            return category
    return "Other"


def get_module(operator):
    """Get the module/PASS for an operator. Returns None if unknown."""
    return OPERATOR_TO_MODULE.get(operator)


def get_category(operator):
    """Get the top-level category for an operator."""
    module = get_module(operator)
    if module:
        return MODULE_TO_CATEGORY.get(module, "Other")
    return "Other"


# =============================================================================
# Convenience: all operators for a platform, grouped by module
# =============================================================================

def get_operators_by_module(platform="b200"):
    """Return dict of module → [operators] for a platform."""
    op_map = get_operator_map(platform)
    result = {}
    for _, operator in op_map.items():
        if operator is None:
            continue
        module = get_module(operator)
        if module:
            result.setdefault(module, []).append(operator)
    # Deduplicate while preserving order
    for mod in result:
        seen = set()
        result[mod] = [x for x in result[mod] if not (x in seen or seen.add(x))]
    return result
