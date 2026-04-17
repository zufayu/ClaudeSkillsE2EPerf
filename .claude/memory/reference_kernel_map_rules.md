---
name: B200 vs MI355X kernel mapping methodology
description: Rules for strict timeline-based kernel mapping between B200 SGLang and MI355X ATOM traces, including pass-level grouping and overlap annotation
type: reference
---

## Mapping Approach: Strict Timeline Alignment

Map MI355X operators in their execution order sequentially against B200 rows. NOT functional mapping (e.g., B200 qkv_a_proj does NOT map to MI355X qkv_a GEMM by function — they map by timeline position).

- B200-only rows (set_mla_kv, splitK_reduce, MoE_input_quant, tensor_copy, MoE_finalize, residual_add) → MI355X columns left blank
- MI355X-only operations (add_rmsnorm_quant, mla_reduce_v1, fused_mxfp4_quant_sort) → placed at the next available B200 row in timeline order
- When MI355X has multiple ops that B200 fuses into one → combine MI355X values (e.g., post_attn_layernorm: 20.0+5.7+5.5=31.2)

## Module Classifications (user-corrected)

- Row 16 (cvt_fp16_to_fp4 after EP_AR#2): module = **MoE_quant** (NOT Proj), operator = **Moe_Expert_quant(BF16→FP4)** (NOT o_proj_quant #2)
- Row 12 MI355X module: **mla_decode:v_up_proj** (not v_up_proj_and_o_proj)
- Overlap references for rows 17/19/20: use **MoE_Quant_GEMM** (not o_proj_GEMM)

## Overlap Column Rules

- B200_Overlap_us: total overlap duration per kernel (10-layer average)
- B200_Overlap_With: per-partner breakdown with format `partner_op(same/cross):Xμs`
- same = same CUDA stream (recorded overlap, not true parallelism due to CUDA Graph replay)
- cross = cross-stream (real parallelism between stream 23 main and stream 385 alt)

## Pass-Level Grouping (for summary comparison)

| Pass | B200 Rows | Description |
|------|-----------|-------------|
| EP_AR before MHA | 1-2 | allreduce + quant (MI355X: reduce_scatter + load_rmsnorm + add_rmsnorm_quant) |
| MHA | 3-11 | qkv_a through uv_gemm (MI355X: qkv_a through mla_reduce) |
| O proj | 12-13, 18 | o_proj_quant + o_proj_GEMM #1 + #2 (MI355X: uv batched_gemm + o_proj gemm_xdl) |
| EP_AR before MOE | 14 | allreduce #2 (MI355X: post_attn combined) |
| MOE | 15-29 | router through residual_add (MI355X: router through moe_mxgemm down) |

## Reference File

`results/b200_dsr_fp4/b200_dsr_fp4_mtp0_ep4_tp4_sglang059_profiling/b200_vs_mi355x_kernel_map.csv`
— EP4 TP4 c=64, 29 B200 rows, strict timeline mapping with overlap + pass summary
