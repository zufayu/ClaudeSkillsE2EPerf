#!/usr/bin/env python3
"""Generate EP1 TP4 kernel map CSV+XLSX from report data."""
import csv
import openpyxl
import sys
import os

ROWS = [
    (1, "EP_AR", "EP_AR+residual+RMSNorm(fused)", "allreduce_fusion_kernel_oneshot_lamport", 13.0, "input_layernorm", "reduce_scatter_cross_device_store", 14.1, "B200 fuses AR+residual+RMSNorm; *estimated from 23.5/2"),
    (2, "", "", "", "", "input_layernorm", "local_device_load_rmsnorm", 4.7, "MI355X only (comm+norm split)"),
    (3, "", "", "", "", "input_layernorm", "dynamic_per_token_scaled_quant<32>", 4.3, "MI355X only (FP8 input quant)"),
    (4, "Attention", "qkv_a_proj_GEMM", "nvjet_splitK_TNT", 19.4, "gemm_a8w8", "kernel_gemm_xdl_cshuffle_v3_multi_d_b_preshuffle (128x16x32)", 11.1, "B200 FP4 splitK; MI355X FP8 preshuffle"),
    (5, "Attention", "qkv_a_splitK_reduce", "splitKreduce_kernel", 3.7, "", "", "", "B200 only"),
    (6, "Attention", "q/k_norm_RMSNorm x2", "RMSNormKernel", 5.5, "q_proj_and_k_up_proj", "fused_qk_rmsnorm_group_quant_kernel", 4.2, "MI355X fuses both norms into 1 kernel"),
    (7, "Attention", "q_b_proj_GEMM", "nvjet_v_bz_TNN", 6.4, "gemm_a8w8", "kernel_gemm_xdl_cshuffle_v3_multi_d_b_preshuffle (256x32x64)", 7.0, ""),
    (8, "Attention", "uk_gemm(K_expansion)", "nvjet_v_bz_TNT", 4.4, "batched_gemm", "batched_gemm_a8w8_M32_N128_K128", 5.8, ""),
    (9, "Attention", "RoPE+KV_cache_write", "RopeQuantizeKernel", 2.7, "rope_and_kv_cache", "fuse_qk_rope_concat_and_cache_mla_per_head_kernel", 4.5, ""),
    (10, "Attention", "set_mla_kv", "set_mla_kv_buffer_kernel", 1.7, "", "", "", "B200 only"),
    (11, "Attention", "Attention(FMHA)", "fmhaSm100fKernel", 20.5, "mla_decode", "mla_a8w8_qh16_qseqlen1_gqaratio16_ps", 26.4, ""),
    (12, "", "", "", "", "mla_decode", "kn_mla_reduce_v1_ps<512,16,1>", 5.5, "MI355X only (MLA reduce)"),
    (13, "Proj", "o_proj_quant(BF16->FP4)", "cvt_fp16_to_fp4", 2.2, "", "dynamic_per_token_scaled_quant<16>", 4.3, ""),
    (14, "Proj", "uv_gemm(V_expansion)", "nvjet_h_bz_TNT", 4.0, "v_up_proj_and_o_proj", "batched_gemm_a8w8_M32_N64_K128", 4.7, ""),
    (15, "Proj", "o_proj_GEMM", "DeviceGemmFp4GemmSm100 (256x256)", 11.2, "v_up_proj_and_o_proj", "FlatmmKernel (FlyDSL cktile)", 11.4, "B200 FP4 Cutlass; MI355X FP8 FlatMM"),
    (16, "EP_AR", "EP_AR+residual+RMSNorm(fused) #2", "allreduce_fusion_kernel_oneshot_lamport", 10.5, "post_attn_layernorm", "reduce_scatter_cross_device_store", 14.1, "*estimated from 23.5/2"),
    (17, "", "", "", "", "post_attn_layernorm", "local_device_load_rmsnorm", 4.7, "MI355X only"),
    (18, "", "", "", "", "", "triton_poi_fused_as_strided_clone x3", 13.5, "MI355X only (tensor reshape)"),
    (19, "MoE_Route", "shared_expert(FP4 GEMMs+SiLU)", "DeviceGemmFp4 x2 + SiLU + cvt x2", 29.0, "gemm_a16w16", "bf16gemm_fp32bf16_tn_64x64_splitk_clean", 8.9, "B200 shared expert parallel w/ MoE; MI355X router GEMM"),
    (20, "MoE_Route", "router_GEMM", "nvjet_h_bz_splitK_TNT", 12.6, "mxfp4_moe", "grouped_topk_opt_sort_kernel", 4.2, ""),
    (21, "MoE_Route", "router_splitK_reduce", "splitKreduce_kernel", 3.5, "mxfp4_moe", "MoeSortingMultiPhaseKernel_P0_v2 + P23", 8.7, ""),
    (22, "MoE_Route", "TopK_select", "routingMainKernel", 4.5, "mxfp4_moe", "mxfp4_quant_moe_sort_kernel<256>", 7.7, ""),
    (23, "MoE_Route", "expert_sort", "routingIndicesClusterKernel", 5.4, "mxfp4_moe", "mxfp4_quant_moe_sort_kernel<8>", 5.6, ""),
    (24, "MoE_Expert", "MoE_input_quant(BF16->FP4)", "quantize_with_block_size", 3.7, "", "", "", "B200 only: FP4 quant for MoE input"),
    (25, "MoE_Expert", "tensor_copy", "unrolled_elementwise_kernel", 3.6, "", "", "", "B200 only"),
    (26, "MoE_Expert", "gate_up_GEMM(+SwiGLU)", "bmm_E2m1_E2m1E2m1", 101.7, "mxfp4_moe", "kernel_moe_mxgemm_2lds<MulABScaleShuffled> (CK)", 102.1, "EP1: FlyDSL tune match (expert=257,topk=9)"),
    (27, "MoE_Expert", "down_GEMM", "bmm_Bfloat16_E2m1E2m1", 66.5, "mxfp4_moe", "kernel_moe_mxgemm_2lds<MulABScaleExpertWeight> (CK)", 57.0, ""),
    (28, "MoE_Expert", "MoE_finalize+residual", "finalizeKernelVecLoad", 8.3, "", "", "", "B200 only"),
    (29, "Residual", "residual_add", "vectorized_elementwise_kernel", 2.1, "", "", "", "B200 only"),
]

PASS_SUMMARY = [
    ("EP_AR before MHA", 13.0, 13.0, 23.1, 37.9, "-39%"),
    ("MHA (qkv->uv)", 64.4, 59.0, 73.5, 68.7, "+7%"),
    ("O_proj", 17.4, 17.0, 20.4, 15.5, "+32%"),
    ("EP_AR before MOE", 10.5, 10.0, 32.3, 17.5, "+85%"),
    ("MOE (router->finalize)", 234.5, 182.0, 185.6, 219.8, "-16%"),
    ("Residual", 5.7, 5.0, 0, 0, "-"),
    ("SUM", 346.1, 285.9, 334.9, 363.8, "-8%"),
]


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/kqian/ClaudeSkillsE2EPerf/results/b200_dsr_fp4/b200_dsr_fp4_mtp0_ep1_tp4_sglang059_profiling"
    os.makedirs(out_dir, exist_ok=True)

    headers = ["#", "B200_Module", "B200_Operator", "B200_Kernel", "B200_us", "MI355X_Module", "MI355X_Kernel", "MI355X_us", "Notes"]

    # CSV
    csv_path = os.path.join(out_dir, "b200_vs_mi355x_kernel_map_c64_ep1.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in ROWS:
            w.writerow(row)
    print(f"CSV: {csv_path}")

    # XLSX
    xlsx_path = os.path.join(out_dir, "b200_vs_mi355x_kernel_map_c64_ep1.xlsx")
    wb = openpyxl.Workbook()

    # Sheet 1: Kernel Map
    ws = wb.active
    ws.title = "EP1_Kernel_Map"
    ws.append(headers)
    for row in ROWS:
        ws.append(list(row))

    # Summary row
    ws.append([])
    ws.append(["", "", "", "", "B200 kernel_sum: 346.1", "", "", "MI355X TOTAL: 334.9", ""])
    ws.append(["", "", "", "", "B200 walltime: 285.9", "", "", "", ""])
    ws.append(["", "", "", "", "B200 overlap: 60.1", "", "", "", ""])

    # Sheet 2: PASS Summary
    ws2 = wb.create_sheet("PASS_Summary")
    ws2.append(["PASS", "B200_kernel_sum", "B200_walltime", "MI355X_EP1", "MI355X_EP4", "EP1_vs_EP4"])
    for row in PASS_SUMMARY:
        ws2.append(list(row))

    # Auto-width
    for sheet in [ws, ws2]:
        for col in sheet.columns:
            max_len = max(len(str(c.value or "")) for c in col)
            sheet.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)

    wb.save(xlsx_path)
    print(f"XLSX: {xlsx_path}")


if __name__ == "__main__":
    main()
