# FP4 性能差距分析：B200 vs MI355X — Breakdown 调查

> **Last updated:** 2026-04-17 v33
> **Model:** DeepSeek-R1-0528-NVFP4-v2, FP4
> **配置：** EP=8, DP=false, c=64, TP=8, chat 1K/1K
> **状态：** B200 trace 完成 ✅；Per-Module Kernel 分析完成 ✅；10 层平均数据完成 ✅；nvjet E4M3 源码考证完成 ✅；权重精度考证完成 ✅；算子级重构完成 ✅；MI355X 复现完成 ✅；MI355X 配置对齐复测完成 ✅；MI355X TPOT 25ms 来源分析完成 ✅；MI355X bs=64 kernel breakdown 修正完成 ✅；**MI355X TP=8+EP 公平对标完成 ✅**；**B200 4GPU torch trace per-layer 分析完成 ✅**；**Multi-stream overlap 分析完成 ✅**；**TP=4 分段执行分析+优化方向完成 ✅**；**v33 PASS 级优化预估完成 ✅**

## 目录

- [端到端性能总表](#端到端性能总表)
- [端到端性能TP=8 C=4跨框架对比表](#端到端性能tp8-c4跨框架对比表)
- [端到端性能TP=4跨框架对比表](#端到端性能tp4跨框架对比表)
- [问题背景](#问题背景)
- [跨平台对齐算子表](#跨平台对齐算子表)
  - [TP8-C64（35 行）](#tp8-c6435-行按逻辑功能对齐)
  - [TP4-C64](#tp4-c64)
    - [算子级对比表（29行）](#算子级对比表29行按执行时序对齐)
    - [PASS 功能分组汇总](#pass-功能分组汇总)
    - [TP=4 单层分段执行分析](#tp4-单层分段执行分析)
  - [分析和结论（v33）](#分析和结论v33)
- [MI355X TPOT 来源分析](#mi355x-tpot-来源分析)
- [精度说明](#精度说明)
- [TP=8 C=4 算子级对比](#tp8-c4-算子级对比)
- [NCU 硬件级 Profiling 进展](#ncu-硬件级-profiling-进展)
- [迭代日志](#迭代日志)

## 端到端性能总表

> MTP=0, chat 1K/1K, c=64, DP=false

| Platform  | Quant  | EP | TP | Env   | Mode   | Total Tput | Output Tput | Interac. | TPOT (ms) | TTFT (ms) |
|----------|-------|----|----|-----|------|------------|-------------|----------|-----------|-----------|
| B200 | NVFP4 | 8 | 8 | post2 | bench | 7577.9 | 3788.1 | 60.96 | 16.4 | 72 |
| B200 | NVFP4 | 1 | 8 | post2 | bench | 7602.5 | 3800.4 | 60.61 | 16.5 | 83 |
| B200 | NVFP4 | 8 | 8 | post2 | profiling | — | — | — | — | — |
| MI355X | MXFP4 | 1 | 8 | rocm721 | bench | 6961.7 | 3480.1 | 55.95 | 17.9 | 94.7 |
| MI355X | MXFP4 | 1 | 8 | rocm711 | bench | 6833.0 | 3415.7 | 54.95 | 18.2 | 93.5 |
| MI355X | MXFP4 | 8 | 8 | rocm721 | bench | 6558.0 | 3278.2 | 52.61 | 19.0 | 96.8 |
| MI355X | MXFP4 | 8 | 8 | rocm711 | bench | 6470.5 | 3234.5 | 52.05 | 19.2 | 90.6 |
| MI355X | MXFP4 | 1 | 4 | rocm721 | bench | 5054.8 | 2526.8 | 40.40 | 24.8 | 104.5 |
| MI355X | MXFP4 | 1 | 4 | rocm711 | bench | 4906.9 | 2452.9 | 39.28 | 25.5 | 104.4 |
| MI355X | MXFP4 | 1 | 8 | rocm711 | profiling | 6302.5 | 3150.6 | 50.5 | 19.8 | 117.3 |
| MI355X | MXFP4 | 8 | 8 | rocm711 | profiling | 6014.6 | 3006.6 | 48.3 | 20.7 | 114.6 |

## 端到端性能TP=8 C=4跨框架对比表

> MTP=0, chat 1K/1K, c=4, DP=false, ratio=0.8
>
> B200: SGLang v0.5.9 (SA InferenceX 同版) vs TRT-LLM rc6.post2
> MI355X: ATOM rocm7.2.2 (0.1.3.dev71) — **v32: 从 rocm7.1.1 升级后重测**

| Platform  | Framework   | Config  | Source    | Total Tput | Per‑GPU | Output Tput | TPOT p50 (ms) | TTFT p50 (ms) | Interactivity |
|----------|-----------|--------|--------|------------|---------|-------------|---------------|---------------|---------------|
| B200 | SGLang | EP8 TP8 | bench | 989.6 | 123.7 | 492.4 | 7.8 | 114.8 | 128.6 |
| **MI355X** | **ATOM 0.1.3** | **EP1 TP8** | **bench** | **882.1** | **110.3** | **438.9** | **8.2** | **70.2** | **121.5** |
| B200 | SGLang | EP8 TP8 | profiling | 872.9 | 109.1 | 434.3 | 7.8 | 134.5 | 114.1 |
| **MI355X** | **ATOM 0.1.3** | **EP8 TP8** | **bench** | **856.0** | **107.0** | **425.9** | **9.0** | **70.7** | **110.8** |
| B200 | TRT-LLM post2 | EP8 TP8 | bench | 848.5 | 106.1 | 422.2 | 9.2 | 55.0 | 109.3 |
| ~~MI355X~~ | ~~ATOM 0.1.1~~ | ~~EP8 TP8~~ | ~~bench~~ | ~~698.2~~ | ~~87.3~~ | ~~347.4~~ | ~~11.1~~ | ~~72.8~~ | ~~90.3~~ |

> **Key findings (v32 updated):**
> - **MI355X ROCm 7.2.2 大幅提升：** ATOM 0.1.3 (rocm722) vs 0.1.1 (rocm711)，EP8 TP8 吞吐量 +22.6%（856.0 vs 698.2），TPOT -19%（9.0 vs 11.1ms）
> - **MI355X EP1 TP8 最快配置（882.1）：** 超过 B200 TRT-LLM post2（848.5），仅落后 B200 SGLang 10.9%
> - **MI355X EP8 TP8（856.0）≈ B200 TRT-LLM post2（848.5）：** 基本持平，差距仅 0.9%
> - **B200 SGLang 仍领先：** vs MI355X EP1 TP8 领先 12.2%（989.6 vs 882.1），主要来自 multi-stream overlap + NVLink 优势
> - **MI355X TTFT 优势：** 70ms vs SGLang 115ms / TRT-LLM 55ms，ATOM scheduler 延迟介于两者之间
> - **Per-GPU 差距大幅缩小：** SGLang 123.7 vs MI355X EP1 110.3 tok/s/GPU（差距从 42% 缩至 12%）

## 端到端性能TP=4跨框架对比表

> MTP=0, chat 1K/1K, c=64, DP=false, ratio=0.8
>
> B200: SGLang v0.5.9 (SA InferenceX 同版) vs TRT-LLM rc6.post2
> MI355X: ATOM rocm7.2.2 (0.1.3.dev71) — **v32: 从 rocm7.1.1 升级后重测**

| Platform  | Framework   | Config  | Source    | Total Tput | Per‑GPU | Output Tput | TPOT p50 (ms) | TTFT p50 (ms) | Interactivity |
|----------|-----------|--------|--------|------------|---------|-------------|---------------|---------------|---------------|
| B200 | SGLang | EP1 TP4 | Ours-bench | 6486.3 | 1621.6 | 3242.4 | 18.8 | 399.2 | 53.25 |
| B200 | TRT-LLM post2 | EP4 TP4 | Ours-bench | 6426.9 | 1606.7 | 3212.7 | 19.4 | 86.1 | 51.61 |
| B200 | SGLang | EP4 TP4 | Ours-bench | 6397.3 | 1599.3 | 3197.9 | 19.04 | 403.5 | 52.52 |
| B200 | SGLang | EP4 TP4 | SA InferenceX | 6000.8 | 1500.2 | 2999.7 | 20.05 | 471.1 | 49.87 |
| **MI355X** | **ATOM 0.1.3** | **EP1 TP4** | **bench** | **5411.0** | **1352.8** | **2704.9** | **22.4** | **90.3** | **44.57** |
| **MI355X** | **ATOM 0.1.3** | **EP4 TP4** | **bench** | **5079.0** | **1269.8** | **2538.9** | **24.6** | **97.9** | **40.66** |
| **MI355X** | **ATOM 0.1.3** | **EP4 TP4** | **profiling** | **4966.7** | **1241.7** | **2482.8** | **25.1** | **101.5** | — |
| B200 | SGLang | EP4 TP4 | Ours-profiling | 6152.7 | 1538.2 | 3075.6 | 19.0 | 401.0 | 52.63 |
| ~~MI355X~~ | ~~ATOM 0.1.1~~ | ~~EP1 TP4~~ | ~~bench~~ | ~~4906.9~~ | ~~1226.7~~ | ~~2452.9~~ | ~~25.5~~ | ~~104.4~~ | ~~39.28~~ |
| ~~MI355X~~ | ~~ATOM 0.1.1~~ | ~~EP4 TP4~~ | ~~bench~~ | ~~4753.6~~ | ~~1188.4~~ | ~~2376.3~~ | ~~26.2~~ | ~~100.9~~ | ~~38.17~~ |

> **Key findings (v32 updated):**
> - **MI355X ROCm 7.2.2 提升：** EP1 TP4 +10.3%（5411 vs 4907），EP4 TP4 +6.8%（5079 vs 4754）
> - **B200 vs MI355X 差距缩小：** B200 SGLang EP1 vs MI355X EP1 从 1.32x 缩至 **1.20x**（6486 vs 5411）
> - **B200 SGLang ≈ TRT-LLM post2**（吞吐量差 <0.5%），但 TRT-LLM TTFT 4.7x 更低（86 vs 404ms）
> - **MI355X EP1 vs EP4:** EP1 仍更快（5411 vs 5079, +6.5%），4GPU 下 EP 通信开销超过收益
> - **MI355X TTFT 优势明显：** 90ms vs SGLang 399ms / TRT-LLM 86ms
> - **Per-GPU 差距：** B200 1621 vs MI355X 1353 tok/s/GPU（差距从 32% 缩至 **20%**）
> - **Profiling overhead:** MI355X ATOM 0.1.3 仅 2.2%（5079→4967），B200 SGLang 3.8%（6397→6153）

## 问题背景

SA InferenceX 报告的 B200 FP4 性能大幅领先 MI355X FP4，需要 breakdown 分析差距来源。

**ATOM 不支持 DP Attention**，原始 SA 对标配置（EP=4, DP=true）无法公平对比。选择 **EP=8, DP=false, c=64** 作为公平对标基准：DP=false 消除 DP Attention 差异；EP=8 是 B200 8GPU 的自然 EP 配置；c=64 是 SA 原始测试点。

## 跨平台对齐算子表

### TP8-C64（35 行，按逻辑功能对齐）

> **数据来源：** B200 nsys trace 10 层平均 / MI355X Kineto trace 全层平均（v21: TP=8+EP, bs=64）
> **对齐原则：** 按逻辑功能（非执行时序）逐行对齐，同一行的 B200 和 MI355X kernel 做同一件事。一端独有的算子另一端留空。
> **GAP(B-M)：** 正值 = B200 更慢，负值 = MI355X 更慢
> **v21 更新：** MI355X 从 TP=4 EP=1 (4GPU) 改为 **TP=8+EP (8GPU)**，与 B200 TP=8 EP=8 **完全对齐**。MoE GEMM per-GPU 权重量相同，差距反映纯算子效率。

| block    | ID | 逻辑算子      | B200 kernel      | B200 μs | MI355X module    | MI355X kernel      | MI355X μs | GAP(B‑M) | 备注         |
|-------|-----|---------|-------------|---------|---------------|---------------|-----------|----------|------|
| pre_attn_comm | 1 | TP_AR+residual+RMSNorm(融合) | moefinalize_lamport | 33.11 | | | 0 | 33.11 | B200独有:EP_AR+加权求和+residual+pre-attn_RMSNorm全融合;与下行qkv_a并行 |
| pre_attn_comm | 2 | TP_reduce_scatter+RMSNorm | | 0 | input_layernorm | reduce_scatter + rmsnorm | 21.12 | -21.12 | MI355X:TP=8 xGMI通信+RMSNorm（v21:从25.08降至21.12,8-way TP） |
| qkv_proj | 3 | per_token_quant(BF16→FP8) | | 0 | per_token_quant_hip | dynamic_per_token_scaled_quant | 5.48 | -5.48 | MI355X独有:qkv_a走FP8需先量化输入 |
| qkv_proj | 4 | qkv_a_proj_GEMM | nvjet_splitK_TNT | 25.12 | gemm_a8w8_bpreshuffle | gemm_xdl_preshuffle | 11.48 | 13.64 | 同一GEMM[64x7168]x[7168x2112];B200=BF16 MI355X=FP8 |
| qkv_proj | 5 | qkv_a_splitK_reduce | splitKreduce(bf16) | 3.68 | | | 0 | 3.68 | B200独有:cuBLAS splitK第二步;MI355X的CK单kernel完成 |
| qkv_proj | 6 | q_norm(RMSNorm) | RMSNormKernel | 2.64 | | | 0 | 2.64 | B200独有:MI355X融入Row8的fused_rms |
| qkv_proj | 7 | k_norm(RMSNorm) | RMSNormKernel(Stream8907) | 2.48 | | | 0 | 2.48 | B200独有:另一stream并行;MI355X融入Row8 |
| qkv_proj | 8 | fused_RMS+FP8_group_quant | | 0 | _fused_rms_fp8_group_quant | fused_rms_fp8_group_quant_kernel | 4.60 | -4.60 | MI355X独有:融合q/k_norm+FP8量化;对标B200 Row6+7(5.12us) |
| qkv_proj | 9 | q_b_proj_GEMM | nvjet_tst_TNN | 5.65 | q_proj_and_k_up_proj | gemm_xdl_preshuffle | 5.82 | -0.17 | 同一GEMM[64x1536]x[1536x3072/TP];Q展开;v21:TP=8维度更小→时间降 |
| qkv_proj | 10 | k_concat | CatArrayBatchedCopy | 4.89 | | | 0 | 4.89 | B200独有:K拼接RoPE部分;MI355X融入rope_kernel |
| qkv_proj | 11 | uk_gemm(K_expansion) | nvjet_tst_TNT | 3.76 | q_proj_and_k_up_proj | batched_gemm_a8w8_quant | 5.82 | -2.06 | 同一GEMM[64x512]x[512x2048/TP];kv_a→K_heads;v21:TP=8 heads/GPU减半 |
| rope_attn | 12 | RoPE+KV_cache_write | applyMLARopeAndAssignQKV | 3.46 | rope_and_kv_cache | fuse_qk_rope_concat_and_cache_mla | 5.12 | -1.66 | 两者都是融合kernel;MI355X额外含concat |
| rope_attn | 13 | Attention(FMHA/MLA) | fmhaSm100f(含reduce) | 20.67 | mla_decode | mla_a8w8_qh16_qseqlen1 | 18.00 | 2.67 | Q×KT→softmax→×V;v21:TP=8 heads/GPU=16→时间从24.08降至18.00;**MI355X反超** |
| rope_attn | 14 | MLA_reduce | | 0 | mla_decode | kn_mla_reduce_v1_ps | 8.92 | -8.92 | MI355X独有:多头reduce;B200已融入fmhaSm100f |
| out_proj | 15 | uv_gemm(V_expansion) | nvjet_tst | 3.74 | v_up_proj_and_o_proj | batched_gemm_a8w8_quant | 5.44 | -1.70 | 同一GEMM[64x512]x[512x2048/TP];v21:TP=8维度更小 |
| out_proj | 16 | o_proj_quant | quantize_with_block_size(FP4) | 2.46 | v_up_proj_and_o_proj | dynamic_per_token_scaled_quant(FP8) | 4.76 | -2.30 | B200=block-scale BF16→FP4;MI355X=per-token BF16→FP8 |
| out_proj | 17 | o_proj_GEMM | nvjet_ootst_FP4 | 6.13 | v_up_proj_and_o_proj | gemm_xdl_preshuffle(FP8) | 8.00 | -1.87 | 同一GEMM[64x2048]x[2048x7168/TP];v21:TP=8维度更小→从13.48降至8.00 |
| post_attn | 18 | TP_AR+residual+RMSNorm | userbuffers_rmsnorm(融合) | 15.15 | post_attn_layernorm | reduce_scatter + rmsnorm | 24.92 | -9.77 | 同一功能:post-attn TP通信+pre-MoE_RMSNorm;MI355X xGMI 8-way |
| post_attn | 19 | residual_AllGather(EP) | userbuffers_allgather | 9.74 | | | 0 | 9.74 | B200独有:EP=8分片后需AG恢复完整residual;**MI355X EP也应有，可能融入其他kernel** |
| router | 20 | router_GEMM | nvjet_tss_splitK | 5.42 | gemm_a16w16 | bf16gemm_splitk | 9.08 | -3.66 | router GEMM[64x7168]x[7168x256]BF16 |
| router | 21 | router_splitK_reduce | splitKreduce(fp32) | 2.73 | | | 0 | 2.73 | B200独有:MI355X的CK内含或无独立reduce |
| router | 22 | MoE_gate_up_quant | quantize_with_block_size | 2.93 | | | 0 | 2.93 | B200独有:BF16→FP4量化;MI355X融入Row26的fused_quant_sort |
| router | 23 | TopK_select | routingMainKernel | 4.30 | mxfp4_moe | grouped_topk_opt_sort | 4.60 | -0.30 | 从256expert选top-8;v21:EP后仅32 local experts路由 |
| router | 24 | expert_sort(phase1) | routingIndicesCluster | 5.12 | mxfp4_moe | MoeSorting_P0_v2 | 4.52 | 0.60 | 按expert_ID分组排序;v21:32 experts更快 |
| router | 25 | expert_sort(phase2+3) | | 0 | mxfp4_moe | MoeSorting_P23 | 5.80 | -5.80 | MI355X独有:3-phase_sort多一个阶段 |
| moe_expert | 26 | gate_up_quant+sort(融合) | | 0 | mxfp4_moe | fused_mxfp4_quant_moe_sort | 8.32 | -8.32 | MI355X独有:BF16→MXFP4量化+排序融合;对标B200 Row22 |
| moe_expert | 27 | gate_up_GEMM(+SwiGLU) | bmm_E2m1(FP4含SwiGLU) | 59.55 | mxfp4_moe | kernel_moe_mxgemm_2lds | 66.52 | -6.97 | **核心MoE;v21:EP后per-GPU权重相同(32exp);MI355X仅1.12x→纯算子效率差** |
| moe_expert | 28 | down_quant+sort(融合) | | 0 | mxfp4_moe | fused_mxfp4_quant_moe_sort | 5.04 | -5.04 | MI355X独有:第二次quant+sort融合 |
| moe_expert | 29 | down_GEMM | bmm_Bfloat16(FP4→BF16) | 32.77 | mxfp4_moe | kernel_moe_mxgemm_2lds(atomic) | 34.24 | -1.47 | **v21:EP后per-GPU权重相同;MI355X仅1.04x→几乎持平** |
| shared_exp | 30 | shared_gate_up_quant | quantize_with_block_size | 3.75 | | | 0 | 3.75 | B200独有:MI355X shared_expert融入Row27+29的MoeMxGemm |
| shared_exp | 31 | shared_gate_up_GEMM | nvjet_ootst_FP4 | 10.03 | | | 0 | 10.03 | B200独有:MI355X作为always-active expert编入grouped_GEMM |
| shared_exp | 32 | SiLU×Mul | silu_and_mul_kernel | 1.64 | | | 0 | 1.64 | B200独有:MI355X SwiGLU融入MoeMxGemm epilogue |
| shared_exp | 33 | shared_down_quant | quantize_with_block_size | 2.17 | | | 0 | 2.17 | B200独有:MI355X融入MoeMxGemm |
| shared_exp | 34 | shared_down_GEMM | nvjet_ootst_FP4 | 3.81 | | | 0 | 3.81 | B200独有:MI355X融入MoeMxGemm |
| moe_finalize | 35 | moe_finalize(=Row1) | moefinalize_lamport | (=Row1) | | | 0 | 0 | 同Row1:层尾=下一层层首;仅计一次不重复 |
| | | | **B200_SUM** | **276.91** | | **MI355X_SUM** | **267.58** | **9.33** | **v21:MI355X TP=8+EP 反超 B200！** |

### TP4-C64

> **数据来源：**
> - B200: SGLang v0.5.9 torch profiler trace，TP=4 EP=4，chat 1K/1K c=64。FMHA 层锚点切分，position-based module 分配，第 10-40 层平均，4410 层样本。
>   - Kernel sum: 350.5μs/layer，Elapsed(关键路径): 283.3μs/layer，Overlap: 67.2μs (19.2%)
>   - 验证: 283.3 × 61 = 17.28ms ≈ decode walltime 17.50ms（误差 1.3%）
> - MI355X: **v32: ATOM rocm722 (0.1.3.dev71)** Kineto trace，TP=4 EP=4，chat 1K/1K c=64。decode_breakdown_c64.xlsx layers 10-40 平均，**363.8μs/layer**
>   - 单流串行执行（HIP Graph 单 stream capture），无 multi-stream overlap
>   - 验证: 363.8 × 61 = 22.2ms ≈ decode walltime 21.57ms（误差 3%）
>   - 旧数据 (rocm711): 399.5μs/layer, decode 23.73ms
> - **配置对齐：** 两平台均 4GPU TP=4 EP=4，per-GPU MoE 权重量相同（64 experts/GPU）
>
> **Multi-stream overlap 说明：**
> - **B200 SGLang** 启用了 `enable_flashinfer_allreduce_fusion=True`（lamport allreduce），该机制使用对称内存在**独立 CUDA stream** 上执行 allreduce+RMSNorm。当 lamport stream 执行 allreduce 时，计算 stream 可以同时执行后续 GEMM（如 qkv_a splitK），形成 multi-stream overlap。
> - **MI355X ATOM** 使用 RCCL fused_allreduce_rmsnorm 在**单一 HIP stream** 上串行执行所有 kernel（包括通信）。HIP Graph capture 确认所有操作在同一 stream 上，无 fork-join pattern。
> - **公平对比原则：** B200 使用 elapsed（关键路径时间，扣除 overlap），MI355X 使用 kernel sum（等于 elapsed，因为无 overlap）。两者都代表实际 GPU 时间线上的关键路径长度。
> - **对齐原则：** 按 B200 执行时序逐行对齐，MI355X 同功能 kernel 对齐到同一行。B200 含 multi-stream overlap 信息。
> - **Overlap 列：** B200 该算子与其他算子在不同 stream 上重叠的时间和对象。

#### 算子级对比表（29行，按执行时序对齐）

| # | B200_Module  | B200_Operator     | B200_Raw_Kernel      | Stream | B200_us | Overlap_us | B200_Overlap_With         | MI355X_Module   | MI355X_Kernel     | MI355X_us | Notes        |
|---|-------------|---------------|-----------------|--------|---------|------------|-------------------|---------------|---------------|-----------|-------|
| 1 | EP_AR | EP_AR+residual+RMSNorm(fused) | allreduce_fusion_kernel_oneshot_lamport | 23 | 25.9 | 24.1 | qkv_a_proj_GEMM(same):24.1μs | input_layernorm | reduce_scatter_cross_device_store + local_device_load_rmsnorm | 29 | B200 fuses AR+residual+RMSNorm; B200 value inflated by same-stream overlap (true ~11μs) |
| 2 | Attention | qkv_a_proj_GEMM | nvjet_splitK_TNT | 23 | 31.9 | 25.5 | EP_AR+residual+RMSNorm(same):24.1μs \| qkv_a_splitK_reduce(same):1.4μs | input_layernorm | add_rmsnorm_quant_kernel<256,8> | 5.5 | MI355X only: additional quant after norm |
| 3 | Attention | qkv_a_splitK_reduce | splitKreduce_kernel | 23 | 3.7 | 2.5 | qkv_a_proj_GEMM(same):1.4μs \| q/k_norm_RMSNorm(same):1.1μs | gemm_a16w16 | bf16gemm_fp32bf16_tn_64x64_splitk_clean | 16.1 | MI355X qkv_a GEMM maps here in strict timeline |
| 4 | Attention | q/k_norm_RMSNorm | RMSNormKernel (stream 23) | 23 | 3.2 | 2.7 | q/k_norm_RMSNorm(cross):1.5μs \| qkv_a_splitK_reduce(same):1.1μs | | | | B200 dual-stream norm #1; no MI355X match at this position |
| 5 | Attention | q/k_norm_RMSNorm | RMSNormKernel (stream 385) | 385 | 2.4 | 1.5 | q/k_norm_RMSNorm(cross):1.5μs | q/k_norm | add_rmsnorm_quant_kernel<64,8> | 10.8 | MI355X combines both norms into 1 kernel |
| 6 | Attention | q_b_proj_GEMM | nvjet_v_bz_TNN | 23 | 6.3 | 1.3 | uk_gemm(K_expansion)(same):1.3μs | q_proj_and_k_up_proj | Cijk_BBS_MT64x32x128_SK3_ISA950 (Tensile) | 11.8 | |
| 7 | Attention | uk_gemm(K_expansion) | nvjet_v_bz_TNT | 23 | 4.4 | 1.3 | q_b_proj_GEMM(same):1.3μs | q_proj_and_k_up_proj | batched_gemm_a8w8_M32_N128_K128 (aiter) | 5.5 | |
| 8 | Attention | RoPE+KV_cache_write | RopeQuantizeKernel | 23 | 2.7 | 0 | | rope_and_kv_cache | fuse_qk_rope_concat_and_cache_mla_per_head_kernel (aiter) | 5.3 | |
| 9 | Attention | set_mla_kv | set_mla_kv_buffer_kernel | 23 | 1.7 | 0 | | | | | B200 only |
| 10 | Attention | Attention(FMHA) | fmhaSm100fKernel | 23 | 20.3 | 0 | | mla_decode | mla_a8w8_qh16_qseqlen1_gqaratio16_ps (aiter) | 25.5 | |
| 11 | Attention | uv_gemm(V_expansion) | nvjet_h_bz_TNT | 23 | 4 | 0 | | mla_decode | kn_mla_reduce_v1_ps<512,16,1> (aiter) | 7.2 | MI355X attention output reduction |
| 12 | Proj | o_proj_quant(BF16→FP4) #1 | cvt_fp16_to_fp4 | 23 | 2.1 | 0.1 | | v_up_proj_and_o_proj | batched_gemm_a8w8_M32_N64_K128 (aiter) | 6.6 | MI355X uv_gemm maps here in strict timeline |
| 13 | Proj | o_proj_GEMM #1 | DeviceGemmFp4GemmSm100 | 23 | 9.4 | 0.1 | | | | | B200 splits o_proj into #1+#2 (9.4+10.2=19.6) |
| 14 | EP_AR | EP_AR+residual+RMSNorm(fused) #2 | allreduce_fusion_kernel_oneshot_lamport | 23 | 11 | 9.2 | router_GEMM(cross):9.1μs | v_up_proj_and_o_proj | Cijk_BBS_MT64x64x128_SK3_ISA950 (Tensile) | 21.4 | MI355X single o_proj; B200 fuses AR+residual+RMSNorm |
| 15 | MoE_Route | router_GEMM | nvjet_h_bz_splitK_TNT | 385 | 13.5 | 14.3 | EP_AR+residual+RMSNorm(cross):9.1μs \| o_proj_quant(cross):2.5μs \| router_splitK_reduce(same):1.4μs \| o_proj_GEMM(cross):1.3μs | post_attn_layernorm | reduce_scatter_cross_device_store + local_device_load_rmsnorm + triton_fused_clone ×2 | 31.2 | MI355X post_attn combined (20.0+5.7+5.5) |
| 16 | MoE_quant | Moe_Expert_quant(BF16→FP4) | cvt_fp16_to_fp4 | 23 | 2.5 | 2.6 | router_GEMM(cross):2.5μs \| router_splitK_reduce(cross):0.2μs | gemm_a16w16 | bf16gemm_fp32bf16_tn_64x64_splitk_clean | 9.2 | MI355X router GEMM maps here in strict timeline |
| 17 | MoE_Route | router_splitK_reduce | splitKreduce_kernel | 385 | 3.6 | 6.3 | MoE_Quant_GEMM(cross):3.4μs \| router_GEMM(same):1.4μs \| MoE_input_quant(same):1.4μs \| o_proj_quant(cross):0.2μs | | | | B200 only: splitK reduction |
| 18 | Proj | o_proj_GEMM #2 | DeviceGemmFp4GemmSm100 | 23 | 10.2 | 12.2 | MoE_input_quant(cross):3.8μs \| router_splitK_reduce(cross):3.4μs \| tensor_copy(cross):3.0μs \| router_GEMM(cross):1.3μs \| TopK_select(cross):1.1μs | | | | (included in B200 o_proj #1+#2 = 19.6) |
| 19 | MoE_Route | MoE_input_quant(BF16→FP4) | quantize_with_block_size | 385 | 3.8 | 5.1 | MoE_Quant_GEMM(cross):3.8μs \| router_splitK_reduce(same):1.4μs | | | | B200 only: FP4 quant for MoE input |
| 20 | MoE_Route | tensor_copy | unrolled_elementwise_kernel | 385 | 3.2 | 3.1 | MoE_Quant_GEMM(cross):3.0μs | | | | B200 tensor_copy; no MI355X match at this position |
| 21 | MoE_Route | TopK_select | routingMainKernel | 385 | 4.5 | 6.3 | SiLU×Mul(cross):2.6μs \| expert_sort(same):2.5μs \| o_proj_GEMM(cross):1.1μs \| shared_quant(cross):1.1μs | triton_poi | triton_poi_fused_as_strided_clone_copy__0 | 5.5 | MI355X tensor_copy maps here in strict timeline |
| 22 | MoE_Route | expert_sort | routingIndicesClusterKernel | 385 | 5.3 | 7.2 | TopK_select(same):2.5μs \| SiLU×Mul(cross):2.0μs \| shared_quant(cross):1.5μs \| shared_GEMM(cross):1.4μs \| gate_up(same):0.2μs | mxfp4_moe | grouped_topk_opt_sort_kernel (aiter) | 5.4 | MI355X TopK maps here in strict timeline |
| 23 | Shared_Exp | SiLU×Mul | act_and_mul_kernel | 23 | 3.2 | 4.7 | TopK_select(cross):2.6μs \| expert_sort(cross):2.0μs | mxfp4_moe | MoeSortingMultiPhaseKernel_P0_v2 + P23 (ck_tile) | 10.8 | MI355X expert_sort maps here in strict timeline; B200 shared expert SiLU parallel with MoE(cross) |
| 24 | Shared_Exp | shared_quant(BF16→FP4) | cvt_fp16_to_fp4 | 23 | 1.7 | 2.2 | expert_sort(cross):1.5μs \| TopK_select(cross):1.1μs | mxfp4_moe | _fused_dynamic_mxfp4_quant_moe_sort_kernel (aiter) | 9.5 | MI355X MoE quant+sort before gate_up |
| 25 | MoE_Expert | gate_up_GEMM(+SwiGLU) | bmm_E2m1_E2m1E2m1 | 385 | 101.7 | 4.3 | shared_GEMM(cross):3.4μs \| down_GEMM(same):0.6μs \| expert_sort(same):0.2μs | mxfp4_moe | kernel_moe_mxgemm_2lds<MulABScaleShuffled> (CK) | 119.5 | |
| 26 | Shared_Exp | shared_GEMM(FP4) | DeviceGemmFp4GemmSm100 | 23 | 5.8 | 6 | gate_up_GEMM(cross):3.4μs \| expert_sort(cross):1.4μs | mxfp4_moe | _fused_dynamic_mxfp4_quant_moe_sort_kernel (aiter) | 5.8 | MI355X quant between gate_up and down; B200 shared expert GEMM parallel with gate_up(cross) |
| 27 | MoE_Expert | down_GEMM | bmm_Bfloat16_E2m1E2m1 | 385 | 55.8 | 2.5 | MoE_finalize(same):0.6μs \| gate_up_GEMM(same):0.6μs | mxfp4_moe | kernel_moe_mxgemm_2lds<MulABScaleExpertWeightShuffled> (CK) | 58 | |
| 28 | MoE_Expert | MoE_finalize+residual | finalizeKernelVecLoad | 385 | 7.5 | 0.6 | down_GEMM(same):0.6μs | | | | B200 only |
| 29 | Residual | residual_add | vectorized_elementwise_kernel | 23 | 2.1 | 0 | | | | | B200 only |
| | | | | | **B200 TOTAL (kernel_sum): 353.2** | | | | | | |
| | | | | | **B200 Walltime: 286.9** | | | | | | |
| | | | | | **B200 Overlap: 66.3** | | | | **MI355X TOTAL** | **399.6** | |

#### PASS 功能分组汇总

| PASS | B200 μs | MI355X rocm711 | MI355X rocm722 | GAP (v32) | 变化 |
|------|---------|---------------|---------------|-----------|------|
| MOE | 214.2 | 223.7 | **219.8** | -5.6 | router(8.3)+topk(4.1)+sort(8.6)+quant(9.5)+gate_up(126.2)+down(63.1) |
| MHA | 80.6 | 88.8 | **68.7** | -11.9 | ~~88.8~~ quant(4.1)+qkv_a(10.9)+fused_norm(4.2)+q_b(7.1)+k_up(4.6)+rope(4.5)+mla(26.1)+mla_reduce(5.9)+v_up(5.9) — **fused norm + FP8 GEMM 优化** |
| O_proj | 21.7 | 21.4 | **15.5** | -6.2 | ~~21.4~~ quant(4.2)+FlatmmKernel(11.3) — **cktile FlatmmKernel 更快** |
| EP_AR before MHA | 25.9 | 34.5 | **37.9** | -12.0 | RS(33.1)+rmsnorm(4.8) — RS 变慢但含更多功能 |
| EP_AR before MOE | 11.0 | 31.2 | **17.5** | -6.5 | ~~31.2~~ RS(13.0)+rmsnorm(4.5) — **triton_poi 消失，RS 大幅缩短** |
| **sum** | **353.4** | **399.6** | **363.8** | **-10.4** | ~~-46.2~~ **总差距从 46→10 μs** |

#### B200 Multi-stream Overlap 分解

> B200 kernel_sum (353.4μs) 与 walltime (288.5μs) 的差 64.9μs 来自两种并行机制：

| 类型 | μs/layer | 占比 | 主要位置 |
|------|----------|------|----------|
| Single-stream (PDL) | 34.4 | 53% | EP_AR#1 ∥ qkv_a 跨层 pipeline (~24μs) + 各 boundary PDL (~10μs) |
| Dual-stream | 30.5 | 47% | EP_AR#2 ∥ router (~9μs) + shared expert ∥ MoE routing (~21μs) |
| **TOTAL** | **64.9** | **100%** | = kernel_sum(353.4) - walltime(288.5) |

> **MI355X 无此优化：** 单 HIP stream 串行执行，kernel_sum ≈ walltime。B200 关键路径 288.5μs vs MI355X 363.8μs 的 **75μs 差距中 ~65μs (87%) 来自 overlap 优势**，仅 ~10μs (13%) 来自算子效率差距。
>
> **v32 更新：** MI355X rocm722 kernel sum 从 399.6→363.8μs (缩小 36μs)，但 walltime 差距仍有 75μs，因为 overlap 是架构级优势无法通过软件栈升级弥补。

#### TP=4 单层分段执行分析

> 基于 29 行时序表原始 Torch Trace 数据，将单层 Transformer 按执行阶段分为 7 段，逐段对比两平台关键路径。
> 数据为统计平均值，可能与精确值存在细微偏差。

##### 总览

| 段 | 功能 | B200 μs | MI355X rocm711 | MI355X rocm722 | GAP (v32) | GAP 主因 |
|----|------|---------|---------------|---------------|-----------|---------|
| 1 | AR+RMSNorm+qkv_a | ~32 | ~45 | ~52.9 | 20.9 | 新增 per_token_quant(4.1)，qkv_a 从 BF16→FP8 GEMM |
| 2 | q/k Norm+q_b+k_up | ~14 | ~33.6 | **~15.9** | **1.9** | ~~19.6~~ **fused_qk_rmsnorm_quant 融合大幅缩短** |
| 3 | RoPE+KV cache | 4.4 | 5.3 | 4.5 | 0.1 | — |
| 4 | MLA Decode+uv | ~24 | ~39.3 | ~37.9 | 13.9 | mla_reduce 仍有独立开销 |
| 5 | O_proj | ~11.5(+10.2‖AR) | 21.4 | **~15.5** | **4.0** | ~~10~~ **FlatmmKernel (cktile) 更快** |
| 6 | MoE routing+shared | ~34 | ~71 | **~43.2** | **9.2** | ~~37~~ **triton_poi 消失 + RS 缩短** |
| 7 | MoE GEMM+finalize | ~167 | ~183 | ~194 | 27 | MoE GEMM 反而变慢 (+6%) |
| **合计** | | **~283** | **~400** | **~364** | **~81** | ~~117~~ **缩小 36μs (31%)** |

##### 段 1：AR + RMSNorm + qkv_a 投影（v32 GAP 20.9μs）

B200 将 AllReduce+residual+RMSNorm 融合为单个 lamport kernel，利用 PDL 在同一 Stream 上提前发射 qkv_a_proj_GEMM，两者重叠执行。含后续 splitKreduce 共 ~32μs。MI355X v32 新增 per_token_quant (FP8 输入量化)，qkv_a 从 BF16 改为 FP8 GEMM (gemm_xdl)，合计 ~53μs。

| B200 算子 | μs | MI355X 算子 (v32 rocm722) | μs |
|-----------|------|---------------------------|------|
| AR+residual+RMSNorm(fused) ‖ qkv_a(PDL) | 25.9 | reduce_scatter | 33.1 |
| qkv_a_proj_GEMM | 31.9(overlap) | load_rmsnorm | 4.8 |
| splitKreduce | 3.7 | per_token_quant (NEW) | 4.1 |
| — | — | gemm_xdl (qkv_a, FP8) | 10.9 |
| **关键路径** | **~32** | **串行合计** | **~52.9** |

> v32 变化：qkv_a 从 bf16gemm (16.1μs) → gemm_xdl FP8 (10.9μs) 快了 32%，但新增 per_token_quant (4.1μs) 量化步骤。RS 从 29→33.1μs 变慢（可能因 TP=4 通信量不变但 RCCL 版本差异）。

##### 段 2：q/k Norm + q_b + k_up 投影（v32 GAP 1.9μs ~~19.6~~）

B200 串行执行 q/k_norm + q_b_proj + uk_gemm，另一 Stream 并行执行第二个 RMSNorm，整体 ~14μs。MI355X v32 **大幅缩短**：fused_qk_rmsnorm_group_quant 将 q/k norm + FP8 量化融合为单个 4.2μs kernel（旧版需要 2 个 add_rmsnorm_quant 共 10.9μs），q_b GEMM 从 Cijk(11.8)→gemm_xdl(7.1)，合计 ~15.9μs。

| B200 算子 | μs | Stream | MI355X 算子 (v32) | μs |
|-----------|------|--------|-------------------|------|
| q/k_norm_RMSNorm | 3.2 | 23 | fused_qk_rmsnorm_group_quant (NEW) | 4.2 |
| q/k_norm_RMSNorm(并行) | 2.4 | 385 | — | — |
| q_b_proj_GEMM | 6.3 | 23 | gemm_xdl (q_b) | 7.1 |
| uk_gemm(K_expansion) | 4.4 | 23 | batched_gemm_a8w8 (k_up) | 4.6 |
| **关键路径** | **~14** | | **串行合计** | **~15.9** |

> v32 变化：**最大改善段**。fused norm+quant 省了 6.7μs，q_b GEMM 快了 4.7μs。GAP 从 19.6→1.9μs，MI355X 几乎追平 B200。

##### 段 3：RoPE + KV cache（GAP 0.9μs）

B200 通过 RoPE+KV_cache_write 与 set_mla_kv 串行实现 4.4μs；MI355X v32 从 5.3→4.5μs。

| B200 算子 | μs | MI355X 算子 (v32) | μs |
|-----------|------|-------------------|------|
| RoPE+KV_cache_write | 2.7 | fuse_qk_rope_concat_cache | 4.5 |
| set_mla_kv | 1.7 | — | — |
| **合计** | **4.4** | | **4.5** |

##### 段 4：MLA Decode + uv_gemm（v32 GAP 13.9μs ~~15.3~~）

B200 串行执行 FMHA + uv_gemm ~24μs。MI355X 需额外 mla_reduce 步骤，v32 三算子串行 ~37.9μs（旧 39.3μs）。

| B200 算子 | μs | MI355X 算子 (v32) | μs |
|-----------|------|-------------------|------|
| fmhaSm100f (FMHA) | 20.3 | mla_a8w8_qh16 | 26.1 |
| nvjet_h_bz_TNT (uv_gemm) | 4.0 | mla_reduce_v1 | 5.9 |
| — | — | batched_gemm_a8w8 (v_up) | 5.9 |
| **合计** | **~24** | **合计** | **~37.9** |

> v32 变化：mla_reduce 7.2→5.9μs，v_up 6.6→5.9μs。

##### 段 5：输出投影（v32 GAP 4.0μs ~~10~~）

B200 执行 o_proj_quant + o_proj_GEMM#1 合计 11.5μs，GEMM#2 (10.2μs) 执行期间同步启动另一 Stream 的融合 Reduce-RMSNorm，成功隐藏约 10μs 延迟。MI355X v32 使用 FlatmmKernel (cktile) 替代 Cijk (Tensile)，从 21.4→15.5μs。

| B200 算子 | μs | MI355X 算子 (v32) | μs |
|-----------|------|-------------------|------|
| o_proj_quant(BF16→FP4) | 2.1 | per_token_quant (NEW) | 4.2 |
| o_proj_GEMM #1 | 9.4 | FlatmmKernel (o_proj, cktile) | 11.3 |
| o_proj_GEMM #2（与段 6 AR 重叠） | 10.2 | — | — |
| **kernel sum** | **21.7** | **合计** | **15.5** |

> v32 变化：**o_proj GEMM 从 Cijk(21.4μs)→FlatmmKernel(11.3μs)** 快了 47%！新增 per_token_quant(4.2μs) 但总体仍省 5.9μs。

##### 段 6：MoE Routing + Shared Expert（GAP 37μs）

B200 双 Stream 交错执行，关键路径 max(34.4, 33.9) ≈ 34μs。MI355X 全部串行 ~71μs。

**B200 双 Stream：**

| Stream A（计算, 34.4μs） | μs | Stream B（路由, 33.9μs） | μs |
|--------------------------|------|--------------------------|------|
| reduce-fused-rms | 11.0 | router_GEMM | 13.5 |
| MoE_Expert_quant(BF16→FP4) | 2.5 | router_splitK_reduce | 3.6 |
| o_proj_GEMM #2 | 10.2 | MoE_input_quant(BF16→FP4) | 3.8 |
| SiLU×Mul | 3.2 | tensor_copy | 3.2 |
| shared_quant(BF16→FP4) | 1.7 | TopK_select | 4.5 |
| shared_GEMM(FP4) | 5.8 | expert_sort | 5.3 |
| **合计** | **34.4** | **合计** | **33.9** |

**MI355X v32（串行, ~43.2μs ~~71~~）：**

| # | 算子 (v32 rocm722) | μs | 旧 (rocm711) |
|---|---------------------|------|-------------|
| 1 | reduce_scatter | 13.0 | 20.0 |
| 2 | load_rmsnorm | 4.5 | 5.7 |
| 3 | bf16gemm (router) | 8.3 | 9.2 |
| 4 | grouped_topk_opt_sort | 4.1 | 5.4 |
| 5 | MoeSorting_P0+P23 | 8.6 | 10.8 |
| 6 | fused_mxfp4_quant_sort | 4.7 | 9.5 |
| | **合计** | **~43.2** | ~~71.6~~ |

> v32 变化：**triton_poi_fused 两个算子消失**（-11μs），RS 20→13μs，quant_sort 9.5→4.7μs。GAP 从 37→9.2μs。

##### 段 7：MoE GEMM + Finalize（v32 GAP 27μs ~~16~~）

B200 单 Stream 串行 ~167μs。MI355X v32 三步串行 **~194μs（旧 183μs，变慢 6%）**。gate_up 和 down GEMM 都变慢了（可能是新 ATOM 版本的 CK kernel 版本变化或 ROCm 7.2.2 的 HIP runtime 差异）。

| B200 算子 | μs | MI355X 算子 (v32) | μs | 旧 |
|-----------|-------|-------------------|-------|------|
| gate_up_GEMM(+SwiGLU) | 101.7 | moe_mxgemm(gate_up) | 126.2 | 119.5 |
| — | — | fused_mxfp4_quant_sort | 4.8 | 5.8 |
| down_GEMM | 55.8 | moe_mxgemm(down) | 63.1 | 58.0 |
| MoE_finalize+residual | 7.5 | — | — | — |
| residual_add | 2.1 | — | — | — |
| **合计** | **~167** | **合计** | **~194** | ~~183~~ |

> v32 变化：gate_up 119.5→126.2μs (+6%)，down 58→63.1μs (+9%)。**MoE GEMM 是唯一变差的部分**，需要排查是 CK kernel 版本回退还是 ROCm 7.2.2 的 runtime 开销。

##### MI355X 优化方向（v32 更新）

> MI355X 无法使用 B200 的 PDL（Programmatic Dependent Launch）和双 stream 并行。以下聚焦单 stream 架构下的算子效率优化空间。

**方向 1：MoE GEMM 回退调查（段 7，预估 10-17μs）**

v32 gate_up 从 119.5→126.2μs (+6%)，down 从 58→63.1μs (+9%)。ROCm 7.2.2 + ATOM 0.1.3 的 CK `moe_mxgemm_2lds` 可能选了不同的 tile config。需要对比新旧 CK kernel 实例名和参数，确认是 kernel 版本回退还是 runtime overhead。aiter tune 调研已确认这些 shape 不在 tuned 配置中（CK grouped GEMM 不可通过 aiter tune 优化）。

**方向 2：MLA + reduce 融合（段 4，预估 0-6μs）**

v32: mla_a8w8_qh16 (26.1μs) → mla_reduce_v1 (5.9μs)。B200 fmhaSm100f 在单 kernel 内完成。v32 mla_reduce 已从 7.2→5.9μs，但仍有融合空间。

**~~方向 3：q_b GEMM tile 调优（段 2）~~  已解决**

v32: q_b GEMM 从 Cijk(11.8μs)→gemm_xdl(7.1μs)，已接近 B200 的 6.3μs。aiter tune 确认已在 tuned 配置中。

**方向 4：MoE 排序精简（段 6，预估 2-3μs）**

MoeSorting_P0+P23 (10.8μs) vs B200 expert_sort (5.3μs)。三阶段排序与 moe_mxgemm input layout 强耦合，改 sort 需联动改 GEMM，性价比低。

**方向 5：gate_up → down 量化融合（段 7，预估 4-5μs）**

B200 的 cuBLAS bmm 在 gate_up epilogue 中完成 SwiGLU + FP4 quantize，直接输出 FP4，gate_up 与 down 之间**无独立 quant kernel**。MI355X 的 moe_mxgemm epilogue 已融合 SwiGLU 但输出 BF16，需额外 `fused_mxfp4_quant_sort` (5.8μs) 转 MXFP4。CK epilogue fusion 框架支持 post-op chain，追加 MXFP4 quantize 是自然扩展。gate_up 与 down 的 token-expert 分配不变、layout 兼容，sort 应为 identity。B200 已证明此模式可行。

| # | 方向 | 段 | 预估 μs | 可行性 |
|---|------|-----|---------|--------|
| 1 | gate_up GEMM 效率 | 7 | 6-12 | 中 |
| 2 | MLA + reduce 融合 | 4 | 0-7 | 不确定 |
| 3 | q_b GEMM tile 调优 | 2 | 3-5 | 高 |
| 4 | MoE 排序精简 | 6 | 2-3 | 低 |
| 5 | gate_up→down quant 融合 | 7 | 4-5 | 中-高 |
| | **总计** | | **15-32** | |

> **小结：** 当前 MI355X ~400μs，上述优化可覆盖 15-32μs（→ 368-385μs）。与 B200 关键路径 ~283μs 的 ~117μs 总差距中，~50-60μs 来自双 stream / PDL 并行优势（段 1/5/6），属架构级差异，软件层面无法弥补。

#### 分析和结论（v33）

> 基于 B200 torch trace 精确 overlap 拆解 + MI355X rocm722 kernel breakdown，重新以 MLA 和 MoE 两大 PASS 进行跨平台对比分析。

##### 数据基础

针对 TP=4 EP=4 C=64 场景：

| 指标 | B200 | MI355X |
|------|-----:|-------:|
| kernel_sum | 353.4μs | 363.75μs |
| walltime | 288.5μs | 363.75μs (单流，无 overlap) |
| overlap | 64.9μs | 0 |

##### B200 Overlap 精确拆解

B200 的 64.9μs overlap 来自两种机制：

| 类型 | μs/layer | 占比 | 来源 |
|------|----------|------|------|
| **Single-stream (PDL)** | **34.4** | **53%** | EP_AR#1 ∥ qkv_a 跨层 pipeline (~22μs，不稳定) + 各 boundary PDL (~12μs，稳定) |
| **Dual-stream** | **30.5** | **47%** | EP_AR#2 ∥ router (~9μs) + shared expert ∥ MoE routing (~21μs) |
| **TOTAL** | **64.9** | **100%** | = kernel_sum(353.4) - walltime(288.5) |

**PDL 关键发现**：EP_AR#1 ∥ qkv_a 的 ~22μs overlap 不是真实计算 savings。通过 trace 分析发现 qkv_a duration 与 EP_AR duration 相关系数 **r=1.000**（slope=1.00），qkv_a 实际计算时间为 **~15μs**（直方图 14-16μs 尖峰），剩余时间是 EP_AR allreduce 等待被 PDL 塞入 qkv_a 的 preamble。真实 PDL boundary savings 仅 **~12μs**（~0.1-2.5μs × 14 个 boundary）。

**双流关键发现**：B200 将 shared expert 从 MoE grouped GEMM 中拆出，放在 stream 23 上与 stream 385 的 MoE routing 并行执行。shared expert 总计 23.4μs（gate_up_quant + gate_up_GEMM + SiLU + down_quant + down_GEMM），完全被 MoE routing 链（~30μs）隐藏，对 critical path 零影响。

##### PASS 级对比

| PASS | B200 walltime μs | MI355X 当前 μs | GAP | 包含算子 |
|------|----------------:|---------------:|----:|---------|
| **MLA decode** | 20.3 | 31.95 | 11.65 | FMHA/MLA attention + multi-head reduce + uv_gemm |
| **MoE** | 190.4 | 223.96 | 33.56 | routing + gate_up + down + shared expert + finalize |
| **Comm+Norm+Proj** | ~78 | ~108 | ~30 | EP_AR ×2 + RMSNorm + qkv_a + q_b + k_up + RoPE + o_proj |
| **总计** | **~288.5** | **~363.8** | **~75** | |

B200 MoE 190.4μs 构成：gate_up(101.7) + down(55.8) + finalize(7.5) + residual(2.1) + shared expert(23.4μs, 双流隐藏不计入 critical path)。
MI355X MoE 223.96μs 构成：gate_up 含 shared(126) + quant(4.8) + down 含 shared(63) + routing 链(topk 4.1 + sort 8.6 + quant_sort 9.5 + router 8.3) = 全部串行。

##### MI355X 优化预估

| # | 优化方向 | 当前 μs | 优化后 μs | 省 μs | 手段 |
|---|---------|--------:|----------:|------:|------|
| 1 | **MLA attn+reduce 融合** | 31.95 | ~20.3 | **10.6** | 将 mla_a8w8 (26.1) + mla_reduce (5.9) 融合为单 kernel，对标 B200 fmhaSm100f |
| 2 | **MoE GEMM (FlyDSL)** | 223.96 | ~190.4 | **~30** | FlyDSL 替换 CK moe_mxgemm，提升至 B200 bmm 效率水平 |
| 3 | **Comm 融合** | ~108 | ~98 | **~10** | 启用 aiter 已有的 rmsnorm_quant 融合 kernel（方案 A） |
| | **总计** | **363.8** | **~309** | **~50** | |

##### 优化后 vs B200

| | B200 walltime | MI355X 当前 | MI355X 优化后 | 差距 |
|--|-------------:|------------:|--------------:|-----:|
| 单层 | 288.5μs | 363.8μs | **~309μs** | **~20μs** |
| 61 层 | 17.6ms | 22.2ms | **~18.8ms** | ~1.2ms |

优化后差距从 75μs 缩小到 **~20μs**。剩余 20μs 构成：
- PDL boundary savings: ~12μs（AMD 无硬件等价机制，结构性差距）
- allreduce 通信效率差异: ~8μs（NVLink5 vs xGMI Infinity Fabric BW 差异 + userbuffers 融合优势）

## MI355X TPOT 来源分析

> **问题：** MI355X benchmark 报告的 Client TPOT 高于 GPU decode walltime，差距从何而来？

### 三层时间栈

| 层级 | 含义 | MI355X v21 (TP8+EP) | MI355X v20 (TP4) | B200 | 数据来源 |
|------|------|---------------------|------------------|------|---------|
| **L1: Kernel 时间** | GPU 算子执行时间之和（单层） | **267.6 μs** | 344.0 μs | 251.1 μs (关键路径) | run_parse_trace.py (bs=64) |
| **L2: Decode walltime** | GPU 端一个完整 decode step (61 层) | **17.22 ms** | 21.56 ms | 15.6 ms | --mark-trace → decode_walltime_trace.csv |
| **L3: Client TPOT** | 客户端观测 per-request TPOT | **18.9 ms** | 24.9 ms | 17.8 ms | benchmark_serving.py |

每层之间都有 gap，需要分别解释：

### Gap 1: L1 → L2（Kernel Sum vs Decode Walltime）

单层 kernel 时间 × 61 层 = 估算 decode walltime。与实测对比：

| | 指标 | Kernel sum/层 | Elapsed/层 | × 61 估算 | 实测 Decode | 拟合误差 |
|---|---|---|---|---|---|---|
| **B200 (4GPU TP4)** | elapsed 关键路径 | 350.5 μs | **283.3 μs** | 17.28 ms | 17.50 ms | **1.3%** |
| **MI355X (4GPU TP4)** | kernel sum=elapsed | 399.5 μs | **399.5 μs** | 24.37 ms | 23.73 ms | **2.7%** |
| B200 (8GPU TP8) | 参考 | 251.1 μs | — | 15.3 ms | 15.6 ms | 1.8% |
| MI355X (8GPU TP8+EP) | 参考 | 267.6 μs | — | 16.3 ms | 17.2 ms | 5.2% |

**B200 elapsed 与 walltime 高度吻合（1.3%误差）**，验证了 multi-stream overlap 分析的正确性。MI355X 单流执行，kernel sum ≈ elapsed ≈ walltime。

### Gap 2: L2 → L3（Decode Walltime vs Client TPOT）

这是 continuous batching 中 **prefill interleaving** 造成的。

#### TPOT 的定义（源码石锤）

`benchmark_serving.py:249`：
```python
tpot = (outputs[i].latency - outputs[i].ttft) / (output_len - 1)
```

`backend_request_func.py:74-106`：
```python
st = time.perf_counter()           # request 开始
async for chunk_bytes in response.content:
    timestamp = time.perf_counter()
    if ttft == 0.0:
        ttft = timestamp - st       # 第一个 token
    else:
        itl.append(timestamp - most_recent_timestamp)  # inter-token latency
    most_recent_timestamp = timestamp
latency = most_recent_timestamp - st  # 最后一个 token
```

TPOT = (最后一个 token 时间 - 第一个 token 时间) / (output_len - 1) = **该 request 所有 ITL 的平均值**。

#### Prefill 打断机制

Continuous batching 下，一个 decode step (bs=64) 同时为 64 个 request 各生成 1 个 token。当有新 request 到达时，scheduler 会**插入一个 prefill step**，期间所有正在 decode 的 request 都暂停等待。

```
时间 →
decode[bs=64] decode[bs=64] decode[bs=64] ... prefill[tok=1024] decode[bs=64] ...
   21ms           21ms           21ms              85ms              21ms
                                    ↑
                          这里所有 64 个 request 的 ITL
                          从 ~21ms 拉长到 ~21+85 = ~106ms
```

某个 request 的 TPOT 取决于它经历了多少次 prefill 打断。如果 request 需要生成 999 个 token (output_len=1000)，每个 token 正常需要 1 个 decode step (~21ms)，但其中有若干步被 prefill 打断变成 ~106ms，则 TPOT > 21ms。

#### Trace 数据验证

**v21 (TP=8+EP)** decode walltime (bs=64): avg=17.22ms, p50=17.16ms, max=86.78ms, 1573 events

**v20 (TP=4 EP=1)** 从 `decode_walltime_trace_chat_c64_tp4_p640.csv`：

| Batch Size | Count | Avg (ms) | P50 (ms) | P99 (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|---|---|
| 1 | 990 | 10.39 | 10.37 | 10.69 | 9.98 | 28.96 |
| 62 | 15 | 22.08 | 21.94 | 22.60 | 21.77 | 22.60 |
| 63 | 54 | 22.03 | 22.03 | 22.68 | 21.35 | 22.68 |
| **64** | **1821** | **21.56** | **21.53** | **22.46** | 18.79 | 110.77 |

- **v21: bs=64 decode 从 21.56→17.22ms (-20%)**，与 kernel sum 降幅 (-22%) 一致
- **bs=1 有 990 次** = 每次 prefill 后的第一个 decode step（bs=1 因为 batch 还没填满）

#### 从 Decode Walltime 重建 TPOT

每个 request 经历 ~999 个 decode step。在 bs=64 稳态下，平均每 ~18 个 bs=64 decode step 后插入一次 prefill（990 次 bs=1 ÷ ~54 个并发 slot 轮转 ≈ 18:1 的 decode:prefill 比例）。

BS-weighted average 重建（从 analyze_prefill_impact.py 对 Kineto trace 的分析）：

```
bs_weighted_avg_gap = sum(gap_ms × bs) / sum(bs)
```

其中 gap = 相邻两次 decode 事件之间的时间差（包含了 prefill 的等待时间）。分析结果：
- 正常 gap（无 prefill 打断）：~22ms，占 ~95.7%
- 打断 gap（跨 prefill）：~85ms，占 ~4.3%
- **BS-weighted average ≈ 25.3ms**，与 benchmark mean TPOT (24.9ms) 仅差 0.4ms

> **结论：** Client TPOT = GPU decode + prefill interleaving overhead。这不是性能 bug，是 continuous batching 的正常行为。
> - **v20 (TP4 EP1):** TPOT 24.9ms = decode 21.6ms + overhead 3.3ms
> - **v21 (TP8+EP):** TPOT 18.9ms = decode 17.2ms + overhead 1.7ms（prefill 也更快了）

### B200 是否也有同样问题？

**是的。** B200 的 benchmark mean TPOT = 17.8ms，也高于 GPU decode walltime 15.6ms，差距 2.2ms。原理完全相同，只是 B200 的 prefill 更快所以 overhead 更小。

此外 B200 TRT-LLM 默认 `stream_interval: 10`（每 10 个 token 发一次 SSE chunk），导致 ITL = 10 × TPOT ≈ 170ms，这是 token batching 而非性能问题。



## 精度说明

> **Kernel 精度判断依据：**
> - `nvjet tst`（无 E4M3 标记、无前置 quantize）= **BF16 GEMM**。MLA 投影（q_a, q_b, kv_a, kv_b）全 61 层在 `hf_quant_config.json` exclude_modules 中排除 FP4
> - `nvjet ootst Avec16UE4M3`（有前置 quantize）= **FP4 GEMM**。kernel 名中 E4M3 指 block scale factor 格式（OCP MXFP4: E2M1 数据 + UE4M3 scale），非数据精度。源码确认：`fp4Quantize.cpp` 输出 E2M1+UE4M3，`kernelParams.h` E2M1 数据配 E4M3 scale
> - `bmm_E2m1` = **FP4 GEMM**，kernel 名直接标注 E2M1
> - `fmhaSm100f QkvE4m3` = **FP8 E4M3 KV cache**
> - `nvjet` 是 cuBLAS 闭源 kernel（`cublasScaledMM.cpp`，需 CUDA 12.8+），`ootst`/`tst` 是 tile scheduler 变体

## 待填充

- [x] B200 EP=8 no-DP 复现 + trace 分析
- [x] Per-Module Kernel 级分析（第 40 层实测）
- [x] 10 层平均数据（第 40-49 层，替换单层快照）
- [x] nvjet E4M3 源码考证（确认 E4M3 = block scale factor 格式）
- [x] NVFP4 权重精度考证（hf_quant_config.json 确认 MLA 投影 BF16，MoE/out_proj FP4）
- [x] 61 层端到端实测数据（20.472ms，单层均值 335.6μs）
- [x] MI355X 复现 + 环境对比（ATOM dev220 vs release，aiter 一致）
- [x] MI355X 配置对齐复测（max-model-len=2248, enforce_eager=false, gpu_memory_util=0.90，ATOM 0.1.3.dev1）
- [x] MI355X TPOT 25ms 来源分析（三层时间栈 + prefill interleaving 证明 + kernel breakdown）
- [x] MI355X bs=64 kernel breakdown 修正（v18 的 165.7μs 系 bs=1 数据，修正为 344.0μs，overhead 从 53% 降至 2.7%）
- [x] MI355X TP=8+EP 公平对标（v21：TP/EP 与 B200 对齐，kernel sum 267.6μs 反超 B200 276.9μs，MoE 差距从 2x→1.1x）
- [x] 4GPU 跨框架对比表（v23：B200 SGLang/TRT-LLM + MI355X ATOM，ratio=0.8 对齐，含 SA InferenceX 基线）
- [x] B200 4GPU per-layer torch trace 分析（v24：SGLang 4GPU TP=4 EP=4，torch profiler，FMHA 层切分+position module 分配，25 算子 × 8 module）

## TP=8 C=4 算子级对比

> B200 SGLang (EP8 TP8, chat c=4) vs MI355X ATOM (EP8 TP8, chat c=4)
> 数据文件: `b200_vs_mi355x_kernel_map_c4_ep8.csv`

### PASS 功能分组汇总 (c=4 EP8)

| Pass | B200 (μs) | MI355X (μs) | B200/MI355X | 说明 |
|------|-----------|-------------|-------------|------|
| EP_AR (pre-MHA) | 26.7 | 19.7 | 1.36x | B200 fused allreduce; MI355X reduce_scatter+load_rmsnorm |
| MHA | 37.7 | 59.8 | 0.63x | MI355X: 含 add_rmsnorm_quant×2 + qkv + rope + mla + batched_gemm |
| O_proj | 20.3 | 11.8 | 1.72x | B200: FP4 quant+GEMM×2; MI355X: single BF16 GEMM |
| EP_AR (pre-MOE) | 8.8 | 21.6 | 0.41x | MI355X: post_attn (15.9) + triton_clone (5.6) |
| MOE | 89.6 | 89.0 | 1.01x | MoE 几乎持平 |
| **TOTAL (kernel_sum)** | **183.1** | **201.8** | **0.91x** | |
| **Walltime** | **122.1** | **~196** | **0.62x** | B200 overlap 61μs (33%); MI355X 无 multi-stream overlap |

**关键发现 (c=4 vs c=64 对比):**
- c=4 下 MoE GEMM 持平 (1.01x)，而 c=64 下 B200 MoE 优势更大 → bs-dependent tile efficiency
- B200 multi-stream overlap 在 c=4 下依然提供 33% kernel_sum 缩减
- MI355X 在 EP_AR 两个 pass 都显著慢于 B200 (reduce_scatter vs fused allreduce)
- MHA pass MI355X 反而更慢 (0.63x)，因为 add_rmsnorm_quant 和 mla_reduce 额外开销

## NCU 硬件级 Profiling 进展

### 运行历史

10 次 B200 NCU workflow (SGLang FP4 EP8, serve mode c=64)，**0 次产生可用 .ncu-rep**:

| # | 日期 | Phase 1 (nsys) | Phase 2 (ncu) | 根因 |
|---|------|----------------|---------------|------|
| 1-2 | 04-10 | 失败 | - | 早期脚本 bug |
| 3 | 04-12 | 5.7GB nsys-rep → 11GB sqlite | 未运行 | `find_decode_region.py` fetchall() OOM |
| 4-9 | 04-13 | launch-skip=321600 ✓ | **Server TimeoutError 3600s** | ncu 跳过 32 万 kernel launch 太慢 |
| 10 | 04-14 | 未运行 | - | zufa_sglang container 不存在 |

### 根因分析

**核心问题：ncu 不适合 serve 模式 c=64**

ncu 逐 kernel 拦截 + replay，每个 kernel 有 10-100x overhead：
- Phase 1 成功后得到 `--launch-skip 321,600`
- Phase 2 需要 ncu 逐个检查并跳过 32 万个 kernel launch（warmup + prefill + ramp-up decode）
- DeepSeek-R1 FP4 8-GPU 在 ncu 下的 server 启动速度极慢
- 3600s timeout 内无法完成跳过 → 永远无法到达稳定态 decode kernel

### 替代方案：基于 torch trace duration 推测稳定态

**思路：** 已有 torch profiler trace 包含每个 decode step 的 walltime。利用这些 duration 数据推测稳定态时间段，绕过 nsys dry-run。

**可行性分析：**

| 方面 | 评估 |
|------|------|
| 稳定态识别 | **可行** — decode walltime CSV 清楚显示 ramp-up → steady → ramp-down |
| 映射到 kernel index | **困难** — torch trace 计时和 ncu 下的 kernel 时间不一致（ncu 本身 10-100x 减速） |
| launch-skip 计算 | **需要换算** — 知道 kernels_per_decode (26-29 per layer × 61 layers ≈ 1600/pass)，但需要准确知道 warmup+prefill 的 kernel 数 |

**结论：** 直接用 torch trace 的稳定态时间段去推测 ncu launch-skip 不够可靠，因为 ncu 下执行时间完全不同。

### 推荐方案：缩小问题规模

改用 **c=4 TP=4 EP=4** 配置：
- c=4 产生的 kernel launch 远少于 c=64（launch-skip 从 ~32 万降到 ~几千）
- TP=4 EP=4 用 4 GPU，模型加载更快
- 可用 `--skip-dry-run --launch-skip N` 手动指定，跳过 Phase 1
- 或用 offline mode (ISL=64, OSL=8) → 完全可控的 kernel 数量

## 迭代日志

| 日期 | 变更 |
|------|------|
| 2026-04-17 v33 | **PASS 级优化预估 + overlap 精确拆解。** B200 overlap 64.9μs 精确分解为 PDL 34.4μs + dual-stream 30.5μs。PDL 中 EP_AR∥qkv_a (~22μs) 不是真实计算 savings（r=1.000 correlation，qkv_a 实际计算 ~15μs），真实 PDL boundary 收益 ~12μs。o_proj 分类修正：EP_AR 后的 DeviceGemmFp4 是 shared expert gate_up（非 o_proj #2）。新增 PASS 级优化预估表：MLA 融合省 10.6μs + MoE FlyDSL 省 30μs + comm 融合省 10μs → MI355X 从 363.8→~309μs，差距从 75→~20μs |
| 2026-04-16 v32 | **MI355X ROCm 7.2.2 升级重测。** zufa_atom 容器升级到 rocm/atom-dev:latest (ROCm 7.2.2, ATOM 0.1.3.dev71, PyTorch 2.10.0)。TP=8 C=4 表：MI355X EP1 TP8 (882.1) 超越 B200 TRT-LLM post2 (848.5)，仅落后 B200 SGLang 10.9%。TP=4 C=64 表：MI355X EP1 TP4 (5411) vs B200 SGLang (6486) 差距从 1.32x 缩至 1.20x。软件栈升级收益：EP8 TP8 c=4 +22.6%, EP1 TP4 c=64 +10.3%, EP4 TP4 c=64 +6.8% |
| 2026-04-14 v30 | **TP=8 C=4 跨框架对比表完整版。** B200 SGLang bench (989.6) vs TRT-LLM post2 bench (848.5) vs MI355X ATOM bench (698.2)。SGLang 1.17x TRT-LLM，1.42x MI355X。TRT-LLM TTFT 最低（55ms）。v29 数据因 GPU 冲突作废（两 job 并行跑在同一 8GPU 上） |
| 2026-04-14 v28 | **MI355X kernel 名修正为 trace 原始名。** 29 行表 MI355X_Kernel 列全部替换为 decode_breakdown.xlsx 原始 kernel 名。关键发现：q_b/o_proj 实际用 Tensile (hipBLASLt) 而非 CK gemm_xdl_preshuffle；uk_gemm/uv_gemm 用 aiter batched_gemm_a8w8；仅 MoE GEMM 用 CK kernel_moe_mxgemm_2lds |
| 2026-04-13 v27 | **数据修正+宽表。** 修正总览表段 2 MI355X 缺失 q/k_norm_RMSNormx2 (10.8μs)：MI355X 22.8→33.6，GAP 8.8→19.6，总 MI355X ~388→~400，总 GAP ~105→~117。段 2 明细表补齐 MI355X q/k_norm 行。全表 CSS 适配 24 寸显示器（min-width: 1400px + nowrap）|
| 2026-04-13 v26 | **TP=4 单层分段执行分析与优化方向。** 将单层 Transformer 按执行阶段分为 7 段，逐段对比 B200 vs MI355X 关键路径（4GPU TP=4 EP=4）。总差距 ~105μs 中 ~50-60μs 来自双 stream/PDL 并行。新增 5 项 MI355X 软件优化方向（15-32μs）：gate_up GEMM 效率、MLA+reduce 融合、q_b GEMM tile 调优、MoE 排序精简、gate_up→down quant epilogue 融合（B200 已实现此模式）|
| 2026-04-06 v24 | **B200 4GPU per-layer torch trace 分析。** 新增 4GPU TP=4 EP=4 SGLang torch trace per-layer kernel 分析（layers 10-40, 4410 样本）。25 算子 × 8 module。per-layer 356.4μs × 61 = 21.74ms ≈ kernel sum 21.69ms（拟合 0.2%）。MoE 48.2% 主导，gate_up 102.0μs + down 58.8μs。4GPU vs 8GPU 对比：MoE GEMM ~1.75x（理论 2x），总层时间 1.29x |
| 2026-04-05 v23 | **4GPU 跨框架对比表。** 新增 4GPU (TP=4) FP4 跨框架对比表：B200 SGLang vs TRT-LLM post2 vs MI355X ATOM，含 SA InferenceX 基线。ratio=0.8 对齐。SGLang ≈ TRT-LLM 吞吐（<0.5%），TRT-LLM TTFT 4.7x 更低。B200 vs MI355X 1.34x。MI355X EP4 vs EP1: EP4 略慢 -3.1%|
| 2026-04-04 v22 | **端到端性能总表。** 新增 5-metric 数据总表（B200 bench/profiling + MI355X bench/profiling × rocm711/721），统一 EP/TP/Env/Mode 维度。B200 profiling 数据待补。删除 fp4-b200-vs-mi355x-comparison.md（重复内容）|
| 2026-04-03 v21 | **MI355X TP=8+EP 公平对标。** MI355X 从 TP=4 EP=1 (4GPU) 改为 TP=8+EP (8GPU)，与 B200 完全对齐。核心发现：(1) MI355X kernel sum 267.6μs **反超** B200 276.9μs（快 3.4%）；(2) MoE GEMM 差距从 2.07x→1.12x (gate_up) / 1.88x→1.04x (down)，证明 v20 的 98% 差距来自 GPU 数差异；(3) 但 decode walltime MI355X 仍慢 10%（17.22 vs 15.6ms），因 B200 moefinalize_lamport 并行遮盖和 NVLink 通信优势；(4) Per-GPU throughput B200 领先 21%（490 vs 406 tok/s/GPU）|
| 2026-04-03 v20 | **跨平台对齐算子表。** 新增 37 行按逻辑功能对齐的 B200 vs MI355X kernel 对比表。关键修正：(1) moefinalize_lamport 融合 EP_AR+residual+pre-attn_RMSNorm（跨层融合）；(2) gemm_a16w16=router GEMM 非 shared_expert；(3) MI355X shared_expert 融入 MoeMxGemm grouped GEMM。总差距 94.35μs 中 MoE GEMM 贡献 98%（GPU 数 4vs8 差异）|
| 2026-04-01 v19 | **MI355X bs=64 kernel breakdown 修正。** v18 的 L1 kernel 数据（165.7μs）系 parse_trace.py 取了 bs=1 decode 事件。用 `run_parse_trace.py --target-bs 64` 重新生成，正确值 344.0μs/层。L1→L2 overhead 从 53% 降至 2.7%（344×61=21.0ms ≈ 实测 21.6ms）。MoE 从 44.8μs→188.0μs 变为绝对主导（55%）。删除 53% overhead 的错误推测（HIP Graph overhead、missing kernels）|
| 2026-04-01 v18 | **MI355X TPOT 25ms 来源分析完成。** 建立三层时间栈模型（L1→L2→L3）。L2→L3 gap: prefill interleaving 导致约 4.3% 的 decode step 被打断（~85ms vs 正常 ~22ms），BS-weighted avg 重建 25.3ms ≈ benchmark 24.9ms。说明跨平台算子对比的 TP/EP 局限性 |
| 2026-03-30 v17 | **MI355X 配置对齐复测完成。** 对齐 SA CI 配置（max-model-len=2248, enforce_eager=false, gpu_memory_util=0.90）。ATOM 0.1.3.dev1 + aiter v0.1.12。Output TPS/GPU=624.9 vs CI 600.7 (+4.0%)，Interactivity=40.09 vs 38.55 (+4.0%)。此前 v16 结果偏差大（-56%）系配置未对齐所致。修正对标数据表和环境对比表 |
| 2026-03-30 v16 | **MI355X 复现完成 + 环境对比。** Output TPS/GPU=328.5 vs CI 297.8 (+10.3%)，Interactivity=21.59 vs 18.98 (+13.7%)。偏差源于 ATOM 版本差异：本机 dev220（多 220 commits）vs CI release 0.1.1。aiter 版本完全一致（`a498c8b62`）。增加 MI355X 复现环境对比表。ATOM 已更新到最新 main（dev466），含 `--mark-trace` 功能 |
| 2026-03-27 v15 | **61 层实测数据。** 增加 61 decode 层端到端实测时间（15.6ms）。关键路径估算（251.1×61=15.3ms）与实测偏差仅 2%，验证 P1 并行模型准确 |
| 2026-03-27 v14 | **10 层平均数据。** 用第 40-49 层平均值替换单层快照。增加 Min/Max 波动列。修正 router 包含 splitK GEMM（12.0μs）、tp_AR+norm 正确归类（15.2μs）。moe_gemm 占比从 24.4% 升至 33.5%（含 quantize），moe_finalize 从 58.9 降至 33.1μs（单层是极端值） |
| 2026-03-27 v13 | **继续精简。** 删除权重精度汇总表（与主表精度列重复）和 nvjet 源码考证（5 条证据→5 行摘要）。合并为"精度说明"注释 |
| 2026-03-27 v12 | **精简报告。** 删除全 trace 统计表、精度判断证据、大类汇总、kernel 明细映射、耗时波动表、hf_quant_config exclude 详情、精度分布统计。保留 15 算子序列表 + 权重精度汇总表 + nvjet 源码考证。后续增加 10 层平均 + 61 层数据 |
| 2026-03-27 v11 | **算子级重构。** 从 25+ kernel 行合并为 15 个逻辑算子，方便跨平台对比。合并 splitKreduce 到 GEMM、quantize 到目标 GEMM。增加 P1 并行组标注、关键路径时间（265.7μs）、大类汇总表、kernel 明细映射表。MI355X 列待补充 |
| 2026-04-15 v31 | **TP=8 C=4 算子级对比表 + NCU 硬件级 profiling 进展总结。** c=4 EP8 kernel map (26 B200 rows vs 23 MI355X rows)。NCU 10 次 run 失败根因：ncu 不适合 c=64 serve mode (launch-skip=321600 timeout)。推荐改用 c=4 TP4 EP4 offline mode |
| 2026-03-27 v10 | **NVFP4 权重精度考证完成。** 从 `hf_quant_config.json` 确认：MLA 4 投影层（q_a, q_b, kv_a, kv_b）全 61 层 exclude → BF16 权重；MoE/out_proj/shared_expert 使用 FP4 权重。解决了 `nvjet tst` kernel 的"待确认"精度：BF16×BF16。增加权重精度汇总表和精度分布统计 |
| 2026-03-27 v9 | **nvjet E4M3 源码考证完成。** 确认 ootst kernel 的 E4M3 指 block scale factor 格式（非数据精度）。ootst 实际执行 FP4(E2M1) GEMM。增加 5 条源码证据（fp4Quantize.cpp, kernelParams.h, KernelMetaInfo.h, fp4GemmTrtllmGen.cpp, cublasScaledMM.cpp）。更新精度判断表和 kernel 表 Precision 列 |
| 2026-03-27 v8 | 增加 B200 % 占比列 + MI355X kernel mapping（时间待补）+ 模块占比汇总表 |
| 2026-03-27 v7 | 精简为仅保留配置 B。删除配置 A 全部内容和 A vs B 对比 |
| 2026-03-27 v6 | 基于第 40 层实测数据重写 Per-Module 表格。增加精度判断证据、量化算子标注、Pipeline 重叠图示、层间波动对比 |
| 2026-03-27 v5 | Per-Module Kernel 级对比：B200 nsys trace (c=64) vs 355X rocprof (c=16) |
| 2026-03-25 v4 | Config B trace 分析完成 + Config A vs B kernel 级对比 |
| 2026-03-25 v3 | 增加配置 B（EP=8, DP=false, c=64）公平对标方案 |
| 2026-03-25 v2 | B200 nsys trace 分析完成 |
| 2026-03-25 v1 | 初版 |
