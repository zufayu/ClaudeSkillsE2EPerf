# FP4 性能差距分析：B200 vs MI355X — Breakdown 调查

> **Last updated:** 2026-04-13 v26
> **Model:** DeepSeek-R1-0528-NVFP4-v2, FP4
> **配置：** EP=8, DP=false, c=64, TP=8, chat 1K/1K
> **状态：** B200 trace 完成 ✅；Per-Module Kernel 分析完成 ✅；10 层平均数据完成 ✅；nvjet E4M3 源码考证完成 ✅；权重精度考证完成 ✅；算子级重构完成 ✅；MI355X 复现完成 ✅；MI355X 配置对齐复测完成 ✅；MI355X TPOT 25ms 来源分析完成 ✅；MI355X bs=64 kernel breakdown 修正完成 ✅；**MI355X TP=8+EP 公平对标完成 ✅**；**B200 4GPU torch trace per-layer 分析完成 ✅**；**Multi-stream overlap 分析完成 ✅**；**TP=4 分段执行分析+优化方向完成 ✅**

## 端到端性能总表

> MTP=0, chat 1K/1K, c=64, DP=false

| Platform | Quant | EP | TP | Env | Mode | Total Tput | Output Tput | Interac. | TPOT (ms) | TTFT (ms) |
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

## 4GPU (TP=4) FP4 跨框架对比

> MTP=0, chat 1K/1K, c=64, DP=false, ratio=0.8
>
> B200: SGLang v0.5.9 (SA InferenceX 同版) vs TRT-LLM rc6.post2
> MI355X: ATOM rocm7.1.1 (EP=4 vs EP=1)

| Platform | Framework | Config | Source | Total Tput | Per-GPU | Output Tput | TPOT p50 (ms) | TTFT p50 (ms) | Interactivity |
|----------|-----------|--------|--------|------------|---------|-------------|---------------|---------------|---------------|
| B200 | SGLang | EP4 TP4 | SA InferenceX | 6000.8 | 1500.2 | 2999.7 | 20.05 | 471.1 | 49.87 |
| B200 | SGLang | EP4 TP4 | Ours-bench | 6397.3 | 1599.3 | 3197.9 | 19.04 | 403.5 | 52.52 |
| B200 | SGLang | EP4 TP4 | Ours-profiling | 6152.7 | 1538.2 | 3075.6 | 19.0 | 401.0 | 52.63 |
| B200 | TRT-LLM post2 | EP4 TP4 | Ours-bench | 6426.9 | 1606.7 | 3212.7 | 19.4 | 86.1 | 51.61 |
| MI355X | ATOM | EP4 TP4 | Ours-bench | 4753.6 | 1188.4 | 2376.3 | 26.2 | 100.9 | 38.17 |
| MI355X | ATOM | EP4 TP4 | Ours-profiling | 4435.6 | 1108.9 | 2217.3 | 28.0 | 111.5 | 35.77 |
| B200 | SGLang | EP1 TP4 | Ours-bench | 6486.3 | 1621.6 | 3242.4 | 18.8 | 399.2 | 53.25 |
| MI355X | ATOM | EP1 TP4 | SA CI | 4806.8 | 1201.7 | 2402.9 | 25.94 | — | 38.55 |
| MI355X | ATOM | EP1 TP4 | Ours-bench | 4906.9 | 1226.7 | 2452.9 | 25.5 | 104.4 | 39.28 |

> **Key findings:**
> - **B200 SGLang ≈ TRT-LLM post2**（吞吐量差 <0.5%），但 TRT-LLM TTFT 4.7x 更低（86 vs 404ms）
> - **B200 vs MI355X（EP4 TP4 bench）:** B200 SGLang 1.34x（6397 vs 4754）
> - **B200 EP1 vs EP4:** EP1 略快（6486 vs 6397, +1.4%），与 MI355X 趋势一致
> - **B200 vs MI355X（EP1 TP4 bench）:** B200 SGLang 1.32x（6486 vs 4907）
> - **MI355X EP4 vs EP1:** EP4 略慢（4754 vs 4907, -3.1%），4GPU 下 EP All-to-All 通信开销超过收益
> - **Profiling overhead:** SGLang ~3.8%, ATOM ~6.7%

## 问题背景

SA InferenceX 报告的 B200 FP4 性能大幅领先 MI355X FP4，需要 breakdown 分析差距来源。

**ATOM 不支持 DP Attention**，原始 SA 对标配置（EP=4, DP=true）无法公平对比。选择 **EP=8, DP=false, c=64** 作为公平对标基准：DP=false 消除 DP Attention 差异；EP=8 是 B200 8GPU 的自然 EP 配置；c=64 是 SA 原始测试点。

## Per-Module Kernel 级分析（10 层平均）

### 跨平台对齐算子表（35 行，按逻辑功能对齐）

> **数据来源：** B200 nsys trace 10 层平均 / MI355X Kineto trace 全层平均（v21: TP=8+EP, bs=64）
> **对齐原则：** 按逻辑功能（非执行时序）逐行对齐，同一行的 B200 和 MI355X kernel 做同一件事。一端独有的算子另一端留空。
> **GAP(B-M)：** 正值 = B200 更慢，负值 = MI355X 更慢
> **v21 更新：** MI355X 从 TP=4 EP=1 (4GPU) 改为 **TP=8+EP (8GPU)**，与 B200 TP=8 EP=8 **完全对齐**。MoE GEMM per-GPU 权重量相同，差距反映纯算子效率。

| block | ID | 逻辑算子 | B200 kernel | B200 μs | MI355X module | MI355X kernel | MI355X μs | GAP(B-M) | 备注 |
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

**v21 对齐表关键发现（vs v20 旧表）：**
- **总差距逆转：** 旧表 MI355X 慢 94.35μs（371.26 vs 276.91），新表 MI355X 快 9.33μs（267.58 vs 276.91）。**TP=8+EP 后 MI355X 单层 kernel sum 反超 B200。**
- **MoE GEMM 差距大幅缩小：** gate_up GEMM 从 2.07x → 1.12x（123.60→66.52μs vs B200 59.55μs）。旧差距 98% 来自 GPU 数差异（4 vs 8 GPU per-GPU 权重 2x），EP 对齐后仅剩纯算子效率差。down GEMM 从 1.88x → 1.04x（61.48→34.24μs vs 32.77μs），**几乎持平**。
- **MLA attention 反超：** MI355X mla_a8w8 从 24.08→18.00μs（TP=8 heads/GPU 减半），**首次快于 B200 fmhaSm100f (20.67μs)**。但 mla_reduce (8.92μs) 使总 attention 26.92μs 仍略慢于 B200 的 20.67μs。
- **o_proj_GEMM 大幅改善：** 从 13.48→8.00μs（TP=8 output 维度更小），差距从 7.35μs 缩小到 1.87μs。
- **通信+Norm 仍是短板：** input_layernorm 21.12μs + post_attn_layernorm 24.92μs = 46.04μs，vs B200 moefinalize_lamport 33.11μs + userbuffers_rmsnorm 15.15μs + allgather 9.74μs = 58.00μs。MI355X 通信反而更快（46 vs 58μs），因为 B200 EP 有额外 allgather。
- **B200 shared_expert 仍是独立开销：** Row 30-34 合计 21.40μs（B200 独有），MI355X 融入 grouped GEMM 无额外开销，这是 MI355X 的架构优势。
- **Row 1 moefinalize_lamport (33.11μs)** 仍与 qkv_a 并行，关键路径被遮盖。

### 4GPU Per-Layer 跨平台对比（B200 vs MI355X）

> **数据来源：**
> - B200: SGLang v0.5.9 torch profiler trace，TP=4 EP=4，chat 1K/1K c=64。FMHA 层锚点切分，position-based module 分配，第 10-40 层平均，4410 层样本。
>   - Kernel sum: 350.5μs/layer，Elapsed(关键路径): 283.3μs/layer，Overlap: 67.2μs (19.2%)
>   - 验证: 283.3 × 61 = 17.28ms ≈ decode walltime 17.50ms（误差 1.3%）
> - MI355X: ATOM rocm711 Kineto trace，TP=4 EP=4，chat 1K/1K c=64。decode_breakdown.xlsx 多层平均（avg 列），399.5μs/layer
>   - 单流串行执行（HIP Graph 单 stream capture），无 multi-stream overlap
>   - 验证: 399.5 × 61 = 24.37ms ≈ decode walltime 23.73ms（误差 2.7%）
> - **配置对齐：** 两平台均 4GPU TP=4 EP=4，per-GPU MoE 权重量相同（64 experts/GPU）
>
> **Multi-stream overlap 说明：**
> - **B200 SGLang** 启用了 `enable_flashinfer_allreduce_fusion=True`（lamport allreduce），该机制使用对称内存在**独立 CUDA stream** 上执行 allreduce+RMSNorm。当 lamport stream 执行 allreduce 时，计算 stream 可以同时执行后续 GEMM（如 qkv_a splitK），形成 multi-stream overlap。
> - **MI355X ATOM** 使用 RCCL fused_allreduce_rmsnorm 在**单一 HIP stream** 上串行执行所有 kernel（包括通信）。HIP Graph capture 确认所有操作在同一 stream 上，无 fork-join pattern。
> - **公平对比原则：** B200 使用 elapsed（关键路径时间，扣除 overlap），MI355X 使用 kernel sum（等于 elapsed，因为无 overlap）。两者都代表实际 GPU 时间线上的关键路径长度。

#### 算子级对比表（29行，按执行时序对齐）

> **数据来源：** B200 nsys trace (SGLang 4GPU TP=4 EP=4, c=64) / MI355X Kineto trace (ATOM 4GPU TP=4, c=64)
> **对齐原则：** 按 B200 执行时序逐行对齐，MI355X 同功能 kernel 对齐到同一行。B200 含 multi-stream overlap 信息。
> **Overlap 列：** B200 该算子与其他算子在不同 stream 上重叠的时间和对象。

| # | B200_Module | B200_Operator | B200_Raw_Kernel | B200_Stream | B200_Avg_us | B200_Overlap_us | B200_Overlap_With | MI355X_Module | MI355X_Kernel | MI355X_Avg_us | Notes |
|---|-------------|---------------|-----------------|-------------|-------------|-----------------|-------------------|---------------|---------------|---------------|-------|
| 1 | EP_AR | EP_AR+residual+RMSNorm(fused) | allreduce_fusion_kernel_oneshot_lamport | 23 | 25.9 | 24.1 | qkv_a_proj_GEMM(same):24.1μs | input_layernorm | reduce_scatter + load_rmsnorm | 29 | B200 fuses AR+residual+RMSNorm; B200 value inflated by same-stream overlap (true ~11μs) |
| 2 | Attention | qkv_a_proj_GEMM | nvjet_splitK_TNT | 23 | 31.9 | 25.5 | EP_AR+residual+RMSNorm(same):24.1μs \| qkv_a_splitK_reduce(same):1.4μs | input_layernorm | add_rmsnorm_quant ×2 | 5.5 | MI355X only: additional quant after norm |
| 3 | Attention | qkv_a_splitK_reduce | splitKreduce_kernel | 23 | 3.7 | 2.5 | qkv_a_proj_GEMM(same):1.4μs \| q/k_norm_RMSNorm(same):1.1μs | gemm_a16w16 | bf16gemm_splitk (qkv_a) | 16.1 | MI355X qkv_a GEMM maps here in strict timeline |
| 4 | Attention | q/k_norm_RMSNorm | RMSNormKernel (stream 23) | 23 | 3.2 | 2.7 | q/k_norm_RMSNorm(cross):1.5μs \| qkv_a_splitK_reduce(same):1.1μs | | | | B200 dual-stream norm #1; no MI355X match at this position |
| 5 | Attention | q/k_norm_RMSNorm | RMSNormKernel (stream 385) | 385 | 2.4 | 1.5 | q/k_norm_RMSNorm(cross):1.5μs | q/k_norm | q/k_norm_RMSNormx2 | 10.8 | MI355X combines both norms into 1 kernel |
| 6 | Attention | q_b_proj_GEMM | nvjet_v_bz_TNN | 23 | 6.3 | 1.3 | uk_gemm(K_expansion)(same):1.3μs | q_proj_and_k_up_proj | gemm_xdl (q_b) | 11.8 | |
| 7 | Attention | uk_gemm(K_expansion) | nvjet_v_bz_TNT | 23 | 4.4 | 1.3 | q_b_proj_GEMM(same):1.3μs | q_proj_and_k_up_proj | batched_gemm_a8w8 | 5.5 | |
| 8 | Attention | RoPE+KV_cache_write | RopeQuantizeKernel | 23 | 2.7 | 0 | | rope_and_kv_cache | fuse_qk_rope_concat_cache | 5.3 | |
| 9 | Attention | set_mla_kv | set_mla_kv_buffer_kernel | 23 | 1.7 | 0 | | | | | B200 only |
| 10 | Attention | Attention(FMHA) | fmhaSm100fKernel | 23 | 20.3 | 0 | | mla_decode | mla_a8w8_qh16 | 25.5 | |
| 11 | Attention | uv_gemm(V_expansion) | nvjet_h_bz_TNT | 23 | 4 | 0 | | mla_decode | mla_reduce_v1 | 7.2 | MI355X attention output reduction |
| 12 | Proj | o_proj_quant(BF16→FP4) #1 | cvt_fp16_to_fp4 | 23 | 2.1 | 0.1 | | mla_decode:v_up_proj | batched_gemm_a8w8 (uv) | 6.6 | MI355X uv_gemm maps here in strict timeline |
| 13 | Proj | o_proj_GEMM #1 | DeviceGemmFp4GemmSm100 | 23 | 9.4 | 0.1 | | | | | B200 splits o_proj into #1+#2 (9.4+10.2=19.6) |
| 14 | EP_AR | EP_AR+residual+RMSNorm(fused) #2 | allreduce_fusion_kernel_oneshot_lamport | 23 | 11 | 9.2 | router_GEMM(cross):9.1μs | v_up_proj_and_o_proj | gemm_xdl (o_proj) | 21.4 | MI355X single o_proj; B200 fuses AR+residual+RMSNorm |
| 15 | MoE_Route | router_GEMM | nvjet_h_bz_splitK_TNT | 385 | 13.5 | 14.3 | EP_AR+residual+RMSNorm(cross):9.1μs \| o_proj_quant(cross):2.5μs \| router_splitK_reduce(same):1.4μs \| o_proj_GEMM(cross):1.3μs | post_attn_layernorm | reduce_scatter + triton_fused_clone ×2 | 31.2 | MI355X post_attn combined (20.0+5.7+5.5) |
| 16 | MoE_quant | Moe_Expert_quant(BF16→FP4) | cvt_fp16_to_fp4 | 23 | 2.5 | 2.6 | router_GEMM(cross):2.5μs \| router_splitK_reduce(cross):0.2μs | gemm_a16w16 | bf16gemm_splitk (router) | 9.2 | MI355X router GEMM maps here in strict timeline |
| 17 | MoE_Route | router_splitK_reduce | splitKreduce_kernel | 385 | 3.6 | 6.3 | MoE_Quant_GEMM(cross):3.4μs \| router_GEMM(same):1.4μs \| MoE_input_quant(same):1.4μs \| o_proj_quant(cross):0.2μs | | | | B200 only: splitK reduction |
| 18 | Proj | o_proj_GEMM #2 | DeviceGemmFp4GemmSm100 | 23 | 10.2 | 12.2 | MoE_input_quant(cross):3.8μs \| router_splitK_reduce(cross):3.4μs \| tensor_copy(cross):3.0μs \| router_GEMM(cross):1.3μs \| TopK_select(cross):1.1μs | | | | (included in B200 o_proj #1+#2 = 19.6) |
| 19 | MoE_Route | MoE_input_quant(BF16→FP4) | quantize_with_block_size | 385 | 3.8 | 5.1 | MoE_Quant_GEMM(cross):3.8μs \| router_splitK_reduce(same):1.4μs | | | | B200 only: FP4 quant for MoE input |
| 20 | MoE_Route | tensor_copy | unrolled_elementwise_kernel | 385 | 3.2 | 3.1 | MoE_Quant_GEMM(cross):3.0μs | | | | B200 tensor_copy; no MI355X match at this position |
| 21 | MoE_Route | TopK_select | routingMainKernel | 385 | 4.5 | 6.3 | SiLU×Mul(cross):2.6μs \| expert_sort(same):2.5μs \| o_proj_GEMM(cross):1.1μs \| shared_quant(cross):1.1μs | triton_poi | triton_fused_clone | 5.5 | MI355X tensor_copy maps here in strict timeline |
| 22 | MoE_Route | expert_sort | routingIndicesClusterKernel | 385 | 5.3 | 7.2 | TopK_select(same):2.5μs \| SiLU×Mul(cross):2.0μs \| shared_quant(cross):1.5μs \| shared_GEMM(cross):1.4μs \| gate_up(same):0.2μs | mxfp4_moe | grouped_topk_opt_sort | 5.4 | MI355X TopK maps here in strict timeline |
| 23 | Shared_Exp | SiLU×Mul | act_and_mul_kernel | 23 | 3.2 | 4.7 | TopK_select(cross):2.6μs \| expert_sort(cross):2.0μs | mxfp4_moe | MoeSorting_P0+P23 | 10.8 | MI355X expert_sort maps here in strict timeline; B200 shared expert SiLU parallel with MoE(cross) |
| 24 | Shared_Exp | shared_quant(BF16→FP4) | cvt_fp16_to_fp4 | 23 | 1.7 | 2.2 | expert_sort(cross):1.5μs \| TopK_select(cross):1.1μs | mxfp4_moe | fused_mxfp4_quant_sort | 9.5 | MI355X MoE quant+sort before gate_up |
| 25 | MoE_Expert | gate_up_GEMM(+SwiGLU) | bmm_E2m1_E2m1E2m1 | 385 | 101.7 | 4.3 | shared_GEMM(cross):3.4μs \| down_GEMM(same):0.6μs \| expert_sort(same):0.2μs | mxfp4_moe | moe_mxgemm(gate_up) | 119.5 | |
| 26 | Shared_Exp | shared_GEMM(FP4) | DeviceGemmFp4GemmSm100 | 23 | 5.8 | 6 | gate_up_GEMM(cross):3.4μs \| expert_sort(cross):1.4μs | mxfp4_moe | fused_mxfp4_quant_sort | 5.8 | MI355X quant between gate_up and down; B200 shared expert GEMM parallel with gate_up(cross) |
| 27 | MoE_Expert | down_GEMM | bmm_Bfloat16_E2m1E2m1 | 385 | 55.8 | 2.5 | MoE_finalize(same):0.6μs \| gate_up_GEMM(same):0.6μs | mxfp4_moe | moe_mxgemm(down) | 58 | |
| 28 | MoE_Expert | MoE_finalize+residual | finalizeKernelVecLoad | 385 | 7.5 | 0.6 | down_GEMM(same):0.6μs | | | | B200 only |
| 29 | Residual | residual_add | vectorized_elementwise_kernel | 23 | 2.1 | 0 | | | | | B200 only |
| | | | | | **B200 TOTAL (kernel_sum): 353.2** | | | | | | |
| | | | | | **B200 Walltime: 286.9** | | | | | | |
| | | | | | **B200 Overlap: 66.3** | | | | **MI355X TOTAL** | **399.6** | |

#### PASS 功能分组汇总

| PASS | B200 μs | MI355X μs | GAP | B200 Kernels | MI355X Kernels |
|------|---------|-----------|-----|--------------|----------------|
| MOE | 214.2 | 223.7 | -9.5 | nvjet_h_bz_splitK_TNT, cvt_fp16_to_fp4, splitKreduce_kernel, quantize_with_block_size, unrolled_elementwise_kernel, routingMainKernel, routingIndicesClusterKernel, act_and_mul_kernel, cvt_fp16_to_fp4, bmm_E2m1_E2m1E2m1, DeviceGemmFp4GemmSm100, bmm_Bfloat16_E2m1E2m1, finalizeKernelVecLoad, vectorized_elementwise_kernel | triton_fused_clone, grouped_topk_opt_sort, MoeSorting_P0+P23, fused_mxfp4_quant_sort, moe_mxgemm(gate_up), fused_mxfp4_quant_sort, moe_mxgemm(down) |
| MHA | 80.6 | 88.8 | -8.2 | nvjet_splitK_TNT, splitKreduce_kernel, RMSNormKernel (stream 23), RMSNormKernel (stream 385), nvjet_v_bz_TNN, nvjet_v_bz_TNT, RopeQuantizeKernel, set_mla_kv_buffer_kernel, fmhaSm100fKernel, nvjet_h_bz_TNT | bf16gemm_splitk (qkv_a), q/k_norm_RMSNormx2, gemm_xdl (q_b), batched_gemm_a8w8, fuse_qk_rope_concat_cache, mla_a8w8_qh16, mla_reduce_v1 |
| O proj | 21.7 | 21.4 | 0.3 | cvt_fp16_to_fp4, DeviceGemmFp4GemmSm100, DeviceGemmFp4GemmSm100 | gemm_xdl (o_proj) |
| EP_AR+residual+RMSNorm(fused) before MHA | 25.9 | 34.5 | -8.6 | allreduce_fusion_kernel_oneshot_lamport | reduce_scatter + load_rmsnorm, add_rmsnorm_quant ×2 |
| EP_AR+residual+RMSNorm(fused) before MOE | 11 | 31.2 | -20.2 | allreduce_fusion_kernel_oneshot_lamport | reduce_scatter + triton_fused_clone ×2 |
| sum | 353.4 | 399.6 | -46.2 | | |

#### B200 Multi-Stream Overlap 分析

**Top Overlap 对（per-layer avg）：**

| Stream A (通信) | Stream B (计算) | Overlap(μs) | 位置 |
|----------------|----------------|-------------|------|
| LAMPORT AR+RMSNorm | qkv_a_splitK_TNT | 9.8 | comm_norm ‖ qkv_proj |
| FP4_GEMM | FP4_block_quant | 3.5 | out_proj ‖ moe_quant |
| FP4_GEMM | tensor_copy | 3.2 | 跨模块 |
| FP4_GEMM | MoE_gate_up | 2.7 | shared_exp ‖ moe_expert |
| splitK_TNT | FP4_convert | 2.5 | qkv_proj ‖ out_proj |
| RMSNorm | RMSNorm | 1.5 | q_norm ‖ k_norm |

> **Overlap 机制解释：**
> - B200 SGLang 使用 `enable_flashinfer_allreduce_fusion=True`，lamport allreduce + RMSNorm 在**通信 stream** 上执行。当通信 stream 正在做 allreduce 时，计算 stream 可以提前开始下一步的 GEMM。
> - 最大 overlap（9.8μs/layer）发生在 lamport AR 与 qkv_a splitK GEMM 之间——这是**层间流水线效果**：上一层的 allreduce 还没完成，当前层的 GEMM 已经开始。
> - 此外 FP4_GEMM（out_proj/shared）与 FP4 量化/MoE GEMM 之间也有 stream overlap（各 2-3μs）。
> - MI355X ATOM 使用 RCCL `fused_allreduce_rmsnorm` 在**单一 HIP stream** 上串行执行，HIP Graph capture 确认无 multi-stream fork-join pattern，因此无 overlap。


#### TP=4 单层分段执行分析

> 基于 29 行时序表原始 Torch Trace 数据，将单层 Transformer 按执行阶段分为 7 段，逐段对比两平台关键路径。
> 数据为统计平均值，可能与精确值存在细微偏差。

##### 总览

| 段 | 功能 | B200 μs | MI355X μs | GAP μs | GAP 主因 |
|----|------|---------|-----------|--------|---------|
| 1 | AR+RMSNorm+qkv_a | ~32 | ~45 | 13 | PDL overlap 隐藏 AR |
| 2 | q/k Norm+q_b+k_up | ~14 | ~22.8 | 8.8 | 双 stream norm 并行 + q_b GEMM |
| 3 | RoPE+KV cache | 4.4 | 5.3 | 0.9 | — |
| 4 | MLA Decode+uv | ~24 | ~39.3 | 15.3 | mla_reduce 独立开销 |
| 5 | O_proj | ~11.5(+10.2‖AR) | 21.4 | ~10 | GEMM#2 与下段 AR 重叠 |
| 6 | MoE routing+shared | ~34 | ~71 | 37 | 双 stream 并行 vs 串行 |
| 7 | MoE GEMM+finalize | ~167 | ~183 | 16 | GEMM 效率 + 中间 quant |
| **合计** | | **~283** | **~388** | **~105** | |

##### 段 1：AR + RMSNorm + qkv_a 投影（GAP 13μs）

B200 将 AllReduce+residual+RMSNorm 融合为单个 lamport kernel，利用 PDL 在同一 Stream 上提前发射 qkv_a_proj_GEMM，两者重叠执行。含后续 splitKreduce 共 ~32μs。MI355X 两个算子串行执行，合计 ~45μs。

| B200 算子 | μs | MI355X 算子 | μs |
|-----------|------|-------------|------|
| AR+residual+RMSNorm(fused) ‖ qkv_a(PDL) | 25.9 | reduce_scatter+load_rmsnorm | 29 |
| qkv_a_proj_GEMM | 31.9(overlap) | bf16gemm_splitk (qkv_a) | 16.1 |
| splitKreduce | 3.7 | — | — |
| **关键路径** | **~32** | **串行合计** | **~45** |

##### 段 2：q/k Norm + q_b + k_up 投影（GAP 8.8μs）

B200 串行执行 q/k_norm + q_b_proj + uk_gemm，另一 Stream 并行执行第二个 RMSNorm，整体 ~14μs。MI355X 四个算子串行 ~22.8μs。

| B200 算子 | μs | Stream | MI355X 算子 | μs |
|-----------|------|--------|-------------|------|
| q/k_norm_RMSNorm | 3.2 | 23 | add_rmsnorm_quant ×2 | 5.5 |
| q/k_norm_RMSNorm(并行) | 2.4 | 385 | — | — |
| q_b_proj_GEMM | 6.3 | 23 | gemm_xdl (q_b) | 11.8 |
| uk_gemm(K_expansion) | 4.4 | 23 | batched_gemm_a8w8 | 5.5 |
| **关键路径** | **~14** | | **串行合计** | **~22.8** |

##### 段 3：RoPE + KV cache（GAP 0.9μs）

B200 通过 RoPE+KV_cache_write 与 set_mla_kv 串行实现 4.4μs；MI355X 单个融合算子 5.3μs。

| B200 算子 | μs | MI355X 算子 | μs |
|-----------|------|-------------|------|
| RoPE+KV_cache_write | 2.7 | fuse_qk_rope_concat_cache | 5.3 |
| set_mla_kv | 1.7 | — | — |
| **合计** | **4.4** | | **5.3** |

##### 段 4：MLA Decode + uv_gemm（GAP 15.3μs）

B200 串行执行 FMHA + uv_gemm ~24μs。MI355X 需额外 mla_reduce 步骤，三算子串行 ~39.3μs。

| B200 算子 | μs | MI355X 算子 | μs |
|-----------|------|-------------|------|
| fmhaSm100f (FMHA) | 20.3 | mla_a8w8_qh16 | 25.5 |
| nvjet_h_bz_TNT (uv_gemm) | 4.0 | mla_reduce_v1 | 7.2 |
| — | — | batched_gemm_a8w8 (uv) | 6.6 |
| **合计** | **~24** | **合计** | **~39.3** |

##### 段 5：输出投影（GAP ~10μs）

B200 执行 o_proj_quant + o_proj_GEMM#1 合计 11.5μs，GEMM#2 (10.2μs) 执行期间同步启动另一 Stream 的融合 Reduce-RMSNorm，成功隐藏约 10μs 延迟。MI355X 单个 GEMM 完成 21.4μs。表面耗时相近，但 B200 利用 GEMM#2 与下段 AR 的重叠，有效缩短了关键路径。

| B200 算子 | μs | MI355X 算子 | μs |
|-----------|------|-------------|------|
| o_proj_quant(BF16→FP4) | 2.1 | — | — |
| o_proj_GEMM #1 | 9.4 | gemm_xdl (o_proj) | 21.4 |
| o_proj_GEMM #2（与段 6 AR 重叠） | 10.2 | — | — |
| **kernel sum** | **21.7** | **合计** | **21.4** |

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

**MI355X（串行, ~71μs）：**

| # | 算子 | μs |
|---|------|------|
| 1 | reduce_scatter + triton_fused_clone ×2 | 31.2 |
| 2 | bf16gemm_splitk (router) | 9.2 |
| 3 | triton_fused_clone | 5.5 |
| 4 | grouped_topk_opt_sort | 5.4 |
| 5 | MoeSorting_P0+P23 | 10.8 |
| 6 | fused_mxfp4_quant_sort | 9.5 |
| | **合计** | **~71.6** |

##### 段 7：MoE GEMM + Finalize（GAP 16μs）

B200 单 Stream 串行 ~167μs。MI355X 三步串行 ~183μs。注意 MI355X 在 gate_up 与 down 之间有独立的 `fused_mxfp4_quant_sort` (5.8μs)，而 B200 无此步骤——cuBLAS bmm 在 gate_up epilogue 中直接完成 SwiGLU + FP4 quantize。

| B200 算子 | μs | MI355X 算子 | μs |
|-----------|-------|-------------|-------|
| gate_up_GEMM(+SwiGLU) | 101.7 | moe_mxgemm(gate_up) | 119.5 |
| — | — | fused_mxfp4_quant_sort | 5.8 |
| down_GEMM | 55.8 | moe_mxgemm(down) | 58.0 |
| MoE_finalize+residual | 7.5 | — | — |
| residual_add | 2.1 | — | — |
| **合计** | **~167** | **合计** | **~183** |

##### MI355X 优化方向（不含双 stream / PDL）

> MI355X 无法使用 B200 的 PDL（Programmatic Dependent Launch）和双 stream 并行。以下仅聚焦单 stream 架构下的算子效率优化空间。

**方向 1：gate_up GEMM 效率（段 7，预估 6-12μs）**

119.5 vs 101.7μs = 1.17x。bs=64 top-8 routing 下每 expert 平均 M≈2，极度 bandwidth-bound，两平台 HBM BW 均 8 TB/s。MI355X 慢 17% 指向 CK `moe_mxgemm_2lds` 的 achieved BW 偏低（wavefront scheduling / LDS bank conflict）。另需排查 shared expert（M=64）融入 grouped GEMM 后与普通 expert（M≈2）的负载不均。

**方向 2：MLA + reduce 融合（段 4，预估 0-7μs）**

mla_a8w8_qh16 (25.5μs) → HBM 写回 → mla_reduce_v1 (7.2μs)。B200 fmhaSm100f 在单 kernel 内完成 attention + head reduce。融合可省 HBM 往返 + kernel launch (~5-7μs)，但 CDNA4 的 register/LDS 容量是否足够在 attention 后保留中间结果做 reduce 需验证——register 饱和会降低 occupancy，可能反效果。

**方向 3：q_b GEMM tile 调优（段 2，预估 3-5μs）**

11.8 vs 6.3μs = 1.87x。MI355X FP8 (gemm_xdl_preshuffle) 理论 TFLOPS 是 B200 BF16 (nvjet_v_bz_TNN) 的 2 倍，却慢近 2 倍——CK 对 shape `[64×1536]×[1536×768]` 的 tile config 明显不优。低挂果实，调整 tile size / split-K 即可。

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

> **小结：** 当前 MI355X ~388μs，上述优化可覆盖 15-32μs（→ 356-373μs）。与 B200 关键路径 ~283μs 的 ~105μs 总差距中，~50-60μs 来自双 stream / PDL 并行优势（段 1/5/6），属架构级差异，软件层面无法弥补。

#### 功能分组对比（单层，绝对 GPU 时间）

> v21: 两平台均为 TP=8 EP=8（MI355X 用 `--enable-expert-parallel`），GPU 数量和并行策略完全对齐。

| 功能块 | B200 μs | MI355X v21 μs | MI355X v20 μs | B200/MI355X (v21) | 差距来源 |
|--------|---------|--------------|--------------|-------------------|---------|
| **MLA Attention** | 97.6 | **82.6** | 96.4 | **1.18x** | **MI355X 反超;TP=8 heads/GPU 减半** |
| └ qkv_a 投影 | 42.6 | 11.5 | 11.6 | 3.70x | B200 BF16 splitK 仍慢于 MI355X FP8 CK（不变） |
| └ q_b + k_up 投影 | 9.5 | 11.6 | 12.5 | 0.82x | v21: TP=8 维度更小，略降 |
| └ attention (fmha/mla) | 20.7 | 26.9 | 33.2 | 0.77x | v21: TP=8 heads 16→8，mla 18.0+reduce 8.9 |
| └ v_proj + out_proj | 12.3 | 18.2 | 24.2 | 0.68x | v21: o_proj GEMM 13.5→8.0μs |
| └ 其他 (norm/rope/concat) | 12.5 | 14.4 | 14.9 | 0.87x | |
| **MoE (含 router/shared)** | 161.8 | **138.1** | 197.3 | **1.17x** | **MI355X 反超;EP 对齐后纯效率差 <12%** |
| └ router + topk + sort | 24.4 | 27.8 | — | 0.88x | MI355X 含 3-phase sort |
| └ gate_up GEMM | 59.6 | 66.5 | 123.6 | 0.90x | **v21: 32exp/GPU 相同;MI355X 仅 1.12x** |
| └ down GEMM | 32.8 | 34.2 | 61.5 | 0.96x | **v21: 几乎持平（1.04x）** |
| └ quant+sort融合 | 0 | 13.4 | 14.6 | — | MI355X 独有:2×fused_mxfp4_quant_moe_sort |
| └ shared expert | 21.4 | 0* | 9.3 | — | B200 独有 5 个 kernel;MI355X 融入 grouped GEMM |
| └ moe_finalize/通信 | 33.1 | 0** | — | — | B200 EP AR+residual+norm;v21 MI355X 可能融入其他 |
| **通信 + Norm** | 24.9 | **46.0** | 50.3 | **0.54x** | B200 NVLink userbuffers 仍有大优势 |
| └ pre-attn norm | 0 (融入Row1) | 21.1 | 27.4 | — | MI355X xGMI reduce_scatter |
| └ post-attn norm | 15.2 | 24.9 | 22.9 | 0.61x | B200 NVLink userbuffers 融合通信 |
| └ residual allgather | 9.7 | 0 | — | — | B200 独有（MI355X EP 的 AG 可能融入其他） |
| **总计（单层 kernel sum）** | **276.9** | **267.6** | **344.0** | **1.03x** | **MI355X kernel sum 反超 B200 3.4%** |

> \* MI355X shared_expert 融入 MoeMxGemm grouped GEMM（gate_up/down 的时间已包含 shared expert）。
> \** MI355X 的 EP AllReduce / moe_finalize 可能融入其他 module（如 layernorm），或使用不同的通信方式（All-to-All dispatch/combine 已融入 MoeMxGemm）。

**v21 对齐表关键发现（vs v20 旧表）：**

- **总差距逆转：** 旧表 MI355X 慢 94.35μs（371.26 vs 276.91），新表 MI355X 快 9.33μs（267.58 vs 276.91）。**TP=8+EP 后 MI355X 单层 kernel sum 反超 B200。**
- **MoE GEMM 差距大幅缩小：** gate_up GEMM 从 2.07x → 1.12x（123.60→66.52μs vs B200 59.55μs）。旧差距 98% 来自 GPU 数差异（4 vs 8 GPU per-GPU 权重 2x），EP 对齐后仅剩纯算子效率差。down GEMM 从 1.88x → 1.04x（61.48→34.24μs vs 32.77μs），**几乎持平**。
- **MLA attention 反超：** MI355X mla_a8w8 从 24.08→18.00μs（TP=8 heads/GPU 减半），**首次快于 B200 fmhaSm100f (20.67μs)**。但 mla_reduce (8.92μs) 使总 attention 26.92μs 仍略慢于 B200 的 20.67μs。
- **o_proj_GEMM 大幅改善：** 从 13.48→8.00μs（TP=8 output 维度更小），差距从 7.35μs 缩小到 1.87μs。
- **通信+Norm 仍是短板：** input_layernorm 21.12μs + post_attn_layernorm 24.92μs = 46.04μs，vs B200 moefinalize_lamport 33.11μs + userbuffers_rmsnorm 15.15μs + allgather 9.74μs = 58.00μs。MI355X 通信反而更快（46 vs 58μs），因为 B200 EP 有额外 allgather。
- **B200 shared_expert 仍是独立开销：** Row 30-34 合计 21.40μs（B200 独有），MI355X 融入 grouped GEMM 无额外开销，这是 MI355X 的架构优势。
- **Row 1 moefinalize_lamport (33.11μs)** 仍与 qkv_a 并行，关键路径被遮盖。

**v21 关键发现：**

1. **MI355X kernel sum 反超 B200**（267.6 vs 276.9μs，快 3.4%），但 decode walltime 仍慢（17.22 vs 15.6ms，1.10x）。差距主要来自 L1→L2 overhead（MI355X 5.2% vs B200 1.8%）。
2. **MoE GEMM EP 对齐后差距极小：** gate_up 1.12x，down 1.04x。v20 的 2.07x/1.88x 差距 **98% 来自 GPU 数差异**，现已消除。两平台 MoE GEMM 效率几乎相同。
3. **MLA Attention MI355X 反超 15μs**（82.6 vs 97.6μs）。B200 qkv_a BF16 (42.6μs) 是最大瓶颈，MI355X FP8 CK 仅 11.5μs。即使 MI355X mla_reduce (8.9μs) 是独有开销，attention 总时间仍更短。
4. **通信 + Norm 仍是 MI355X 短板**（46.0 vs 24.9μs，1.85x）。B200 NVLink userbuffers 的融合 AR+norm 效率显著优于 MI355X xGMI reduce_scatter 两步式通信。
5. **B200 decode walltime 快 10% 的原因：** 虽然 kernel sum MI355X 更短，但 B200 的 moefinalize_lamport (33.11μs) 与 qkv_a (25.12μs) 并行执行，关键路径只看 max(33.1, 25.1)=33.1μs 而非 sum。B200 关键路径 251.1μs × 61 = 15.3ms ≈ 实测 15.6ms。MI355X 无此并行优势。
6. **Per-GPU throughput B200 仍领先 21%**（490.1 vs 405.8 tok/s/GPU），因为 B200 decode walltime 更短 + NVLink 通信优势。



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

### MI355X Kernel 级 Breakdown（bs=64, --mark-trace, 全层平均）

> **数据来源：** MI355X Kineto trace，MXFP4, **TP=8+EP (v21)**, c=64, chat 1K/1K
> **工具：** `run_parse_trace.py --target-bs 64` → `decode_breakdown.xlsx`
> **统计口径：** 全层平均值（avg sum per module），bs=64 稳态 decode 事件

| Module | v21 TP8+EP μs | v20 TP4 EP1 μs | 变化 | % | Kernel(s) | 精度 |
|--------|--------------|----------------|------|---|-----------|------|
| **mxfp4_moe** | **129.0** | 188.0 | **-31%** | **48.2%** | topk_sort + MoeSorting ×2 + mxfp4_quant ×2 + MoeMxGemm ×2 | MXFP4 |
| mla_decode | 26.9 | 33.2 | -19% | 10.1% | mla_a8w8 + mla_reduce_v1 | FP8 |
| post_attn_layernorm | 24.9 | 22.9 | +9% | 9.3% | reduce_scatter + rmsnorm | BF16 |
| input_layernorm | 21.1 | 27.4 | -23% | 7.9% | reduce_scatter + rmsnorm | BF16 |
| v_up_proj_and_o_proj | 18.2 | 24.2 | -25% | 6.8% | batched_gemm_a8w8 + per_token_quant + gemm_preshuffle | FP8 |
| q_proj_and_k_up_proj | 11.6 | 12.5 | -7% | 4.3% | gemm_preshuffle (q_b) + batched_gemm_a8w8 (k_up) | FP8 |
| gemm_a8w8_bpreshuffle | 11.5 | 11.6 | -1% | 4.3% | gemm_xdl_cshuffle_v3 (qkv_a) | FP8 |
| gemm_a16w16 | 9.1 | 9.3 | -2% | 3.4% | bf16gemm_splitk (router) | BF16 |
| per_token_quant_hip | 5.5 | 5.0 | +10% | 2.0% | dynamic_per_token_scaled_quant | BF16→FP8 |
| rope_and_kv_cache | 5.1 | 5.1 | 0% | 1.9% | fuse_qk_rope_concat_and_cache_mla | BF16 |
| _fused_rms_fp8_group_quant | 4.6 | 4.8 | -4% | 1.7% | fused RMSNorm + FP8 group quantize | BF16→FP8 |
| **TOTAL** | **267.6** | **344.0** | **-22%** | **100%** | | |

> **验证：** 267.6 × 61 = 16.32ms ≈ 实测 decode walltime 17.22ms（偏差 5.2%），kernel breakdown 覆盖了主要 GPU 执行时间。
>
> **TP=8+EP vs TP=4 EP=1 变化分析：**
> - **MoE GEMM -31% (188→129μs)：** EP 后每 GPU 仅 32 experts（全宽度），vs TP=4 时 256 experts（1/4 宽度）。gate_up 从 123.6→66.5μs，down 从 61.5→34.2μs。
> - **MLA attention -19% (33.2→26.9μs)：** TP=8 每 GPU 16 heads（vs TP=4 的 32 heads），mla_a8w8 从 24.1→18.0μs。
> - **v_up/o_proj -25% (24.2→18.2μs)：** TP=8 维度更小，o_proj GEMM 从 13.5→8.0μs。
> - **post_attn_layernorm +9% (22.9→24.9μs)：** 8-way xGMI 通信略慢于 4-way（更多 hops）。
> - **整体 -22%（344→267.6μs）：** EP 的 MoE 分片优势远大于 8-way 通信的额外开销。

### 跨平台 Per-GPU 时间栈总结

| 层级 | B200 (8×GPU, TP8 EP8) | MI355X TP8+EP (v21) | MI355X TP4 EP1 (旧) | B200/MI355X (v21) |
|------|----------------------|---------------------|---------------------|-------------------|
| **L1: Kernel/层** | 251.1 μs (关键路径) / 276.9 (sum) | **267.6 μs** | 344.0 μs | 0.94x (sum) |
| **L2: Decode step** | 15.6 ms | **17.22 ms** | 21.6 ms | 0.91x |
| **L3: Client TPOT** | 17.8 ms | **18.9 ms** | 24.9 ms | 0.94x |
| **L1→L2 overhead** | 1.8% | **5.2%** | 2.7% | — |
| **L2→L3 overhead** | +2.2 ms (prefill) | **+1.7 ms** | +3.3 ms | — |
| **Output TPS (total)** | 3921 | **3247** | ~2500 | 1.21x |
| **Output TPS/GPU** | 490.1 | **405.8** | 624.9 | 1.21x |

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

## 迭代日志

| 日期 | 变更 |
|------|------|
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
