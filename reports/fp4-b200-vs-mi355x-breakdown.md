# FP4 性能差距分析：B200 vs MI355X — Breakdown 调查

> **Last updated:** 2026-03-27
> **Model:** DeepSeek-R1-0528, FP4
> **状态：** 配置 A（EP4 DP）trace 完成 ✅；配置 B（EP8 no-DP）trace 完成 ✅；跨配置对比完成 ✅；Per-Module Kernel 对比完成 ✅

## 问题背景

SA InferenceX 报告的 B200 FP4 性能大幅领先 MI355X FP4，需要 breakdown 分析差距来源。

**ATOM 不支持 DP Attention**，原始 SA 对标配置（EP=4, DP=true）无法公平对比。因此增加 **配置 B（EP=8, DP=false, c=64）** 作为更公平的对标基准。

## 两组对比配置

### 配置 A：原始 SA 对标（EP=4, DP=true, c=256）

| 项目 | B200 (SA InferenceX) | MI355X (ATOM) | 差异 |
|------|---------------------|---------------|------|
| **Image** | rc6.post2 | ATOM 0.1.1 | TRT-LLM vs ATOM |
| **GPU / TP / EP** | 4 / 4 / 4 | 4 / 4 / 1 | **EP 不同** |
| **DP Attention** | True | False | **B200 有 DP，355X 不支持** |
| **Concurrency** | 256 | 256 | 相同 |

| Metric | B200 | MI355X | 差距 | 355X/B200 |
|--------|------|--------|------|-----------|
| **Output TPS /GPU** | 1,954.9 | 1,154.7 | **-40.9%** | 59.1% |
| **Interactivity** | 32.16 | 18.43 | **-42.7%** | 57.3% |

> **不公平因素：** B200 EP=4 + DP Attention 使 attention 负载降为 1/4，且 MoE 通信模式不同。41% 差距中包含 DP Attention 和 EP 策略的额外优势。

### 配置 B：公平对标（EP=8, DP=false, c=64）✅

| 项目 | B200 (SA InferenceX) | B200 (复现) | MI355X (ATOM) | 差异 |
|------|---------------------|------------|---------------|------|
| **Image** | rc6.post2 | rc6.post2 | ATOM 0.1.1 | TRT-LLM vs ATOM |
| **GPU / TP / EP** | 8 / 8 / 8 | 8 / 8 / 8 | 4 / 4 / 1 | GPU 数不同，需按 /GPU 比 |
| **DP Attention** | **False** | **False** | False | **相同** |
| **Concurrency** | 64 | 64 | 待定 | |

| Metric | B200 (SA) | B200 (复现) | 复现偏差 | MI355X | 差距 |
|--------|-----------|------------|---------|--------|------|
| **Output TPS /GPU** | 473.4 | 490.1 | +3.5% | 待测 | 待填 |
| **Interactivity** | 60.93 | 63.12 | +3.6% | 待测 | 待填 |

> **复现验证：** 同 rc6.post2 docker，复现结果 +3.5% vs SA，在正常波动范围内。
> **为什么选这个配置：** DP=false 消除 DP Attention 差异；EP=8 是 B200 8GPU 的自然 EP 配置；c=64 是 SA 原始测试点。

### 与 FP8 对比的参考（chat 1K/1K c=256，来自已有数据）

| Config | B200 Output TPS (total) | 355X Output TPS (total) | 差距 |
|--------|------------------------|------------------------|------|
| **FP8 TP=8 EP=1 MTP0** | 7,706.9 (8GPU) | 5,970.6 (8GPU) | -22.5% |
| **FP4 TP=4 EP=4/1 DP** | 7,819.5 (4GPU×1954.9) | 4,618.9 (4GPU×1154.7) | **-40.9%** |

> FP4 配置 A 差距（41%）远大于 FP8（22.5%），其中 DP Attention 差异是重要因素。配置 B 消除此因素后差距应缩小。

## Breakdown 计划

### 配置 A：EP=4, DP=true, c=256

#### 第一步：复现 B200 数据 ✅

**结果目录：** `results_b200_mtp0_fp4_ep4_conc256_post2/`

#### 第二步：B200 Trace 分析 ✅

**Trace 文件：** `nsys_fp4_throughput_chat_tp4_ep4_c256_iter100-150_dp.nsys-rep` (37 MB)
**配置：** FP4, TP=4, EP=4, DP Attention, chat 1K/1K, c=256, iter 100-150

#### Kernel 级 GPU 时间分布

| 类别 | 占比 | 时间 | 实例数 | 关键 Kernel |
|------|------|------|--------|------------|
| **MoE Expert GEMM** | **48.7%** | 2.840 s | 23,664 | CUTLASS SM100 MXF4 PtrArray grouped GEMM (两个变体: 31.6% + 17.1%) |
| **Dense GEMM (non-MoE)** | **~20%** | ~1.17 s | — | cutlass block_scaled (~3.8%) + nvjet SM100 kernels (~15%) + cublasLt splitK (~1.5%) |
| **MoE All-to-All 通信** | **9.5%** | 556 ms | — | moeCombine (5.6%, 323ms) + moeDispatch (2.9%, 168ms) + prepare/sanitize (~1%) |
| **MoE 支持 Kernel** | **10.2%** | 595 ms | — | expandInputRows (2.8%) + doActivation/SiLU (2.5%) + quantize (2.8%) + buildExpertMaps (0.8%) + topk (0.8%) + computeStrides (0.5%) |
| **Attention (FMHA)** | **3.8%** | 224 ms | — | fmhaSm100fKernel (FP4 quantized KV cache) |
| **Norm** | **2.6%** | 152 ms | — | RMSNorm (1.5%) + FusedAddRMSNorm (1.1%) |
| **MLA RoPE** | **1.4%** | 82 ms | — | applyMlaRope kernel |
| **Other** | **~4%** | ~230 ms | — | memcpy, memset, misc |

#### 分类汇总

| 大类 | 占比 | 说明 |
|------|------|------|
| **MoE 相关 (GEMM + Comm + Support)** | **~68.4%** | MoE 是绝对瓶颈 |
| **Dense GEMM (non-MoE linear)** | **~20%** | 包括 input/output projection, shared expert dense |
| **Attention** | **~3.8%** | DP Attention + FP4 KV 使得 attention 非常轻量 |
| **Norm + RoPE + Other** | **~8%** | 辅助开销 |

#### 关键发现

1. **MoE Expert GEMM 独占近一半 GPU 时间（48.7%）。** 两个 CUTLASS SM100 MXF4 PtrArray grouped GEMM 变体——大的 155μs/次（gate+up projection），小的 84μs/次（down projection）——构成 FP4 性能的核心。MI355X 的 FP4 MoE GEMM 效率直接决定差距大小。

2. **MoE All-to-All 通信占 9.5%。** EP=4 配置下每步 decode 需要 dispatch（发送 token 到对应 expert 所在 GPU）和 combine（收集结果），总计 ~556ms。**MI355X EP=1 不需要这部分通信**——但 EP=1 意味着每个 GPU 要计算所有 expert，compute 负载更大。EP=4 的 All-to-All 9.5% 开销 vs EP=1 的额外 compute 开销——哪个更大需要进一步量化。

3. **Attention 仅占 3.8%——DP Attention 的威力。** EP=4 + DP Attention 使得每个 GPU 只处理 1/4 的 attention 负载。MI355X EP=1 无 DP Attention，attention 是全量计算。但即使 355X attention 是 4 倍，3.8%×4 = 15.2% 仍不是主要瓶颈。

4. **CUTLASS SM100 MXF4 是 B200 的硬件优势。** FP4 block-scaled GEMM 使用 B200 SM100 架构的原生 MXF4 tensor core 指令。MI355X 的 FP4 GEMM 实现（CDNA4 矩阵核心）的效率对比是关键。

5. **Dense GEMM ~20% 不容忽视。** 包括 MoE 之外的 linear projection（attention QKV proj, output proj 等），使用 nvjet 和 cutlass block_scaled kernel。两平台的 dense GEMM 效率差异也贡献性能差距。

#### MoE GEMM 性能估算

FP4 MoE Expert GEMM 的两个变体性能参考：

| 变体 | 总时间 | 实例数 | 平均耗时 | 推测功能 |
|------|--------|--------|---------|---------|
| 大 (31.6%) | 1.842 s | 11,832 | 155.7 μs | gate_proj + up_proj (fused) |
| 小 (17.1%) | 998 ms | 11,832 | 84.4 μs | down_proj |

> 实例数相同（11,832 = ~197 steps × 60 experts/step），大变体时间约为小变体的 1.84 倍，与 DeepSeek MoE 结构中 gate+up 维度是 down 维度 2 倍一致。

### 配置 B：EP=8, DP=false, c=64（公平对标）

#### 第三步：复现 B200 DP=false 数据 ✅

**结果目录：** `results_b200_mtp0_fp4_ep8_c64_dp0/`
**复现结果：** Output TPS/GPU = 490.1（SA 473.4，+3.5%），Interactivity = 63.12（SA 60.93，+3.6%）

#### 第四步：配置 B Trace 分析 ✅

**Trace 文件：** `nsys_fp4_throughput_chat_tp8_ep8_c64_iter100-150.nsys-rep`
**配置：** FP4, TP=8, EP=8, DP=false, chat 1K/1K, c=64, iter 100-150
**验证：** nsys UI 确认 gen_reqs ≈ 64，并发配置正确

#### Kernel 级 GPU 时间分布

| 类别 | 占比 | 时间 | 实例数 | 关键 Kernel |
|------|------|------|--------|------------|
| **MoE Expert GEMM** | **31.5%** | 2,246 ms | 47,328 | bmm_E2m1 FP4 GEMM (20.5%: 3 变体) + bmm_Bfloat16 FP4→BF16 GEMM (11.0%: 4 变体) |
| **Dense GEMM (nvjet)** | **28.8%** | 2,049 ms | — | nvjet splitK (12.7%) + nvjet ootst (7.1%) + 其他 nvjet ~9% + splitKreduce (2.2%) |
| **MoE Comm (moefinalize)** | **10.6%** | 756 ms | 23,664 | moefinalize_allreduce_fusion_kernel_oneshot_lamport（EP=8 allreduce 模式） |
| **Attention (FMHA)** | **7.2%** | 514 ms | 24,888 | fmhaSm100fKernel (FP4 KV cache，无 DP = 全量 attention) |
| **Norm + Comm** | **8.5%** | 604 ms | — | userbuffers_rmsnorm (5.0%) + userbuffers_allgather (3.3%) + RMSNorm (1.8%) |
| **Quantize** | **3.8%** | 270 ms | 97,104 | quantize_with_block_size |
| **MoE Routing** | **3.1%** | 226 ms | — | routingIndicesCluster (1.7%) + routingMain (1.4%) |
| **RoPE** | **1.2%** | 88 ms | 24,888 | applyMLARopeAndAssignQKV |
| **MoE Activation** | **0.6%** | 41 ms | 24,888 | silu_and_mul |
| **Other** | **4.7%** | ~327 ms | — | CatArrayBatchedCopy (1.7%), NCCL AllGather (0.3%), reduce, memcpy 等 |

> **注：** MoE backend = TRTLLM（Config A 用 CUTLASS），kernel 命名不同：Config A 的 CUTLASS MXF4 PtrArray grouped GEMM → Config B 的 bmm_E2m1 / bmm_Bfloat16 系列。实例总数 47,328 = 2 × 23,664，与 Config A 相同。

#### 分类汇总

| 大类 | 占比 | 说明 |
|------|------|------|
| **MoE 相关 (GEMM + Comm + Routing + Activation)** | **~45.8%** | 仍是最大占比，但低于 Config A 的 68.4% |
| **Dense GEMM (non-MoE linear)** | **~28.8%** | 占比上升（Config A ~20%） |
| **Norm + Comm (fused)** | **~8.5%** | userbuffers fused norm+allreduce |
| **Attention** | **~7.2%** | 无 DP，全量 attention（Config A 3.8%） |
| **Quantize + RoPE + Other** | **~9.7%** | 辅助开销 |

#### 第五步：配置 A vs 配置 B Kernel 级对比 ✅

| 类别 | Config A (EP4,DP,c256) | Config B (EP8,noDP,c64) | 变化 | 分析 |
|------|----------------------|------------------------|------|------|
| **MoE Expert GEMM** | **48.7%** (2,840ms) | **31.5%** (2,246ms) | **-17.2pp** | TRTLLM backend vs CUTLASS，EP=8 每 GPU 更少 expert |
| **Dense GEMM** | ~20% (~1,170ms) | **28.8%** (2,049ms) | **+8.8pp** | 绝对时间也增加，nvjet kernel 比重上升 |
| **MoE Comm** | 9.5% (556ms, A2A) | 10.6% (756ms, allreduce) | +1.1pp | EP=8 用 allreduce 替代 A2A dispatch+combine，绝对时间增 36% |
| **MoE Support** | 10.2% (595ms) | 3.7% (267ms) | **-6.5pp** | routing+activation 开销大幅降低 |
| **Attention** | **3.8%** (224ms) | **7.2%** (514ms) | **+3.4pp** | **无 DP = 全量 attention，绝对时间 2.3×** |
| **Norm** | 2.6% (152ms) | 8.5% (604ms) | +5.9pp | userbuffers fused norm+comm 在 TP=8 下更重 |
| **总 GPU 时间** | 5.83s | 7.12s | **+22%** | Config B 整体更慢 |

#### 关键发现

1. **Attention 3.8% → 7.2%：量化了 DP Attention 的影响。** 无 DP 后 attention 绝对时间翻倍（224ms → 514ms），但仍仅占 7.2%——**DP Attention 不是 41% 差距的主因**，即使完全消除也只影响 ~3.4pp。

2. **MoE GEMM 48.7% → 31.5%：MoE backend 差异显著。** Config A 用 CUTLASS MXF4 PtrArray grouped GEMM，Config B 用 TRTLLM 的 bmm 系列。占比下降但仍是最大单项。MoE GEMM 效率仍是跨平台对比的核心因素。

3. **MoE 通信模式完全改变但占比相近。** A2A dispatch+combine (9.5%) → moefinalize allreduce (10.6%)。EP=8 的 allreduce 绝对时间反而增加 36%（556ms → 756ms），说明 EP=8 通信并不比 EP=4 A2A 更高效。

4. **Dense GEMM 占比上升到 28.8%。** Config B 下 nvjet kernel 占比大幅增加，表明非 MoE 线性层（attention QKV/output proj 等）在无 DP 配置下成为更重要的性能因素。

5. **Norm+Comm 融合开销 2.6% → 8.5%。** TP=8 下 userbuffers fused norm+allreduce/allgather 成为可观开销，Config A 的 TP=4 通信更轻量。

## 差距来源分解（基于双配置 trace 对比）

基于 Config A + Config B 两组 B200 trace 数据，41% 性能差距的来源分解：

| 因素 | Config A 占比 | Config B 占比 | 对差距的预估贡献 | 说明 |
|------|-------------|-------------|----------------|------|
| **FP4 MoE GEMM 效率** | 48.7% | 31.5% | **高** | B200 SM100 MXF4 tensor core vs MI355X CDNA4；两种配置下均为最大单项 |
| **Dense GEMM 效率** | ~20% | 28.8% | **中-高** | Config B 下占比上升，nvjet kernel 效率差异是重要因素 |
| **MoE 通信** | 9.5% (A2A) | 10.6% (allreduce) | **中** | 两种 EP 策略通信开销相近；355X EP=1 无此开销但 compute 更重 |
| **DP Attention** | 3.8% | 7.2% | **低（已量化）** | Config A→B 仅增 3.4pp。即使 355X attention 效率更差，影响有限 |
| **Norm + TP 通信** | 2.6% | 8.5% | **低-中** | TP=8 下 fused norm+comm 开销更大 |
| **Framework 差异** | 全局 | 全局 | **低-中** | TRT-LLM C++ runtime vs ATOM/vLLM PyTorch overhead |

> **核心判断：** FP4 MoE Expert GEMM + Dense GEMM 效率（合计 ~60% GPU time）是差距的主要来源。DP Attention 影响已量化为仅 3.4pp——不是 41% 差距的主因。跨平台对比应聚焦 GEMM kernel 效率（datatype、shape、算法选择）。

## Per-Module Kernel 级对比：B200 vs MI355X（Config B）

> **⚠️ 非 Apple-to-Apple 对比说明：**
>
> - **B200 数据来源：** nsys trace，Config B（FP4, TP=8, EP=8, DP=false, **c=64**），trtllm-bench 离线模式，iter 100-150，nsys UI 手动逐层读取 + sqlite 统计
> - **MI355X 数据来源：** ATOM 内部 rocprof trace，TP=8, EP=8, DSMXFP4, **c=16**，ATOM 0.1.1
> - **并发不同：** B200 c=64 vs 355X c=16。并发影响 batch size，进而影响 GEMM shape 和 kernel 选择。c=64 下 MoE GEMM 的 M 维度更大，kernel 耗时更长。**因此 B200 的 per-kernel 绝对时间偏高，直接比较数值不公平。**
> - **EP 策略相同：** 两者均为 EP=8，MoE 通信模式一致（allreduce），每 GPU 处理 8 experts
> - **用途：** 此表用于识别各模块的 kernel 实现差异和相对瓶颈分布，不作为绝对性能优劣判断

### MoE Layer（layers 3-60, 58 层）— 单层 Kernel 序列

| Module | B200 Kernel | B200 (μs) | 355X Kernel | 355X (μs) | 说明 |
|--------|-------------|-----------|-------------|-----------|------|
| **pre-attn norm** | RMSNormKernel<8,bf16> | 2.9 | local_device_load_rmsnorm (aiter) | 4.2 | B200 独立 kernel；355X fused allreduce+norm |
| **QKV_A proj** | nvjet_tst_64x32 splitK TNT (FP8 E4M3) | 10.6 | bf16gemm_fp32bf16_tn_32x64_splitk (hipBLASLt) | 13.7 | [M,7168]×[7168,2112]，B200 用 FP8 nvjet，355X 用 BF16 hipBLASLt |
| splitK reduce | cublasLt::splitKreduce | 3.0 | (included above) | — | |
| **q_norm** | RMSNormKernel<8,bf16> | 2.5 | add_rmsnorm_quant_kernel<256,8> | 4.3 | 355X fused norm+quant |
| **k_norm** | RMSNormKernel<8,bf16> | 2.5 | add_rmsnorm_quant_kernel<64,8> | 4.1 | |
| **Q_B proj** | nvjet_tst_128x32 TNT (FP8 E4M3) | 3.9 | bf16gemm_fp32bf16_tn_64x64_splitk | 8.8 | [M,1536]×[1536,3072] |
| **q×up_k** | (fused into RoPE kernel) | — | batched_gemm_a8w8 M16_N128_K128 | 4.4 | B200 在 RoPE kernel 中融合；355X 独立 batched GEMM |
| **RoPE + KV cache** | applyMLARopeAndAssignQKV | 3.5 | fuse_qk_rope_concat_and_cache_mla_per_head | 4.1 | 功能相同，实现不同 |
| **MLA Attention** | fmhaSm100f E4M3 Qk576 V512 PagedKV | **20.5** | mla_a8w8_qh16_qseqlen1_gqaratio16 | **9.8** | B200 单 fused kernel；355X 分 attention+reduce 两步 |
| mla reduce | (included above) | — | kn_mla_reduce_v1 | 7.9 | 355X MLA reduce 独立 kernel |
| **O proj** | nvjet_tst_64x16 TNT | 3.7 | batched_gemm_a8w8 M16_N32_K128 | 5.6 | |
| **quantize (FP4)** | quantize_with_block_size | 2.4 | (fused into allreduce) | — | B200 独立量化 kernel |
| **O proj pt2** | nvjet_ootst_128x128 E4M3 | 6.1 | bf16gemm_fp32bf16_tn_32x64_splitk | 12.0 | [M,2048]×[2048,7168] |
| **TP allreduce+norm** | userbuffers_fp16_sum_gpu_mc_rmsnorm | 14.5 | reduce_scatter_cross_device_store (aiter) | 11.7 | B200 fused allreduce+RMSNorm |
| | | | local_device_load_rmsnorm (aiter) | 4.2 | 355X 分 scatter+load+norm |
| **residual allgather** | userbuffers_fp16_sum_inplace_res_allgather | 9.8 | (included in allreduce) | — | |
| **Router GEMM** | nvjet_tss splitK TNT | 5.5 | bf16gemm_fp32bf16_tn_80x64_splitk | 8.4 | [M,7168]×[7168,384] |
| splitK reduce | splitKreduce | 2.7 | (included above) | — | |
| **quantize (router)** | quantize_with_block_size | 2.9 | — | — | |
| **MoE routing** | routingMainKernel | 4.3 | grouped_topk_opt_sort (aiter) | 3.8 | topk 路由 |
| | routingIndicesClusterKernel | 5.2 | ck_tile::MoeSortingKernel ×2 | 11.1 | 355X 用 CK tile sorting |
| **MoE quant+sort** | — | — | _fused_dynamic_mxfp4_quant_moe_sort | 4.8 | 355X fused quant+sort |
| **MoE gate+up GEMM** | bmm_E2m1_E2m1E2m1_Fp32 swiGlu (FP4×FP4→FP32) | **62** | ck::kernel_moe_mxgemm_2lds (MXFP4) | **40.4** | **核心瓶颈。** B200 1.53× 慢（但 c=64 vs c=16，M 维不同） |
| **MoE activation** | (fused swiGlu in gate+up GEMM) | — | act_and_mul_kernel silu | 5.7 | B200 fused 在 GEMM 内；355X 独立 kernel |
| **MoE down GEMM** | bmm_Bfloat16_E2m1E2m1_Fp32 (FP4×FP4→BF16) | **35** | ck::kernel_moe_mxgemm_2lds (MXFP4) | **24.5** | B200 1.43× 慢（同上 batch size 差异） |
| **MoE out quantize** | quantize_with_block_size | 3.8 | _fused_dynamic_mxfp4_quant_moe_sort | 4.0 | |
| **Shared Expert gate+up** | nvjet_ootst E4M3 | 10.0 | (fused into MoE pipeline at EP=8) | — | B200 独立 shared expert；355X 可能融合 |
| **SiLU activation** | silu_and_mul | 1.6 | — | — | |
| **Shared Expert quant** | quantize_with_block_size | 2.2 | — | — | |
| **Shared Expert down** | nvjet_ootst E4M3 | 3.8 | — | — | |
| **MoE EP allreduce** | moefinalize_allreduce_lamport_oneshot | **28** | reduce_scatter_cross_device_store | **13.1** | B200 Lamport allreduce 2.14× 慢 |
| **overlap: next QKV splitK** | nvjet_tst splitK (与 moefinalize 重叠) | 34 | — | — | B200 pipeline 下一层 QKV 与 MoE allreduce 重叠执行 |
| **MoE 层合计** | | **~255** | | **~213** | **B200 1.20× 慢（c=64 vs c=16）** |

### Dense Layer（layers 0-2, 3 层）— 无 MoE routing/GEMM

| Module | B200 Kernel | B200 (μs) | 说明 |
|--------|-------------|-----------|------|
| pre-attn norm | RMSNormKernel<8,bf16> | 2.9 | |
| QKV_A proj | nvjet_tst_64x32 splitK TNT | 10.6 | |
| splitK reduce | cublasLt::splitKreduce | 3.0 | |
| q_norm + k_norm | RMSNormKernel ×2 | 5.0 | |
| Q_B proj | nvjet_tst_128x32 TNT | 3.9 | |
| RoPE + KV cache | applyMLARopeAndAssignQKV | 3.5 | |
| MLA Attention | fmhaSm100f E4M3 | 20.5 | |
| O proj | nvjet_tst + ootst | 9.8 | |
| fused allreduce+norm+FP4 quant | userbuffers_rmsnorm_quant_fp4 | 18.7 | Dense 层特有：fused AR+norm+quant |
| Shared Expert gate+up | nvjet_ootst E4M3 | 11.7 | |
| SiLU | silu_and_mul | 1.8 | |
| Shared Expert down | nvjet_ootst E4M3 | 6.5 | |
| allreduce+norm | userbuffers_rmsnorm | 12.8 | |
| **Dense 层合计** | | **~120** | 无 MoE，约 MoE 层的一半时间 |

### 单步 Decode 汇总

| Component | B200 (μs) | 355X (μs) | 说明 |
|-----------|-----------|-----------|------|
| 3 Dense layers | ~360 | ~400 (est.) | |
| 58 MoE layers | ~14,790 | ~12,362 | 主要差距来源 |
| Postprocess (sampling, NCCL) | ~350 | ~240 (est.) | |
| **Total per decode step** | **~15,500** | **~13,000** | B200 c=64, 355X c=16 |
| **TPOT (observed)** | 15.84 ms | ~13 ms | |

### 关键发现

1. **MoE GEMM 是 355X 领先的主要来源。** B200 `bmm_E2m1` gate+up = 62μs vs 355X `kernel_moe_mxgemm` = 40.4μs（1.53×）；down proj = 35μs vs 24.5μs（1.43×）。58 层 × ~32μs 差距 = **~1.9ms/step**，是最大单项差距贡献者。但需注意 **c=64 vs c=16 的 batch size 差异**——B200 的 M 维度更大，GEMM 计算量更多，直接数值比较不公平。

2. **MoE EP allreduce 通信 B200 2.14× 慢。** B200 `moefinalize_allreduce_lamport` 28μs vs 355X `reduce_scatter` 13.1μs。58 层 × 15μs = ~870μs/step。B200 用 Lamport-clock 全局 allreduce，355X 用 reduce-scatter，通信拓扑和实现不同。

3. **Dense GEMM（非 MoE）B200 全面领先。** nvjet FP8 E4M3 kernel 一致优于 355X hipBLASLt BF16 kernel：QKV_A 10.6 vs 13.7μs（0.77×），Q_B 3.9 vs 8.8μs（0.44×），O proj 6.1 vs 12.0μs（0.51×）。但 Dense GEMM 占总时间较少，不足以抵消 MoE GEMM 劣势。

4. **MLA Attention 两平台接近。** B200 fmhaSm100f = 20.5μs（单 fused kernel）vs 355X mla_a8w8 + reduce = 17.7μs（两步）。B200 略慢 1.16×，但 attention 占比仅 7%，影响有限。

5. **Kernel 融合策略不同。** B200 将 swiGLU 融合进 MoE gate+up GEMM（`bmm_E2m1...swiGlu`），355X 使用独立 `act_and_mul_kernel`。B200 将 allreduce+RMSNorm 融合（`userbuffers_rmsnorm`），355X 分步执行。B200 pipeline 化 moefinalize 与下一层 QKV GEMM 重叠执行。

6. **数据类型差异显著。** Dense GEMM：B200 用 FP8 E4M3（nvjet kernel），355X 用 BF16（hipBLASLt）。MoE GEMM：两者均用 MXFP4×MXFP4，但实现不同（B200 bmm 系列 vs 355X CK tile kernel_moe_mxgemm）。MLA：B200 FP8 E4M3 KV cache，355X INT8 (A8W8) KV cache。

## 待填充

- [x] 配置 A：B200 EP=4 DP 复现 + trace 分析
- [x] 配置 B：B200 EP=8 no-DP 复现 + trace 分析
- [x] 配置 A vs B kernel 分布对比
- [x] Per-Module Kernel 级对比（B200 c=64 vs 355X c=16，非 apple-to-apple）
- [ ] 配置 B vs 355X 公平对标结果（同 concurrency）
- [ ] 355X c=64 trace 数据（公平 per-kernel 对比）

## 迭代日志

| 日期 | 变更 |
|------|------|
| 2026-03-27 v5 | Per-Module Kernel 级对比：B200 nsys trace (c=64) vs 355X rocprof (c=16)。标注非 apple-to-apple（batch size 不同）。识别 MoE GEMM (1.53×)、MoE allreduce (2.14×) 为 B200 劣势项，Dense GEMM (0.44-0.77×) 为优势项 |
| 2026-03-25 v4 | Config B trace 分析完成 + Config A vs B kernel 级对比。复现验证 +3.5% vs SA。关键发现：DP Attention 仅影响 3.4pp，MoE GEMM 仍是核心因素 |
| 2026-03-25 v3 | 增加配置 B（EP=8, DP=false, c=64）公平对标方案。ATOM 不支持 DP Attention，消除此差异后重新对比 |
| 2026-03-25 v2 | B200 nsys trace 分析完成：kernel 级分布、分类汇总、差距来源初步分解 |
| 2026-03-25 v1 | 初版：SA 报告数据 + MI355X 数据对比表格，Breakdown 计划 |
