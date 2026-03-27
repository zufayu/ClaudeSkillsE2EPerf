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

## Per-Module Kernel 级分析（Config B，第 40 层实测数据）

> **数据来源：** B200 nsys trace，Config B（FP4, TP=8, EP=8, DP=false, c=64），trtllm-bench 离线模式，iter 100-150
> **分析方法：** nsys UI Events View 手动逐 kernel 读取第 40 层（典型 MoE 层）
> **MI355X 数据：** 待补充（后续会增加 355X 列用于跨平台对比）

### 精度判断证据

| Kernel 类型 | 名字中的证据 | 精度判断 |
|------------|-------------|---------|
| `nvjet_ootst_...Avec16U**E4M3**_Bvec16U**E4M3**` | A、B 矩阵均标注 E4M3 | **FP8 E4M3 × FP8 E4M3** |
| `bmm_**E2m1**_**E2m1E2m1**_Fp32...swiGlu` | E2M1 = MXFP4，累加器 FP32 | **FP4 × FP4 → FP32** |
| `bmm_**Bfloat16**_**E2m1E2m1**_Fp32` | 输出 BF16，输入 E2M1 | **FP4 × FP4 → BF16** |
| `fmhaSm100f_**QkvE4m3**OBfloat16` | QKV 标注 E4M3，输出 BF16 | **FP8 E4M3 KV cache** |
| `quantize_with_block_size<**Type::0**, __nv_bfloat16, 16>` | Type 0 = FP4 block scaling | **BF16 → FP4 量化** |
| `nvjet_tst_...splitK_TNT`（QKV_A 等） | 名字中**无数据类型标记** | 待确认（推测 FP8） |

> **结论：** MoE Expert 权重 = FP4 (MXFP4)，Dense 权重（out_proj, shared_expert）= FP8 E4M3，MLA = FP8 KV cache。`tst` 系列 kernel（QKV_A, Q_B, o×up_v）名字中未标注精度。

### MoE Layer 单层 Kernel 序列（第 40 层实测）

> **层间 Pipeline 重叠说明：** 上一层末尾的 `moefinalize` 与本层的 `QKV_A proj` 在 GPU 上并行执行（仅差 2μs 启动）。`userbuffers_rmsnorm`（#11）融合了本层的 TP allreduce 和下一层的 pre-attn RMSNorm，因此 MoE 层没有独立的 pre-attn norm kernel。

| # | Module | Kernel | Duration (μs) | Precision | 说明 |
|---|--------|--------|---------------|-----------|------|
| — | fused_qkv_a_proj | nvjet_sm100_tst_64x32 splitK TNT | ⟨pipeline 重叠⟩ | 待确认 | 与上一层 moefinalize 并行执行，wall time 含重叠 |
| — | fused_qkv_a_proj (续) | cublasLt::splitKreduce | ⟨pipeline 重叠⟩ | BF16 | splitK 归约 |
| 1 | q_norm | RMSNormKernel<8, bf16> (Stream 7) | 2.7 | BF16 | |
| 2 | k_norm | RMSNormKernel<8, bf16> (Stream 8907) | 2.5 | BF16 | 与 q_norm 并行 |
| 3 | q_b_proj | nvjet_sm100_tst_24x64 TNN | 5.8 | 待确认 | [M,1536]×[1536,3072] |
| 4 | (k concat) | CatArrayBatchedCopy (Stream 8907) | 5.0 | — | 与 #3 并行，k_norm 结果拼接 |
| 5 | q × up_k | nvjet_sm100_tst_128x32 TNT | 3.6 | 待确认 | |
| 6 | cache_update | applyMLARopeAndAssignQKV | 3.5 | BF16 | RoPE 旋转位置编码 + KV cache 写入 |
| 7 | **mla** | **fmhaSm100fKernel QkvE4m3 Qk576 V512** | **20.6** | **FP8 E4M3** | MLA attention, PagedKV, 单 fused kernel |
| 8 | o × up_v | nvjet_sm100_tst_64x16 TNT | 4.1 | 待确认 | |
| 9 | (quantize → out_proj) | quantize_with_block_size | 2.6 | BF16→FP4 | FP4 block 量化 |
| 10 | out_proj | nvjet_sm100_ootst_128x128 E4M3 | 6.1 | **FP8 E4M3** | [M,2048]×[2048,7168] |
| 11 | **allreduce+addrmsnorm** | **userbuffers_fp16_sum_gpu_mc_rmsnorm** | **15.6** | BF16 | fused TP allreduce + 下一模块 pre-norm |
| 12 | (residual allgather) | userbuffers_fp16_sum_inplace_res_allgather | 9.7 | BF16 | 残差连接的 allgather |
| 13 | Router gemm | nvjet_sm100_tss splitK TNT | 5.4 | 待确认 | [M,7168]×[7168,384] |
| 14 | Router gemm (续) | cublasLt::splitKreduce | 2.8 | — | splitK 归约 |
| 15 | (quantize → routing) | quantize_with_block_size | 3.0 | BF16→FP4 | routing 输入量化 |
| 16 | topk | routingMainKernel (DeepSeek, topk=8) | 4.4 | FP32/BF16 | expert 路由选择 |
| 17 | sort | routingIndicesClusterKernel | 5.1 | — | token→expert 分组排序 |
| 18 | **moe (gate+up)** | **bmm_E2m1 swiGlu dynBatch** | **48.4** | **FP4×FP4→FP32** | MoE gate+up proj, fused SwiGLU 激活 |
| 19 | **moe (down)** | **bmm_Bfloat16 dynBatch** | **27.9** | **FP4×FP4→BF16** | MoE down proj |
| 20 | (quantize → SE) | quantize_with_block_size | 3.6 | BF16→FP4 | MoE 输出量化 |
| 21 | shared_expert (gate+up) | nvjet_sm100_ootst_128x128 E4M3 | 9.9 | **FP8 E4M3** | |
| 22 | shared_expert (激活) | silu_and_mul_kernel | 1.9 | BF16 | SiLU 激活 |
| 23 | shared_expert (量化) | quantize_with_block_size | 2.2 | BF16→FP4 | |
| 24 | shared_expert (down) | nvjet_sm100_ootst_128x128 E4M3 | 3.9 | **FP8 E4M3** | |
| 25 | **moe (EP allreduce)** | **moefinalize_allreduce_lamport_oneshot** | **58.9** | BF16 | EP=8 MoE 结果聚合 |
| — | ⟨pipeline⟩ 下一层 qkv_a | nvjet_sm100_tst_64x32 splitK TNT | (65.4) | 待确认 | 与 #25 并行执行，Start 仅差 2μs |

> **Pipeline 重叠图示（第 40→41 层边界）：**
> ```
> 时间 →    .451483s                              .451542s
> 第40层 #25: |████ moefinalize (58.9μs) ████████████|
> 第41层 QKV:   |████ nvjet splitK (65.4μs) ████████████|
>            .451485s                                .451550s
>            ↑ 间隔 2μs，GPU 并行执行两个 kernel
> ```

### MoE 层各模块占比（第 40 层，不含 pipeline 重叠区）

| Module | Duration (μs) | 说明 |
|--------|---------------|------|
| q_norm + k_norm (#1-2) | 5.2 | |
| q_b_proj + q×up_k (#3,5) | 9.4 | |
| cache_update (#6) | 3.5 | |
| **mla (#7)** | **20.6** | |
| o×up_v + quantize + out_proj (#8-10) | 12.8 | |
| **allreduce+addrmsnorm + allgather (#11-12)** | **25.3** | |
| Router gemm + reduce (#13-14) | 8.2 | |
| quantize + topk + sort (#15-17) | 12.5 | |
| **moe gate+up + down (#18-19)** | **76.3** | 最大单项 |
| quantize + shared_expert (#20-24) | 21.5 | |
| **moefinalize (#25)** | **58.9** | 第二大，波动很大 |
| **合计（#1-#25）** | **253.7** | 不含 pipeline 重叠区的 QKV_A |

### Kernel 耗时波动（第 40 层 vs 第 41 层）

| Kernel | 第 40 层 | 第 41 层 | 波动 | 原因 |
|--------|---------|---------|------|------|
| bmm_E2m1 (gate+up) | 48.4μs | 60.0μs | ±12μs | expert 路由不同，M 维度变化 |
| bmm_Bfloat16 (down) | 27.9μs | 33.4μs | ±5μs | 同上 |
| moefinalize | 58.9μs | 27.8μs | ±31μs | 通信负载波动大 |
| fmhaSm100f | 20.6μs | 20.8μs | ±0.2μs | 非常稳定（shape 固定） |
| 其他 dense kernel | ±0.3μs | ±0.3μs | 稳定 | shape 固定 |

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
| 2026-03-27 v6 | 基于第 40 层实测数据重写 Per-Module 表格。增加精度判断证据（FP8/FP4 从 kernel 名解析）、量化算子标注、Pipeline 重叠图示、层间波动对比。删除 MI355X 列（待补充） |
| 2026-03-27 v5 | Per-Module Kernel 级对比：B200 nsys trace (c=64) vs 355X rocprof (c=16)。标注非 apple-to-apple（batch size 不同） |
| 2026-03-25 v4 | Config B trace 分析完成 + Config A vs B kernel 级对比。复现验证 +3.5% vs SA。关键发现：DP Attention 仅影响 3.4pp，MoE GEMM 仍是核心因素 |
| 2026-03-25 v3 | 增加配置 B（EP=8, DP=false, c=64）公平对标方案。ATOM 不支持 DP Attention，消除此差异后重新对比 |
| 2026-03-25 v2 | B200 nsys trace 分析完成：kernel 级分布、分类汇总、差距来源初步分解 |
| 2026-03-25 v1 | 初版：SA 报告数据 + MI355X 数据对比表格，Breakdown 计划 |
