# FP4 性能差距分析：B200 vs MI355X — Breakdown 调查

> **Last updated:** 2026-03-25
> **Model:** DeepSeek-R1-0528, FP4
> **状态：** 配置 A（EP4 DP）trace 完成 ✅；配置 B（EP8 no-DP）复现+trace 进行中 🔄

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

### 配置 B：公平对标（EP=8, DP=false, c=64）🔄 进行中

| 项目 | B200 (SA InferenceX) | MI355X (ATOM) | 差异 |
|------|---------------------|---------------|------|
| **Image** | rc6.post2 | ATOM 0.1.1 | TRT-LLM vs ATOM |
| **GPU / TP / EP** | 8 / 8 / 8 | 4 / 4 / 1 | GPU 数不同，需按 /GPU 比 |
| **DP Attention** | **False** | False | **相同** |
| **Concurrency** | 64 | 待定 | |

| Metric | B200 (SA 数据) | MI355X | 差距 |
|--------|---------------|--------|------|
| **Output TPS /GPU** | 473.4 | 待测 | 待填 |
| **Interactivity** | 60.93 | 待测 | 待填 |

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

#### 第三步：复现 B200 DP=false 数据 🔄 进行中

在 B200 上使用 SA InferenceX 配置 B（rc6.post2, FP4, TP=8, EP=8, DP=false, c=64）复现性能 + 抓 trace + 分析。

**命令：**
```
bash scripts/collect_nsys_trace.sh \
  --model /home/models/models--DeepSeek-R1-0528 \
  --mode serve --scenario chat --concurrency 64 \
  --quant fp4 --config throughput --tp 8 --ep 8 --dp false \
  --trace-dir results_b200_mtp0_fp4_ep8_conc64_post2 --iter-range 100-150
```

**预期结果：** SA 报告 Output TPS/GPU = 473.4，Interactivity = 60.93

#### 第四步：配置 B Trace 分析

分析 DP=false 配置下的 kernel 分布，与配置 A 对比：
- MoE All-to-All 通信占比变化（EP=8 vs EP=4）
- Attention 占比变化（无 DP Attention → 全量 attention）
- MoE GEMM 占比变化（EP=8 每 GPU 仅处理 ~8 个 expert）

#### 第五步：跨配置 + 跨平台 Kernel 级对比

| 对比 | 目的 |
|------|------|
| B200 配置 A vs 配置 B | 量化 DP Attention + EP 策略对 kernel 分布的影响 |
| B200 配置 B vs 355X | 公平对比：消除 DP 差异后的 kernel 级性能差距 |

## 差距来源初步分解

基于 B200 trace 数据，41% 性能差距的可能来源：

| 因素 | 影响范围 | 对差距的预估贡献 | 说明 |
|------|---------|----------------|------|
| **FP4 MoE GEMM 效率** | 48.7% GPU time | **高** | B200 SM100 MXF4 tensor core vs MI355X CDNA4；这是最大单一因素 |
| **EP=4 vs EP=1 (MoE compute)** | 48.7% + 10.2% | **中-高** | EP=1 每 GPU 算全部 expert（更多 compute），EP=4 算 1/4 expert + 9.5% 通信开销。净效应取决于 expert 并行效率 |
| **Dense GEMM 效率** | ~20% GPU time | **中** | 非 MoE 线性层效率差异 |
| **DP Attention vs 全量 Attention** | 3.8% GPU time | **低** | 即使 355X attention 4 倍慢，仅贡献 ~11% GPU time |
| **Framework 差异** | 全局 | **低-中** | TRT-LLM C++ runtime vs ATOM/vLLM PyTorch overhead |

> **核心判断：** FP4 MoE Expert GEMM 效率（48.7% of GPU time）+ EP 并行策略差异是 41% 差距的主要来源。Attention 和 Norm 不是瓶颈。

## 待填充

- [x] 配置 A：B200 EP=4 DP 复现 + trace 分析
- [ ] 配置 B：B200 EP=8 no-DP 复现 + trace 分析
- [ ] 配置 A vs B kernel 分布对比
- [ ] 配置 B vs 355X 公平对标结果
- [ ] 355X trace 分析（如有 rocprof 数据）

## 迭代日志

| 日期 | 变更 |
|------|------|
| 2026-03-25 v3 | 增加配置 B（EP=8, DP=false, c=64）公平对标方案。ATOM 不支持 DP Attention，消除此差异后重新对比 |
| 2026-03-25 v2 | B200 nsys trace 分析完成：kernel 级分布、分类汇总、差距来源初步分解 |
| 2026-03-25 v1 | 初版：SA 报告数据 + MI355X 数据对比表格，Breakdown 计划 |
