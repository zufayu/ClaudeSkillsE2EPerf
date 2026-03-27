# FP4 性能差距分析：B200 vs MI355X — Breakdown 调查

> **Last updated:** 2026-03-27 v14
> **Model:** DeepSeek-R1-0528-NVFP4-v2, FP4
> **配置：** EP=8, DP=false, c=64, TP=8, chat 1K/1K
> **状态：** B200 trace 完成 ✅；Per-Module Kernel 分析完成 ✅；10 层平均数据完成 ✅；nvjet E4M3 源码考证完成 ✅；权重精度考证完成 ✅；算子级重构完成 ✅

## 问题背景

SA InferenceX 报告的 B200 FP4 性能大幅领先 MI355X FP4，需要 breakdown 分析差距来源。

**ATOM 不支持 DP Attention**，原始 SA 对标配置（EP=4, DP=true）无法公平对比。选择 **EP=8, DP=false, c=64** 作为公平对标基准：DP=false 消除 DP Attention 差异；EP=8 是 B200 8GPU 的自然 EP 配置；c=64 是 SA 原始测试点。

## 对标配置

| 项目 | B200 (SA InferenceX) | B200 (复现) | MI355X (ATOM) | 差异 |
|------|---------------------|------------|---------------|------|
| **Image** | rc6.post2 | rc6.post2 | ATOM 0.1.1 | TRT-LLM vs ATOM |
| **GPU / TP / EP** | 8 / 8 / 8 | 8 / 8 / 8 | 4 / 4 / 1 | GPU 数不同，需按 /GPU 比 |
| **DP Attention** | **False** | **False** | False | **相同** |
| **Concurrency** | 64 | 64 | 待定 | |

| Metric | B200 (SA) | B200 (复现) | 复现偏差 | MI355X | 差距 |
|--------|-----------|------------|---------|--------|------|
| **Allput TPS /GPU** | 待查 | 待查 | | 待测 | 待填 |
| **TPOT p50 (ms)** | 待查 | 待查 | | 待测 | 待填 |
| **TTFT p50 (ms)** | 待查 | 待查 | | 待测 | 待填 |
| **Interactivity** | 60.93 | 63.12 | +3.6% | 待测 | 待填 |

> **复现验证：** 同 rc6.post2 docker，复现结果 +3.5% vs SA，在正常波动范围内。

## Per-Module Kernel 级分析（10 层平均）

> **数据来源：** B200 nsys trace，FP4, TP=8, EP=8, DP=false, c=64，trtllm-bench 离线模式，iter 100-150
> **统计口径：** 第 40-49 层（连续 10 层）平均值，含 min/max 波动范围
> **算子定义：** 按逻辑功能合并 kernel（splitKreduce 合入 GEMM，quantize 合入目标 GEMM），方便跨平台对比
> **并行组：** 标注 **P1** 的算子在不同 stream 上并行执行，关键路径 = max(组内算子时间)
> **MI355X 数据：** 待补充（rocprof trace）

| # | 算子 | 计算内容 | B200 Kernel(s) | Avg μs | % | Min | Max | 并行 | 精度 | MI355X μs |
|---|------|---------|---------------|--------|------|-----|-----|------|------|-----------|
| 1 | **qkv_a_proj** | q_a+kv_a 低秩压缩 [7168→2112] | nvjet tst splitK + reduce | **42.6** | 15.0% | 28.5 | 69.1 | **P1** | BF16×BF16 | |
| 2 | q/k_norm | Q、K RMSNorm ×2 | RMSNormKernel ×2 | 4.6 | 1.6% | 2.4 | 5.3 | | BF16 | |
| 3 | q_b_proj | Q 展开 [1536→nhead×192] | nvjet tst | 5.7 | 2.0% | 5.4 | 5.8 | | BF16×BF16 | |
| 4 | k_concat | K 拼接（RoPE 部分） | CatArrayBatchedCopy | 4.4† | 1.6% | 0.0 | 5.1 | | — | |
| 5 | uk_gemm | kv_b K 展开 [512→nhead×128] | nvjet tst | 3.8 | 1.3% | 3.6 | 4.0 | | BF16×BF16 | |
| 6 | rope_cache | RoPE + KV cache 写入 | applyMLARopeAndAssignQKV | 3.5 | 1.2% | 3.3 | 3.6 | | BF16 | |
| 7 | **fmha** | MLA attention | fmhaSm100f QkvE4m3 | **20.7** | 7.3% | 20.1 | 21.7 | | FP8 E4M3 KV | |
| 8 | uv_gemm | kv_b V 投影 | nvjet tst | 3.7 | 1.3% | 3.5 | 4.1 | | BF16×BF16 | |
| 9 | **out_proj** | BF16→FP4 量化 + o_proj GEMM | quantize + nvjet ootst | **8.6** | 3.0% | 8.4 | 8.8 | | FP4×FP4 | |
| 10 | **tp_allreduce+norm** | TP AR + residual add + pre-MLP norm | userbuffers_rmsnorm | **15.2** | 5.3% | 13.1 | 17.3 | | BF16 | |
| 11 | residual_ag | residual allgather | userbuffers_allgather | 9.7 | 3.4% | 9.0 | 10.1 | | BF16 | |
| 12 | router | Router splitK + topK + sort | nvjet splitK + reduce + routing ×2 | 12.0 | 4.2% | 8.0 | 13.2 | | BF16 | |
| 13 | **moe_gemm** | quantize + gate+up+SwiGLU + down | quantize + bmm_E2m1 + bmm_BF16 | **95.3** | 33.5% | 79.4 | 104.9 | | FP4×FP4 | |
| 14 | shared_expert | quantize×2 + gate+up + SiLU + down | quantize×2 + ootst×2 + silu | 21.4 | 7.5% | 20.9 | 21.8 | | FP4×FP4 | |
| 15 | **moe_finalize** | 加权求和 + EP allreduce + residual | moefinalize_lamport | **33.1** | 11.7% | 19.0 | 58.9 | **P1** | BF16 | |

> † k_concat 和 q/k_norm 的第二个 RMSNorm 在 Stream 8907 上执行，部分层 copy-paste 丢失，平均值略偏低。
> **高方差算子：** qkv_a_proj（std=11.4μs）和 moe_finalize（std=11.1μs）层间波动大，前者受 splitK 调度影响，后者受 EP allreduce 跨 GPU 同步影响。

#### 合计

| 口径 | B200 μs | 说明 |
|------|---------|------|
| **GPU 总时间** | **284.2** | 所有算子 GPU 执行时间求和（10 层平均） |
| **关键路径时间** | **251.1** | 扣除 P1 并行隐藏的 moe_finalize 33.1μs |
| **并行节省** | 33.1 | moe_finalize 被 qkv_a_proj 完全遮盖 |

> **P1 并行组（层间 pipeline）：**
> ```
> 时间 →
> #15 moe_finalize: |████ 33.1μs（avg）████|
> #1  qkv_a_proj:   |████ 42.6μs（avg）████████████|
>                                            ↑ qkv_a 多跑 ~9.5μs
> 关键路径 = max(33.1, 42.6) = 42.6μs，moefinalize 完全隐藏
> ```

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
- [ ] 第 61 层数据（最后一层，无 MoE）
- [ ] MI355X 数据补充（355X 列 + ratio 列）
- [ ] 配置 B vs 355X 公平对标结果（同 concurrency）

## 迭代日志

| 日期 | 变更 |
|------|------|
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
