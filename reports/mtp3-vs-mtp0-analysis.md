# MTP3 vs MTP0 收益分析：B200 vs MI355X

> **Last updated:** 2026-03-23
> **Data date:** 2026-03-21/22 (B200), 2026-03-23 (355X)
> **Model:** DeepSeek-R1-0528 FP8, 8-GPU TP=8
> **B200 framework:** TRT-LLM 1.2.0rc6 | **MI355X framework:** ATOM (613e57af)

## 核心结论

1. **记分牌 8:10 (B200:355X)**，355X 在 18 个可比点中拿到更多相对增益。
2. **DAR 差距因场景而异：** B200 在 chat 场景 DAR 优势巨大（80% vs 49%），但在 reasoning 场景差距很小（74% vs 65%）。这直接解释了为什么 reasoning 场景 355X 能追平 B200。
3. **355X DAR 随并发基本稳定，B200 DAR 未按并发实测。** 355X per-concurrency DAR 波动仅 ±3%，说明 DAR 主要由模型/场景决定，与并发关系不大。B200 DAR 来自离线 trtllm-bench（单一并发），是否在 serving 高并发下衰减仍未知。
4. **两个平台的 DAR 利用效率均偏低（6-53%）。** 即使 DAR 数据正确，实际 TPS 增益远低于理论值。最大优化空间在 serving 路径的 MTP 执行效率。

## 数据表格

> 由 `compare_mtp.py` 生成：
> ```bash
> python scripts/compare_mtp.py --cross \
>   --b200-mtp0 runs/8xb200-fp8-20260321-mtp0-ep1.json \
>   --b200-mtp3 runs/8xb200-fp8-20260322-mtp3-ep1.json \
>   --mi355x-mtp0 runs/atom-mi355x-deepseek-r1-0528.json \
>   --mi355x-mtp3 runs/atom-mi355x-deepseek-r1-0528-mtp3.json \
>   --md
> ```

### Chat (1K/1K)

| Conc | B200 mtp0 | B200 mtp3 | B200 Gain | 355X mtp0 | 355X mtp3 | 355X Gain | Winner |
|------|-----------|-----------|-----------|-----------|-----------|-----------|--------|
| 1 | 115.6 | 231.5 | +100.3% | - | - | - | B200 only |
| 4 | 402.2 | 740.5 | +84.1% | 324.7 | 602.0 | +85.4% | 355X |
| 8 | 784.4 | 1186.6 | +51.3% | 665.2 | 978.9 | +47.2% | B200 |
| 16 | 1274.5 | 1872.0 | +46.9% | 1019.5 | 1602.9 | +57.2% | 355X |
| 32 | 2085.7 | 3012.9 | +44.5% | 1757.2 | 2537.7 | +44.4% | B200 |
| 64 | 3290.2 | 4679.1 | +42.2% | 2928.4 | 3398.5 | +16.1% | B200 |
| 128 | 5198.7 | 6943.5 | +33.6% | 4475.6 | 5503.5 | +23.0% | B200 |
| 256 | - | - | - | 5970.6 | 7520.2 | +26.0% | 355X only |

### Reasoning (1K/8K)

| Conc | B200 mtp0 | B200 mtp3 | B200 Gain | 355X mtp0 | 355X mtp3 | 355X Gain | Winner |
|------|-----------|-----------|-----------|-----------|-----------|-----------|--------|
| 1 | 113.5 | 278.5 | +145.2% | - | - | - | B200 only |
| 4 | 411.2 | 893.3 | +117.2% | 350.1 | 702.5 | +100.6% | B200 |
| 8 | 789.0 | 1357.8 | +72.1% | 647.8 | 1117.8 | +72.5% | 355X |
| 16 | 1277.4 | 2088.0 | +63.5% | 1154.4 | 1920.8 | +66.4% | 355X |
| 32 | 2055.1 | 3274.6 | +59.3% | 1782.4 | 3207.9 | +80.0% | 355X |
| 64 | 3098.2 | 5063.5 | +63.4% | 3005.5 | 4904.7 | +63.2% | B200 |
| 128 | 4966.2 | 7867.8 | +58.4% | 4712.3 | 7729.5 | +64.0% | 355X |

### Summarize (8K/1K)

| Conc | B200 mtp0 | B200 mtp3 | B200 Gain | 355X mtp0 | 355X mtp3 | 355X Gain | Winner |
|------|-----------|-----------|-----------|-----------|-----------|-----------|--------|
| 1 | 109.6 | 248.2 | +126.5% | - | - | - | B200 only |
| 4 | 371.0 | 680.6 | +83.4% | 286.4 | 565.1 | +97.3% | 355X |
| 8 | 674.0 | 978.5 | +45.2% | 570.5 | 778.5 | +36.5% | B200 |
| 16 | 996.4 | 1339.5 | +34.4% | 826.3 | 1138.8 | +37.8% | 355X |
| 32 | 1437.0 | 1882.2 | +31.0% | 1257.3 | 1644.5 | +30.8% | B200 |
| 64 | 1983.7 | 2388.1 | +20.4% | 1671.0 | 2177.7 | +30.3% | 355X |
| 128 | 2503.5 | 2816.6 | +12.5% | 2235.7 | 2646.3 | +18.4% | 355X |
| 256 | - | - | - | 2599.3 | 2955.0 | +13.7% | 355X only |

### Scoreboard

| | B200 | 355X | Total |
|------|------|------|-------|
| TPS gain winner | 8 | 10 | 18 |

## 分析

### 1. DAR 差距因场景而异——解释了 TPS 表现

**DAR 对比总览：**

| 场景 | B200 DAR | 355X DAR 范围 | DAR 差距 | B200 TPS 领先 (MTP3) |
|------|----------|-------------|---------|---------------------|
| chat (1K/1K) | 80.2% | 48-55% | **B200 大幅领先** | +17~38% |
| reasoning (1K/8K) | 73.8% | 63-69% | **差距较小** | +1.8~27% |
| summarize (8K/1K) | 72.9% | 54-64% | B200 中等领先 | +6.5~20% |

**核心发现：** 355X reasoning DAR (65%) 接近 B200 (74%)，差距仅 ~12%。这直接解释了为什么 reasoning 场景 355X TPS 几乎追平 B200（c=128 仅差 1.8%）。之前误将 chat-only 的 49% 当作全局 DAR，严重低估了 355X 在 reasoning 场景的 MTP 效果。

**B200 领先幅度变化（MTP0→MTP3）：**

| 场景 | 并发 | B200 领先 (MTP0) | B200 领先 (MTP3) | 领先变化 | 355X DAR |
|------|------|-----------------|-----------------|---------|----------|
| chat | c=4 | +23.9% | +23.0% | -0.9% | 55.1% |
| chat | c=64 | +12.4% | +37.7% | +25.3% | 48.6% |
| chat | c=128 | +16.2% | +26.2% | +10.0% | 48.8% |
| reasoning | c=32 | +15.3% | +2.1% | **-13.2%** | 65.4% |
| reasoning | c=128 | +5.4% | +1.8% | **-3.6%** | 62.9% |
| summarize | c=64 | +18.7% | +9.7% | **-9.1%** | 57.6% |

**规律：** B200 领先缩小的点（reasoning/summarize）恰好是 355X DAR 较高的场景（55-65%），与 B200 DAR（73-80%）差距较小。B200 领先扩大的点（chat 高并发）恰好是 355X DAR 最低的场景（48-49%），B200 DAR 优势最大。**DAR 差距与 TPS 领先变化高度一致。**

### 2. DAR 数据来源与可信度

**B200 DAR 来源：** `trtllm-bench throughput` 离线基准测试，200 requests，avg_concurrent=30。这是**离线环境**，不经过 HTTP serving 路径，scheduler 行为与实际 serving 不同。B200 有 3 个场景的 DAR 数据，但**每个场景仅单一值**（不区分并发）。

**355X DAR 来源：** ATOM CI benchmark run（#23408094038），覆盖全部 3 个场景 × 7 个并发级别 = **21 个 per-concurrency DAR 数据点**。从 CI job log 中的 `[MTP Stats]` 行提取。

**可比性分析：**
1. B200 DAR 为离线值，不代表 serving 实际行为；355X DAR 直接来自 serving benchmark，更贴近真实
2. 355X per-concurrency DAR 波动仅 ±3%（同场景内），说明 DAR 主要由模型+场景决定，并发影响有限
3. B200 每场景仅一个 DAR 值（离线 avg_conc≈30），缺少 per-concurrency 数据

**DAR 效率分析（实际 TPS 增益 / 理论最大增益）：**

| 场景 | 并发 | B200 DAR | B200 效率 | 355X DAR | 355X 效率 |
|------|------|----------|----------|----------|----------|
| chat | c=4 | 80.2% | 35% | 55.1% | 52% |
| chat | c=8 | 80.2% | 21% | 49.8% | 32% |
| chat | c=32 | 80.2% | 18% | 49.2% | 30% |
| chat | c=64 | 80.2% | 18% | 48.6% | 11% |
| chat | c=128 | 80.2% | 14% | 48.8% | 16% |
| reasoning | c=4 | 73.8% | 53% | 68.5% | 49% |
| reasoning | c=32 | 73.8% | 27% | 65.4% | 41% |
| reasoning | c=128 | 73.8% | 26% | 62.9% | 33% |
| summarize | c=4 | 72.9% | 38% | 63.6% | 52% |
| summarize | c=64 | 72.9% | 9% | 57.6% | 17% |
| summarize | c=128 | 72.9% | 6% | 53.9% | 10% |

> 效率 = 实际 TPS 增益 / (DAR × MTP_layers)。B200 MTP3 理论最大增益 = acceptance_len（~3.4x），355X MTP3 理论最大增益 = avg_toks_fwd × 对应 DAR。

**两个平台效率都偏低（6-53%）。** 高并发时效率急剧下降，说明 MTP 的瓶颈不在 DAR，而在 serving 路径的执行效率（scheduler overhead、KV cache 管理、batch 拼接）。

### 3. 高并发增益衰减的场景差异

**Chat/Summarize（短 output）：** 355X 高并发增益衰减明显，B200 衰减更慢。
- chat c=64: B200 +42.2% vs 355X +16.1%

**Reasoning（长 output）：** 两者衰减程度几乎相同。
- reasoning c=64: B200 +63.4% vs 355X +63.2%
- reasoning c=128: B200 +58.4% vs 355X +64.0%

**解释：** Reasoning 的 decode 步数约 8K 步，MTP speculative 的加速效果能充分摊薄高并发带来的 scheduling overhead。Chat/Summarize 只有 ~1K 步 decode，overhead 占比更大。

### 4. B200 绝对性能领先幅度

MTP3 后 B200 仍在所有可比点领先，但 reasoning 场景的领先幅度极小：

| 场景 | 并发 | B200 mtp3 | 355X mtp3 | B200 领先 |
|------|------|----------|----------|----------|
| reasoning | c=128 | 7868 | 7730 | +1.8% |
| reasoning | c=32 | 3275 | 3208 | +2.1% |
| chat | c=128 | 6944 | 5504 | +26.2% |
| summarize | c=128 | 2817 | 2646 | +6.5% |

Reasoning 场景 355X 几乎追平 B200。原因已确认：**355X reasoning DAR (63-69%) 接近 B200 (74%)**，DAR 差距仅 ~12%，不足以拉开 TPS 差距。Chat 场景 B200 领先更多是因为 DAR 差距大（80% vs 49%，差 63%）。

## DAR 原始数据

### B200（trtllm-bench 离线测试，200 requests，avg_conc≈30）

| 场景 | DAR avg | DAR p50 | Acc Len avg | Acc Len p50 |
|------|---------|---------|-------------|-------------|
| chat | 80.2% | 83.2% | 3.41 / 4 | 3.49 / 4 |
| reasoning | 73.8% | 73.4% | 3.21 / 4 | 3.20 / 4 |
| summarize | 72.9% | 73.7% | 3.19 / 4 | 3.21 / 4 |

### 355X（ATOM CI benchmark run #23408094038，per-scenario per-concurrency）

| 场景 | Conc | DAR avg | avg_toks_fwd |
|------|------|---------|-------------|
| chat | c=4 | 55.1% | 2.66 |
| chat | c=8 | 49.8% | 2.49 |
| chat | c=16 | 48.0% | 2.44 |
| chat | c=32 | 49.2% | 2.48 |
| chat | c=64 | 48.6% | 2.46 |
| chat | c=128 | 48.8% | 2.46 |
| chat | c=256 | 48.6% | 2.46 |
| reasoning | c=4 | 68.5% | 3.06 |
| reasoning | c=8 | 64.8% | 2.94 |
| reasoning | c=16 | 65.3% | 2.96 |
| reasoning | c=32 | 65.4% | 2.96 |
| reasoning | c=64 | 64.2% | 2.93 |
| reasoning | c=128 | 62.9% | 2.89 |
| reasoning | c=256 | 63.2% | 2.90 |
| summarize | c=4 | 63.6% | 2.91 |
| summarize | c=8 | 55.6% | 2.67 |
| summarize | c=16 | 54.9% | 2.65 |
| summarize | c=32 | 54.7% | 2.64 |
| summarize | c=64 | 57.6% | 2.73 |
| summarize | c=128 | 53.9% | 2.62 |
| summarize | c=256 | 54.1% | 2.62 |

**DAR 按场景差异显著：** reasoning (63-69%) >> summarize (54-64%) > chat (48-55%)。同场景内并发变化影响有限（±3%），但 chat c=4 是异常高点（55.1%），可能因为低并发下 batch 更小、speculation 更准确。

> 数据来源：ATOM CI run #23408094038（commit 613e57af，2026-03-22）。该 run 整体 conclusion=failure（reasoning c=256 单个 job 失败），但其余 job 均为 success。DAR 从各 MTP3 job log 的 `[MTP Stats]` 累计行提取。

## 下一步行动

### P0：实测 B200 serving 路径下的 per-concurrency DAR

B200 DAR 来自 trtllm-bench 离线环境（每场景仅单一值），而 355X 已有 21 个 per-concurrency 数据点。B200 是否在高并发 serving 下 DAR 衰减，是当前最大的未知数。

**具体行动：** 关注 TRT-LLM 后续版本是否在 `/perf_metrics` 中暴露 DAR 指标；或在 server log 中寻找 acceptance rate 输出。当前 rc6 的 `/perf_metrics` 不返回 speculative_decoding 指标。

### P1：提升 MTP 执行效率

两个平台的 DAR 利用效率都偏低（6-53%）。DAR 本身不是瓶颈（355X reasoning DAR 65% 已接近 B200 的 74%），瓶颈在 serving 路径将 DAR 转化为 TPS 的效率。

**优化方向：**
1. 降低 MTP step 的 scheduling overhead（尤其是高并发场景）
2. 优化 KV cache 管理（speculative tokens 的 cache 分配/回收）
3. 重点关注 chat/summarize 短 output 场景的高并发衰减（decode 步数少，overhead 占比大）

## 待验证问题

- [x] 355X 高并发 MTP 增益衰减 — 场景相关：chat/summarize 衰减明显，reasoning 不受影响
- [x] DAR 数据 — 355X 已补全 3 场景 × 7 并发 = 21 个数据点；B200 有 3 场景离线值
- [x] 355X DAR 是否为全局值 — **否，per-scenario 差异显著**：reasoning 65% >> chat 49%
- [x] DAR 是否随并发变化 — 355X 同场景内并发波动仅 ±3%，DAR 主要由场景决定
- [x] 355X reasoning/summarize DAR — **已获取**：reasoning 63-69%, summarize 54-64%
- [ ] **P0: B200 serving 时的 per-concurrency DAR**（当前仅离线单一值，`/perf_metrics` 暂不支持）
- [ ] EP>1 配置下 MTP 收益差异（B200 EP=8 MTP3 有数据，EP=8 MTP0 缺失）

## 迭代日志

| 日期 | 变更 |
|------|------|
| 2026-03-23 v4 | **355X per-scenario per-concurrency DAR 补全**：从 ATOM CI run #23408094038 提取全部 21 个 DAR 数据点。核心发现：reasoning DAR 65% 接近 B200 的 74%，解释了 reasoning TPS 接近的现象。重写全部 DAR 分析章节，消除"矛盾"结论 |
| 2026-03-23 v3 | 修正关键错误：355X DAR 49% 并非全局值，而是仅 chat 1024/1024 c=128。重构 DAR 矛盾分析为 chat-only 同场景对比，标注 reasoning/summarize DAR 对比无效。重排优先级（P0→补全 355X 多场景 DAR，P1→B200 serving DAR） |
| 2026-03-23 v2 | 重写分析：发现 DAR 与 TPS 增益的逻辑矛盾，B200 高 DAR 未转化为预期优势。增加 DAR 效率分析、B200 领先变化追踪。修正 import_results.py DAR 提取 bug |
| 2026-03-23 v1 | 数据更新：355X 数据修正，新增 reasoning c=64/128/256，DAR 数据。记分牌从 5:11 变为 8:10 |
| 2026-03-20 | 初版分析，基于 2026-03-19 数据，16 个可比数据点 |
