# MTP3 vs MTP0 收益分析：B200 vs MI355X

> **Last updated:** 2026-03-23
> **Data date:** 2026-03-21/22 (B200), 2026-03-23 (355X)
> **Model:** DeepSeek-R1-0528 FP8, 8-GPU TP=8
> **B200 framework:** TRT-LLM 1.2.0rc6（NVIDIA TensorRT-LLM，C++ runtime + speculative decoding）
> **MI355X framework:** ATOM (commit 613e57af)（ROCm/ATOM，基于 vLLM 的 PyTorch serving 后端）

## DAR 跨平台对比的方法论限制

> **重要：** B200 (TRT-LLM) 与 355X (ATOM) 的 DAR 对比是**工程选型参考**，不代表同一投机算法下的理论接受率对比。以下因素导致两者 DAR 不严格可比：
>
> 1. **执行栈不同：** TRT-LLM 使用 C++ runtime + 自有调度器 + CUDA kernel；ATOM 基于 vLLM PyTorch 路径 + ROCm HIP kernel。调度策略、采样实现、投机解码的验证/拒绝逻辑均可能不同。
> 2. **DAR 采集环境不同：** B200 DAR 来自 `trtllm-bench` 离线 LLM API（非 HTTP serving），每场景仅单一值；355X DAR 来自 ATOM CI 的 serving benchmark，per-concurrency 采集。
> 3. **采样与确定性：** 两个 benchmark 均使用随机输入 token（SemiAnalysis InferenceX 方法论），但采样参数（greedy vs top-k/top-p）、random seed 未对齐。FP8 量化在不同硬件上的精度差异也可能影响 token 预测分布，进而影响 DAR。
> 4. **MTP 配置：** 两者均为 MTP3（3 个 draft token），但投机模型的实现细节（draft head 架构、权重加载方式）可能不同。
>
> **结论限定：** 本报告的 DAR 对比反映的是**两个完整系统（硬件 + 框架 + 配置）在相同 workload 下的端到端接受率差异**。DAR 差距来自后端实现与调度的综合影响，不代表单一算法的优劣。

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

### Chat (1K/1K) — B200 DAR: 80.2%

| Conc | B200 MTP3 | B200 Gain | 355X MTP3 | 355X Gain | 355X DAR | B200 领先 | Winner |
|------|----------|-----------|----------|-----------|----------|----------|--------|
| 1 | 231.5 | +100.3% | - | - | - | - | B200 only |
| 4 | 740.5 | +84.1% | 602.0 | +85.4% | 55.1% | +23.0% | 355X |
| 8 | 1186.6 | +51.3% | 978.9 | +47.2% | 49.8% | +21.2% | B200 |
| 16 | 1872.0 | +46.9% | 1602.9 | +57.2% | 48.0% | +16.8% | 355X |
| 32 | 3012.9 | +44.5% | 2537.7 | +44.4% | 49.2% | +18.7% | B200 |
| 64 | 4679.1 | +42.2% | 3398.5 | +16.1% | 48.6% | +37.7% | B200 |
| 128 | 6943.5 | +33.6% | 5503.5 | +23.0% | 48.8% | +26.2% | B200 |
| 256 | - | - | 7520.2 | +26.0% | 48.6% | - | 355X only |

### Reasoning (1K/8K) — B200 DAR: 73.8%

| Conc | B200 MTP3 | B200 Gain | 355X MTP3 | 355X Gain | 355X DAR | B200 领先 | Winner |
|------|----------|-----------|----------|-----------|----------|----------|--------|
| 1 | 278.5 | +145.2% | - | - | - | - | B200 only |
| 4 | 893.3 | +117.2% | 702.5 | +100.6% | 68.5% | +27.2% | B200 |
| 8 | 1357.8 | +72.1% | 1117.8 | +72.5% | 64.8% | +21.5% | 355X |
| 16 | 2088.0 | +63.5% | 1920.8 | +66.4% | 65.3% | +8.7% | 355X |
| 32 | 3274.6 | +59.3% | 3207.9 | +80.0% | 65.4% | +2.1% | 355X |
| 64 | 5063.5 | +63.4% | 4904.7 | +63.2% | 64.2% | +3.2% | B200 |
| 128 | 7867.8 | +58.4% | 7729.5 | +64.0% | 62.9% | +1.8% | 355X |

### Summarize (8K/1K) — B200 DAR: 72.9%

| Conc | B200 MTP3 | B200 Gain | 355X MTP3 | 355X Gain | 355X DAR | B200 领先 | Winner |
|------|----------|-----------|----------|-----------|----------|----------|--------|
| 1 | 248.2 | +126.5% | - | - | - | - | B200 only |
| 4 | 680.6 | +83.4% | 565.1 | +97.3% | 63.6% | +20.4% | 355X |
| 8 | 978.5 | +45.2% | 778.5 | +36.5% | 55.6% | +25.7% | B200 |
| 16 | 1339.5 | +34.4% | 1138.8 | +37.8% | 54.9% | +17.6% | 355X |
| 32 | 1882.2 | +31.0% | 1644.5 | +30.8% | 54.7% | +14.4% | B200 |
| 64 | 2388.1 | +20.4% | 2177.7 | +30.3% | 57.6% | +9.7% | 355X |
| 128 | 2816.6 | +12.5% | 2646.3 | +18.4% | 53.9% | +6.4% | 355X |
| 256 | - | - | 2955.0 | +13.7% | 54.1% | - | 355X only |

### Scoreboard

| | B200 | 355X | Total |
|------|------|------|-------|
| MTP3 增益更大 | 8 | 10 | 18 |

## 分析

### 1. DAR 差距因场景而异——解释了 TPS 表现

| 场景 | B200 DAR | 355X DAR | DAR 差距 | B200 TPS 领先 (MTP3) |
|------|----------|----------|---------|---------------------|
| chat | 80.2% | 48-55% | **大** | +17~38% |
| reasoning | 73.8% | 63-69% | **小** | +1.8~27% |
| summarize | 72.9% | 54-64% | 中 | +6.5~26% |

**规律：** DAR 差距越大的场景，B200 TPS 领先越多。Reasoning 场景 DAR 差距仅 ~12%（74% vs 65%），355X 几乎追平 B200（c=128 仅差 1.8%）。Chat 场景 DAR 差距 ~63%（80% vs 49%），B200 高并发领先高达 38%。

高并发下 B200 在 chat 的领先反而扩大（c=64 +37.7%），因为 355X 低 DAR 场景 MTP 增益衰减更快（chat c=64 仅 +16.1%）。

### 2. DAR 数据来源与可比性

> 跨平台 DAR 对比的方法论限制见文档开头。

**B200 DAR 来源：** `trtllm-bench throughput` 离线基准测试（TRT-LLM C++ runtime，非 HTTP serving），200 requests，avg_concurrent=30。每场景仅单一值（不区分并发）。采样方式：随机输入 token，greedy/top-k 参数未记录。

**355X DAR 来源：** ATOM CI benchmark run #23408094038（vLLM PyTorch runtime，HTTP serving 路径），覆盖 3 场景 × 7 并发 = **21 个数据点**。从 job log `[MTP Stats]` 累计行提取。采样方式：随机输入 token，参数由 ATOM CI 配置决定。

**同平台内的可信度：**
1. 355X per-concurrency DAR 波动仅 ±3%（同场景内），说明 DAR 主要由模型+场景决定，与并发关系不大
2. B200 每场景仅一个离线值，无法验证是否随并发变化

**跨平台对比的局限性：**
1. 采集环境不同（离线 vs serving），不能排除环境差异对 DAR 的影响
2. TRT-LLM 与 ATOM/vLLM 的投机解码实现（验证/拒绝逻辑、batch 管理）不同，DAR 差异可能部分来自框架实现而非硬件能力
3. 本报告的 DAR 差距（如 chat: 80% vs 49%）反映的是完整系统差异，不能简单归因于"硬件更强"或"算法更优"

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

MTP3 后 B200 仍在所有可比点领先（见数据表格的「B200 领先」列），但 reasoning 场景领先幅度极小（c=128 仅 +1.8%，c=32 仅 +2.1%），与 DAR 差距小（74% vs 65%）一致。Chat 场景 B200 领先更多（c=128 +26.2%），与 DAR 差距大（80% vs 49%）一致。

## DAR 原始数据

### B200（TRT-LLM C++ runtime，trtllm-bench 离线测试，200 requests，avg_conc≈30）

| 场景 | DAR avg | DAR p50 | Acc Len avg | Acc Len p50 |
|------|---------|---------|-------------|-------------|
| chat | 80.2% | 83.2% | 3.41 / 4 | 3.49 / 4 |
| reasoning | 73.8% | 73.4% | 3.21 / 4 | 3.20 / 4 |
| summarize | 72.9% | 73.7% | 3.19 / 4 | 3.21 / 4 |

### 355X（ATOM/vLLM PyTorch runtime，CI serving benchmark #23408094038）

355X per-concurrency DAR 已整合到上方各场景数据表格的「355X DAR」列。汇总：

| 场景 | DAR 范围 | avg_toks_fwd 范围 | 特征 |
|------|---------|------------------|------|
| chat | 48-55% | 2.44-2.66 | 最低，高并发稳定 ~49% |
| reasoning | 63-69% | 2.89-3.06 | 最高，接近 B200 |
| summarize | 54-64% | 2.62-2.91 | 中等，c=4 异常高（63.6%） |

同场景内并发波动 ±3%，DAR 主要由场景决定。

> 数据来源：ATOM CI run #23408094038（commit 613e57af，2026-03-22）。该 run 整体 conclusion=failure（reasoning c=256 单个 job 失败），但其余 job 均为 success。

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
| 2026-03-23 v6 | 精简表格：主表增加 355X DAR 和 B200 领先列，去掉 mtp0 绝对值列；删除分析中的重复表格（DAR 对比总览、领先变化表、B200 领先表、355X DAR 完整 21 行表） |
| 2026-03-23 v5 | 增加跨平台 DAR 对比的方法论限制章节：明确 TRT-LLM (C++ runtime) vs ATOM (vLLM PyTorch runtime) 的执行栈差异、采集环境差异、采样/确定性问题。限定结论为工程选型参考，非严格算法对比 |
| 2026-03-23 v4 | **355X per-scenario per-concurrency DAR 补全**：从 ATOM CI run #23408094038 提取全部 21 个 DAR 数据点。核心发现：reasoning DAR 65% 接近 B200 的 74%，解释了 reasoning TPS 接近的现象。重写全部 DAR 分析章节，消除"矛盾"结论 |
| 2026-03-23 v3 | 修正关键错误：355X DAR 49% 并非全局值，而是仅 chat 1024/1024 c=128。重构 DAR 矛盾分析为 chat-only 同场景对比，标注 reasoning/summarize DAR 对比无效。重排优先级（P0→补全 355X 多场景 DAR，P1→B200 serving DAR） |
| 2026-03-23 v2 | 重写分析：发现 DAR 与 TPS 增益的逻辑矛盾，B200 高 DAR 未转化为预期优势。增加 DAR 效率分析、B200 领先变化追踪。修正 import_results.py DAR 提取 bug |
| 2026-03-23 v1 | 数据更新：355X 数据修正，新增 reasoning c=64/128/256，DAR 数据。记分牌从 5:11 变为 8:10 |
| 2026-03-20 | 初版分析，基于 2026-03-19 数据，16 个可比数据点 |
