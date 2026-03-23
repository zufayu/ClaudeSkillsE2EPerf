# MTP3 vs MTP0 收益分析：B200 vs MI355X

> **Last updated:** 2026-03-23
> **Data date:** 2026-03-21/22 (B200), 2026-03-23 (355X)
> **Model:** DeepSeek-R1-0528 FP8, 8-GPU TP=8
> **B200 framework:** TRT-LLM 1.2.0rc6 | **MI355X framework:** ATOM (613e57af)

## 核心结论

1. **记分牌 8:10 (B200:355X)**，355X 在 18 个可比点中拿到更多相对增益。但这个指标意义有限——问题出在为什么 B200 的高 DAR 没有转化为预期优势。
2. **B200 chat DAR (80%) 远高于 355X chat DAR (49%)，但 MTP3 后 B200 的性能领先并未扩大，反而在多数点缩小。** 对于 chat 场景这是一个严格的同场景对比矛盾。
3. **关键发现：355X DAR 49% 仅来自 chat 1024/1024 c=128，并非全局值。** ATOM CI 仅对该单一配置运行 MTP benchmark，reasoning/summarize 场景的 355X DAR 完全未知。因此 reasoning/summarize 的 DAR 对比分析无效。
4. 两个不可比因素叠加：(a) B200 DAR 是离线 trtllm-bench 测得，不代表 serving 时的实际 DAR；(b) 355X DAR 仅覆盖 chat 场景。**下一步：B200 需在 serving 路径实测 DAR，355X 需扩展 MTP benchmark 覆盖 reasoning/summarize。**

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

### 1. DAR 与 TPS 增益的矛盾（核心问题）

**现象：** B200 chat DAR (80.2%) 远高于 355X chat DAR (49.0%)，理论上 B200 从 MTP3 获得的加速应该更大。但实际数据显示 355X 在多数 chat 可比点的相对增益反而更高或接近。

**量化矛盾（仅 chat 场景有效，因为 355X 仅有 chat DAR 数据）：**

B200 chat 每步有效 token = 3.41 (acceptance_len)，355X chat = 2.47 (avg_toks_fwd)。如果两者 MTP 执行开销相同，B200 的 MTP3 加速应比 355X 多 38% (3.41/2.47)。即：如果 MTP0 下 B200 领先 355X 20%，MTP3 下应领先 ~58%。

实际数据（chat 场景）：

| 并发 | B200 领先 (MTP0) | B200 领先 (MTP3) | 领先变化 |
|------|-----------------|-----------------|---------|
| c=4 | +23.9% | +23.0% | -0.9% |
| c=8 | +17.9% | +21.2% | +3.3% |
| c=16 | +25.0% | +16.8% | -8.2% |
| c=32 | +18.7% | +18.7% | 0.0% |
| c=64 | +12.4% | +37.7% | +25.3% |
| c=128 | +16.2% | +26.2% | +10.0% |

**Chat 场景混合结果：** 低并发（c=4/16）B200 领先缩小，但高并发（c=64/128）B200 领先反而扩大。这与 B200 DAR 80% vs 355X 49% 的预期一致——在高并发下 355X 的 MTP 增益衰减更快（chat c=64: B200 +42.2% vs 355X +16.1%），B200 的高 DAR 优势在此显现。

**跨场景领先变化（reasoning/summarize 的 DAR 对比无效，仅记录现象）：**

| 场景 | 并发 | B200 领先 (MTP0) | B200 领先 (MTP3) | 领先变化 |
|------|------|-----------------|-----------------|---------|
| reasoning | c=32 | +15.3% | +2.1% | -13.2% |
| reasoning | c=128 | +5.4% | +1.8% | -3.6% |
| summarize | c=64 | +18.7% | +9.7% | -9.1% |

Reasoning/summarize 场景 B200 领先幅度在 MTP3 后普遍缩小，但由于**缺少 355X 这两个场景的 DAR 数据**，无法判断这是因为 355X reasoning/summarize DAR 高于 chat 的 49%，还是其他原因（如 B200 serving DAR 低于离线值）。

### 2. DAR 数据的可信度分析

**B200 DAR 来源：** `trtllm-bench throughput` 离线基准测试，200 requests，avg_concurrent=30，固定 ISL/OSL=1024/1024。这是**离线环境**，不经过 HTTP serving 路径，scheduler 行为与实际 serving 不同。B200 有 3 个场景的 DAR 数据（chat/reasoning/summarize），但均为离线测量。

**355X DAR 来源：** ATOM CI 的 MTP benchmark，**仅 chat 1024/1024 c=128 场景**（非全局值）。经查验 ATOM CI 最近 5 次 benchmark run，MTP benchmark 仅测试 `isl=1024 osl=1024 c=128` 配置。Reasoning (1K/8K) 和 Summarize (8K/1K) 场景**无 DAR 数据**。

**两个 DAR 的局限性：**
1. B200 DAR 在离线 batch 环境测得，验证/拒绝的动态行为可能与 serving 不同
2. 355X DAR 仅覆盖 chat 场景——reasoning 场景（长 output、重复 pattern 多）的 DAR 可能远高于 49%，也可能更低，**完全未知**
3. B200 chat DAR 80% 但实际 chat TPS 增益仅 33-84%（理论应 241%），说明**即使 DAR 数据正确，高并发下 MTP 的执行效率也远低于理论值**

**DAR 效率分析（实际增益 / 理论最大增益）：**

仅 chat 场景可做有效对比（两者均有 DAR 数据）：

| 场景 | 并发 | B200 效率 | 355X 效率 |
|------|------|----------|----------|
| chat | c=4 | 35% | 58% |
| chat | c=8 | 21% | 32% |
| chat | c=32 | 18% | 30% |
| chat | c=64 | 18% | 11% |
| chat | c=128 | 14% | 16% |

低并发（c=4~32）355X 效率高于 B200；高并发（c=64）B200 效率反超。这与 chat c=64 355X MTP 增益大幅衰减（+16.1%）一致。

Reasoning/summarize 场景的 355X 效率无法计算（缺少 DAR），以下仅展示 B200：

| 场景 | 并发 | B200 效率 |
|------|------|----------|
| reasoning | c=4 | 53% |
| reasoning | c=32 | 27% |
| summarize | c=128 | 6% |

B200 的 DAR 利用效率整体偏低（6-53%），说明**即使离线 DAR 数据正确，serving 路径的 MTP 执行也存在大量效率损失**。可能原因：serving scheduler overhead、KV cache 管理、batch 拼接开销等。

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

Reasoning 场景 355X 几乎追平 B200。由于 355X reasoning DAR 未知，无法确认原因。两种假设：(a) 355X reasoning DAR 远高于 chat 的 49%（长 output 重复 pattern 多→更高接受率），(b) B200 serving 时 reasoning DAR 低于离线测的 73.8%。两者可能同时存在。

## DAR 原始数据

### B200（trtllm-bench 离线测试，200 requests，avg_conc≈30）

| 场景 | DAR avg | DAR p50 | Acc Len avg | Acc Len p50 |
|------|---------|---------|-------------|-------------|
| chat | 80.2% | 83.2% | 3.41 / 4 | 3.49 / 4 |
| reasoning | 73.8% | 73.4% | 3.21 / 4 | 3.20 / 4 |
| summarize | 72.9% | 73.7% | 3.19 / 4 | 3.21 / 4 |

### 355X（ATOM CI MTP benchmark，仅 chat 1024/1024 c=128）

| 指标 | 值 | 备注 |
|------|-----|------|
| DAR avg | 49.0% | 仅 chat 场景 |
| avg_toks_fwd | 2.47 / 3 | 仅 chat 场景 |
| reasoning DAR | **未知** | ATOM CI 未测试 |
| summarize DAR | **未知** | ATOM CI 未测试 |

DAR 分布（0/1/2/3 accepted，chat 场景）：22.5% / 32.9% / 19.9% / 24.7%

> **注意：** 经查验 ATOM CI 最近 5 次 benchmark run（2026-03-17 ~ 2026-03-23），MTP benchmark 均仅测试 `isl=1024 osl=1024 c=128`（对应 chat 场景）。`fetch_competitors.py` 已支持 per-scenario DAR 注入，但上游 CI 不提供 reasoning/summarize 的 MTP benchmark 数据。

## 下一步行动

### P0：补全 355X reasoning/summarize DAR

当前 355X DAR 仅覆盖 chat 场景，reasoning/summarize 完全缺失。这导致跨场景对比分析无法进行。

**具体行动：** 在 ATOM CI 中增加 reasoning (1K/8K) 和 summarize (8K/1K) 的 MTP benchmark 配置，或在 355X 实机上手动运行 MTP benchmark 获取这两个场景的 DAR。`fetch_competitors.py` 已支持 per-scenario DAR 注入，只需上游 CI 提供数据。

### P1：实测 B200 serving 路径下的 DAR

B200 DAR 来自 trtllm-bench 离线环境，不代表 HTTP serving 时的实际行为。当前 B200 TRT-LLM rc6 的 `/perf_metrics` 不返回 speculative_decoding 指标，暂不支持 serving 路径 DAR 采集。

**具体行动：** 关注 TRT-LLM 后续版本是否在 `/perf_metrics` 中暴露 DAR 指标；或在 server log 中寻找 acceptance rate 相关输出。

### P2：验证 DAR 是否随并发变化

理论上，高并发时 batch 变大，MTP 的 draft token 验证效率可能下降（KV cache 压力、attention 计算量增大）。如果 B200 serving DAR 在高并发时大幅低于离线测的 80%，就能解释为什么高 DAR 没有转化为高 TPS 增益。355X 同理——chat c=64 MTP 增益衰减到 +16.1%，可能是高并发 DAR 下降导致。

### P2：优化方向

根据当前数据的一个确定性结论：**两个平台都没有充分利用 MTP 的理论加速空间**（B200 效率 6-53%，355X chat 效率 11-58%）。最大的优化空间在于：
1. 降低 MTP step 的 overhead（scheduling、KV cache 管理）
2. 提高高并发场景下的 DAR 维持能力
3. 重点关注 chat/summarize 场景的高并发衰减

## 待验证问题

- [x] 355X 高并发 MTP 增益衰减 — 场景相关：chat/summarize 衰减明显，reasoning 不受影响
- [x] DAR 数据 — 已收集但存在局限性（B200 离线、355X 仅 chat）
- [x] 355X DAR 是否为全局值 — **否，仅 chat 1024/1024 c=128**（ATOM CI 仅测试该配置）
- [ ] **P0: 355X reasoning/summarize DAR**（ATOM CI 未测试，需扩展 MTP benchmark 配置）
- [ ] **P1: B200 serving 时的实际 DAR**（当前 80% 来自离线 trtllm-bench，`/perf_metrics` 暂不支持）
- [ ] P2: DAR 是否随并发级别变化
- [ ] EP>1 配置下 MTP 收益差异（B200 EP=8 MTP3 有数据，EP=8 MTP0 缺失）

## 迭代日志

| 日期 | 变更 |
|------|------|
| 2026-03-23 v3 | 修正关键错误：355X DAR 49% 并非全局值，而是仅 chat 1024/1024 c=128。重构 DAR 矛盾分析为 chat-only 同场景对比，标注 reasoning/summarize DAR 对比无效。重排优先级（P0→补全 355X 多场景 DAR，P1→B200 serving DAR） |
| 2026-03-23 v2 | 重写分析：发现 DAR 与 TPS 增益的逻辑矛盾，B200 高 DAR 未转化为预期优势。增加 DAR 效率分析、B200 领先变化追踪。修正 import_results.py DAR 提取 bug |
| 2026-03-23 v1 | 数据更新：355X 数据修正，新增 reasoning c=64/128/256，DAR 数据。记分牌从 5:11 变为 8:10 |
| 2026-03-20 | 初版分析，基于 2026-03-19 数据，16 个可比数据点 |
