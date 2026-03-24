# MTP3 vs MTP0 收益分析：B200 vs MI355X

> **Last updated:** 2026-03-24
> **Data date:** 2026-03-21/22 (B200), 2026-03-23 (355X)
> **Model:** DeepSeek-R1-0528 FP8, 8-GPU TP=8
> **B200 framework:** TRT-LLM 1.2.0rc6.post3（NVIDIA TensorRT-LLM，C++ runtime + speculative decoding）
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

1. **MTP0 基线差距主要来自 TRT-LLM 版本迭代，而非硬件差异。** SA InferenceX 对标数据显示：配置完全一致时，B200 rc6.post2 ≈ 355X ATOM（差距 ±5%）。我们 B200 用 rc6.post3（3 个 MoE kernel 优化 PR），比 post2 快 ~15%。报告中的 MTP0 基线差距（3-33%）大部分是 framework 红利。
2. **355X MTP 表现优于 B200（记分牌 10:8）。** MTP 本身对 355X 是正面因素，尤其 reasoning 场景 MTP 增益（+64-80%）持续高于 B200（+58-63%），有效弥补基线差距（MTP0 差距 +15% → MTP3 差距 +1.8%）。
3. **Chat 场景是例外：DAR 差距叠加导致 MTP3 后 B200 领先反而扩大。** 355X chat DAR 仅 49%（vs B200 80%），MTP 增益受限。c=64 MTP0 差距仅 +12.4%，MTP3 后扩大到 +37.7%。
4. **DAR 差距因场景而异：** chat 差距大（80% vs 49%），reasoning 差距小（74% vs 65%）。这直接解释了 MTP3 在不同场景对性能差距的影响方向不同。

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

| Conc | B200 MTP0 | B200 MTP3 | B200 Gain | 355X MTP0 | 355X MTP3 | 355X Gain | 355X DAR | MTP0 差距 | MTP3 差距 | Winner |
|------|----------|----------|-----------|----------|----------|-----------|----------|----------|----------|--------|
| 1 | 115.6 | 231.5 | +100.3% | 86.7 | - | - | - | +33.2% | - | - |
| 4 | 402.2 | 740.5 | +84.1% | 324.7 | 602.0 | +85.4% | 55.1% | +23.9% | +23.0% | 355X |
| 8 | 784.4 | 1186.6 | +51.3% | 665.2 | 978.9 | +47.2% | 49.8% | +17.9% | +21.2% | B200 |
| 16 | 1274.5 | 1872.0 | +46.9% | 1019.5 | 1602.9 | +57.2% | 48.0% | +25.0% | +16.8% | 355X |
| 32 | 2085.7 | 3012.9 | +44.5% | 1757.2 | 2537.7 | +44.4% | 49.2% | +18.7% | +18.7% | B200 |
| 64 | 3290.2 | 4679.1 | +42.2% | 2928.5 | 3398.5 | +16.1% | 48.6% | +12.4% | +37.7% | B200 |
| 128 | 5198.7 | 6943.5 | +33.6% | 4475.6 | 5503.5 | +23.0% | 48.8% | +16.2% | +26.2% | B200 |
| 256 | 7706.9 | - | - | 5970.6 | 7520.2 | +26.0% | 48.6% | +29.1% | - | - |

### Reasoning (1K/8K) — B200 DAR: 73.8%

| Conc | B200 MTP0 | B200 MTP3 | B200 Gain | 355X MTP0 | 355X MTP3 | 355X Gain | 355X DAR | MTP0 差距 | MTP3 差距 | Winner |
|------|----------|----------|-----------|----------|----------|-----------|----------|----------|----------|--------|
| 1 | 113.5 | 278.5 | +145.2% | 96.3 | - | - | - | +17.9% | - | - |
| 4 | 411.2 | 893.3 | +117.2% | 350.1 | 702.5 | +100.6% | 68.5% | +17.4% | +27.2% | B200 |
| 8 | 789.0 | 1357.8 | +72.1% | 647.8 | 1117.8 | +72.5% | 64.8% | +21.8% | +21.5% | 355X |
| 16 | 1277.4 | 2088.0 | +63.5% | 1154.4 | 1920.8 | +66.4% | 65.3% | +10.6% | +8.7% | 355X |
| 32 | 2055.1 | 3274.6 | +59.3% | 1782.4 | 3207.9 | +80.0% | 65.4% | +15.3% | +2.1% | 355X |
| 64 | 3098.2 | 5063.5 | +63.4% | 3005.5 | 4904.7 | +63.2% | 64.2% | +3.1% | +3.2% | B200 |
| 128 | 4966.2 | 7867.8 | +58.4% | 4712.3 | 7729.5 | +64.0% | 62.9% | +5.4% | +1.8% | 355X |

### Summarize (8K/1K) — B200 DAR: 72.9%

| Conc | B200 MTP0 | B200 MTP3 | B200 Gain | 355X MTP0 | 355X MTP3 | 355X Gain | 355X DAR | MTP0 差距 | MTP3 差距 | Winner |
|------|----------|----------|-----------|----------|----------|-----------|----------|----------|----------|--------|
| 1 | 109.6 | 248.2 | +126.5% | 91.4 | - | - | - | +19.9% | - | - |
| 4 | 371.0 | 680.6 | +83.4% | 286.4 | 565.1 | +97.3% | 63.6% | +29.5% | +20.4% | 355X |
| 8 | 674.0 | 978.5 | +45.2% | 570.6 | 778.5 | +36.5% | 55.6% | +18.1% | +25.7% | B200 |
| 16 | 996.4 | 1339.5 | +34.4% | 826.3 | 1138.8 | +37.8% | 54.9% | +20.6% | +17.6% | 355X |
| 32 | 1437.0 | 1882.2 | +31.0% | 1257.3 | 1644.5 | +30.8% | 54.7% | +14.3% | +14.4% | B200 |
| 64 | 1983.7 | 2388.1 | +20.4% | 1671.0 | 2177.7 | +30.3% | 57.6% | +18.7% | +9.7% | 355X |
| 128 | 2503.5 | 2816.6 | +12.5% | 2235.7 | 2646.3 | +18.4% | 53.9% | +12.0% | +6.4% | 355X |
| 256 | 2889.7 | - | - | 2599.3 | 2955.0 | +13.7% | 54.1% | +11.2% | - | - |

### Scoreboard

| | B200 | 355X | Total |
|------|------|------|-------|
| MTP3 增益更大 | 8 | 10 | 18 |

## 分析

### 1. MTP0 基线差距是性能差异的主要来源

B200 在 MTP0（无投机解码）配置下始终领先 355X，这是两个平台性能差异的基础：

| 场景 | MTP0 差距范围 | MTP3 差距范围 | MTP3 后差距变化 |
|------|-------------|-------------|---------------|
| chat | +12~33% | +17~38% | 高并发扩大（DAR 差距叠加） |
| reasoning | +3~22% | +1.8~27% | 高并发显著缩小 |
| summarize | +11~30% | +6~26% | 普遍缩小 |

**关键发现：** 355X 在 MTP0 基线上落后 B200，但通过 MTP3 的更高相对增益，成功缩小了差距（尤其是 reasoning 和 summarize）。MTP 本身对 355X 是正面因素。

**Reasoning 场景最能说明问题：**
- c=32: MTP0 差距 +15.3% → MTP3 差距 +2.1%（差距缩小 86%）
- c=128: MTP0 差距 +5.4% → MTP3 差距 +1.8%（差距缩小 67%）
- 355X MTP 增益（+64-80%）持续高于 B200（+58-63%），有效弥补基线劣势

**Chat 场景是例外：** MTP0 差距（+12-25%）在 MTP3 后反而扩大（+17-38%），因为 355X chat DAR 仅 49%（vs B200 80%），MTP 无法有效转化。c=64 是极端案例：MTP0 差距仅 +12.4%，MTP3 后扩大到 +37.7%。

### 2. MTP0 基线差距的来源：TRT-LLM 版本迭代（已验证）

**SA InferenceX 对标数据证实：MTP0 基线差距主要来自 TRT-LLM 版本差异，而非硬件能力差异。**

#### SA 标准化测试对标（chat 1K/1K, 8×GPU, TP=8, FP8, MTP0）

SA InferenceX 使用**完全相同的 benchmark 配置**（`max_seq_len=8192`, `random_range_ratio=0.8`, `kv_cache_dtype=fp8`, `piecewise_cuda_graph=true`, `moe_backend=TRTLLM`），唯一差异是 docker 版本。

| Conc | SA B200 (rc6.post2) | 我们 B200 (rc6.post3) | 我们 355X (ATOM) | SA B200 vs 我们 355X |
|------|---------------------|----------------------|-----------------|---------------------|
| 8 | 694.5 | 784.4 | 665.2 | +4.4% |
| 64 | 2791.9 | 3290.2 | 2928.5 | **-4.7%** |
| 128 | 4461.7 | 5198.7 | 4475.6 | **-0.3%** |

> 数据来源：SA InferenceX B200 run (2026-01-29, rc6.post2, [workflow](https://github.com/SemiAnalysisAI/InferenceX/actions/workflows/run-sweep.yml))；355X 数据来自我们的 ATOM CI 提取。

**关键发现：**
1. **SA 的 B200 (rc6.post2) ≈ 我们的 355X (ATOM)**（差距在 ±5% 以内），在 SA 标准化测试中两平台基本打平
2. **我们的 B200 (rc6.post3) 比 SA 的 B200 (rc6.post2) 高 13-17%**，而 355X 数据高度一致（SA 561.9/gpu vs 我们 559.5/gpu，差 0.4%）
3. **在配置完全一致的前提下，~15% 的差距只能来自 TRT-LLM rc6.post2 → rc6.post3 的优化**

#### TRT-LLM rc6.post2 → rc6.post3 的优化内容

post2 (2026-01-22) → post3 (2026-02-05) 仅包含 **5 个 PR**，其中 3 个直接优化 MoE 性能，全部针对 DeepSeek：

| PR | 类型 | 内容 | 影响 |
|----|------|------|------|
| [#11143](https://github.com/NVIDIA/TensorRT-LLM/pull/11143) | **MoE 融合** | Shared Expert 与 Sparse Expert 融合为一次 grouped GEMM | **高** — 减少 kernel launch 和显存读写，DeepSeek 有 58 层 MoE |
| [#11104](https://github.com/NVIDIA/TensorRT-LLM/pull/11104) | **FP8 kernel** | 更新 FP8 MoE cubins + 优化 finalize kernel | **中高** — 更快的 MoE 计算 kernel |
| [#11160](https://github.com/NVIDIA/TensorRT-LLM/pull/11160) | **调度修复** | 修复 MoE cost estimation，改善 multi-stream 调度 | **中** — 更好的 CUDA stream 利用 |
| [#11174](https://github.com/NVIDIA/TensorRT-LLM/pull/11174) | **稳定性** | NCCL Symmetric fallback（修复 B200/B300 segfault） | 稳定性修复 |
| #11224 | 版本号 | Bump to post3 | 无 |

> DeepSeek-R1 有 58 个 MoE 层，97% 的模型权重在 MoE 中。3 个 MoE 优化 PR 的效果在每一层都累积，解释了 ~15% 的端到端性能提升。

#### 结论

| 因素 | 贡献 | 证据 |
|------|------|------|
| **TRT-LLM 版本迭代（post2→post3）** | **~15%** | SA 对标：配置一致，仅版本不同 |
| 硬件基础差异 | **~0-5%** | SA 中 B200 post2 ≈ 355X ATOM |

**报告中的 MTP0 基线差距（3-33%）大部分来自 TRT-LLM post2→post3 的 MoE kernel 优化。** 355X (ATOM) 的基线性能并不弱于 B200 在 rc6.post2 上的表现。这意味着 ATOM 框架如果获得类似的 MoE kernel 优化，基线差距可以进一步缩小。

### 3. DAR 差距因场景而异——解释了 MTP 对差距的不同影响

| 场景 | B200 DAR | 355X DAR | DAR 差距 | MTP 对差距的影响 |
|------|----------|----------|---------|-----------------|
| chat | 80.2% | 48-55% | **大** | 扩大（355X MTP 增益受限） |
| reasoning | 73.8% | 63-69% | **小** | 缩小（355X MTP 有效弥补基线） |
| summarize | 72.9% | 54-64% | 中 | 缩小 |

**规律：** DAR 差距小的场景（reasoning），MTP 帮助 355X 缩小基线差距；DAR 差距大的场景（chat），MTP 反而扩大差距。这是因为 MTP 增益 ∝ DAR × 执行效率，DAR 差距过大时 355X 的 MTP 加速不足以抵消 B200 的 MTP 加速。

### 4. 高并发增益衰减的场景差异

**Chat/Summarize（短 output）：** 355X 高并发增益衰减明显，B200 衰减更慢。
- chat c=64: B200 +42.2% vs 355X +16.1%

**Reasoning（长 output）：** 两者衰减程度几乎相同。
- reasoning c=64: B200 +63.4% vs 355X +63.2%
- reasoning c=128: B200 +58.4% vs 355X +64.0%

**解释：** Reasoning 的 decode 步数约 8K 步，MTP speculative 的加速效果能充分摊薄高并发带来的 scheduling overhead。Chat/Summarize 只有 ~1K 步 decode，overhead 占比更大。

### 5. DAR 数据来源与可比性

> 跨平台 DAR 对比的方法论限制见文档开头。

**B200 DAR 来源：** `trtllm-bench throughput` 离线基准测试（TRT-LLM C++ runtime，非 HTTP serving），200 requests，avg_concurrent=30。每场景仅单一值（不区分并发）。采样方式：随机输入 token，greedy/top-k 参数未记录。

**355X DAR 来源：** ATOM CI benchmark run #23408094038（vLLM PyTorch runtime，HTTP serving 路径），覆盖 3 场景 × 7 并发 = **21 个数据点**。从 job log `[MTP Stats]` 累计行提取。采样方式：随机输入 token，参数由 ATOM CI 配置决定。

**同平台内的可信度：**
1. 355X per-concurrency DAR 波动仅 ±3%（同场景内），说明 DAR 主要由模型+场景决定，与并发关系不大
2. B200 每场景仅一个离线值，无法验证是否随并发变化

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

### P0：缩小 MTP0 基线差距（已定位根因：MoE kernel 优化）

SA 对标数据已证实：355X ATOM 基线 ≈ B200 rc6.post2，差距来自 rc6.post3 的 3 个 MoE 优化。优化方向：
1. **MoE Shared+Sparse Expert 融合：** TRT-LLM PR#11143 将 shared expert 融入 sparse expert grouped GEMM，减少 kernel launch。ATOM/vLLM 是否有类似优化？
2. **FP8 MoE kernel 更新：** TRT-LLM PR#11104 更新了 FP8 cubins + finalize kernel。ROCm HIP 侧的 FP8 MoE kernel 优化空间？
3. **Multi-stream MoE 调度：** TRT-LLM PR#11160 修复了 MoE cost estimation。ATOM 的 MoE 调度是否有类似问题？

### P1：提升 MTP 执行效率（尤其 chat 场景）

两个平台的 DAR 利用效率都偏低（6-53%）。Chat 场景 355X 的 MTP 执行效率在高并发下急剧衰减（c=64 仅 11%），是 chat 差距扩大的直接原因。

**优化方向：**
1. 降低 MTP step 的 scheduling overhead（尤其是高并发场景）
2. 优化 KV cache 管理（speculative tokens 的 cache 分配/回收）
3. 重点关注 chat/summarize 短 output 场景的高并发衰减（decode 步数少，overhead 占比大）

### P2：实测 B200 serving 路径下的 per-concurrency DAR

B200 DAR 来自 trtllm-bench 离线环境（每场景仅单一值），而 355X 已有 21 个 per-concurrency 数据点。B200 是否在高并发 serving 下 DAR 衰减，是当前最大的未知数。

**具体行动：** 关注 TRT-LLM 后续版本是否在 `/perf_metrics` 中暴露 DAR 指标；或在 server log 中寻找 acceptance rate 输出。当前 rc6 的 `/perf_metrics` 不返回 speculative_decoding 指标。

## 待验证问题

- [x] 355X 高并发 MTP 增益衰减 — 场景相关：chat/summarize 衰减明显，reasoning 不受影响
- [x] DAR 数据 — 355X 已补全 3 场景 × 7 并发 = 21 个数据点；B200 有 3 场景离线值
- [x] 355X DAR 是否为全局值 — **否，per-scenario 差异显著**：reasoning 65% >> chat 49%
- [x] DAR 是否随并发变化 — 355X 同场景内并发波动仅 ±3%，DAR 主要由场景决定
- [x] 355X reasoning/summarize DAR — **已获取**：reasoning 63-69%, summarize 54-64%
- [x] **MTP0 基线差距根因** — **已定位：TRT-LLM rc6.post2→post3 的 3 个 MoE 优化 PR（#11143, #11104, #11160）贡献 ~15%**。SA 对标证实 355X ATOM ≈ B200 rc6.post2
- [ ] **P2: B200 serving 时的 per-concurrency DAR**（当前仅离线单一值，`/perf_metrics` 暂不支持）
- [ ] EP>1 配置下 MTP 收益差异（B200 EP=8 MTP3 有数据，EP=8 MTP0 缺失）
- [x] TRT-LLM 版本对基线的影响 — **已量化：rc6.post2→post3 带来 ~15% 提升**（3 个 MoE PR：shared+sparse 融合、FP8 cubins 更新、multi-stream 调度修复）

## 迭代日志

| 日期 | 变更 |
|------|------|
| 2026-03-24 v8 | **SA 对标 + TRT-LLM 版本分析。** 新增 SA InferenceX 对标数据（B200 rc6.post2 vs 355X ATOM，配置完全一致），证实 MTP0 基线差距来自 TRT-LLM post2→post3 的 3 个 MoE 优化 PR（~15% 提升）而非硬件差异。修正 B200 framework 为 rc6.post3。核心结论第一条重写 |
| 2026-03-24 v7 | **分析重构：MTP0 基线差距分析。** 恢复 B200 MTP0 和 355X MTP0 原始数据列；新增 MTP0 差距 / MTP3 差距对比列；核心结论从"MTP 增益对比"转为"基线差距 + MTP 弥补效果"；新增 MTP0 基线差距来源分析（框架成熟度 vs 硬件）；优先级调整（P0→缩小基线差距，P1→MTP 效率，P2→B200 serving DAR） |
| 2026-03-23 v6 | 精简表格：主表增加 355X DAR 和 B200 领先列，去掉 mtp0 绝对值列；删除分析中的重复表格（DAR 对比总览、领先变化表、B200 领先表、355X DAR 完整 21 行表） |
| 2026-03-23 v5 | 增加跨平台 DAR 对比的方法论限制章节：明确 TRT-LLM (C++ runtime) vs ATOM (vLLM PyTorch runtime) 的执行栈差异、采集环境差异、采样/确定性问题。限定结论为工程选型参考，非严格算法对比 |
| 2026-03-23 v4 | **355X per-scenario per-concurrency DAR 补全**：从 ATOM CI run #23408094038 提取全部 21 个 DAR 数据点。核心发现：reasoning DAR 65% 接近 B200 的 74%，解释了 reasoning TPS 接近的现象。重写全部 DAR 分析章节，消除"矛盾"结论 |
| 2026-03-23 v3 | 修正关键错误：355X DAR 49% 并非全局值，而是仅 chat 1024/1024 c=128。重构 DAR 矛盾分析为 chat-only 同场景对比，标注 reasoning/summarize DAR 对比无效。重排优先级（P0→补全 355X 多场景 DAR，P1→B200 serving DAR） |
| 2026-03-23 v2 | 重写分析：发现 DAR 与 TPS 增益的逻辑矛盾，B200 高 DAR 未转化为预期优势。增加 DAR 效率分析、B200 领先变化追踪。修正 import_results.py DAR 提取 bug |
| 2026-03-23 v1 | 数据更新：355X 数据修正，新增 reasoning c=64/128/256，DAR 数据。记分牌从 5:11 变为 8:10 |
| 2026-03-20 | 初版分析，基于 2026-03-19 数据，16 个可比数据点 |
