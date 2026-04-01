# FP4 性能差距分析：B200 vs MI355X — Breakdown 调查

> **Last updated:** 2026-04-01 v18
> **Model:** DeepSeek-R1-0528-NVFP4-v2, FP4
> **配置：** EP=8, DP=false, c=64, TP=8, chat 1K/1K
> **状态：** B200 trace 完成 ✅；Per-Module Kernel 分析完成 ✅；10 层平均数据完成 ✅；nvjet E4M3 源码考证完成 ✅；权重精度考证完成 ✅；算子级重构完成 ✅；MI355X 复现完成 ✅；MI355X 配置对齐复测完成 ✅；MI355X TPOT 25ms 来源分析完成 ✅

## 问题背景

SA InferenceX 报告的 B200 FP4 性能大幅领先 MI355X FP4，需要 breakdown 分析差距来源。

**ATOM 不支持 DP Attention**，原始 SA 对标配置（EP=4, DP=true）无法公平对比。选择 **EP=8, DP=false, c=64** 作为公平对标基准：DP=false 消除 DP Attention 差异；EP=8 是 B200 8GPU 的自然 EP 配置；c=64 是 SA 原始测试点。

## 对标配置

| 项目 | B200 (SA InferenceX) | B200 (复现) | MI355X (ATOM CI) | MI355X (复现) | 差异 |
|------|---------------------|------------|------------------|--------------|------|
| **Image** | rc6.post2 | rc6.post2 | rocm/atom:rocm7.1.1-ubuntu24.04-pytorch2.9-atom0.1.1-MI350x | 同机器（见下方环境对比） | TRT-LLM vs ATOM |
| **GPU / TP / EP** | 8 / 8 / 8 | 8 / 8 / 8 | 4 / 4 / 1 | 4 / 4 / 1 | GPU 数不同，需按 /GPU 比 |
| **DP Attention** | **False** | **False** | False | False | **相同** |
| **Concurrency** | 64 | 64 | 64 | 64 | **相同** |
| **Scenario** | chat 1K/1K | chat 1K/1K | chat 1K/1K | chat 1K/1K | **相同** |

| Metric | B200 (SA) | B200 (复现) | B200 复现偏差 | MI355X (CI) | MI355X (对齐复测) | MI355X 复测偏差 | B200 vs MI355X |
|--------|-----------|------------|-------------|-------------|-------------------|----------------|----------------|
| **Output TPS /GPU** | 473.4 | 490.1 | +3.5% | 600.7 | **624.9** | **+4.0%** | B200 0.78x |
| **Interactivity** | 60.93 | 63.12 | +3.6% | 38.55 | **40.09** | **+4.0%** | B200 1.57x |

> **B200 复现验证：** 同 rc6.post2 docker，复现结果 +3.5% vs SA，在正常波动范围内。
>
> **MI355X 配置对齐复测 (v17)：** 对齐 SA CI 配置后（max-model-len=2248, enforce_eager=false, gpu_memory_util=0.90），Output TPS/GPU=624.9 vs CI 600.7 (+4.0%)，Interactivity=40.09 vs 38.55 (+4.0%)。偏差在正常范围内，确认最新 ATOM (0.1.3.dev1) 性能略优于 CI 基准。
>
> **此前 v16 复现偏差较大的原因：** 配置未对齐 SA CI（max-model-len=8192 vs 2248、enforce_eager=true、gpu_memory_util=0.80），导致 Output TPS/GPU 仅 260.2，误判为性能退步。

### MI355X 复现环境对比

| 组件 | ATOM CI (2026-02-25) | 复现机器 (v17) | 差异 |
|------|---------------------|---------------|------|
| **Ubuntu** | 24.04 | 24.04.3 LTS | 一致 |
| **ROCm** | 7.1.1 | 7.1.1 | **一致** |
| **PyTorch** | 2.9 | 2.11.0+rocm7.1 | 不同（pip 重装） |
| **aiter** | `a498c8b62` (v0.1.9.post1+20, 2026-01-09) | `2bca98ced` (v0.1.12, 2026-03-30) | **不同：最新 main** |
| **ATOM** | 0.1.1 (release) | **0.1.3.dev1** (`df6ab2c`, 最新 main) | **不同：最新 main** |
| **max-model-len** | 2248 | 2248 | **一致** |
| **enforce-eager** | false（默认） | false | **一致** |
| **gpu-memory-utilization** | 0.90（默认） | 0.90 | **一致** |
| **模型** | amd/DeepSeek-R1-0528-MXFP4-Preview | DeepSeek-R1-0528-MTP-MoE-MXFP4-Attn-PTPC-FP8 | 不同 checkpoint |

> **复现结果：** Output TPS/GPU=624.9 vs CI 600.7 (+4.0%)，Interactivity=40.09 vs 38.55 (+4.0%)。配置对齐后偏差在正常范围内。ATOM 和 aiter 均使用最新 main，PyTorch 从原装 2.9 升级到 2.11+rocm7.1，模型 checkpoint 不同但量化方式相同（MXFP4）。+4% 的提升可能来自 ATOM/aiter 的累积优化。

## Per-Module Kernel 级分析（10 层平均）

> **数据来源：** B200 nsys trace，FP4, TP=8, EP=8, DP=false, c=64，trtllm-bench 离线模式，iter 100-150
> **统计口径：** 第 40-49 层（连续 10 层）平均值，含 min/max 波动范围
> **算子定义：** 按逻辑功能合并 kernel（splitKreduce 合入 GEMM，quantize 合入目标 GEMM），方便跨平台对比
> **并行组：** 标注 **P1** 的算子在不同 stream 上并行执行，关键路径 = max(组内算子时间)
> **MI355X 数据：** 待补充（rocprof trace）

| # | 算子 | Type | GEMM Shape (per GPU) | 计算内容 | B200 Kernel(s) | Avg μs | % | Min | Max | 并行 | 精度 | MI355X μs |
|---|------|------|---------------------|---------|---------------|--------|------|-----|-----|------|------|-----------|
| 1 | **qkv_a_proj** | GEMM | `[bs,7168]×[7168,2112]` 不随TP split | q_a+kv_a 低秩压缩 | nvjet tst splitK + reduce | **42.6** | 15.0% | 28.5 | 69.1 | **P1** | BF16×BF16 | |
| 2 | q/k_norm | Norm | — | Q、K RMSNorm ×2 | RMSNormKernel ×2 | 4.6 | 1.6% | 2.4 | 5.3 | | BF16 | |
| 3 | q_b_proj | GEMM | `[bs,1536]×[1536,3072]` 128h/8=16, 16×192 | Q 展开 | nvjet tst | 5.7 | 2.0% | 5.4 | 5.8 | | BF16×BF16 | |
| 4 | k_concat | Mem | — | K 拼接（RoPE 部分） | CatArrayBatchedCopy | 4.4† | 1.6% | 0.0 | 5.1 | | — | |
| 5 | uk_gemm | GEMM | `[bs,512]×[512,2048]` 16×128 | kv_b K 展开 | nvjet tst | 3.8 | 1.3% | 3.6 | 4.0 | | BF16×BF16 | |
| 6 | rope_cache | Mem | — | RoPE + KV cache 写入 | applyMLARopeAndAssignQKV | 3.5 | 1.2% | 3.3 | 3.6 | | BF16 | |
| 7 | **fmha** | Attn | — | MLA attention | fmhaSm100f QkvE4m3 | **20.7** | 7.3% | 20.1 | 21.7 | | FP8 E4M3 KV | |
| 8 | uv_gemm | GEMM | `[bs,512]×[512,2048]` 16×128 | kv_b V 投影 | nvjet tst | 3.7 | 1.3% | 3.5 | 4.1 | | BF16×BF16 | |
| 9 | **out_proj** | GEMM | `[bs,2048]×[2048,7168]` + allreduce | o_proj GEMM（含 BF16→FP4 量化） | quantize + nvjet ootst | **8.6** | 3.0% | 8.4 | 8.8 | | FP4×FP4 | |
| 10 | **tp_allreduce+norm** | Comm+Norm | — | TP AR + residual add + pre-MLP norm | userbuffers_rmsnorm | **15.2** | 5.3% | 13.1 | 17.3 | | BF16 | |
| 11 | residual_ag | Comm | — | residual allgather | userbuffers_allgather | 9.7 | 3.4% | 9.0 | 10.1 | | BF16 | |
| 12 | router | Route | `[bs,7168]×[7168,256]` splitK | Router GEMM + topK + sort | nvjet splitK + reduce + routing ×2 | 12.0 | 4.2% | 8.0 | 13.2 | | BF16 | |
| 13 | **moe_gemm** | GEMM | grouped: `[7168,4096]`+`[2048,7168]` ×32exp/GPU | gate+up+SwiGLU+down（含量化） | quantize + bmm_E2m1 + bmm_BF16 | **95.3** | 33.5% | 79.4 | 104.9 | | FP4×FP4 | |
| 14 | shared_expert | GEMM | `[bs,7168]×[7168,?]`+`[bs,?,7168]` 2 GEMMs | gate+up+SiLU+down（含量化×2） | quantize×2 + ootst×2 + silu | 21.4 | 7.5% | 20.9 | 21.8 | | FP4×FP4 | |
| 15 | **moe_finalize** | Comm | — | 加权求和 + EP allreduce + residual | moefinalize_lamport | **33.1** | 11.7% | 19.0 | 58.9 | **P1** | BF16 | |

> † k_concat 和 q/k_norm 的第二个 RMSNorm 在 Stream 8907 上执行，部分层 copy-paste 丢失，平均值略偏低。
> **高方差算子：** qkv_a_proj（std=11.4μs）和 moe_finalize（std=11.1μs）层间波动大，前者受 splitK 调度影响，后者受 EP allreduce 跨 GPU 同步影响。

#### 合计

| 口径 | B200 μs | 说明 |
|------|---------|------|
| **GPU 总时间（单层）** | **284.2** | 所有算子 GPU 执行时间求和（10 层平均） |
| **关键路径（单层）** | **251.1** | 扣除 P1 并行隐藏的 moe_finalize 33.1μs |
| **并行节省** | 33.1 | moe_finalize 被 qkv_a_proj 完全遮盖 |
| **61 decode 层实测** | **15,600** | nsys 端到端实测 15.6ms |
| **关键路径 × 61 估算** | **15,317** | 251.1 × 61 = 15.3ms，与实测偏差仅 2% |

> **P1 并行组（层间 pipeline）：**
> ```
> 时间 →
> #15 moe_finalize: |████ 33.1μs（avg）████|
> #1  qkv_a_proj:   |████ 42.6μs（avg）████████████|
>                                            ↑ qkv_a 多跑 ~9.5μs
> 关键路径 = max(33.1, 42.6) = 42.6μs，moefinalize 完全隐藏
> ```

## MI355X TPOT 25ms 来源分析

> **问题：** MI355X benchmark 报告 mean TPOT = 24.9ms，但 GPU trace 显示 decode 一步 (bs=64) 只需 21.6ms。差了 ~3ms 从何而来？

### 三层时间栈

| 层级 | 含义 | MI355X 数值 | B200 数值 | 数据来源 |
|------|------|------------|----------|---------|
| **L1: Kernel 时间** | GPU 算子执行时间之和（单层） | 165.7 μs | 251.1 μs (关键路径) | parse_trace.py → decode_breakdown.xlsx |
| **L2: Decode walltime** | GPU 端一个完整 decode step (61 层) | 21.56 ms (bs=64 p50) | 15.6 ms | --mark-trace → decode_walltime_trace.csv / nsys |
| **L3: Client TPOT** | 客户端观测 per-request TPOT | 24.9 ms (mean) | 17.8 ms | benchmark_serving.py |

每层之间都有 gap，需要分别解释：

### Gap 1: L1 → L2（Kernel Sum vs Decode Walltime）

单层 kernel 时间 × 61 层 = 估算 decode walltime。与实测对比：

| | Kernel/层 | × 61 层估算 | 实测 Decode | 差距 | Overhead % |
|---|---|---|---|---|---|
| **B200** | 251.1 μs | 15.3 ms | 15.6 ms | 0.3 ms | **1.8%** |
| **MI355X** | 165.7 μs | 10.1 ms | 21.6 ms | **11.5 ms** | **53.1%** |

**B200：** kernel 估算与实测仅差 2%，CUDA Graph replay 将整个 decode step 录制为一个 graph，几乎消除了所有 kernel 间的 CPU dispatch 开销。

**MI355X：** kernel 总和仅占 decode walltime 的 47%。剩余 53%（每层 187.7μs）是 kernel 之间的 gap。可能来源：

1. **decode_breakdown.xlsx 未捕获全部 kernel** — parse_trace.py 只提取 `--mark-trace` 标注的 module 内的 kernel launch，模块边界之间的 kernel（如 sampling、token selection、allreduce 同步等）可能未被计入
2. **HIP Graph replay overhead** — 即使使用 HIP Graph，replay 的 overhead 可能高于 CUDA Graph
3. **通信同步等待** — `reduce_scatter_cross_device_store` 的实际等待时间可能大于 kernel 本身的执行时间

> **待验证：** 需要在 MI355X 上用 rocprof 抓完整 GPU timeline（不限于 --mark-trace 标注的 module），确认 11.5ms gap 中有多少是未捕获的 kernel vs 真正的 GPU idle。

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

从 `decode_walltime_trace_chat_c64_tp4_p640.csv`（collect_atom_trace.sh 抓取的 GPU 端 decode 事件）：

| Batch Size | Count | Avg (ms) | P50 (ms) | P99 (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|---|---|
| 1 | 990 | 10.39 | 10.37 | 10.69 | 9.98 | 28.96 |
| 62 | 15 | 22.08 | 21.94 | 22.60 | 21.77 | 22.60 |
| 63 | 54 | 22.03 | 22.03 | 22.68 | 21.35 | 22.68 |
| **64** | **1821** | **21.56** | **21.53** | **22.46** | 18.79 | 110.77 |

- **bs=64 的 decode step 正常耗时 21.5ms**，max 110ms 是被 prefill 打断的极端值
- **bs=1 有 990 次** = 每次 prefill 后的第一个 decode step（bs=1 因为 batch 还没填满）
- 总 decode step = 990 + 15 + 54 + 1821 = 2880 次
- 其中 bs=64 的 1821 步中，被 prefill 打断的约占 4-5%（max 远大于 p99）

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

> **结论：** Client TPOT 24.9ms = GPU decode 21.6ms + prefill interleaving overhead ~3.3ms。这不是性能 bug，是 continuous batching 的正常行为：为了维持高 throughput，必须在 decode 间隙插入新 request 的 prefill，代价是每个 request 的 TPOT 略高于纯 decode 时间。

### B200 是否也有同样问题？

**是的。** B200 的 benchmark mean TPOT = 17.8ms，也高于 GPU decode walltime 15.6ms，差距 2.2ms。原理完全相同，只是 B200 的 prefill 更快所以 overhead 更小。

此外 B200 TRT-LLM 默认 `stream_interval: 10`（每 10 个 token 发一次 SSE chunk），导致 ITL = 10 × TPOT ≈ 170ms，这是 token batching 而非性能问题。

### MI355X Kernel 级 Breakdown（--mark-trace, 全层平均）

> **数据来源：** MI355X Kineto trace，MXFP4, TP=4, EP=1, c=64, chat 1K/1K
> **工具：** ATOM `--mark-trace` + `parse_trace.py` → `decode_breakdown.xlsx`
> **统计口径：** 全层平均值（avg sum per module）

| Module | Avg μs | % | Kernel(s) |
|--------|--------|---|-----------|
| mxfp4_moe | 44.8 | 27.0% | topk_sort + MoeSorting + elementwise + MoeFlatmm×2 + act_and_mul |
| v_up_proj_and_o_proj | 21.5 | 13.0% | batched_gemm_a8w8 + per_token_quant + gemm_preshuffle |
| post_attn_layernorm | 21.4 | 12.9% | reduce_scatter_cross_device_store + local_device_load_rmsnorm |
| input_layernorm | 16.7 | 10.1% | reduce_scatter_cross_device_store + local_device_load_rmsnorm |
| mla_decode | 14.8 | 8.9% | mla_a8w8 + mla_reduce_v1 |
| gemm_a8w8_bpreshuffle | 12.4 | 7.5% | gemm_xdl_cshuffle_v3 (qkv_a) |
| q_proj_and_k_up_proj | 11.1 | 6.7% | gemm_preshuffle (q_b) + batched_gemm_a8w8 (k_up) |
| gemm_a16w16 | 8.6 | 5.2% | bf16gemm_splitk (shared expert) |
| per_token_quant_hip | 4.8 | 2.9% | dynamic_per_token_scaled_quant |
| _fused_rms_fp8_group_quant | 4.8 | 2.9% | fused RMSNorm + FP8 group quantize |
| rope_and_kv_cache | 4.8 | 2.9% | fuse_qk_rope_concat_and_cache_mla |
| **TOTAL** | **165.7** | **100%** | |

> **注意：** 165.7μs 仅为 --mark-trace 标注范围内的 kernel 时间总和，未覆盖整个 decode step。实测 decode walltime 21.56ms 对应单层 353.4μs，差距 187.7μs/层（53%）的来源待进一步排查。

### 跨平台对比的局限性

B200（TP=8, EP=8）和 MI355X（TP=4, EP=1）的算子级直接对比受以下限制：

1. **MoE 计算量不同** — B200 EP=8 每 GPU 算 32 experts（full width），MI355X EP=1 每 GPU 算 256 experts（1/4 width by TP=4），总 FLOPs/GPU 比为 1:2
2. **MLA 头数不同** — B200 TP=8 每 GPU 16 heads，MI355X TP=4 每 GPU 32 heads
3. **通信拓扑不同** — B200 NVLink (userbuffers)，MI355X xGMI (reduce_scatter)

公平对比需要归一化到效率指标（TFLOP/s 或 GB/s），或在相同 TP/EP 配置下重新测量。当前数据只能做 per-GPU throughput 级别的对比。

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
- [ ] MI355X 53% overhead 排查（完整 GPU timeline vs --mark-trace 子集）
- [ ] MI355X Per-Module Kernel 数据补充（--mark-trace + parse_trace.py）

## 迭代日志

| 日期 | 变更 |
|------|------|
| 2026-04-01 v18 | **MI355X TPOT 25ms 来源分析完成。** 建立三层时间栈模型（L1 kernel 165.7μs → L2 decode 21.6ms → L3 TPOT 24.9ms）。L1→L2 gap: kernel 总和仅占 decode walltime 47%，53% overhead 待排查（可能是 --mark-trace 未覆盖全部 kernel）。L2→L3 gap: prefill interleaving 导致约 4.3% 的 decode step 被打断（~85ms vs 正常 ~22ms），BS-weighted avg 重建 25.3ms ≈ benchmark 24.9ms。增加 MI355X kernel breakdown 表（11 modules，全层平均）。说明跨平台算子对比的 TP/EP 局限性 |
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
