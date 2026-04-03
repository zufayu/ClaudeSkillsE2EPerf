# FP4 性能差距分析：B200 vs MI355X — Breakdown 调查

> **Last updated:** 2026-04-03 v20
> **Model:** DeepSeek-R1-0528-NVFP4-v2, FP4
> **配置：** EP=8, DP=false, c=64, TP=8, chat 1K/1K
> **状态：** B200 trace 完成 ✅；Per-Module Kernel 分析完成 ✅；10 层平均数据完成 ✅；nvjet E4M3 源码考证完成 ✅；权重精度考证完成 ✅；算子级重构完成 ✅；MI355X 复现完成 ✅；MI355X 配置对齐复测完成 ✅；MI355X TPOT 25ms 来源分析完成 ✅；MI355X bs=64 kernel breakdown 修正完成 ✅

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

### 跨平台对齐算子表（37 行，按逻辑功能对齐）

> **数据来源：** B200 nsys trace 10 层平均 / MI355X Kineto trace 全层平均 / kernel-pairs-corrected.tsv
> **对齐原则：** 按逻辑功能（非执行时序）逐行对齐，同一行的 B200 和 MI355X kernel 做同一件事。一端独有的算子另一端留空。
> **GAP(B-M)：** 正值 = B200 更慢，负值 = MI355X 更慢

| block | ID | 逻辑算子 | B200 kernel | B200 μs | MI355X module | MI355X kernel | MI355X μs | GAP(B-M) | 备注 |
|-------|-----|---------|-------------|---------|---------------|---------------|-----------|----------|------|
| pre_attn_comm | 1 | TP_AR+residual+RMSNorm(融合) | moefinalize_lamport | 33.11 | | | 0 | 33.11 | B200独有:EP_AR+加权求和+residual+pre-attn_RMSNorm全融合;与下行qkv_a并行 |
| pre_attn_comm | 2 | TP_reduce_scatter | | 0 | input_layernorm | reduce_scatter_cross_device_store | 18.28 | -18.28 | MI355X独有:TP=4 xGMI通信第一步 |
| pre_attn_comm | 3 | local_load+RMSNorm | | 0 | input_layernorm | local_device_load_rmsnorm | 6.80 | -6.80 | MI355X独有:通信第二步+RMSNorm |
| qkv_proj | 4 | per_token_quant(BF16→FP8) | | 0 | per_token_quant_hip | dynamic_per_token_scaled_quant | 5.72 | -5.72 | MI355X独有:qkv_a走FP8需先量化输入 |
| qkv_proj | 5 | qkv_a_proj_GEMM | nvjet_splitK_TNT | 25.12 | gemm_a8w8_bpreshuffle | gemm_xdl_preshuffle | 11.00 | 14.12 | 同一GEMM[64x7168]x[7168x2112];B200=BF16 MI355X=FP8 |
| qkv_proj | 6 | qkv_a_splitK_reduce | splitKreduce(bf16) | 3.68 | | | 0 | 3.68 | B200独有:cuBLAS splitK第二步;MI355X的CK单kernel完成 |
| qkv_proj | 7 | q_norm(RMSNorm) | RMSNormKernel | 2.64 | | | 0 | 2.64 | B200独有:MI355X融入Row9的fused_rms |
| qkv_proj | 8 | k_norm(RMSNorm) | RMSNormKernel(Stream8907) | 2.48 | | | 0 | 2.48 | B200独有:另一stream并行;MI355X融入Row9 |
| qkv_proj | 9 | fused_RMS+FP8_group_quant | | 0 | _fused_rms_fp8_group_quant | fused_rms_fp8_group_quant_kernel | 5.52 | -5.52 | MI355X独有:融合q/k_norm+FP8量化;对标B200 Row7+8(5.12us) |
| qkv_proj | 10 | q_b_proj_GEMM | nvjet_tst_TNN | 5.65 | q_proj_and_k_up_proj | gemm_xdl_preshuffle | 7.12 | -1.47 | 同一GEMM[64x1536]x[1536x3072/TP];Q展开 |
| qkv_proj | 11 | k_concat | CatArrayBatchedCopy | 4.89 | | | 0 | 4.89 | B200独有:K拼接RoPE部分;MI355X可能融入rope_kernel |
| qkv_proj | 12 | uk_gemm(K_expansion) | nvjet_tst_TNT | 3.76 | q_proj_and_k_up_proj | batched_gemm_a8w8_quant | 5.88 | -2.12 | 同一GEMM[64x512]x[512x2048/TP];kv_a→K_heads |
| rope_attn | 13 | RoPE+KV_cache_write | applyMLARopeAndAssignQKV | 3.46 | rope_and_kv_cache | fuse_qk_rope_concat_and_cache_mla | 5.20 | -1.74 | 两者都是融合kernel;MI355X额外含concat |
| rope_attn | 14 | Attention(FMHA/MLA) | fmhaSm100f(含reduce) | 20.67 | mla_decode | mla_a8w8_qh16_qseqlen1 | 24.08 | -3.41 | Q×KT→softmax→×V;B200单kernel含reduce;MI355X TP=4读2x_KV_cache |
| rope_attn | 15 | MLA_reduce | | 0 | mla_decode | kn_mla_reduce_v1_ps | 6.72 | -6.72 | MI355X独有:多头reduce;B200已融入fmhaSm100f |
| out_proj | 16 | uv_gemm(V_expansion) | nvjet_tst | 3.74 | v_up_proj_and_o_proj | batched_gemm_a8w8_quant | 6.64 | -2.90 | 同一GEMM[64x512]x[512x2048/TP];kv_a→V_heads |
| out_proj | 17 | o_proj_quant | quantize_with_block_size(FP4) | 2.46 | v_up_proj_and_o_proj | dynamic_per_token_scaled_quant(FP8) | 5.40 | -2.94 | B200=block-scale BF16→FP4;MI355X=per-token BF16→FP8 |
| out_proj | 18 | o_proj_GEMM | nvjet_ootst_FP4 | 6.13 | v_up_proj_and_o_proj | gemm_xdl_preshuffle(FP8) | 13.48 | -7.35 | 同一GEMM[64x2048]x[2048x7168/TP];MI355X效率极低3.4%BW |
| post_attn | 19 | TP_AR+residual+RMSNorm | userbuffers_rmsnorm(融合) | 15.15 | post_attn_layernorm | reduce_scatter_cross_device_store | 19.76 | -4.61 | 同一功能:post-attn TP通信+pre-MoE_RMSNorm |
| post_attn | 20 | local_load+RMSNorm(step2) | | 0 | post_attn_layernorm | local_device_load_rmsnorm | 5.24 | -5.24 | MI355X独有第二步;B200已融入Row19 |
| post_attn | 21 | residual_AllGather(EP) | userbuffers_allgather | 9.74 | | | 0 | 9.74 | B200独有:EP=8分片后需AG恢复完整residual;MI355X EP=1无需 |
| router | 22 | router_GEMM | nvjet_tss_splitK | 5.42 | gemm_a16w16 | bf16gemm_splitk | 9.24 | -3.82 | router GEMM[64x7168]x[7168x256]BF16;MI355X shared_expert融入MoeMxGemm |
| router | 23 | router_splitK_reduce | splitKreduce(fp32) | 2.73 | | | 0 | 2.73 | B200独有:MI355X的CK内含或无独立reduce |
| router | 24 | MoE_gate_up_quant | quantize_with_block_size | 2.93 | | | 0 | 2.93 | B200独有:BF16→FP4量化;MI355X融入Row28的fused_quant_sort |
| router | 25 | TopK_select | routingMainKernel | 4.30 | mxfp4_moe | grouped_topk_opt_sort | 5.16 | -0.86 | 从256expert选top-8;数据量极小Kernel-Lat |
| router | 26 | expert_sort(phase1) | routingIndicesCluster | 5.12 | mxfp4_moe | MoeSorting_P0_v2 | 5.20 | -0.08 | 按expert_ID分组排序 |
| router | 27 | expert_sort(phase2+3) | | 0 | mxfp4_moe | MoeSorting_P23 | 5.20 | -5.20 | MI355X独有:3-phase_sort多一个阶段 |
| moe_expert | 28 | gate_up_quant+sort(融合) | | 0 | mxfp4_moe | fused_mxfp4_quant_moe_sort | 9.44 | -9.44 | MI355X独有:BF16→MXFP4量化+排序融合;对标B200 Row24 |
| moe_expert | 29 | gate_up_GEMM(+SwiGLU) | bmm_E2m1(FP4含SwiGLU) | 59.55 | mxfp4_moe | kernel_moe_mxgemm_2lds | 123.60 | -64.05 | 核心MoE;2.07x=per-GPU权重2x(4vs8GPU);两者~95%BW |
| moe_expert | 30 | down_quant+sort(融合) | | 0 | mxfp4_moe | fused_mxfp4_quant_moe_sort | 5.12 | -5.12 | MI355X独有:第二次quant+sort融合 |
| moe_expert | 31 | down_GEMM | bmm_Bfloat16(FP4→BF16) | 32.77 | mxfp4_moe | kernel_moe_mxgemm_2lds(atomic) | 61.48 | -28.71 | 1.88x≈per-GPU权重2x;MI355X用atomic_add |
| shared_exp | 32 | shared_gate_up_quant | quantize_with_block_size | 3.75 | | | 0 | 3.75 | B200独有:MI355X shared_expert融入Row29+31的MoeMxGemm |
| shared_exp | 33 | shared_gate_up_GEMM | nvjet_ootst_FP4 | 10.03 | | | 0 | 10.03 | B200独有:MI355X作为always-active expert编入grouped_GEMM |
| shared_exp | 34 | SiLU×Mul | silu_and_mul_kernel | 1.64 | | | 0 | 1.64 | B200独有:MI355X SwiGLU融入MoeMxGemm epilogue |
| shared_exp | 35 | shared_down_quant | quantize_with_block_size | 2.17 | | | 0 | 2.17 | B200独有:MI355X融入MoeMxGemm |
| shared_exp | 36 | shared_down_GEMM | nvjet_ootst_FP4 | 3.81 | | | 0 | 3.81 | B200独有:MI355X融入MoeMxGemm |
| moe_finalize | 37 | moe_finalize(=Row1) | moefinalize_lamport | (=Row1) | | | 0 | 0 | 同Row1:层尾=下一层层首;仅计一次不重复 |
| | | | **B200_SUM** | **276.91** | | **MI355X_SUM** | **371.26** | **-94.35** | |

**对齐表关键发现：**
- **Row 1 moefinalize_lamport** 融合了 EP AllReduce + 加权求和 + residual add + pre-attn RMSNorm，是层尾=下一层层首的跨层融合 kernel，与下一行 qkv_a 并行执行（关键路径被 qkv_a 遮盖）
- **MI355X 无独立 shared_expert kernel**（Row 32-36 全空），shared expert 作为 always-active expert 编入 MoeMxGemm grouped GEMM
- **MI355X gemm_a16w16 (Row 22, 9.24μs) = router GEMM**，非 shared expert（执行顺序在 topK 之前，权重/时间分析确认）
- **MoE sparse GEMM (Row 29+31)** 差距 92.76μs，其中 ~95% 来自 GPU 数差异（4 vs 8 GPU → per-GPU 权重 2x），两平台 HBM BW 利用率均 ~95%
- **总差距 94.35μs**：MoE GEMM 贡献 92.76μs (98%)，通信+Norm 贡献 ~25μs，被 B200 qkv_a BF16 劣势（-14μs）和 shared expert 劣势（-21μs）部分抵消

## MI355X TPOT 25ms 来源分析

> **问题：** MI355X benchmark 报告 mean TPOT = 24.9ms，但 GPU trace 显示 decode 一步 (bs=64) 只需 21.6ms。差了 ~3ms 从何而来？

### 三层时间栈

| 层级 | 含义 | MI355X 数值 | B200 数值 | 数据来源 |
|------|------|------------|----------|---------|
| **L1: Kernel 时间** | GPU 算子执行时间之和（单层） | 344.0 μs (avg) | 251.1 μs (关键路径) | run_parse_trace.py (bs=64) → decode_breakdown.xlsx |
| **L2: Decode walltime** | GPU 端一个完整 decode step (61 层) | 21.56 ms (bs=64 p50) | 15.6 ms | --mark-trace → decode_walltime_trace.csv / nsys |
| **L3: Client TPOT** | 客户端观测 per-request TPOT | 24.9 ms (mean) | 17.8 ms | benchmark_serving.py |

每层之间都有 gap，需要分别解释：

### Gap 1: L1 → L2（Kernel Sum vs Decode Walltime）

单层 kernel 时间 × 61 层 = 估算 decode walltime。与实测对比：

| | Kernel/层 | × 61 层估算 | 实测 Decode | 差距 | Overhead % |
|---|---|---|---|---|---|
| **B200** | 251.1 μs | 15.3 ms | 15.6 ms | 0.3 ms | **1.8%** |
| **MI355X** | 344.0 μs | 21.0 ms | 21.6 ms | 0.6 ms | **2.7%** |

**两个平台的 kernel 估算与实测 decode walltime 偏差均 <3%。** CUDA Graph (B200) 和 HIP Graph (MI355X) 都有效消除了 kernel 间 CPU dispatch 开销。

> **v18 勘误：** 此前报告 MI355X overhead 53% 系 parse_trace.py 取了 bs=1 的 decode 事件（prefill 后第一个 decode，batch 尚未填满），导致 kernel 时间被严重低估（165.7μs vs 正确的 344.0μs）。修复后使用 `run_parse_trace.py --target-bs 64` 自动选取稳态 bs=64 decode 事件，估算与实测吻合。

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

### MI355X Kernel 级 Breakdown（bs=64, --mark-trace, 全层平均）

> **数据来源：** MI355X Kineto trace，MXFP4, TP=4, EP=1, c=64, chat 1K/1K
> **工具：** `run_parse_trace.py --target-bs 64` → `decode_breakdown.xlsx`
> **统计口径：** 全层平均值（avg sum per module），bs=64 稳态 decode 事件

| Module | Avg μs | % | Kernel(s) | 精度 |
|--------|--------|---|-----------|------|
| **mxfp4_moe** | **188.0** | **54.7%** | topk_sort + MoeSorting ×2 + mxfp4_quant ×2 + MoeMxGemm ×2 | MXFP4 |
| mla_decode | 33.2 | 9.7% | mla_a8w8 + mla_reduce_v1 | FP8 |
| input_layernorm | 27.4 | 8.0% | reduce_scatter_cross_device_store + local_device_load_rmsnorm | BF16 |
| v_up_proj_and_o_proj | 24.2 | 7.0% | batched_gemm_a8w8 + per_token_quant + gemm_preshuffle | FP8 |
| post_attn_layernorm | 22.9 | 6.7% | reduce_scatter_cross_device_store + local_device_load_rmsnorm | BF16 |
| q_proj_and_k_up_proj | 12.5 | 3.6% | gemm_preshuffle (q_b) + batched_gemm_a8w8 (k_up) | FP8 |
| gemm_a8w8_bpreshuffle | 11.6 | 3.4% | gemm_xdl_cshuffle_v3 (qkv_a) | FP8 |
| gemm_a16w16 | 9.3 | 2.7% | bf16gemm_splitk (shared expert) | BF16 |
| per_token_quant_hip | 5.0 | 1.5% | dynamic_per_token_scaled_quant | BF16→FP8 |
| rope_and_kv_cache | 5.1 | 1.5% | fuse_qk_rope_concat_and_cache_mla | BF16 |
| _fused_rms_fp8_group_quant | 4.8 | 1.4% | fused RMSNorm + FP8 group quantize | BF16→FP8 |
| **TOTAL** | **344.0** | **100%** | | |

> **验证：** 344.0 × 61 = 20.98ms ≈ 实测 decode walltime 21.56ms（偏差 2.7%），kernel breakdown 覆盖了几乎全部 GPU 执行时间。
>
> **vs v18 bs=1 数据：** bs=1 时 TOTAL 仅 165.7μs，主要因为 MoE 在 bs=1 下只路由 8 experts（而非 bs=64 的 512 tokens × 8 experts/token），gate_up + down GEMM 的计算量差异巨大。

### 跨平台 Per-GPU 时间栈总结

| 层级 | B200 (8×GPU, TP8 EP8) | MI355X (4×GPU, TP4 EP1) | B200/MI355X |
|------|----------------------|------------------------|-------------|
| **L1: Kernel/层** | 251.1 μs (关键路径) | 344.0 μs | 0.73x |
| **L2: Decode step** | 15.6 ms | 21.6 ms | 0.72x |
| **L3: Client TPOT** | 17.8 ms | 24.9 ms | 0.72x |
| **L1→L2 overhead** | 1.8% | 2.7% | 两者均 <3% |
| **L2→L3 overhead** | +2.2 ms (prefill) | +3.3 ms (prefill) | — |
| **Output TPS/GPU** | 490.1 | 624.9 | 0.78x |

### 功能分组算子对比（单层，绝对 GPU 时间）

> 算子时间是绝对物理时间，TP/EP 不同不影响直接对比——差异本身就是 TP/EP 配置选择的结果。

| 功能块 | B200 μs | MI355X μs | B200/MI355X | 差距来源 |
|--------|---------|-----------|-------------|---------|
| **MLA Attention** | 97.6 | 96.4 | **1.01x** | 几乎持平 |
| └ qkv_a 投影 | 42.6 | 11.6 | 3.67x | B200 BF16 splitK 慢于 MI355X FP8 CK |
| └ q_b + k_up 投影 | 9.5 | 12.5 | 0.76x | |
| └ attention (fmha/mla) | 20.7 | 33.2 | 0.62x | MI355X 32h vs B200 16h (TP差异) |
| └ v_proj + out_proj | 12.3 | 24.2 | 0.51x | MI355X 含量化+双 GEMM |
| └ 其他 (norm/rope/concat) | 12.5 | 14.9 | 0.84x | |
| **MoE (含 router/finalize)** | 161.8 | 197.3 | **0.82x** | |
| └ router | 12.0 | — | — | MI355X topk 在 mxfp4_moe 内 |
| └ sparse experts GEMM | 95.3 | 188.0* | 0.51x | EP 差异：B200 32exp/GPU vs MI355X 全256exp |
| └ shared expert | 21.4 | 9.3 | 2.30x | B200 FP4 vs MI355X BF16 splitK |
| └ moe_finalize/通信 | 33.1 | — | — | B200 EP=8 allreduce，MI355X EP=1 无需 |
| **通信 + Norm** | 24.9 | 50.3 | **0.50x** | |
| └ TP allreduce + norm | 15.2 | 27.4 + 22.9 | 0.30x | B200 NVLink userbuffers vs MI355X xGMI reduce_scatter |
| └ residual allgather | 9.7 | — | — | B200 独有 |
| **总计（单层）** | **284.2** (251.1 关键路径) | **344.0** | **0.73x** | |

> \* MI355X mxfp4_moe 188.0μs 含 router + topk_sort + quant + 2×MoeMxGemm，不含 shared_expert。
>
> **关键发现：**
> 1. **MLA Attention 两平台几乎持平**（97.6 vs 96.4μs）。B200 的 qkv_a 慢（BF16 splitK 42.6μs），但 attention 和 v/out_proj 快，两者抵消。
> 2. **MoE 差距 35.5μs**（161.8 vs 197.3μs），是 decode 时间差距的主要来源。B200 EP=8 每 GPU 仅算 32 experts（full width），MI355X EP=1 全部 256 experts（1/4 width by TP=4），计算量差异大但 MI355X 单卡效率更高（Output TPS/GPU 624.9 vs 490.1）。
> 3. **通信 + Norm 差距 25.4μs**（24.9 vs 50.3μs），B200 NVLink userbuffers 的融合 allreduce+norm 显著快于 MI355X xGMI reduce_scatter。
> 4. **两个平台 L1→L2 overhead 均 <3%**，CUDA/HIP Graph replay 有效。v18 报告的 53% overhead 系 bs=1 数据错误。
> 5. **L2→L3 的 ~3ms gap 是 prefill interleaving 的正常开销**，非性能 bug。

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

## 迭代日志

| 日期 | 变更 |
|------|------|
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
