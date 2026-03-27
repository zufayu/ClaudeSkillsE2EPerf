# FP4 性能差距分析：B200 vs MI355X — Breakdown 调查

> **Last updated:** 2026-03-27 v11
> **Model:** DeepSeek-R1-0528-NVFP4-v2, FP4
> **配置：** EP=8, DP=false, c=64, TP=8, chat 1K/1K
> **状态：** B200 trace 完成 ✅；Per-Module Kernel 分析完成 ✅；nvjet E4M3 源码考证完成 ✅；权重精度考证完成 ✅；算子级重构完成 ✅

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
| **Output TPS /GPU** | 473.4 | 490.1 | +3.5% | 待测 | 待填 |
| **Interactivity** | 60.93 | 63.12 | +3.6% | 待测 | 待填 |

> **复现验证：** 同 rc6.post2 docker，复现结果 +3.5% vs SA，在正常波动范围内。

## B200 Trace 分析

**Trace 文件：** `nsys_fp4_throughput_chat_tp8_ep8_c64_iter100-150.nsys-rep`
**配置：** FP4, TP=8, EP=8, DP=false, chat 1K/1K, c=64, iter 100-150
**结果目录：** `results_b200_mtp0_fp4_ep8_c64_dp0/`

### Kernel 级 GPU 时间分布（全 trace 统计）

| 类别 | 占比 | 时间 | 实例数 | 关键 Kernel |
|------|------|------|--------|------------|
| **MoE Expert GEMM** | **31.5%** | 2,246 ms | 47,328 | bmm_E2m1 FP4 GEMM (20.5%: 3 变体) + bmm_Bfloat16 FP4→BF16 GEMM (11.0%: 4 变体) |
| **Dense GEMM (nvjet)** | **28.8%** | 2,049 ms | — | nvjet splitK 12.7%（BF16 MLA投影）+ nvjet ootst 7.1%（FP4 out/shared_expert）+ 其他 nvjet ~9% + splitKreduce 2.2% |
| **MoE Comm (moefinalize)** | **10.6%** | 756 ms | 23,664 | moefinalize_allreduce_fusion_kernel_oneshot_lamport（EP=8 allreduce 模式） |
| **Attention (FMHA)** | **7.2%** | 514 ms | 24,888 | fmhaSm100fKernel (FP8 E4M3 KV cache，无 DP = 全量 attention) |
| **Norm + Comm** | **8.5%** | 604 ms | — | userbuffers_rmsnorm (5.0%) + userbuffers_allgather (3.3%) + RMSNorm (1.8%) |
| **Quantize** | **3.8%** | 270 ms | 97,104 | quantize_with_block_size |
| **MoE Routing** | **3.1%** | 226 ms | — | routingIndicesCluster (1.7%) + routingMain (1.4%) |
| **RoPE** | **1.2%** | 88 ms | 24,888 | applyMLARopeAndAssignQKV |
| **MoE Activation** | **0.6%** | 41 ms | 24,888 | silu_and_mul |
| **Other** | **4.7%** | ~327 ms | — | CatArrayBatchedCopy (1.7%), NCCL AllGather (0.3%), reduce, memcpy 等 |

### 分类汇总

| 大类 | 占比 | 说明 |
|------|------|------|
| **MoE 相关 (GEMM + Comm + Routing + Activation)** | **~45.8%** | 最大占比 |
| **Dense GEMM (non-MoE linear)** | **~28.8%** | BF16 MLA 投影 ~16.3% + FP4 out/shared_expert ~7.1% + Router/其他 ~5.4% |
| **Norm + Comm (fused)** | **~8.5%** | userbuffers fused norm+allreduce |
| **Attention** | **~7.2%** | 无 DP，全量 attention |
| **Quantize + RoPE + Other** | **~9.7%** | 辅助开销 |

## Per-Module Kernel 级分析（第 40 层实测数据）

> **数据来源：** B200 nsys trace，FP4, TP=8, EP=8, DP=false, c=64，trtllm-bench 离线模式，iter 100-150
> **分析方法：** nsys UI Events View 手动逐 kernel 读取第 40 层（典型 MoE 层）
> **MI355X 数据：** 待补充（后续会增加 355X 列用于跨平台对比）

### 精度判断证据

| Kernel 类型 | 名字中的证据 | 精度判断 | 源码证据 |
|------------|-------------|---------|---------|
| `nvjet_ootst_...Avec16UE4M3_Bvec16UE4M3` | A、B 矩阵标注 E4M3 | **FP4(E2M1) data + E4M3 block scale** | E4M3 指 block scale factor 格式，非数据精度（见下方源码考证） |
| `bmm_E2m1_E2m1E2m1_Fp32...swiGlu` | E2M1 = MXFP4，累加器 FP32 | **FP4 × FP4 → FP32** | kernel 名直接标注 E2M1 |
| `bmm_Bfloat16_E2m1E2m1_Fp32` | 输出 BF16，输入 E2M1 | **FP4 × FP4 → BF16** | kernel 名直接标注 |
| `fmhaSm100f_QkvE4m3OBfloat16` | QKV 标注 E4M3，输出 BF16 | **FP8 E4M3 KV cache** | kernel 名直接标注 QkvE4m3 |
| `quantize_with_block_size<Type::0, __nv_bfloat16, 16>` | Type::0 = NVFP4 block scaling；**全 trace 仅此一种变体**（97,104 实例） | **BF16 → FP4(E2M1) + E4M3 scale** | `fp4Quantize.cpp`: 输出 E2M1 data + UE4M3 scale factor |
| `nvjet_tst_...bz_TNT`（QKV_A, Q_B, UK_BGEMM, UV_GEMM） | shortName = demangledName，**无数据类型标记**；前面**无 quantize** kernel | **BF16 × BF16**（权重 BF16） | `hf_quant_config.json`: q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj 全 61 层 exclude |

> **各投影层权重精度与计算精度：**
>
> | 投影 | Kernel | 权重精度 | Activation 精度 | 前置量化算子 | 证据 |
> |------|--------|---------|----------------|------------|------|
> | **QKV_A proj** (q_a + kv_a) | nvjet tst splitK TNT | **BF16** | **BF16** | **无** | `hf_quant_config.json` exclude: q_a_proj, kv_a_proj_with_mqa（全 61 层） |
> | **Q_B proj** | nvjet tst 24x64 TNN | **BF16** | **BF16** | **无** | `hf_quant_config.json` exclude: q_b_proj（全 61 层） |
> | **UK_BGEMM** (kv_b_proj K部分) | nvjet tst 128x32 TNT | **BF16** | **BF16** | **无** | `hf_quant_config.json` exclude: kv_b_proj（全 61 层） |
> | **UV_GEMM** (kv_b_proj V部分) | nvjet tst 64x16 TNT | **BF16** | **BF16** | **无** | 同上，kv_b_proj 的 V 投影部分 |
> | **WO_GEMM** (o_proj) | nvjet ootst Avec16UE4M3 | **FP4** | **FP4**（←#9 quantize） | **有：#9 quantize_with_block_size<Type::0>（BF16→FP4）** | 不在 exclude 中；ootst E4M3 = block scale 格式 |
> | **shared_expert** | nvjet ootst Avec16UE4M3 | **FP4** | **FP4**（←#20/#23 quantize） | **有：#20/#23 quantize_with_block_size<Type::0>（BF16→FP4）** | 不在 exclude 中 |
> | **MoE experts** | bmm_E2m1 / bmm_Bfloat16 | **FP4** | **FP4**（←#15 quantize） | **有：#15 quantize_with_block_size<Type::0>（BF16→FP4）** | 不在 exclude 中；kernel 名含 E2M1 |
> | **Router** | nvjet tss splitK TNT | **BF16** | **BF16** | **无** | Router 权重未量化（小矩阵） |
>
> **quantize 实例确认（SQLite 查询结果）：**
> - 全 trace 仅一种 quantize kernel：`quantize_with_block_size<BlockScaleQuantizationType::0, __nv_bfloat16, 16>`
> - 共 97,104 个实例，**无其他量化变体**
> - 因此 #9（→out_proj）、#15（→MoE GEMM）、#20（→shared_expert gate+up）、#23（→shared_expert down）**均为 BF16→FP4 量化**
> - 每次量化产出两个 tensor：E2M1 数据 + UE4M3 block scale factor（源码：`fp4Quantize.cpp:38-39`）
>
> **已确认事实：**
> - MoE Expert GEMM：`bmm_E2m1`（名字含 E2M1）= FP4 权重 × FP4 activation，前置 #15 quantize（BF16→FP4）
> - MLA attention：`fmhaSm100f QkvE4m3` = FP8 E4M3 KV cache（名字含 QkvE4m3）
> - out_proj / shared_expert：`nvjet ootst Avec16UE4M3` = FP4 权重 × FP4 activation。E4M3 指 block scale factor 格式
> - **QKV_A / Q_B / kv_b K / kv_b V：`nvjet tst` = BF16 权重 × BF16 activation。** `hf_quant_config.json` 将 q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj 全 61 层排除。kv_b_proj 在 runtime 拆分为 UK_BGEMM(#5, K部分) 和 UV_GEMM(#8, V部分)，均 BF16
> - Router：BF16 权重 × BF16 activation（未量化）
> - 第 61 层（最后一层）：`model.layers.61*` 整体排除，所有 linear 层 BF16

### MoE Layer 单层算子序列（第 40 层实测）

> **数据来源：** B200 nsys trace，FP4, TP=8, EP=8, DP=false, c=64，trtllm-bench 离线模式，iter 100-150
> **算子定义：** 按逻辑功能合并 kernel（splitKreduce 合入 GEMM，quantize 合入目标 GEMM），方便跨平台对比
> **并行组：** 标注 **P1** 的算子在不同 stream 上并行执行，关键路径 = max(组内算子时间)
> **MI355X 数据：** 待补充（rocprof trace）

| # | 算子 | 计算内容 | B200 Kernel(s) | B200 μs | % | 并行 | 精度 | MI355X Kernel | MI355X μs |
|---|------|---------|---------------|---------|------|------|------|--------------|-----------|
| 1 | **qkv_a_proj** | q_a+kv_a 低秩压缩 [7168→2112] | nvjet tst splitK + reduce | 65.4† | 20.1% | **P1** | BF16×BF16 | ck GEMM | |
| 2 | q/k_norm | Q、K RMSNorm ×2 | RMSNormKernel ×2 | 5.2 | 1.6% | | BF16 | rms_norm ×2 | |
| 3 | q_b_proj | Q 展开 [1536→nhead×192] | nvjet tst | 5.8 | 1.8% | | BF16×BF16 | ck GEMM | |
| 4 | k_concat | K 拼接（RoPE 部分） | CatArrayBatchedCopy | 5.0 | 1.5% | | — | concat | |
| 5 | uk_gemm | kv_b K 展开 [512→nhead×128] | nvjet tst | 3.6 | 1.1% | | BF16×BF16 | ck GEMM | |
| 6 | rope_cache | RoPE + KV cache 写入 | applyMLARopeAndAssignQKV | 3.5 | 1.1% | | BF16 | rotary_emb | |
| 7 | **fmha** | MLA attention | fmhaSm100f QkvE4m3 | **20.6** | 6.3% | | FP8 E4M3 KV | flash_attn (ck) | |
| 8 | uv_gemm | kv_b V 投影 | nvjet tst | 4.1 | 1.3% | | BF16×BF16 | ck GEMM | |
| 9 | **out_proj** | BF16→FP4 量化 + o_proj GEMM | quantize + nvjet ootst | **8.7** | 2.7% | | FP4×FP4 | (quant +) ck GEMM | |
| 10 | **tp_allreduce+norm** | TP AR + residual add + pre-MLP norm | userbuffers_rmsnorm | **15.6** | 4.8% | | BF16 | rccl AR + add + norm | |
| 11 | residual_ag | residual allgather | userbuffers_allgather | 9.7 | 3.0% | | BF16 | rccl AG | |
| 12 | router | Router GEMM + topK + sort | nvjet+reduce + routing ×2 | 17.7 | 5.5% | | BF16 | ck GEMM + topk | |
| 13 | **moe_gemm** | quantize + gate+up+SwiGLU + down | quantize + bmm_E2m1 + bmm_BF16 | **79.3** | 24.4% | | FP4×FP4 | fused_moe | |
| 14 | shared_expert | quantize×2 + gate+up + SiLU + down | quantize×2 + ootst×2 + silu | 21.5 | 6.6% | | FP4×FP4 | ck ×2 + silu | |
| 15 | **moe_finalize** | 加权求和 + EP allreduce + residual | moefinalize_lamport | **58.9** | 18.1% | **P1** | BF16 | local reduce（无 EP AR） | |

> † qkv_a_proj 65.4μs 为 splitK 实测值（第 41 层），splitKreduce 未单独计时，实际略长。

#### 合计

| 口径 | B200 μs | 说明 |
|------|---------|------|
| **GPU 总时间** | **324.6** | 所有算子 GPU 执行时间求和 |
| **关键路径时间** | **265.7** | 扣除 P1 并行隐藏的 moe_finalize 58.9μs |
| **并行节省** | 58.9 | moe_finalize 被 qkv_a_proj 完全遮盖 |

> **P1 并行组图示（第 40→41 层边界）：**
> ```
> 时间 →
> #15 moe_finalize: |████ 58.9μs ████████████████|
> #1  qkv_a_proj:     |████ 65.4μs ████████████████████|
>                                                 ↑ qkv_a 多跑 ~6.5μs
> 关键路径 = max(58.9, 65.4) = 65.4μs，moefinalize 完全隐藏
> ```

#### 大类汇总

| 大类 | 算子 | B200 μs | GPU % | 关键路径 % | MI355X 对应 |
|------|------|---------|-------|-----------|------------|
| **MLA 投影 (BF16)** | #1-5 | 85.0 | 26.2% | 32.0% | ck GEMM ×4 + norm ×2 + concat |
| **MLA Attention** | #6-7 | 24.1 | 7.4% | 9.1% | rotary + flash_attn |
| **输出投影** | #8-9 | 12.8 | 3.9% | 4.8% | ck GEMM ×2 + quant |
| **TP 通信** | #10-11 | 25.3 | 7.8% | 9.5% | rccl AR + AG + norm |
| **MoE 路由** | #12 | 17.7 | 5.5% | 6.7% | ck GEMM + topk + sort |
| **MoE 计算 (FP4)** | #13 | 79.3 | 24.4% | 29.8% | fused_moe |
| **Shared Expert (FP4)** | #14 | 21.5 | 6.6% | 8.1% | ck ×2 + silu |
| **EP 通信** | #15 | 58.9 | 18.1% | ⟨P1 隐藏⟩ | 无（EP=1） |
| | **合计** | **324.6** | **100%** | | |

> **MI355X 跨平台对比要点：**
> - **EP 通信（18.1%）：** MI355X EP=1 无此开销，是 B200 EP=8 的独有成本。对比时应从 B200 关键路径中扣除。
> - **TP 通信（7.8%）：** B200 用 userbuffers 融合 allreduce+norm 为单 kernel；MI355X 用 RCCL + 独立 norm，kernel 数更多但可能总时间不同。
> - **MoE 计算（24.4%）：** B200 拆分为 quantize+bmm×2 共 3 kernel；MI355X 的 fused_moe 可能合并为 1-2 个 kernel。对比时应按组对比（B200 #13 整体 vs MI355X fused_moe 整体）。
> - **MoE 路由（5.5%）：** MI355X 的 routing 可能融合在 fused_moe 内部，无法单独测量。可与 MoE 计算合并对比（B200 #12+#13 = 97.0μs vs MI355X fused_moe 整体）。

### B200 Kernel 明细（算子→kernel 映射）

> 以下为 15 个算子对应的原始 kernel 列表，供 trace 验证和深入分析参考。

| 算子 | B200 Kernel | μs | 说明 |
|------|------------|-----|------|
| #1 qkv_a_proj | nvjet tst splitK TNT | 65.4† | Fuse_A_GEMM (q_a + kv_a)，与上一层 #15 并行 |
| #2 q/k_norm | RMSNormKernel bf16 ×2 | 2.7 + 2.5 | Q norm + K norm |
| #3 q_b_proj | nvjet tst 24x64 TNN | 5.8 | UQ_QR_GEMM |
| #4 k_concat | CatArrayBatchedCopy | 5.0 | K 拼接 |
| #5 uk_gemm | nvjet tst 128x32 TNT | 3.6 | UK_BGEMM (kv_b K 部分) |
| #6 rope_cache | applyMLARopeAndAssignQKV | 3.5 | RoPE + cache write |
| #7 fmha | fmhaSm100f QkvE4m3 | 20.6 | Flash MHA (FP8 KV cache) |
| #8 uv_gemm | nvjet tst 64x16 TNT | 4.1 | UV_GEMM (kv_b V 部分) |
| #9 out_proj | quantize_with_block_size + nvjet ootst Avec16UE4M3 | 2.6 + 6.1 | BF16→FP4 量化 + WO_GEMM |
| #10 tp_allreduce+norm | userbuffers_rmsnorm | 15.6 | 融合 TP AR + residual + pre-MLP norm |
| #11 residual_ag | userbuffers_allgather | 9.7 | residual allgather |
| #12 router | nvjet tss splitK + reduce + routingMain + routingIndicesCluster | 5.4 + 2.8 + 4.4 + 5.1 | Router GEMM + topK + sort |
| #13 moe_gemm | quantize_with_block_size + bmm_E2m1_swiGlu + bmm_Bfloat16 | 3.0 + 48.4 + 27.9 | act 量化 + gate+up+SwiGLU + down |
| #14 shared_expert | quantize ×2 + ootst ×2 + silu_and_mul | 3.6 + 9.9 + 1.9 + 2.2 + 3.9 | gate+up + SiLU + down（各含前置量化） |
| #15 moe_finalize | moefinalize_allreduce_lamport | 58.9 | 加权和 + EP allreduce + residual，与下层 #1 并行 |

### Kernel 耗时波动（第 40 层 vs 第 41 层）

| Kernel | 第 40 层 | 第 41 层 | 波动 | 原因 |
|--------|---------|---------|------|------|
| bmm_E2m1 (gate+up) | 48.4μs | 60.0μs | ±12μs | expert 路由不同，M 维度变化 |
| bmm_Bfloat16 (down) | 27.9μs | 33.4μs | ±5μs | 同上 |
| moefinalize | 58.9μs | 27.8μs | ±31μs | 通信负载波动大 |
| fmhaSm100f | 20.6μs | 20.8μs | ±0.2μs | 非常稳定（shape 固定） |
| 其他 dense kernel | ±0.3μs | ±0.3μs | 稳定 | shape 固定 |

## nvjet E4M3 源码考证（已确认）

> **结论：** nvjet kernel 名中的 `E4M3` 指 **block scale factor 的数据格式**，不是 GEMM 数据元素的精度。ootst kernel 实际执行的是 **FP4 (E2M1) GEMM**，输入数据是 E2M1，block scale 是 E4M3。

### 证据 1：NVFP4 量化输出 = E2M1 数据 + UE4M3 block scale

**源文件：** `cpp/tensorrt_llm/thop/fp4Quantize.cpp:38-39`
```
// self_fp4: [M, K / 2], FLOAT4_E2M1X2
// self_block_scale_factors: ceil(M/128)*128 * ceil(K/sfVecSize/4)*4, SF_DTYPE (UE4M3 or UE8M0)
```

**源文件：** `cpp/tensorrt_llm/thop/thUtils.h:60-61`
```cpp
constexpr auto FLOAT4_E2M1X2 = torch::ScalarType::Byte; // uint8_t (两个 E2M1 packed)
constexpr auto SF_DTYPE = torch::ScalarType::Byte;       // uint8_t (UE4M3 scale factor)
```

FP4 量化 (`quantize_with_block_size<Type::0>`) 产出两个 tensor：
- **数据 tensor：** E2M1 格式（每 2 个 FP4 值 packed 成 1 个 uint8）
- **Scale factor tensor：** UE4M3 格式（每 16 个 E2M1 元素共享 1 个 E4M3 scale）

### 证据 2：blockscaleGemm TMA descriptor 对 E2M1 输入使用 E4M3 scale

**源文件：** `cpp/tensorrt_llm/kernels/trtllmGenKernels/blockscaleGemm/kernelParams.h:474-476`
```cpp
if (options.mDtypeElt == Data_type::DATA_TYPE_E2M1)
{
    const Data_type dTypeSf = Data_type::DATA_TYPE_E4M3;  // scale factor dtype = E4M3
    // ... 构建 A/B 的 block scale TMA descriptor，dtype = E4M3
}
```

当数据元素类型是 E2M1 (FP4) 时，block scale factor 固定使用 **E4M3** 格式。

### 证据 3：trtllmGen gemm cubin 命名规则区分 E2M1 和 E4M3

**源文件：** `cpp/tensorrt_llm/kernels/trtllmGenKernels/gemm/trtllmGen_export/KernelMetaInfo.h:34-45`

12 个 cubin 中存在两类：
- **FP4 kernel（E2M1）：** `GemmKernel_Bfloat16_E2m1_Fp32_...`，`GemmKernel_Fp16_E2m1_Fp32_...`，`GemmKernel_Fp32_E2m1_Fp32_...` — mMmaK=64
- **FP8 kernel（E4M3）：** `GemmKernel_E4m3_E4m3_Fp32_...`，`GemmKernel_Bfloat16_E4m3_Fp32_...` — mMmaK=32

cubin 命名格式为 `GemmKernel_{OutputDtype}_{EltDtype}_{AccDtype}_...`，E2M1 和 E4M3 是**不同的 element dtype**。

### 证据 4：FP4 GEMM 入口确认 eltType = E2m1

**源文件：** `cpp/tensorrt_llm/thop/fp4GemmTrtllmGen.cpp:39`
```cpp
auto eltType = tensorrt_llm::kernels::Dtype::E2m1;  // FP4 GEMM 的 element type
```

FP4 GEMM 调用 `TrtllmGenGemmRunner`，匹配 `mDtypeElt == E2m1` 的 cubin（不是 E4M3）。

### 证据 5：nvjet 是闭源 cuBLAS kernel

**源文件：** `cpp/tensorrt_llm/thop/cublasScaledMM.cpp:233`
```cpp
#if CUDART_VERSION < 12080
    // nvjet is not supported
```

`nvjet` 是 cuBLAS 内部的闭源 kernel 系列名。`ootst` / `tst` 是 cuBLAS 内部 tile scheduler 变体。kernel 名中的 `Avec16UE4M3` 含义为：A matrix, vec16, Unsigned E4M3 — 描述的是 **block scale factor 的数据类型**。

### 综合结论

| nsys kernel 名 | 数据元素精度 | Block Scale 格式 | 证据 |
|---------------|------------|-----------------|------|
| `nvjet ootst Avec16UE4M3` | **FP4 (E2M1)** | **UE4M3** | 前置 quantize 输出 E2M1+UE4M3；源码确认 E2M1 数据配 E4M3 scale |
| `bmm_E2m1` | **FP4 (E2M1)** | E4M3（同上） | kernel 名直接标注 E2M1 |
| `nvjet tst` (QKV_A/Q_B/kv_b K/kv_b V) | **BF16** | — | `hf_quant_config.json` exclude: 全 61 层 MLA 投影排除 FP4 |

> **之前的"矛盾"已解决：** ootst kernel 消费 FP4 数据（来自 quantize），但 kernel 名标注 E4M3——因为 **E4M3 在这里指 block scale factor 格式**，不是数据本身。NVFP4 的完整格式是 E2M1 数据 + E4M3 block scale（每 16 个元素共享 1 个 scale），这是 OCP MXFP4 标准的一部分。

## NVFP4 权重精度分析（已确认）

> **数据来源：** `/home/models/DeepSeek-R1-0528-NVFP4-v2/hf_quant_config.json`
> **量化工具：** modelopt 0.34.1.dev3
> **量化算法：** NVFP4（group_size=16），KV cache FP8

### hf_quant_config.json exclude_modules

`exclude_modules` 列出了**排除在 FP4 量化之外、保持 BF16 权重**的模块：

| 排除模块 | 对应 kernel | 层范围 | 说明 |
|----------|-----------|--------|------|
| `lm_head` | 最终输出头 | 全局 | 词表投影，精度敏感 |
| `q_a_proj`（全 61 层） | fused_qkv_a_proj 的 Q 部分 | layer 0-60 | MLA Q 压缩投影 |
| `kv_a_proj_with_mqa`（全 61 层） | fused_qkv_a_proj 的 KV 部分 | layer 0-60 | MLA KV 压缩投影 |
| `q_b_proj`（全 61 层） | #3 nvjet tst | layer 0-60 | MLA Q 展开投影 |
| `kv_b_proj`（全 61 层） | #5 nvjet tst | layer 0-60 | MLA K 展开投影（512→128×nhead） |
| `model.layers.61*` | 最后一层所有 linear | layer 61 | 最后一层整体排除 |

**注意：** 排除列表仅包含 4 种 attention 投影层（全 61 层 × 4 = 244 个）+ lm_head + 第 61 层通配符。**所有 MoE、shared_expert、out_proj、Router 均未排除 → 使用 FP4 权重。**

### 各模块权重精度汇总

| 模块 | 权重精度 | Activation 精度 | GEMM 类型 | 占 GPU 时间 | 证据 |
|------|---------|----------------|-----------|-----------|------|
| **q_a_proj** | BF16 | BF16 | BF16 × BF16 | ~12.7%（nvjet splitK） | exclude_modules |
| **kv_a_proj_with_mqa** | BF16 | BF16 | BF16 × BF16 | ↑（与 q_a fused） | exclude_modules |
| **q_b_proj** | BF16 | BF16 | BF16 × BF16 | ~2.2% | exclude_modules |
| **kv_b_proj K部分** (UK_BGEMM) | BF16 | BF16 | BF16 × BF16 | ~1.4% | exclude_modules |
| **kv_b_proj V部分** (UV_GEMM) | BF16 | BF16 | BF16 × BF16 | ~1.6% | exclude_modules（同 kv_b_proj） |
| **o_proj** (WO_GEMM) | **FP4** | **FP4**（←quantize） | FP4 × FP4 | ~2.4%（ootst） | 不在 exclude 中 |
| **MoE gate+up** | **FP4** | **FP4**（←quantize） | FP4 × FP4 | ~18.7%（bmm_E2m1） | 不在 exclude 中 |
| **MoE down** | **FP4** | **FP4**（←quantize） | FP4 × FP4 | ~10.8%（bmm_Bfloat16） | 不在 exclude 中 |
| **shared_expert** | **FP4** | **FP4**（←quantize） | FP4 × FP4 | ~5.3%（ootst ×2） | 不在 exclude 中 |
| **Router** | BF16 | BF16 | BF16 × BF16 | ~3.2% | 小矩阵未量化 |
| **lm_head** | BF16 | BF16 | BF16 × BF16 | — | exclude_modules |

### 精度分布统计

| 精度 | 占 GEMM GPU 时间 | 模块 |
|------|-----------------|------|
| **FP4 × FP4** | **~37.2%**（MoE 29.5% + out_proj 2.4% + shared_expert 5.3%） | MoE experts, o_proj, shared_expert |
| **BF16 × BF16** | **~21.1%**（QKV_A ~12.7% + Q_B 2.2% + kv_b K 1.4% + kv_b V 1.6% + Router 3.2%） | MLA 投影（q_a, q_b, kv_b K/V）+ Router |
| **FP8 E4M3** | **~7.9%** | FMHA（KV cache） |

> **关键发现：**
> 1. "FP4 模型"并非所有层都是 FP4。MLA attention 的 4 个投影层（q_a, q_b, kv_a, kv_b）全部 61 层保留 BF16 精度。
> 2. kv_b_proj 在 TRT-LLM 中拆分为 K 和 V 两部分（UK_BGEMM #5 + UV_GEMM #8），均为 BF16，对应 trace 中两个独立的 `nvjet tst` kernel。
> 3. trace 中所有 `nvjet tst`（无 E4M3 标记、无前置 quantize）= BF16 GEMM；所有 `nvjet ootst Avec16UE4M3`（有前置 quantize）= FP4 GEMM。
> 4. **模块命名对应（来自 TRT-LLM 官方 blog）：** Module1=Fuse_A_GEMM(q_a+kv_a), Module3=UQ_QR_GEMM(q_b), Module4=UK_BGEMM(kv_b K), Module7=UV_GEMM(kv_b V), Module8=WO_GEMM(o_proj)。

## 待填充

- [x] B200 EP=8 no-DP 复现 + trace 分析
- [x] Per-Module Kernel 级分析（第 40 层实测）
- [x] nvjet E4M3 源码考证（确认 E4M3 = block scale factor 格式）
- [x] NVFP4 权重精度考证（hf_quant_config.json 确认 MLA 投影 BF16，MoE/out_proj FP4）
- [ ] MI355X 数据补充（355X 列 + ratio 列）
- [ ] 配置 B vs 355X 公平对标结果（同 concurrency）

## 迭代日志

| 日期 | 变更 |
|------|------|
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
