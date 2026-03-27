# FP4 性能差距分析：B200 vs MI355X — Breakdown 调查

> **Last updated:** 2026-03-27 v10
> **Model:** DeepSeek-R1-0528-NVFP4-v2, FP4
> **配置：** EP=8, DP=false, c=64, TP=8, chat 1K/1K
> **状态：** B200 trace 完成 ✅；Per-Module Kernel 分析完成 ✅；nvjet E4M3 源码考证完成 ✅；权重精度考证完成 ✅

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

### MoE Layer 单层 Kernel 序列（第 40 层实测）

> **层间 Pipeline 重叠说明：** 上一层末尾的 `moefinalize` 与本层的 `QKV_A proj` 在 GPU 上并行执行（仅差 2μs 启动）。`userbuffers_rmsnorm`（#11）融合了本层的 TP allreduce 和下一层的 pre-attn RMSNorm，因此 MoE 层没有独立的 pre-attn norm kernel。
>
> **MI355X 结构差异：** ① EP=1 → 无 EP allreduce（#25 不存在）；② 无 userbuffers → allreduce 和 norm 是分离 kernel；③ MoE 可能使用 fused_moe triton kernel（#15-19 合并）。MI355X kernel 名待 rocprof trace 确认。

| # | Module | B200 Kernel | B200 μs | B200 % | Precision | MI355X Kernel | MI355X μs |
|---|--------|-------------|---------|--------|-----------|---------------|-----------|
| — | fused_qkv_a_proj | nvjet tst splitK TNT | ⟨重叠⟩ | — | **BF16×BF16**（权重 exclude） | ck/hipblaslt GEMM | |
| — | fused_qkv_a_proj (续) | splitKreduce | ⟨重叠⟩ | — | — | （同上） | |
| 1 | q_norm | RMSNormKernel bf16 | 2.7 | 1.0% | BF16 | rms_norm (triton) | |
| 2 | k_norm | RMSNormKernel bf16 | 2.5 | 1.0% | BF16 | rms_norm (triton) | |
| 3 | q_b_proj | nvjet tst 24x64 TNN | 5.8 | 2.2% | **BF16×BF16**（权重 exclude） | ck/hipblaslt GEMM | |
| 4 | k concat | CatArrayBatchedCopy | 5.0 | 1.9% | — | concat/reshape | |
| 5 | UK_BGEMM (kv_b_proj K部分) | nvjet tst 128x32 TNT | 3.6 | 1.4% | **BF16×BF16**（权重 exclude） | ck/hipblaslt GEMM | |
| 6 | cache_update | applyMLARopeAndAssignQKV | 3.5 | 1.4% | BF16 | rotary_embedding | |
| 7 | **mla** | **fmhaSm100f QkvE4m3** | **20.6** | **7.9%** | **FP8 E4M3 KV cache** | **flash_attn (ck)** | |
| 8 | UV_GEMM (kv_b_proj V部分) | nvjet tst 64x16 TNT | 4.1 | 1.6% | **BF16×BF16**（权重 exclude，kv_b_proj 的 V 投影） | ck/hipblaslt GEMM | |
| 9 | quantize→out_proj | quantize_with_block_size<Type::0> | 2.6 | 1.0% | **BF16→FP4**（Type::0，全 trace 仅此一种） | scaled_fp4_quant | |
| 10 | out_proj | nvjet ootst Avec16UE4M3 | 6.1 | 2.4% | **FP4(E2M1) data + E4M3 scale**（←#9 quantize） | ck/hipblaslt GEMM | |
| 11 | **allreduce+norm** | **userbuffers_rmsnorm** | **15.6** | **6.0%** | BF16 | **rccl allreduce + rms_norm（分离）** | |
| 12 | residual allgather | userbuffers_allgather | 9.7 | 3.7% | BF16 | rccl allgather | |
| 13 | Router gemm | nvjet tss splitK TNT | 5.4 | 2.1% | **BF16×BF16** | ck/hipblaslt GEMM | |
| 14 | Router gemm (续) | splitKreduce | 2.8 | 1.1% | — | （同上） | |
| 15 | quantize→MoE | quantize_with_block_size<Type::0> | 3.0 | 1.2% | **BF16→FP4** | fused_moe 内部 | |
| 16 | topk | routingMainKernel | 4.4 | 1.7% | — | fused_moe 内部 | |
| 17 | sort | routingIndicesCluster | 5.1 | 2.0% | — | fused_moe 内部 | |
| 18 | **moe (gate+up)** | **bmm_E2m1 swiGlu** | **48.4** | **18.7%** | **FP4(E2M1)×FP4→FP32**（←#15） | **fused_moe gate+up (triton/ck)** | |
| 19 | **moe (down)** | **bmm_Bfloat16** | **27.9** | **10.8%** | **FP4(E2M1)×FP4→BF16** | **fused_moe down (triton/ck)** | |
| 20 | quantize→shared_expert | quantize_with_block_size<Type::0> | 3.6 | 1.4% | **BF16→FP4** | scaled_fp4_quant | |
| 21 | shared_expert (gate+up) | nvjet ootst Avec16UE4M3 | 9.9 | 3.8% | **FP4(E2M1) data + E4M3 scale**（←#20 quantize） | ck/hipblaslt GEMM | |
| 22 | shared_expert (激活) | silu_and_mul | 1.9 | 0.7% | BF16 | silu_mul (triton) | |
| 23 | shared_expert (量化) | quantize_with_block_size<Type::0> | 2.2 | 0.8% | **BF16→FP4** | scaled_fp4_quant | |
| 24 | shared_expert (down) | nvjet ootst Avec16UE4M3 | 3.9 | 1.5% | **FP4(E2M1) data + E4M3 scale**（←#23 quantize） | ck/hipblaslt GEMM | |
| 25 | **moefinalize** | **moefinalize_allreduce_lamport** | **58.9** | **22.7%** | BF16 | **N/A（EP=1 无 EP allreduce）** | |
| — | ⟨pipeline⟩ 下一层 qkv_a | nvjet tst splitK TNT | (65.4) | — | — | ck/hipblaslt GEMM | |
| | **合计 (#1-#25)** | | **259.2** | **100%** | | | |

> **Pipeline 重叠图示（第 40→41 层边界）：**
> ```
> 时间 →    .451483s                              .451542s
> 第40层 #25: |████ moefinalize (58.9μs) ████████████|
> 第41层 QKV:   |████ nvjet splitK (65.4μs) ████████████|
>            .451485s                                .451550s
>            ↑ 间隔 2μs，GPU 并行执行两个 kernel
> ```
>
> **MI355X 关键结构差异：**
> - **#25 moefinalize（22.7%）：** MI355X EP=1 不需要 EP allreduce，此项完全不存在。这是 B200 EP=8 的独有开销。
> - **#11 userbuffers_rmsnorm（6.0%）：** B200 用 userbuffers 融合 allreduce+norm 为单 kernel；MI355X 用 RCCL allreduce + 独立 rms_norm，至少 2 个 kernel。
> - **#15-19 MoE（34.4%）：** B200 分离为 quantize+routing+sort+bmm×2 共 5 个 kernel；MI355X 的 fused_moe 可能将部分步骤合并。

### MoE 层各模块占比（第 40 层）

| Module Group | B200 μs | B200 % | MI355X 对应 | MI355X μs |
|--------------|---------|--------|-------------|-----------|
| Norm (#1-2) | 5.2 | 2.0% | rms_norm ×2 | |
| MLA proj (#3-5, BF16) | 14.4 | 5.6% | ck GEMM ×3 | |
| cache_update (#6) | 3.5 | 1.4% | rotary_emb | |
| **MLA attention (#7)** | **20.6** | **7.9%** | **flash_attn** | |
| UV_GEMM + quant + WO (#8-10) | 12.8 | 4.9% | ck GEMM ×2 + quant | |
| **TP comm (#11-12)** | **25.3** | **9.8%** | **rccl AR + AG + norm** | |
| MoE routing (#13-17) | 20.7 | 8.0% | fused_moe routing | |
| **MoE GEMM (#18-19)** | **76.3** | **29.4%** | **fused_moe GEMM** | |
| shared_expert (#20-24) | 21.5 | 8.3% | quant + ck GEMM + silu | |
| **EP allreduce (#25)** | **58.9** | **22.7%** | **N/A (EP=1)** | |
| **合计** | **259.2** | **100%** | | |

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
