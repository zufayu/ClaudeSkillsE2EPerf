---
name: Cross-Platform Kernel Catalog (B200 vs MI355X)
description: Known kernel families, structural gaps, and verified per-operator ratios for DeepSeek-R1 inference on B200 (TRT-LLM/SGLang) vs MI355X (ATOM). Consult before claiming any "new discovery" in profiling analysis.
type: reference
originSessionId: ef6a5f82-0e8a-4423-8898-b406f3722dca
---
# Cross-Platform Kernel Catalog — B200 vs MI355X

> Consult this catalog before labeling any profiling finding as "new."
> Data sources: `reports/fp4-b200-vs-mi355x-breakdown.md` (v28), memory files `reference_b200_pdl.md`, `reference_dual_stream_overlap.md`.

## 1. B200-Specific Kernel Families

| Family | Trace keywords | Mechanism | Perf impact |
|--------|---------------|-----------|-------------|
| **PDL overlap** | `cudaGridDependencySynchronize`, tight same-stream kernel gaps | SM100+ Programmatic Dependent Launch: tail of kernel A overlaps preamble of kernel B on same stream | ~27μs/layer (28 boundaries × ~2μs) |
| **Dual-stream parallelism** | stream 23 + stream 385, `lamport`, cross-stream events | Shared expert + MoE routing on separate CUDA stream from main compute | ~39μs/layer cross-stream overlap |
| **userbuffers_rmsnorm** | `userbuffers_rmsnorm` | Fused AllReduce + RMSNorm + residual add, eliminates separate norm launch | TP8: 15.15μs vs MI355X 24.92μs (save ~10μs) |
| **moefinalize_lamport** | `moefinalize_lamport` | Lamport-style EP AllReduce + weighted sum + residual + pre-attn RMSNorm, all fused | 33.11μs (TP8), overlaps with qkv_a_proj via PDL |
| **allreduce_fusion_kernel_oneshot_lamport** | `allreduce_fusion_kernel_oneshot_lamport` | SGLang flashinfer lamport allreduce on symmetric memory, separate stream | 25.9μs (TP4), overlaps with qkv_a GEMM |
| **SM100-native attention** | `fmhaSm100f`, `QkvE4m3` | Blackwell-native FlashMHA with FP8 E4M3 KV cache, includes multi-head reduce | 20.3-20.7μs (fused reduce) |
| **SM100 FP4 GEMM** | `DeviceGemmFp4GemmSm100`, `nvjet_ootst`, `bmm_E2m1` | Blackwell FP4 (E2M1) GEMM via cuBLAS/CUTLASS | gate_up: 59.55μs(TP8), 101.7μs(TP4) |
| **splitK two-phase** | `nvjet_splitK_TNT` + `splitKreduce` | cuBLAS splits GEMM into parallel tiles then reduces; MI355X CK does single-kernel | 3.7μs overhead per splitK GEMM |
| **Block-scale FP4 quantize** | `quantize_with_block_size`, `cvt_fp16_to_fp4` | BF16→FP4 block-scale quantization, standalone kernel (not always fused) | ~2-4μs per occurrence |

## 2. MI355X-Specific Kernel Families

| Family | Trace keywords | Mechanism | Notes |
|--------|---------------|-----------|-------|
| **ATOM --mark-trace** | `--mark-trace` flag | Built-in per-module Kineto annotation, superseded roctx markers | Always use this for MI355X trace collection |
| **aiter MoE GEMM** | `kernel_moe_mxgemm_2lds` (CK) | MXFP4 grouped GEMM from aiter library, CK tile-based | gate_up: 66.52μs(TP8), 119.5μs(TP4) |
| **aiter batched GEMM** | `batched_gemm_a8w8_M32_N*_K*` | FP8 batched GEMM for MLA projections | Used for uk_gemm, uv_gemm |
| **aiter fused norm+quant** | `add_rmsnorm_quant_kernel`, `_fused_rms_fp8_group_quant` | Fused RMSNorm + FP8 quantization in single kernel | Saves separate quant launch |
| **aiter fused MoE quant+sort** | `_fused_dynamic_mxfp4_quant_moe_sort_kernel` | BF16→MXFP4 quantize + expert sort in single kernel | 8-9μs per occurrence |
| **aiter MLA attention** | `mla_a8w8_qh16_qseqlen1` | FP8 MLA decode attention | 18-25.5μs depending on TP |
| **MLA reduce (separate)** | `kn_mla_reduce_v1_ps` | Multi-head reduce after MLA attention, separate kernel | 7.2-8.9μs; B200 fuses this into fmhaSm100f |
| **3-phase MoE sort** | `MoeSorting_P0_v2` + `MoeSorting_P23` | Multi-phase expert sorting; B200 uses single-phase | Extra ~5.8μs vs B200 |
| **RCCL reduce_scatter** | `reduce_scatter_cross_device_store`, `local_device_load_rmsnorm` | TP allreduce via xGMI Infinity Fabric | Single-stream, no overlap with compute |
| **Tensile GEMM** | `Cijk_BBS_MT*_SK*_ISA950` | AMD Tensile BF16 GEMM for non-quantized paths | qkv_a: 16.1μs(TP4), router: 9.2μs(TP4) |
| **Single-stream execution** | All ops on one HIP stream | HIP Graph captures entire layer on single stream, no multi-stream overlap | **0μs overlap** vs B200's ~66μs |
| **Shared expert fused into MoE** | No separate shared_expert kernels | Shared expert treated as always-active expert inside grouped GEMM | Eliminates separate shared expert kernels |

## 3. Structural Gaps (Architecture-Level)

| Gap | B200 | MI355X | Delta | Reducible? |
|-----|------|--------|-------|-----------|
| **PDL overlap** | ~27μs/layer | 0 (no equivalent in ROCm) | 27μs | No — hardware/runtime feature gap |
| **Dual-stream overlap** | ~39μs/layer (TP4: shared exp ∥ MoE routing) | 0 (single HIP stream) | 39μs | Partially — AMD could implement multi-stream, but HIP Graph currently single-stream |
| **Total overlap** | ~66μs/layer (19% of kernel sum) | 0 | 66μs | Partially — structural advantage |
| **NVLink5 vs IF4 BW** | 1,800 GB/s | 1,075 GB/s | 1.67x | No — interconnect hardware |
| **TP allreduce+norm fusion** | userbuffers_rmsnorm (15.15μs) | reduce_scatter + rmsnorm (24.92μs) | 10μs | Yes — fusion kernel development |
| **MLA reduce fusion** | Fused into fmhaSm100f | Separate kn_mla_reduce_v1 (7-9μs) | 7-9μs | Yes — kernel development |
| **FP4 quant standalone** | 2-4μs per quantize kernel | N/A (MXFP4 fused into aiter) | — | MI355X is better here |

## 4. Verified Per-Operator Ratios

### TP=8, EP=8, c=64, chat 1K/1K (FP4/MXFP4)

| Operator | B200 μs | MI355X μs | Ratio (MI/B) | Notes |
|----------|---------|-----------|-------------|-------|
| gate_up GEMM (+SwiGLU) | 59.55 | 66.52 | 1.12x | Pure GEMM efficiency, per-GPU weights identical |
| down GEMM | 32.77 | 34.24 | 1.04x | Near parity |
| FMHA/MLA | 20.67 | 18.00 + 8.92 = 26.92 | 1.30x | MI355X split into attention + separate reduce |
| qkv_a GEMM | 25.12 + 3.68 = 28.80 | 11.48 | 0.40x | **MI355X wins**: FP8 GEMM vs B200 BF16 + splitK |
| TP allreduce+norm | 15.15 | 24.92 | 1.64x | userbuffers fusion advantage |
| EP allreduce (moefinalize) | 33.11 | 21.12 | 0.64x | **MI355X wins**: simpler TP RS vs lamport full EP |
| **Layer sum** | **276.91** | **267.58** | **0.97x** | **MI355X kernel sum is actually faster** |

### TP=4, EP=4, c=64, chat 1K/1K (FP4/MXFP4, SGLang vs ATOM)

| Metric | B200 | MI355X | Ratio |
|--------|------|--------|-------|
| Kernel sum | 353.2 μs | 399.6 μs | 1.13x |
| Overlap | 66.3 μs (19.2%) | 0 μs | — |
| Critical path (walltime) | 286.9 μs | 399.6 μs | 1.39x |
| E2E decode (61 layers) | ~17.5 ms | ~23.7 ms | 1.35x |

**Key insight**: At TP=8, MI355X kernel sum is actually **faster** than B200 (267.58 vs 276.91). The E2E gap comes entirely from B200's ~66μs/layer overlap savings + framework-level differences.

## 5. Framework Toggle/Env Table

| Toggle | Platform | Effect on trace shape |
|--------|----------|----------------------|
| `speculative_config.num_nextn_predict_layers: 3` | B200 TRT-LLM | Enables MTP3, trades compute for TPOT |
| `enable_piecewise_cuda_graph: true` | B200 TRT-LLM | Piecewise CUDA graph, reduces launch overhead |
| `kv_cache_config.dtype: fp8` | B200 TRT-LLM | FP8 KV cache, enables fmhaSm100f E4M3 |
| `moe_config.backend: TRTLLM/CUTLASS/DEEPGEMM` | B200 TRT-LLM | Changes MoE GEMM kernel family entirely |
| `enable_flashinfer_allreduce_fusion=True` | B200 SGLang | Lamport allreduce on separate stream, enables overlap |
| `--enable-mtp` | MI355X ATOM | Enables MTP, DAR varies by scenario |
| `--enforce-eager false` | MI355X ATOM | Allow HIP graphs (default, better perf) |
| `--max-model-len` | MI355X ATOM | Limits KV cache, affects memory pressure |
| EP size (1/4/8) | Both | Changes per-GPU expert count, allreduce pattern |
| TP size (4/8) | Both | Changes per-GPU dimension, GEMM shapes |
