---
name: B200 Dual-Stream Overlap Analysis
description: How B200 uses two CUDA streams (23 main, 385 aux) for cross-stream kernel parallelism in DeepSeek-V3. Verified from trace with exact overlap breakdown.
type: reference
originSessionId: ef6a5f82-0e8a-4423-8898-b406f3722dca
---
# Dual-Stream Overlap on B200 (DeepSeek-V3)

## Stream Assignment

- **Stream 23 (main)**: EP_AR, qkv_a, MLA attention chain, o_proj, EP_AR#2, shared expert (quant+GEMM+SiLU+quant+GEMM), residual
- **Stream 385 (aux)**: q/k_norm#2, router, router_splitK, MoE routing (quant, copy, TopK, sort), gate_up, down, MoE_finalize

## Overlap Breakdown (exact, no double-counting)

Computed from per-stream walltime subtraction (10-layer avg, TP=4 EP=4 c=64):

| Type | μs/layer | % of total | Mechanism |
|------|----------|-----------|-----------|
| **Single-stream (PDL)** | **34.4** | **53%** | Same-stream kernel tail/preamble overlap |
| **Dual-stream** | **30.5** | **47%** | Cross-stream concurrent execution |
| **TOTAL** | **64.9** | **100%** | = kernel_sum(353.4) - walltime(288.5) |

Formula: `pdl = Σ(stream_ksum - stream_wall)`, `dual = total - pdl`

## PDL Overlap (34.4μs) — where it happens

| Boundary | μs | Notes |
|----------|---:|-------|
| EP_AR#1 ∥ qkv_a | ~24 | Cross-layer pipeline: moefinalize tail overlaps with next-layer qkv_a preamble |
| qkv_a ∥ splitK_reduce | ~1.5 | |
| splitK ∥ q/k_norm | ~1.2 | |
| q_b ∥ uk_gemm | ~1.4 | |
| router ∥ splitK_reduce | ~1.4 | On stream 385 |
| Other boundaries (~8 more) | ~5 | ~0.5-1μs each |

## Dual-Stream Overlap (30.5μs) — where it happens

| Window | Stream 23 | Stream 385 | overlap μs |
|--------|-----------|------------|----------:|
| EP_AR#2 ∥ router | EP_AR(11μs) | router_GEMM(13.5μs) | ~9 |
| shared expert ∥ MoE routing | shared_quant+GEMM(12.5μs) | splitK+quant+copy+TopK+sort(~20μs) | ~12 |
| shared_GEMM(down) ∥ gate_up start | shared_GEMM(5.2μs) | gate_up start | ~3 |
| SiLU+quant ∥ routing tail | SiLU+quant(~5μs) | TopK+sort tail | ~4 |
| q/k_norm dual | norm(3.2μs) | norm(2.4μs) | ~2.5 |

**Note**: o_proj_GEMM does NOT overlap with router. o_proj finishes → EP_AR starts → router starts 1.7μs after EP_AR. This was previously reported incorrectly.

## Architecture Dependence

| Model Architecture | PDL savings | Dual-stream savings | Total | % of kernel sum |
|---|---|---|---|---|
| Dense Transformer (Llama-style) | ~18μs | ~0 | ~18μs | ~5-8% |
| Standard MoE (no shared expert) | ~25μs | ~5μs | ~30μs | ~8-10% |
| **DeepSeek-V3 (MLA+MoE+shared)** | **~34μs** | **~31μs** | **~65μs** | **~18%** |

**MI355X gap**: ROCm has no PDL equivalent (~34μs structural). Dual-stream (~31μs) partially addressable via HIP multi-stream, but HIP Graph currently single-stream.
