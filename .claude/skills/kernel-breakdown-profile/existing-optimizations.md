# Existing Optimizations & Known Fix Paths

Mapping from diagnosed kernel bottlenecks to known optimizations on B200 and MI355X.

Data sources: `reports/fp4-b200-vs-mi355x-breakdown.md`, `reports/mtp3-vs-mtp0-analysis.md`, TRT-LLM changelog, ATOM/aiter release notes.

## Bottleneck -> Optimization Map

| Bottleneck | Diagnosis | B200 Known Optimization | MI355X Known Optimization |
|-----------|-----------|------------------------|--------------------------|
| **MoE GEMM slow** (moe_gemm >30%) | MoE grouped GEMM dominates decode | TRT-LLM post3 MoE fusion (PR#11143): 3 kernel optimizations, ~15% E2E speedup over post2. `bmm_E2m1` for FP4, `bmm_BF16` for FP8 | ATOM MXFP4 quantization. aiter MoE kernel updates. Check aiter version for latest grouped GEMM improvements |
| **Attention slow** (fmha >10%) | FP8 KV cache overhead or suboptimal attention variant | `fmhaSm100f QkvE4m3`: SM100-native FP8 E4M3 KV cache attention. Verify SM100 kernel selected (not SM90 fallback) | FlashAttention ROCm build. Check `SGLANG_ATTENTION_BACKEND` or vLLM attention backend config |
| **TP AllReduce + Norm** (tp_allreduce+norm >8%) | Communication overhead on TP=8 | `userbuffers_rmsnorm`: Fused AllReduce + RMSNorm + residual add. Eliminates separate norm kernel launch | RCCL AllReduce. No known fused norm+AR for MI355X yet |
| **EP AllReduce** (moe_finalize >15%) | EP=8 cross-GPU synchronization | `moefinalize_lamport`: Lamport-style EP AllReduce, lower latency than NCCL for small messages. High variance (19-59 us) due to GPU sync | Not applicable (MI355X typically uses TP=4, EP=1) |
| **qkv_a_proj high variance** (std >10 us) | splitK scheduling instability | nvjet splitK + splitKreduce: cuBLAS auto-tunes tile shape. Variance is expected for small-batch GEMMs with splitK | N/A |
| **Quantize overhead** (quantize >5%) | FP4/FP8 quantize not fused into GEMM | TRT-LLM fuses quantize into GEMM pipeline (out_proj, moe_gemm). Standalone quantize kernels indicate fusion not applied -- check config | ATOM MXFP4 handles quantize in aiter kernels |
| **MTP DAR low** (<60% on MI355X) | Draft acceptance rate drops at high concurrency | N/A (B200 TRT-LLM DAR ~80% across concurrencies) | ATOM MTP3 schedule optimization. Chat c>=64 DAR drops to ~49% -- known issue. Reasoning DAR ~65% is acceptable |
| **TTFT high** (>200ms at c>=64) | Prefill scheduling overhead or prefill-decode interference | B200 TTFT improves dramatically with MTP3 at high concurrency (c=64: 494ms->123ms). Check piecewise CUDA graphs | MI355X prefill-decode interleaving analysis: use `scripts/analyze_prefill_impact.py` to detect prefill blocking decode |
| **Memory operations** (memcpy/memset >3%) | Excessive data copies | Check for unnecessary D2D copies. KV cache block reuse can reduce memcpy. `enable_block_reuse: false` in current config | Similar: check for unnecessary tensor copies in PyTorch path |
| **Router overhead** (router >5%) | TopK + sort on 256 experts | Router GEMM `[bs,7168]x[7168,256]` + topK + sort. Normal overhead for 256-expert MoE | Same architecture, similar overhead expected |

## Framework-Level Optimizations

### B200 TRT-LLM Config Knobs

| Knob | Effect | When to Use |
|------|--------|-------------|
| `moe_config.backend: TRTLLM` | Use TRT-LLM native MoE (default) | EP>=1, best for FP4/FP8 |
| `moe_config.backend: CUTLASS` | Use CUTLASS grouped GEMM | DP Attention enabled |
| `moe_config.backend: DEEPGEMM` | Use DeepGEMM | FP8 + DP + high concurrency (SM90 only, not SM100) |
| `enable_piecewise_cuda_graph: true` | Piecewise CUDA graph capture | High concurrency, reduces launch overhead |
| `kv_cache_config.dtype: fp8` | FP8 KV cache | Always (reduces memory, enables fmha E4M3) |
| `speculative_config.num_nextn_predict_layers: 3` | MTP3 | Latency config (trades compute for lower TPOT) |
| `batch_wait_timeout_iters / max_tokens_ratio` | Delay batching | FP8 high concurrency (c>=64 chat) |

### MI355X ATOM Config Knobs

| Knob | Effect | When to Use |
|------|--------|-------------|
| `--max-model-len 2248` | Limit KV cache allocation | Standard SA benchmark config |
| `--gpu-memory-utilization 0.90` | GPU memory fraction | Default, matches SA CI |
| `--enforce-eager false` | Allow CUDA graphs | Default, better performance |
| `--enable-mtp` | Enable Multi-Token Prediction | Latency config |

## B200 Kernel Name Reference

Understanding TRT-LLM kernel names:

| Kernel Pattern | Meaning |
|---------------|---------|
| `nvjet tst` | cuBLAS GEMM (BF16), `tst` = tile scheduler type |
| `nvjet ootst Avec16UE4M3` | cuBLAS FP4 GEMM (E2M1 data + E4M3 block scale) |
| `bmm_E2m1` | MoE grouped FP4 GEMM (explicit E2M1 in name) |
| `bmm_BF16` | MoE grouped BF16 GEMM |
| `fmhaSm100f QkvE4m3` | SM100 FlashMHA with FP8 E4M3 QKV |
| `splitKreduce` | splitK reduction pass (follows a splitK GEMM) |
| `RMSNormKernel` | Standalone RMSNorm |
| `userbuffers_rmsnorm` | Fused AllReduce + RMSNorm |
| `userbuffers_allgather` | AllGather via userbuffers |
| `moefinalize_lamport` | Lamport EP AllReduce + MoE finalize |
| `applyMLARopeAndAssignQKV` | Fused RoPE + KV cache assignment |

## Precision Verification

When diagnosing kernel performance, verify the actual precision:

1. **Check `hf_quant_config.json`** in model directory: `exclude_modules` lists layers that stay BF16 even in FP4 models (MLA projections: q_a, q_b, kv_a, kv_b).
2. **Kernel name patterns**: `nvjet tst` (no E4M3) = BF16; `nvjet ootst` with preceding `quantize` = FP4; `bmm_E2m1` = FP4.
3. **FP8 E4M3 in fmha**: Refers to KV cache precision, not compute precision.

## Performance Targets (SA InferenceX Reference)

| Config | Metric | B200 Reference | MI355X Reference |
|--------|--------|---------------|-----------------|
| FP8 chat 1K/1K c=128 | Output TPS/GPU | ~650 (post3) | ~550 (ATOM mtp0) |
| FP8 reasoning 1K/8K c=64 | Output TPS/GPU | ~630 (post3) | ~550 (ATOM mtp0) |
| FP4 chat 1K/1K c=64 | Output TPS/GPU | ~490 (post2) | ~625 (ATOM MXFP4) |
