# DeepSeek R1 Profiling Results (SGLang)
## B200 Torch Profiler Trace

| Config | Scenario | CONC | Total Tput | Output Tput | TPOT (ms) | TTFT (ms) |
|--------|----------|------|------------|-------------|-----------|-----------|
| profiling | chat | 4 | 872.9 | 434.3 | 7.8 | 134.5 |

## Kernel Breakdown (per decode step, averaged)

```
==============================================================================================================
Per-Decode-Step Kernel Breakdown (averaged over 179 steps, total=11159.7μs)
==============================================================================================================
  1 | comm: lamport_AR+RMSNorm                      |   1916.8 |  17.2 | 121.6 | void flashinfer::trtllm_allreduce_fus...
  2 | other: void fused_a_gemm_kernel<1, 2112, 7168, 16, 8, 256 |   1539.2 |  13.8 |  61.3 | void fused_a_gemm_kernel<1, 2112, 716...
  3 | out_proj/shared: FP4_GEMM                     |   1460.0 |  13.1 | 183.9 | _ZN7cutlass13device_kernelIN10flashin...
  4 | moe: gate_up_GEMM                             |    870.5 |   7.8 |  58.3 | bmm_E2m1_E2m1E2m1_Fp32_bA16_bB16_bC16...
  5 | rope_attn: Attention_FMHA                     |    797.3 |   7.1 |  61.3 | fmhaSm100fKernel_QkvE4m3OBfloat16HQk5...
  6 | moe: down_GEMM                                |    670.7 |   6.0 |  58.3 | bmm_Bfloat16_E2m1E2m1_Fp32_bA16_bB16_...
```
