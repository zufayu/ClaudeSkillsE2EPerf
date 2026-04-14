# DeepSeek R1 Profiling Results (SGLang)
## B200 Torch Profiler Trace

| Config | Scenario | CONC | Total Tput | Output Tput | TPOT (ms) | TTFT (ms) |
|--------|----------|------|------------|-------------|-----------|-----------|
| profiling | chat | 4 | 889.0 | 442.3 | 7.7 | 131.1 |

## Kernel Breakdown (per decode step, averaged)

```
==============================================================================================================
Per-Decode-Step Kernel Breakdown (averaged over 179 steps, total=10909.9μs)
==============================================================================================================
  1 | comm: lamport_AR+RMSNorm                      |   1796.9 |  16.5 | 121.6 | void flashinfer::trtllm_allreduce_fus...
  2 | other: void fused_a_gemm_kernel<1, 2112, 7168, 16, 8, 256 |   1430.6 |  13.1 |  61.3 | void fused_a_gemm_kernel<1, 2112, 716...
  3 | out_proj/shared: FP4_GEMM                     |   1375.6 |  12.6 | 183.9 | _ZN7cutlass13device_kernelIN10flashin...
  4 | moe: gate_up_GEMM                             |    881.1 |   8.1 |  58.3 | bmm_E2m1_E2m1E2m1_Fp32_bA16_bB16_bC16...
  5 | rope_attn: Attention_FMHA                     |    807.9 |   7.4 |  61.3 | fmhaSm100fKernel_QkvE4m3OBfloat16HQk5...
  6 | moe: down_GEMM                                |    660.8 |   6.1 |  58.3 | bmm_Bfloat16_E2m1E2m1_Fp32_bA16_bB16_...
```
