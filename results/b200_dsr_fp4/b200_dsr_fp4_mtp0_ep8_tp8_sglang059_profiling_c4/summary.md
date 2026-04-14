# DeepSeek R1 Profiling Results (SGLang)
## B200 Torch Profiler Trace

| Config | Scenario | CONC | Total Tput | Output Tput | TPOT (ms) | TTFT (ms) |
|--------|----------|------|------------|-------------|-----------|-----------|
| profiling | chat | 4 | 876.3 | 436.0 | 7.8 | 133.9 |

## Kernel Breakdown (per decode step, averaged)

```
==============================================================================================================
Per-Decode-Step Kernel Breakdown (averaged over 179 steps, total=11122.5μs)
==============================================================================================================
  1 | comm: lamport_AR+RMSNorm                      |   1899.4 |  17.1 | 121.6 | void flashinfer::trtllm_allreduce_fus...
  2 | other: void fused_a_gemm_kernel<1, 2112, 7168, 16, 8, 256 |   1492.1 |  13.4 |  61.3 | void fused_a_gemm_kernel<1, 2112, 716...
  3 | out_proj/shared: FP4_GEMM                     |   1425.7 |  12.8 | 183.9 | _ZN7cutlass13device_kernelIN10flashin...
  4 | moe: gate_up_GEMM                             |    859.0 |   7.7 |  58.3 | bmm_E2m1_E2m1E2m1_Fp32_bA16_bB16_bC16...
  5 | rope_attn: Attention_FMHA                     |    806.5 |   7.3 |  61.3 | fmhaSm100fKernel_QkvE4m3OBfloat16HQk5...
  6 | moe: down_GEMM                                |    660.3 |   5.9 |  58.3 | bmm_Bfloat16_E2m1E2m1_Fp32_bA16_bB16_...
```
