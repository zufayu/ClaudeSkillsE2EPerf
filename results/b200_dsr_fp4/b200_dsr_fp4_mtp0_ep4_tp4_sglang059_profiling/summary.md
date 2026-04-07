# DeepSeek R1 Profiling Results (SGLang)
## B200 Torch Profiler Trace

| Config | Scenario | CONC | Total Tput | Output Tput | TPOT (ms) | TTFT (ms) |
|--------|----------|------|------------|-------------|-----------|-----------|
| profiling | chat | 64 | 6060.3 | 3029.4 | 19.0 | 402.6 |

## Kernel Breakdown (per decode step, averaged)

```
==============================================================================================================
Per-Decode-Step Kernel Breakdown (averaged over 147 steps, total=21436.4μs)
==============================================================================================================
  1 | moe: gate_up_GEMM                             |   5969.0 |  27.8 |  58.4 | bmm_E2m1_E2m1E2m1_Fp32_bA16_bB16_bC16...
  2 | moe: down_GEMM                                |   3302.8 |  15.4 |  58.4 | bmm_Bfloat16_E2m1E2m1_Fp32_bA16_bB16_...
  3 | qkv_proj: qkv_a_proj_GEMM                     |   2639.3 |  12.3 | 119.8 | nvjet_sm100_tst_64x32_64x16_2x1_2cta_...
  4 | comm: lamport_AR+RMSNorm                      |   2154.0 |  10.0 | 121.8 | void flashinfer::trtllm_allreduce_fus...
  5 | out_proj/shared: FP4_GEMM                     |   1691.3 |   7.9 | 184.2 | _ZN7cutlass13device_kernelIN10flashin...
  6 | rope_attn: Attention_FMHA                     |   1249.2 |   5.8 |  61.4 | fmhaSm100fKernel_QkvE4m3OBfloat16HQk5...
```
