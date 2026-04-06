# DeepSeek R1 Profiling Results (SGLang)
## B200 Torch Profiler Trace

| Config | Scenario | CONC | Total Tput | Output Tput | TPOT (ms) | TTFT (ms) |
|--------|----------|------|------------|-------------|-----------|-----------|
| profiling | chat | 64 | 6025.6 | 3012.1 | 19.1 | 405.9 |

## Kernel Breakdown (per decode step, averaged)

```
==============================================================================================================
Per-Decode-Step Kernel Breakdown (averaged over 147 steps, total=21798.4μs)
==============================================================================================================
  1 | moe: gate_up_GEMM                             |   5986.2 |  27.5 |  58.4 | bmm_E2m1_E2m1E2m1_Fp32_bA16_bB16_bC16...
  2 | moe: down_GEMM                                |   3293.0 |  15.1 |  58.4 | bmm_Bfloat16_E2m1E2m1_Fp32_bA16_bB16_...
  3 | qkv_proj: qkv_a_proj_GEMM                     |   2801.5 |  12.9 | 119.8 | nvjet_sm100_tst_64x32_64x16_2x1_2cta_...
  4 | comm: lamport_AR+RMSNorm                      |   2325.8 |  10.7 | 121.8 | void flashinfer::trtllm_allreduce_fus...
  5 | out_proj/shared: FP4_GEMM                     |   1722.1 |   7.9 | 184.3 | _ZN7cutlass13device_kernelIN10flashin...
  6 | rope_attn: Attention_FMHA                     |   1239.2 |   5.7 |  61.4 | fmhaSm100fKernel_QkvE4m3OBfloat16HQk5...
```
