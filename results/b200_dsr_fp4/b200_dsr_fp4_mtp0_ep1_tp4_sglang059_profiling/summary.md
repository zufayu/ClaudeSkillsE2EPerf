# DeepSeek R1 Profiling Results (SGLang)
## B200 Torch Profiler Trace

| Config | Scenario | CONC | Total Tput | Output Tput | TPOT (ms) | TTFT (ms) |
|--------|----------|------|------------|-------------|-----------|-----------|
| profiling | chat | 64 | 5893.7 | 2946.2 | 19.1 | 412.6 |

## Kernel Breakdown (per decode step, averaged)

```
==============================================================================================================
Per-Decode-Step Kernel Breakdown (averaged over 147 steps, total=21018.5μs)
==============================================================================================================
  1 | moe: gate_up_GEMM                             |   5993.2 |  28.5 |  58.4 | bmm_E2m1_E2m1E2m1_Fp32_bA16_bB16_bC16...
  2 | moe: down_GEMM                                |   3905.8 |  18.6 |  58.4 | bmm_Bfloat16_E2m1E2m1_Fp32_bA16_bB16_...
  3 | out_proj/shared: FP4_GEMM                     |   2033.2 |   9.7 | 184.3 | _ZN7cutlass13device_kernelIN10flashin...
  4 | qkv_proj: qkv_a_proj_GEMM                     |   1915.3 |   9.1 | 119.8 | nvjet_sm100_tst_64x32_64x16_2x1_2cta_...
  5 | comm: lamport_AR+RMSNorm                      |   1426.1 |   6.8 | 121.8 | void flashinfer::trtllm_allreduce_fus...
  6 | rope_attn: Attention_FMHA                     |   1262.2 |   6.0 |  61.4 | fmhaSm100fKernel_QkvE4m3OBfloat16HQk5...
```
