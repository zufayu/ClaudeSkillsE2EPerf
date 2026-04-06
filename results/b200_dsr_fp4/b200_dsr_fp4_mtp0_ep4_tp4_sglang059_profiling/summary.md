# DeepSeek R1 Profiling Results (SGLang)
## B200 Torch Profiler Trace

| Config | Scenario | CONC | Total Tput | Output Tput | TPOT (ms) | TTFT (ms) |
|--------|----------|------|------------|-------------|-----------|-----------|
| profiling | chat | 64 | 6145.5 | 3072.1 | 19.0 | 406.2 |

## Kernel Breakdown (per decode step, averaged)

```
==============================================================================================================
Per-Decode-Step Kernel Breakdown (averaged over 20 steps, total=21862.1μs)
==============================================================================================================
  1 | moe: gate_up_GEMM                             |   5661.3 |  25.9 |  60.9 | bmm_E2m1_E2m1E2m1_Fp32_bA16_bB16_bC16...
  2 | moe: down_GEMM                                |   3127.4 |  14.3 |  60.9 | bmm_Bfloat16_E2m1E2m1_Fp32_bA16_bB16_...
  3 | qkv_proj: qkv_a_proj_GEMM                     |   2991.1 |  13.7 | 125.0 | nvjet_sm100_tst_64x32_64x16_2x1_2cta_...
  4 | comm: lamport_AR+RMSNorm                      |   2480.2 |  11.3 | 127.0 | void flashinfer::trtllm_allreduce_fus...
  5 | out_proj/shared: FP4_GEMM                     |   1775.9 |   8.1 | 192.2 | _ZN7cutlass13device_kernelIN10flashin...
  6 | rope_attn: Attention_FMHA                     |   1257.2 |   5.8 |  64.0 | fmhaSm100fKernel_QkvE4m3OBfloat16HQk5...
```
