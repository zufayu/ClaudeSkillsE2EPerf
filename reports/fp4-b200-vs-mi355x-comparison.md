# FP4 B200 vs MI355X Performance Comparison (MTP0, Chat c=64)

**Date:** 2026-04-04
**Scenario:** Chat (ISL=1024, OSL=1024), Concurrency=64, MTP=0 (throughput config)
**B200:** 8x B200, TRT-LLM 1.2.0rc6.post2, NVFP4
**MI355X:** 8x MI355X, ATOM 0.1.3.dev1, MXFP4

## 1. Five-Metric Comparison Table

### B200 NVFP4

| Config | Total Tput (tok/s) | Output Tput (tok/s) | TPOT p50 (ms) | TTFT p50 (ms) | Interactivity (tok/s/user) |
|--------|-------------------:|--------------------:|--------------:|--------------:|---------------------------:|
| EP1-TP8 | 7602.5 | 3800.4 | 16.5 | 83.0 | 60.6 |
| EP8-TP8 | 6307.3 | 3153.0 | 18.9 | 632.0 | 52.9 |
| **EP1 vs EP8** | **+20.5%** | **+20.5%** | **-12.7%** | **-86.9%** | **+14.6%** |

### MI355X MXFP4 (ROCm 7.2.1 — zufa_atom2)

| Config | Total Tput (tok/s) | Output Tput (tok/s) | TPOT p50 (ms) | TTFT p50 (ms) | Interactivity (tok/s/user) |
|--------|-------------------:|--------------------:|--------------:|--------------:|---------------------------:|
| EP1-TP8 | 6961.7 | 3480.1 | 17.9 | 94.7 | 55.9 |
| EP8-TP8 | 6558.0 | 3278.2 | 19.0 | 96.8 | 52.6 |
| EP1-TP4 | 5054.8 | 2526.8 | 24.8 | 104.5 | 40.4 |
| **EP1 vs EP8** | **+6.2%** | **+6.2%** | **-5.8%** | **-2.2%** | **+6.3%** |
| **EP1-TP8 vs EP1-TP4** | **+37.7%** | **+37.7%** | **-27.8%** | **-9.4%** | **+38.5%** |

### MI355X MXFP4 (ROCm 7.1.1 — zufa_atom)

| Config | Total Tput (tok/s) | Output Tput (tok/s) | TPOT p50 (ms) | TTFT p50 (ms) | Interactivity (tok/s/user) |
|--------|-------------------:|--------------------:|--------------:|--------------:|---------------------------:|
| EP1-TP8 | 6833.0 | 3415.7 | 18.2 | 93.5 | 55.0 |
| EP8-TP8 | 6470.5 | 3234.5 | 19.2 | 90.6 | 52.1 |
| EP1-TP4 | 4906.9 | 2452.9 | 25.5 | 104.4 | 39.3 |

### ROCm Version Impact (7.1.1 vs 7.2.1)

| Config | 7.2.1 Output Tput | 7.1.1 Output Tput | Delta |
|--------|-------------------:|-------------------:|------:|
| EP1-TP8 | 3480.1 | 3415.7 | +1.9% |
| EP8-TP8 | 3278.2 | 3234.5 | +1.4% |
| EP1-TP4 | 2526.8 | 2452.9 | +3.0% |

ROCm 7.2.1 consistently ~1.5-3% faster than 7.1.1.

## 2. Head-to-Head: B200 vs MI355X (best configs)

Using B200 post2 and MI355X ROCm 7.2.1 (best available for each):

| Metric | B200 EP1-TP8 | MI355X EP1-TP8 | B200 Lead |
|--------|-------------:|---------------:|----------:|
| Total Tput (tok/s) | 7602.5 | 6961.7 | **+9.2%** |
| Output Tput (tok/s) | 3800.4 | 3480.1 | **+9.2%** |
| TPOT p50 (ms) | 16.5 | 17.9 | **-7.8%** (lower=better) |
| TTFT p50 (ms) | 83.0 | 94.7 | **-12.4%** (lower=better) |
| Interactivity | 60.6 | 55.9 | **+8.4%** |

| Metric | B200 EP8-TP8 | MI355X EP8-TP8 | B200 Lead |
|--------|-------------:|---------------:|----------:|
| Total Tput (tok/s) | 6307.3 | 6558.0 | **-3.8%** |
| Output Tput (tok/s) | 3153.0 | 3278.2 | **-3.8%** |
| TPOT p50 (ms) | 18.9 | 19.0 | **-0.5%** (lower=better) |
| TTFT p50 (ms) | 632.0 | 96.8 | **+553%** (B200 much worse) |
| Interactivity | 52.9 | 52.6 | **+0.6%** |

## 3. Key Findings

### EP Configuration Impact

1. **B200 strongly favors EP1:** EP1 is 20.5% faster than EP8 in throughput. EP8 also has a catastrophic TTFT regression (83ms → 632ms), likely due to expert routing overhead across all 8 GPUs at prefill time.

2. **MI355X mildly favors EP1:** EP1 is only 6.2% faster than EP8. No TTFT regression (94.7 vs 96.8ms). ATOM's expert parallel implementation appears more efficient.

3. **TP4 vs TP8 on MI355X:** TP8 is 37.7% faster than TP4, showing strong scaling benefit from more GPUs in tensor parallel.

### Cross-Platform Comparison

4. **EP1-TP8 (best config for both):** B200 leads by ~9% across all metrics. This is significantly smaller than the ~15% gap seen in FP8 (which was attributed to TRT-LLM post2→post3 MoE optimizations).

5. **EP8-TP8:** MI355X actually **beats B200 by 3.8%** in throughput, and has **6.5x better TTFT** (96.8 vs 632ms). B200's EP8 implementation has a clear TTFT bottleneck.

6. **The "best config" matters:** If deploying with EP1, B200 wins by 9%. If deploying with EP8 (e.g., for cost/power reasons), MI355X wins by 4% with dramatically better latency.

### ROCm Version Impact

7. **ROCm 7.2.1 vs 7.1.1:** Consistent 1.5-3% improvement, smaller than the ~6% gap seen in FP8 CI validation. FP4 kernels may be less affected by ROCm version changes.

## 4. Cross-Validation

| Source | Config | Output Tput | Notes |
|--------|--------|------------:|-------|
| Bench (rocm721) | MI355X EP1-TP8 | 3480.1 | Standard bench run |
| Profiling | MI355X EP1-TP8 | 3150.6 | ~9.5% lower (profiling overhead, expected) |
| Bench (rocm721) | MI355X EP8-TP8 | 3278.2 | Standard bench run |
| Profiling | MI355X EP8-TP8 | 3006.6 | ~8.3% lower (profiling overhead, expected) |
| CI reference | MI355X EP1-TP8 | ~3400-3500 | Matches bench within noise |

All data sources are internally consistent. Profiling runs show expected 8-10% throughput reduction due to trace capture overhead.

## 5. Data Sources

| Directory | Platform | Config | Source |
|-----------|----------|--------|--------|
| `results/b200_dsr_fp4/b200_dsr_fp4_mtp0_ep1_tp8/` | B200 | EP1-TP8 | Self-run bench |
| `results/b200_dsr_fp4/b200_dsr_fp4_mtp0_ep8_tp8/` | B200 | EP8-TP8 | Self-run bench |
| `results/mi355x_dsr_mxfp4/mi355x_dsr_mxfp4_mtp0_ep1_tp8_rocm721/` | MI355X | EP1-TP8 | Self-run bench (ROCm 7.2.1) |
| `results/mi355x_dsr_mxfp4/mi355x_dsr_mxfp4_mtp0_ep8_tp8_rocm721/` | MI355X | EP8-TP8 | Self-run bench (ROCm 7.2.1) |
| `results/mi355x_dsr_mxfp4/mi355x_dsr_mxfp4_mtp0_ep1_tp4_rocm721/` | MI355X | EP1-TP4 | Self-run bench (ROCm 7.2.1) |
| `results/mi355x_dsr_mxfp4/mi355x_dsr_mxfp4_mtp0_ep1_tp8_rocm711/` | MI355X | EP1-TP8 | Self-run bench (ROCm 7.1.1) |
| `results/mi355x_dsr_mxfp4/mi355x_dsr_mxfp4_mtp0_ep8_tp8_rocm711/` | MI355X | EP8-TP8 | Self-run bench (ROCm 7.1.1) |
| `results/mi355x_dsr_mxfp4/mi355x_dsr_mxfp4_mtp0_ep1_tp4_rocm711/` | MI355X | EP1-TP4 | Self-run bench (ROCm 7.1.1) |
| `results/mi355x_dsr_mxfp4/mi355x_dsr_mxfp4_mtp0_ep1_tp8_profiling/` | MI355X | EP1-TP8 | Profiling run |
| `results/mi355x_dsr_mxfp4/mi355x_dsr_mxfp4_mtp0_ep8_tp8_profiling/` | MI355X | EP8-TP8 | Profiling run |
