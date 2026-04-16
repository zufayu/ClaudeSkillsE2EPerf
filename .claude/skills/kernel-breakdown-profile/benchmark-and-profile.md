# Kernel-Level Benchmark & Profile Workflow

Four-level profiling methodology for DeepSeek R1 671B inference on B200 and MI355X.

---

## Output Contract — Column Specifications

All triage reports must use these exact column formats. See `SKILL.md` for the three-table structure.

### Table 1: Kernel Table columns

| Column | Type | Description |
|--------|------|-------------|
| `#` | int | Row number (1-indexed) |
| `Kernel` | string | GPU kernel function name (use trace-original name, never simplify) |
| `Category` | enum | Auto-classified: MoE/Expert, Attention, GEMM/MatMul, Communication, Normalization, Quantization, RoPE, Activation, Memory, Sampling, Other |
| `GPU time (μs)` | float | Total GPU time in microseconds for this kernel across all launches |
| `Share%` | float | `GPU time / total GPU time × 100`, one decimal place |
| `Launches` | int | Number of kernel launches |
| `Platform` | enum | B200 / MI355X |
| `Source` | string | Trace file path or profiling run identifier |

### Table 2: Cross-Platform Delta Table columns

| Column | Type | Description |
|--------|------|-------------|
| `#` | int | Operator sequence number (1-15 for standard DeepSeek R1 decode layer) |
| `Operator` | string | Logical operator name from 15-operator decomposition |
| `B200 kernel` | string | B200 kernel name(s) for this operator |
| `B200 μs` | float | B200 GPU time |
| `MI355X kernel` | string | MI355X kernel name(s) |
| `MI355X μs` | float | MI355X GPU time |
| `GAP(B-M)` | float | `B200 μs - MI355X μs`. Positive = B200 slower |
| `GAP%` | float | `abs(GAP) / max(B200, MI355X) × 100` |
| `Bottleneck side` | enum | `B200` (B200 > MI355X by >10%), `MI355X` (MI355X > B200 by >10%), `parity` (within 10%) |

### Table 3: Optimization Opportunity Table columns

| Column | Type | Description |
|--------|------|-------------|
| `Operator` | string | From Table 2 |
| `GAP source` | enum | `kernel_efficiency` / `fusion_gap` / `overlap_gap` / `comm_bw` / `arch_difference` / `quant_difference` |
| `Known fix` | string | From `existing-optimizations.md` or catalog |
| `Catalog status` | enum | `existing` (known, may be applied/missing), `in-flight` (PR/development), `structural` (hardware gap), `new opportunity` |
| `Priority` | enum | `P1` (>10μs gap), `P2` (3-10μs), `P3` (<3μs), `skip` (structural, not actionable) |

---

## Cross-Platform Triage Workflow

### Prerequisites

1. Both platforms profiled with **same scenario** (chat/reasoning/summarize)
2. Same concurrency level (c=4 for low-latency, c=64 for throughput)
3. Same EP/TP configuration where possible (EP=8 TP=8 for 8-GPU parity, or EP=4 TP=4 for 4-GPU)
4. Decode-only data isolated (use `--phase decode` or steady-state batch selection)

### Steps

1. **Collect traces** (can run in parallel on different machines):
   ```bash
   # B200 (nsys or SGLang torch trace)
   bash scripts/collect_nsys_trace.sh --model $MODEL --mode bench --scenario chat --concurrency 64 --quant fp4 --config throughput
   # or: SGLang torch profiler
   bash scripts/collect_sglang_trace.sh --model $MODEL --scenario chat --concurrency 64

   # MI355X (ATOM Kineto)
   bash scripts/collect_atom_trace.sh --model $MODEL --scenario chat --concurrency 64 --result-dir ./results_mi355x_trace
   ```

2. **Parse into per-operator breakdown**:
   ```bash
   python3 scripts/parse_torch_trace.py b200_trace.json.gz --phase decode --csv b200_kernel_breakdown.csv
   python3 scripts/parse_torch_trace.py mi355x_trace.json.gz --phase decode --csv mi355x_kernel_breakdown.csv
   ```

3. **Align operators** using the 15-operator skeleton (see Level 2 below)

4. **Generate three tables** — read both CSVs, classify kernels, compute deltas

5. **Check Table 3 against catalog** — consult `reference_cross_platform_catalog.md` memory and `existing-optimizations.md`

### Overlap Accounting

B200 traces with multi-stream overlap require special handling:
- **B200 kernel_sum** ≠ **B200 walltime** (overlap must be subtracted)
- **MI355X kernel_sum** = **MI355X walltime** (single-stream, no overlap)
- For fair comparison: use B200 **elapsed/walltime** (critical path), not kernel_sum
- Report both kernel_sum and walltime; note the overlap amount

### Data Integrity Rules

1. **Never reconstruct** operator shapes/names/types from memory — always Read the CSV/trace
2. **Use trace-original kernel names** — do not rename or simplify
3. **Verify total**: sum of all operator μs should match within 5% of measured TPOT × 1000 / 61 layers
4. **Version-stamp** all data: framework version, container image, date

---

## Kernel Auto-Classification Reference

The keyword patterns below are used by `parse_torch_trace.py` and `compare_traces.py`. Check in listed order — first match wins.

```
1. Attention:  fmha, flash_fwd, flash_attn, mha_, mla_, merge_attn, set_mla_kv, mla_reduce
2. MoE/Expert: moe::, expert, routing, swiglu, nvjet_sm100, bmm_E2m1, bmm_BF16, 
               moe_mxgemm, MoeSorting, fused_mxfp4_quant_moe
3. Communication: allreduce, reduce_scatter, all_gather, nccl, rccl, userbuffers, 
                  device_load, device_store, moefinalize, lamport
4. GEMM/MatMul: gemm, gemv, cutlass, cublas, nvjet, splitKreduce, matmul, bmm, Cijk_
5. Normalization: rmsnorm, layernorm, batchnorm, groupnorm, Norm
6. Activation: silu, gelu, relu, act_and_mul, swiglu, sigmoid
7. Quantization: quantize, dequant, cvt_fp16_to_fp4, fp4, fp8, mxfp
8. Memory: memcpy, memset, copy, transpose
9. RoPE: rope, rotary, RopeQuantize, fuse_qk_rope
10. Sampling: sample, argmax, topk_sampling, multinomial
11. Other: everything else
```

**Important**: Attention must be checked before GEMM (fmha kernels may contain "gemm" substring).

---

## Level 0: E2E Benchmark (Existing)

Establish the performance baseline before drilling into kernels.

### B200
```bash
bash scripts/sa_bench_b200.sh \
  --model-fp8 /home/models/models--DeepSeek-R1-0528/ \
  --configs fp8-throughput --scenario chat --concurrency 128 \
  --ep-sizes 1 --result-dir ./results_b200_fp8_ep1
```

### MI355X
```bash
bash scripts/sa_bench_mi355x.sh \
  --model /path/to/DeepSeek-R1-0528 \
  --scenario chat --concurrency 128 \
  --result-dir ./results_mi355x
```

Key metrics: Output TPS/GPU, Total TPS, TPOT p50/p99, TTFT p50/p99, Interactivity.

---

## Level 1: Kernel Category Breakdown

Classify all GPU kernels into functional categories and identify the dominant bottleneck.

### Categories

| Category | Pattern Matches | Typical % (B200 FP4 chat c=64) |
|----------|----------------|------|
| **MoE** | moe, expert, expandInput, topk, buildExpert, Dispatch, Combine, Prepare, PtrArray | ~45% |
| **Attention** | fmha, flash, attention | ~7% |
| **GEMM** | gemm, cutlass, cublas, nvjet, splitKreduce, bmm | ~20% |
| **NCCL/Comm** | nccl, allreduce, AllGather, userbuffers | ~10% |
| **Norm** | Norm, rmsnorm | ~2% |
| **RoPE** | rope, rotary | ~1% |
| **Quantize** | quantize, dequant | ~3% |
| **Memory** | memcpy, memset | ~1% |
| **Other** | everything else | ~11% |

### B200: nsys Category Breakdown

```bash
# Capture trace
bash scripts/collect_nsys_trace.sh \
  --model /home/models/models--DeepSeek-R1-0528 \
  --mode bench --scenario chat --concurrency 32 \
  --quant fp8 --config throughput --iter-range 100-150

# Analyze
bash scripts/analyze_nsys_trace.sh --trace traces/nsys_*.nsys-rep --top 30
```

Output: Top N kernels by GPU time, category breakdown (MoE/Attn/GEMM/NCCL/Norm/RoPE/Quant/Mem), NVTX layer events.

### MI355X: Kineto Trace

```bash
bash scripts/collect_atom_trace.sh \
  --model /path/to/DeepSeek-R1-0528 \
  --scenario chat --concurrency 64 \
  --result-dir ./results_mi355x_trace
```

Output: Kineto JSON trace, decode walltime CSV (avg/p50/p99 per batch size).

### Interpreting Category Results

| Dominant Category | Likely Bottleneck | Next Step |
|-------------------|-------------------|-----------|
| MoE > 40% | MoE GEMM or routing overhead | Level 2: per-module MoE breakdown |
| NCCL/Comm > 15% | TP AllReduce or EP AllReduce too large | Check EP config, userbuffers fusion |
| Attention > 10% | KV cache or attention implementation | Check FP8 KV cache, fmha variant |
| GEMM > 25% (non-MoE) | MLA projection kernels | Level 3: ncu bandwidth analysis |
| Quantize > 5% | FP4/FP8 quantize overhead | Check if quantize fused into GEMM |

---

## Level 2: Per-Module Kernel Analysis

Break down a single decode iteration into 15 logical operators to enable cross-platform comparison.

### Operator Sequence (DeepSeek R1 671B, single decode layer)

| # | Operator | Type | Description |
|---|----------|------|-------------|
| 1 | qkv_a_proj | GEMM | Q+KV low-rank projection `[bs,7168]x[7168,2112]` |
| 2 | q/k_norm | Norm | RMSNorm on Q and K (x2) |
| 3 | q_b_proj | GEMM | Q expansion `[bs,1536]x[1536,3072]` |
| 4 | k_concat | Memory | K concatenation (RoPE portion) |
| 5 | uk_gemm | GEMM | kv_b K projection `[bs,512]x[512,2048]` |
| 6 | rope_cache | Memory | RoPE + KV cache write |
| 7 | fmha | Attention | MLA attention (FP8 E4M3 KV on B200) |
| 8 | uv_gemm | GEMM | kv_b V projection |
| 9 | out_proj | GEMM | Output projection + BF16->FP4 quantize |
| 10 | tp_allreduce+norm | Comm+Norm | TP AllReduce + residual + pre-MLP norm |
| 11 | residual_ag | Comm | Residual AllGather |
| 12 | router | Route | Router GEMM + topK + sort |
| 13 | moe_gemm | GEMM | gate+up+SwiGLU+down (grouped GEMM, FP4) |
| 14 | shared_expert | GEMM | Shared expert (2 GEMMs + SiLU) |
| 15 | moe_finalize | Comm | Weighted sum + EP AllReduce + residual |

**Pipeline Parallelism**: Operators #1 and #15 overlap across layers (P1 group). Critical path = max(qkv_a_proj, moe_finalize).

### B200 Reference Data (FP4 chat c=64, 10-layer avg)

From `reports/fp4-b200-vs-mi355x-breakdown.md`:
- **moe_gemm**: 95.3 us (33.5%) -- dominant
- **qkv_a_proj**: 42.6 us (15.0%)
- **moe_finalize**: 33.1 us (11.7%) -- hidden by P1 overlap
- **fmha**: 20.7 us (7.3%)
- **shared_expert**: 21.4 us (7.5%)
- **tp_allreduce+norm**: 15.2 us (5.3%)
- Single layer critical path: ~251 us
- 61 layers estimated: 15.3 ms (vs 15.6 ms measured, 2% error)

### How to collect per-module data

**B200**: Use nsys SQLite with time-range filtering to isolate a single decode iteration, then manually classify each kernel into the 15-operator sequence.

```bash
# Get the SQLite
bash scripts/analyze_nsys_trace.sh --trace traces/your_trace.nsys-rep --export-only

# Query per-layer using Python
python3 -c "
import sqlite3
conn = sqlite3.connect('traces/your_trace.sqlite')
# Filter by NVTX decode iteration markers or time range
# Group kernels by shortName, classify into 15 operators
"
```

**MI355X**: Use ATOM's `--mark-trace` flag + `parse_trace.py` to get per-operator timings.

---

## Level 3: Deep Kernel Analysis (ncu / rocprof)

Profile individual kernels for hardware utilization metrics.

### B200: ncu Analysis

```bash
# Discovery: find top bottleneck kernels
bash scripts/ncu_kernel_analysis.sh \
  --model /home/models/models--DeepSeek-R1-0528 \
  --mode discovery --scenario chat --concurrency 32

# Targeted: deep-dive on specific kernel
bash scripts/ncu_kernel_analysis.sh \
  --model /home/models/models--DeepSeek-R1-0528 \
  --mode targeted --kernel-name "bmm_E2m1" \
  --scenario chat --concurrency 32
```

Output CSV columns: `kernel, count, total_us, avg_us, dram_pct, sm_pct, occupancy_pct, diagnosis`

### ncu Metric Interpretation

| Metric | Value | Diagnosis | Action |
|--------|-------|-----------|--------|
| DRAM throughput | > 60% | **Memory-bound** | Reduce data movement, fuse kernels, use lower precision |
| SM throughput | > 50% | **Compute-bound** | Optimize arithmetic, use tensor cores, reduce register pressure |
| Occupancy | < 25% | **Latency-bound** (low occupancy) | Increase block size, reduce shared memory, reduce registers |
| DRAM < 40%, SM < 30% | Balanced / under-utilized | Check launch config, consider kernel fusion |

### MI355X: rocprof Analysis

```bash
# Basic kernel profiling
rocprof --stats your_binary

# With hardware counters
rocprof -i input.txt --hip-trace your_binary
```

### Before/After Comparison

```bash
# Compare two nsys traces (category + kernel level)
python3 scripts/compare_traces.py \
  --baseline traces/baseline.sqlite \
  --current traces/optimized.sqlite \
  --md --top 30

# Compare ncu CSV metrics
python3 scripts/compare_traces.py \
  --baseline ncu_reports/baseline_metrics.csv \
  --current ncu_reports/optimized_metrics.csv

# Compare MI355X decode walltime
python3 scripts/compare_traces.py \
  --baseline results_old/decode_walltime_*.csv \
  --current results_new/decode_walltime_*.csv --md

# Cross-platform comparison (JSON format)
python3 scripts/compare_traces.py \
  --baseline b200_categories.json \
  --current mi355x_categories.json --cross-platform
```

---

## Cross-Platform Comparison Methodology

### Rules for fair comparison

1. **Normalize per GPU**: Use `/GPU` metrics (Output TPS/GPU) since GPU count may differ (B200 8-GPU vs MI355X 4-GPU).
2. **Align configurations**: DP=false for both (ATOM doesn't support DP Attention), same concurrency and scenario.
3. **DAR is not directly comparable**: B200 (TRT-LLM C++ runtime) vs MI355X (ATOM PyTorch runtime) have different speculative decoding implementations. DAR reflects system-level behavior, not algorithm-level.
4. **Category % is comparable, absolute time is less so**: Different clock speeds, memory bandwidth, and pipeline depths make absolute kernel time less meaningful than category proportions.

### Cross-platform JSON format

For `compare_traces.py --cross-platform`:

```json
{
  "MoE": 150000000,
  "Attention": 30000000,
  "GEMM": 60000000,
  "NCCL/Comm": 40000000,
  "Norm": 5000000,
  "Other": 15000000
}
```
Values in nanoseconds. The tool normalizes to percentages for comparison.

---

## Checklist Before Concluding Analysis

- [ ] E2E metrics reproduced and within 5% of reference (SA InferenceX or ATOM CI)
- [ ] Category breakdown accounts for >90% of total GPU time (no large "Other")
- [ ] Top 5 kernels identified with function names and time percentages
- [ ] Bottleneck category clearly identified (MoE / Attention / Comm / GEMM)
- [ ] For ncu: diagnosis assigned (memory-bound / compute-bound / latency-bound)
- [ ] Optimization path identified from `existing-optimizations.md`
- [ ] Before/after delta table generated if comparing two configs
- [ ] Results documented in `reports/` with version tag and data provenance
