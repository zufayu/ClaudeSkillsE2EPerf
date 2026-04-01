# Kernel-Level Breakdown & Profile Skill

## When to use this skill

Activate this skill when the user needs to:

1. **Diagnose benchmark anomalies** -- E2E results (from `sa_bench_b200.sh` / `sa_bench_mi355x.sh`) show unexpected regressions or cross-platform gaps, and the user wants to know _which kernel category_ is responsible.
2. **Locate GPU bottlenecks** -- Identify whether a workload is MoE-bound, Attention-bound, GEMM-bound, or communication-bound, and drill down to individual kernel metrics (DRAM bandwidth, SM throughput, occupancy).
3. **Compare before/after** -- Two traces (different configs, framework versions, or optimizations) need a delta table showing per-category and per-kernel improvements.
4. **Cross-platform kernel analysis** -- Compare B200 (TRT-LLM / nsys / ncu) vs MI355X (ATOM / Kineto / rocprof) at the operator level.
5. **Plan optimizations** -- Map a diagnosed bottleneck to known optimization paths (fused kernels, quantization changes, communication overlapping).

## Platform matrix

| Capability | B200 (NVIDIA) | MI355X (AMD) |
|-----------|--------------|-------------|
| **Framework** | TRT-LLM (rc6.post3) | ATOM (vLLM-based) |
| **Level 1: Category Breakdown** | `collect_nsys_trace.sh` + `analyze_nsys_trace.sh` | `collect_atom_trace.sh` (Kineto) |
| **Level 2: Per-Module Analysis** | nsys SQLite per-layer queries | `--mark-trace` + `parse_trace.py` |
| **Level 3: Deep Kernel Analysis** | `ncu_kernel_analysis.sh` (ncu --set full) | `rocprof` / `omniperf` |
| **Comparison** | `compare_traces.py` (SQLite / ncu CSV / JSON) | `compare_traces.py` (decode CSV / JSON) |
| **Environment Check** | `kernel_env.py detect` / `check-tools` | `kernel_env.py detect` / `check-tools` |

## Preflight

Before running any profiling, verify the environment:

```bash
# 1. Detect platform and GPU inventory
python3 scripts/kernel_env.py detect

# 2. Check profiling tools
python3 scripts/kernel_env.py check-tools

# 3. Find idle GPUs (for ncu which needs exclusive access)
python3 scripts/kernel_env.py idle-gpus --count 1

# 4. Get suggested commands for your scenario
python3 scripts/kernel_env.py suggest --scenario chat --concurrency 64
```

## Reference documents

| Document | Purpose |
|----------|---------|
| [benchmark-and-profile.md](benchmark-and-profile.md) | **Core reference**: 4-level profiling workflow (E2E -> Category -> Per-Module -> Deep Kernel) with commands, interpretation, and checklists |
| [existing-optimizations.md](existing-optimizations.md) | Bottleneck-to-optimization mapping table: diagnosed issue -> known fix for B200 and MI355X |
| [nsight-profiler.md](nsight-profiler.md) | Nsight Systems/Compute tool reference adapted for LLM inference (commands, metric interpretation, roofline analysis) |

## Key scripts

| Script | Purpose |
|--------|---------|
| `scripts/collect_nsys_trace.sh` | Capture nsys trace (bench/serve modes) |
| `scripts/collect_atom_trace.sh` | Capture Kineto trace on MI355X |
| `scripts/analyze_nsys_trace.sh` | Analyze nsys SQLite: Top N kernels, category breakdown, NVTX events |
| `scripts/ncu_kernel_analysis.sh` | ncu deep kernel analysis (targeted/discovery modes) |
| `scripts/compare_traces.py` | Before/after trace delta tables (SQLite, ncu CSV, decode CSV, JSON) |
| `scripts/kernel_env.py` | Environment preflight (detect platform, check tools, find idle GPUs) |
| `scripts/analyze_prefill_impact.py` | Prefill interruption impact on TPOT (MI355X Kineto) |
| `scripts/run_parse_trace.py` | MI355X decode kernel breakdown (steady-state bs selection) |
| `scripts/upload_profiling.sh` | Upload profiling results to remote |
| `scripts/compare_mtp.py` | MTP0 vs MTP3 cross-platform comparison |

## Quick start examples

### B200: Full kernel analysis workflow
```bash
# Step 1: Capture nsys trace
bash scripts/collect_nsys_trace.sh \
  --model /home/models/models--DeepSeek-R1-0528 \
  --mode bench --scenario chat --concurrency 32 \
  --quant fp8 --config throughput --iter-range 100-150

# Step 2: Analyze category breakdown
bash scripts/analyze_nsys_trace.sh --trace traces/nsys_*.nsys-rep --top 30

# Step 3: Deep kernel analysis (discovery mode)
bash scripts/ncu_kernel_analysis.sh \
  --model /home/models/models--DeepSeek-R1-0528 \
  --mode discovery --scenario chat --concurrency 32

# Step 4: Compare before/after
python3 scripts/compare_traces.py \
  --baseline traces/baseline.sqlite --current traces/new.sqlite --md
```

### MI355X: Decode bottleneck analysis
```bash
# Step 1: Capture Kineto trace
bash scripts/collect_atom_trace.sh \
  --model /path/to/DeepSeek-R1-0528 \
  --scenario chat --concurrency 64 \
  --result-dir ./results_mi355x_trace

# Step 2: Compare decode walltime
python3 scripts/compare_traces.py \
  --baseline results_old/decode_walltime_*.csv \
  --current results_new/decode_walltime_*.csv --md

# Step 3: Analyze prefill impact on decode
python3 scripts/analyze_prefill_impact.py traces/*.json.gz
```
