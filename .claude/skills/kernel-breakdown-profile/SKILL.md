# Cross-Platform Kernel Profiling Skill

## When to Use

Activate when the user needs to:

1. **Diagnose benchmark anomalies** — E2E results show unexpected regressions or cross-platform gaps
2. **Locate GPU bottlenecks** — Identify the dominant kernel category (MoE / Attention / Comm / GEMM)
3. **Compare before/after** — Two traces need a delta table per-category and per-kernel
4. **Cross-platform kernel analysis** — Compare B200 vs MI355X at the operator level
5. **Plan optimizations** — Map diagnosed bottleneck to known optimization paths

## Platform Matrix

| Capability | B200 (NVIDIA) | MI355X (AMD) |
|-----------|--------------|-------------|
| **Framework** | TRT-LLM (rc6.post3) / SGLang 0.5.9 | ATOM (vLLM-based, rocm 7.1.1/7.2.1) |
| **Level 1: Category Breakdown** | `collect_nsys_trace.sh` → `analyze_nsys_trace.sh` | `collect_atom_trace.sh` (Kineto) |
| **Level 2: Per-Module Analysis** | nsys SQLite per-layer / SGLang torch trace | `--mark-trace` + `parse_torch_trace.py` |
| **Level 3: Deep Kernel Analysis** | `ncu_kernel_analysis.sh` (ncu --set full) | `rocprof` / `omniperf` |
| **Comparison** | `compare_traces.py` (SQLite / ncu CSV / JSON) | `compare_traces.py` (decode CSV / JSON) |
| **Environment Check** | `kernel_env.py detect` / `check-tools` | `kernel_env.py detect` / `check-tools` |

## Output Contract — Fixed Three-Table Report

Every triage must produce the same three tables. Ad-hoc formats are not allowed.

### Table 1: Kernel Table

Per-platform kernel breakdown sorted by GPU time share descending. Only render rows >= 1.0% share.

```
| # | Kernel | Category | GPU time (μs) | Share% | Launches | Platform | Source |
```

- **Category**: Auto-classified by keyword (see Kernel Classification below)
- **Source**: Trace file or profiling run that produced the data
- Prefill and decode should be reported separately when stage data is available

### Table 2: Cross-Platform Delta Table

Operator-aligned B200 vs MI355X comparison. Use the 15-operator DeepSeek R1 decomposition as the alignment skeleton.

```
| # | Operator | B200 kernel | B200 μs | MI355X kernel | MI355X μs | GAP(B-M) | GAP% | Bottleneck side |
```

- **GAP(B-M)**: Positive = B200 slower, negative = MI355X slower
- **Bottleneck side**: `B200` / `MI355X` / `parity` (within 10%)
- Must use verified data from trace files — never reconstruct from memory

### Table 3: Optimization Opportunity Table

Map each gap to known fixes, catalog status, and priority.

```
| Operator | GAP source | Known fix | Catalog status | Priority |
```

- **GAP source**: What causes the difference (kernel efficiency / fusion gap / overlap gap / comm BW)
- **Known fix**: From `existing-optimizations.md` or cross-platform catalog
- **Catalog status**: `existing` / `in-flight` / `structural` / `new opportunity`
- **Priority**: `P1` (>10μs gap) / `P2` (3-10μs) / `P3` (<3μs) / `skip` (structural, not actionable)

## Triage Workflows

### 1. Single-Platform Triage (one trace)

For quick diagnosis of a single platform's bottleneck distribution.

```bash
# B200: nsys trace → category breakdown
bash scripts/collect_nsys_trace.sh --model $MODEL --mode bench --scenario chat --concurrency 64 --quant fp4 --config throughput
bash scripts/analyze_nsys_trace.sh --trace traces/nsys_*.nsys-rep --top 30

# MI355X: Kineto trace → kernel breakdown
bash scripts/collect_atom_trace.sh --model $MODEL --scenario chat --concurrency 64 --result-dir ./results_mi355x_trace

# Either platform: torch trace → kernel table
python3 scripts/parse_torch_trace.py trace.json.gz --csv kernel_breakdown.csv
```

Output: **Table 1 only** (Kernel Table for that platform).

### 2. Cross-Platform Triage (B200 + MI355X traces)

The primary workflow for this project. Pairs one B200 trace with one MI355X trace.

**Prerequisites**:
- Both traces must use the same scenario (chat/reasoning/summarize), same concurrency, same EP/TP if possible
- Use `--phase decode` to isolate decode-only kernels
- Normalize per-GPU (B200 8GPU vs MI355X 8GPU or 4GPU)

**Steps**:
1. Parse both traces into per-operator breakdown
2. Align operators using the 15-operator DeepSeek R1 skeleton
3. Produce all three tables
4. Check Table 3 against the catalog before concluding

```bash
# Cross-platform comparison
python3 scripts/compare_traces.py \
  --baseline b200_categories.json --current mi355x_categories.json --cross-platform --md
```

Output: **All three tables**.

### 3. Before/After Triage (same platform, two configs)

For measuring the effect of a config change, framework upgrade, or optimization.

```bash
# Same-platform delta
python3 scripts/compare_traces.py \
  --baseline traces/before.sqlite --current traces/after.sqlite --md --top 30
```

Output: **Table 1** (kernel table for each) + **delta table** with per-kernel change%.

Delta sign convention (from CLAUDE.md):
- TPS: `(ref - cmp) / cmp` — positive green = Ref higher throughput
- TPOT/TTFT: `(cmp - ref) / ref` — positive green = Ref lower latency

## Catalog-First Rule

**Before claiming any finding as "new" or "novel", check in this order:**

1. **Cross-platform catalog** (memory: `reference_cross_platform_catalog.md`) — known B200/MI355X kernel families, structural gaps, verified ratios
2. **Existing optimizations** (skill: [existing-optimizations.md](existing-optimizations.md)) — bottleneck → known fix mapping
3. **Verified report data** (`reports/fp4-b200-vs-mi355x-breakdown.md`) — per-operator measurements

If a finding matches any catalog entry, report it as:
- An **existing** pattern that is present / missing / regressed
- A **structural gap** that is not actionable at the kernel level
- An **in-flight** optimization that exists but isn't applied

Only call it a **new opportunity** when no catalog row fits.

## Confidence Labels for AI Judgment

When exact catalog matching is inconclusive but the pattern looks semantically similar:

| Label | Meaning |
|-------|---------|
| `high` | Very likely the same pattern family; naming drift or version difference |
| `medium` | Several signals align, but one important piece is ambiguous |
| `low` | Weak resemblance; mention only if worth human follow-up |

Format: "No exact catalog match. AI similarity judgment: **{level}** — {one sentence rationale}"

## Kernel Classification (Auto, Keyword-Based)

Used by `parse_torch_trace.py` and `compare_traces.py`. Handles both TRT-LLM and ATOM/CK kernel names.

| Category | Pattern matches (case-insensitive) |
|----------|-----------------------------------|
| **MoE/Expert** | moe, expert, expandInput, topk, buildExpert, Dispatch, Combine, bmm_E2m1, bmm_BF16, routingMain, routingIndices, moe_mxgemm, MoeSorting, fused_mxfp4_quant_moe |
| **Attention** | fmha, flash_attn, flash_fwd, attention, mla_, mha_, merge_attn, set_mla_kv, mla_reduce |
| **GEMM/MatMul** | gemm, gemv, cutlass, cublas, nvjet, splitKreduce, matmul, bmm, Cijk_, dot_product |
| **Communication** | nccl, rccl, allreduce, reduce_scatter, allgather, all_gather, userbuffers, device_load, device_store, moefinalize, lamport |
| **Normalization** | rmsnorm, layernorm, batchnorm, groupnorm, Norm |
| **Quantization** | quantize, dequant, cvt_fp16_to_fp4, fp4, fp8, mxfp |
| **RoPE** | rope, rotary, RopeQuantize, fuse_qk_rope |
| **Activation** | silu, gelu, relu, act_and_mul, swiglu, sigmoid |
| **Memory** | memcpy, memset, copy, transpose |
| **Sampling** | sample, argmax, topk_sampling, multinomial |

Priority: check in listed order (first match wins). Attention before GEMM (fmha kernels may contain "gemm").

## Reference Documents

| Document | Purpose |
|----------|---------|
| [benchmark-and-profile.md](benchmark-and-profile.md) | 4-level profiling workflow, output contract column specs, cross-platform triage details |
| [existing-optimizations.md](existing-optimizations.md) | Bottleneck → known fix mapping for B200 and MI355X |
| [nsight-profiler.md](nsight-profiler.md) | Nsight Systems/Compute tool reference for LLM inference |

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/collect_nsys_trace.sh` | Capture nsys trace (bench/serve modes) |
| `scripts/collect_atom_trace.sh` | Capture Kineto trace on MI355X |
| `scripts/collect_sglang_trace.sh` | Capture SGLang torch profiler trace on B200 |
| `scripts/analyze_nsys_trace.sh` | Analyze nsys SQLite: Top N kernels, category breakdown |
| `scripts/ncu_kernel_analysis.sh` | ncu deep kernel analysis (targeted/discovery modes) |
| `scripts/parse_torch_trace.py` | Generic torch trace → kernel breakdown (B200 + MI355X) |
| `scripts/compare_traces.py` | Delta tables (SQLite, ncu CSV, decode CSV, JSON) |
| `scripts/compare_mtp.py` | MTP0 vs MTP3 cross-platform comparison |
| `scripts/kernel_env.py` | Environment preflight (detect platform, check tools) |
| `scripts/analyze_prefill_impact.py` | Prefill interruption impact on TPOT (MI355X) |

## Preflight

Before any profiling run:

```bash
python3 scripts/kernel_env.py detect        # Platform + GPU inventory
python3 scripts/kernel_env.py check-tools   # nsys/ncu/rocprof availability
python3 scripts/kernel_env.py idle-gpus --count 1  # For ncu exclusive access
```

## Checklist Before Concluding Analysis

- [ ] E2E metrics reproduced within 5% of reference (SA InferenceX or ATOM CI)
- [ ] Category breakdown accounts for >90% of total GPU time
- [ ] Top 5 kernels identified with function names and time percentages
- [ ] Bottleneck category clearly identified
- [ ] Table 3 checked against catalog — no known pattern mislabeled as "new"
- [ ] For ncu: diagnosis assigned (memory-bound / compute-bound / latency-bound)
- [ ] Before/after delta table generated if comparing two configs
- [ ] Results stored in `results/{platform}_{model}_{quant}/{...}_profiling/` per directory convention
