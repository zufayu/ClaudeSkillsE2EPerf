# Profiling Methodology (R8d wide-sample steady-state)

> **TL;DR** — All kernel-level profiling on B300 / B200 (sglang) and MI355X
> (ATOM) uses the same R8d methodology: skip the first 5 decode steps,
> aggregate over the next 20 steady-state decodes (~600 samples per
> operator), report mean / median / p95 / std / N. No cherry-picked
> windows. Goal: data that fits real production decode average AND is
> apples-to-apples comparable across NV vs AMD architectures.

## Why this methodology

Two requirements drove the design:

1. **Mean must reflect real production decode state.** Capacity planning,
   regression detection, optimization prioritization all consume the mean.
   Anything that systematically biases the mean (cherry-picked stable
   windows, outlier removal that includes real production noise) makes
   downstream decisions wrong.

2. **NV ↔ AMD comparison must be apples-to-apples.** A delta of 15% on a
   GEMM kernel could be architecture, or it could be one platform sampling
   from a tighter window than the other. If methodology differs between
   sides, the delta is uninterpretable. So both sides MUST use identical
   sampling rules.

Wide-sample steady-state aggregation satisfies both. It's the standard
cross-architecture benchmark convention used in MLPerf, NVIDIA Nsight's
ranged collection guide, and academic GPU benchmarking literature.

## Rules

### Sample selection

| rule | value | rationale |
|---|---|---|
| Skip warmup decodes | first **5** | excludes cold-cache / JIT-compile / lazy-init artifacts |
| Aggregate steady-state decodes | **20** | gives ~600 (decode × layer) samples → mean SE ≈ CV/√600 ≈ CV/24 (4× tighter than N=30) |
| Layer range | **10–40** (DSR1, 30 layers) | first ~10 layers are dense (no MoE), last ~20 include lm_head and other unstable; middle is steady |
| Layers per decode | **61** (DSR1, configurable) | layer-anchored step splitting, not gap-based (continuous batching has no inter-step gaps) |
| Prefill contamination filter | reject step if its median FMHA > 2× global median | prefill FMHA is much longer than decode; layer-anchored slicing breaks if prefill is mid-trace |

### Statistics reported

| metric | use |
|---|---|
| **mean (avg_us)** | primary number — represents real production decode average |
| **median (median_us)** | robust check — if much smaller than mean, the kernel has a long tail |
| **p95 (p95_us)** | tail latency — the optimization target for capacity / SLA work |
| std, CV% | distribution width; CV is informational, not gate |
| N samples | sanity check; if N << 600, downstream estimates have wider SE |

### What we explicitly DON'T do

- **No cherry-picked "best-aligned" windows.** This was the legacy
  `--legacy-best-aligned` mode. It picked the 10 consecutive layers in one
  step with lowest within-window variance — a textbook survivorship bias
  that systematically underestimated CV by 30–50%.
- **No outlier removal.** Outliers in production are real production state.
  Remove them and you stop measuring production.
- **No GPU clock locking** (`nvidia-smi -lgc`). Production runs with DVFS;
  locking clocks measures something other than production. Reserve clock
  locking for cross-run regression isolation work, not for "what does
  production look like" data.
- **No single-decode picking.** The old `--skip-ratio 0.5` (pick one
  median-position decode) is back-compat only via `--max-steps 1`. It
  produces N=30 samples and a CV that's 20–50% off the true value just
  due to undersampling outliers.

## Tooling

### Per-platform analyzers

All three apply the same R8d defaults; output schema is aligned:

| platform | script | calls | inputs |
|---|---|---|---|
| B300 / B200 (sglang) | `scripts/trace_layer_detail.py` | 12 wf refs | torch profiler `.fix.json.gz` |
| MI355X (ATOM, primary) | `scripts/run_parse_trace.py` | 5 wf refs | torch profiler `.json.gz` (uses ATOM `parse_trace.py`) |
| MI355X (ATOM, secondary) | `scripts/decode_kernel_breakdown.py` | 0 wf refs (unused) | torch profiler `.json.gz` |

Common args: `--skip-warmup 5 --max-steps 20 --layer-range 10-40` (B300)
or `--skip-warmup 5 --max-steps 20 --layers 10-40 --target-bs N` (MI355X).

### Cross-platform comparison

`scripts/compare_b300_mi355x.py` reads two R8d CSVs and emits markdown:

- Section A: totals + B300 pass-level breakdown
- Section B: top-N kernels each platform side by side, with mean / median /
  p95 / tail (p95/mean) so dominant kernels and long-tail optimization
  candidates are both visible
- Section C: caveats (sample-size sanity check, methodology audit)

Per-op NV ↔ AMD kernel name mapping is intentionally not yet automated —
kernel naming diverges (`Attention(FMHA)` ↔ `mla_a8w8`,
`gate_up_GEMM` ↔ `moe_mxgemm`, etc.) and any auto-mapping that's wrong is
worse than no mapping. Section B side-by-side ranks let humans match.

### Provenance

Every `server_${TAG}.log` produced by `collect_sglang_trace.sh` /
`collect_atom_trace.sh` starts with a provenance header (image identity,
framework versions, repo short-shas, GPU model). This makes any historical
artifact reproducible months later — the alternative ("what was running
when this number was produced?") is unanswerable without it.

## Validation history

R8c (2026-04-23): proposed cross-step layer accumulation; first run on
B300 hit several orthogonal blockers (CI container chown, Python
UnboundLocalError, host fabricmanager). All fixed.

R8d (2026-04-24, B300 dispatch run #24882599928): first end-to-end run
with the full methodology. Key validations:

- Trace contained 84 decode steps (5125 FMHA / 61); 0 prefill-contaminated
  steps filtered.
- STEADY (skip 5, take 20) achieved exactly **N=600** samples per operator.
- WARMUP (skip 0, take 5) vs STEADY mean shift = **1.4%** → confirms
  collect-side `start_step=total/4` already provides warmup buffer; the
  analyzer-side `--skip-warmup-steps 5` is redundant safety margin (no harm,
  no required removal).
- N=30 (old) reported avg CV 12.0%; N=600 (R8d) reports **14.5%** — the
  higher number is the truth, the lower one was undersampled outliers. The
  apparent "regression" is the methodology working as intended.
- p95 / mean ratios exposed real long-tail kernels that LEGACY's cherry-pick
  hid: `shared_GEMM(FP4)` 2.0×, `gate_up_GEMM` 1.6×, `fused_a_gemm` 1.5×.
  These are the actual optimization priorities.

MI355X end-to-end validation pending (analyzer ported, no production run
with new defaults yet).

## Operational defaults

When dispatching a profiling workflow, the default args inherit R8d
methodology automatically — no per-workflow opt-in needed. Workflows
that pass explicit `--max-steps` or `--legacy-best-aligned` are using
non-standard methodology and their data is not directly comparable.

The pilot workflow `b300_test_profiling.yml` runs both STEADY (the
methodology) and WARMUP (skip 0, take 5) so its own Phase 4 self-check
documents that warmup exclusion and methodology stability are working.
This serves as a regression sanity check against future analyzer changes.

## References

- Memory: `project_r8d_methodology.md` — operational context, decision
  rationale, validation numbers
- Memory: `project_mi355x_parallel_slots.md` — slot-a / slot-b architecture
  for parallel TP=4 runs on a single 8-GPU MI355X node
- Code: `scripts/trace_layer_detail.py`, `scripts/run_parse_trace.py`,
  `scripts/decode_kernel_breakdown.py` — analyzer implementations
- Code: `scripts/compare_b300_mi355x.py` — cross-platform comparator
- Workflow: `.github/workflows/b300_test_profiling.yml` — pilot /
  regression self-check
