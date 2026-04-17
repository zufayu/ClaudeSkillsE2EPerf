---
name: Warmup aligned with SA InferenceX
description: collect_atom_trace.sh warmup changed from CONC×10 to CONC×2 to match SA InferenceX methodology
type: feedback
---

Warmup prompts should be `CONC × 2`, matching SA InferenceX's `benchmark_lib.sh`.

**Why:** Previously was `CONC × 10` (with min=20 floor), 5× more than SA. No need for extra warmup — SA's 2× is sufficient.

**How to apply:** When modifying benchmark scripts, keep warmup = `2 × concurrency`. SA reference: `--num-warmups "$((2 * max_concurrency))"` in `benchmarks/benchmark_lib.sh`.
