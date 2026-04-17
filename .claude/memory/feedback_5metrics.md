---
name: 5 standard benchmark metrics
description: All benchmark reports and comparisons must use these 5 metrics consistently, with Total Throughput as primary (not Output Throughput alone)
type: feedback
---

Reports and data tables must always include these 5 metrics:

1. **Total Throughput** (tok/s) — `total_token_throughput`, input + output tokens/sec, primary throughput metric
2. **Output Throughput** (tok/s) — `output_throughput`, output tokens/sec only
3. **TPOT p50** (ms) — `median_tpot_ms` / `tpot_p50`, time per output token median
4. **TTFT p50** (ms) — `median_ttft_ms` / `ttft_p50`, time to first token median
5. **Interactivity** — `output_throughput / (1000/tpot_p50)`, measures how interactive the system feels under load

**Why:** User corrected that Total Throughput should be primary, not just Output Throughput. SA InferenceX reports use Total Tput as the headline number. Per-GPU calculations should also be based on Total Throughput.

**How to apply:** Every comparison table, dashboard column, and report must include all 5 metrics. Per-GPU efficiency = Total Throughput / N_GPUs. Never show only Output Throughput without Total Throughput.
