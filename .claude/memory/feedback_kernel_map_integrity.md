---
name: Kernel map integrity rules
description: Strict rules for B200 vs MI355X kernel map generation - no missing operators, no name changes, source tracing, sum verification
type: feedback
---

Kernel map CSV must satisfy 4 invariants:

1. **No missing operators** — every kernel from B200 layer_kernel_avg.csv AND every kernel from MI355X decode_breakdown.xlsx must appear in the final map. No operator can be silently dropped.

2. **Preserve original names** — use the exact operator/kernel names from source data. Do not rename, abbreviate, or "prettify" kernel names. If a shortname is needed, keep the original name as a separate column.

3. **Mark data source** — each row must clearly indicate where the data comes from (which CSV/xlsx file, which column).

4. **Sum verification** — B200 total must equal sum of all B200_Avg_us values. MI355X total must equal sum of all MI355X_avg_us values. No double-counting, no gaps.

**Why:** User found previous maps had missing operators and modified names, making it hard to trace back to source data and verify correctness.

**How to apply:** When generating kernel maps (manually or via script), always run a sum-check at the end. List any B200 or MI355X operators that weren't mapped. Never substitute a "friendly" name for the raw kernel name.
