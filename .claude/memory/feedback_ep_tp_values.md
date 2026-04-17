---
name: Never reduce EP/TP values
description: User repeatedly corrected me for reducing EP/TP counts in workflows - always use the correct SA config values
type: feedback
---

Never reduce EP or TP values in benchmark/profiling workflows. Always match the SA InferenceX reference config exactly.

**Why:** I have repeatedly set EP=1 when the correct SA config is EP=4 TP=4 for 4-GPU B200. This changes the benchmark behavior fundamentally (EP=1 means all experts on each GPU, EP=4 means experts distributed). The user has flagged this as a recurring error.

**How to apply:** For B200 4-GPU SGLang FP4: always use `--ep 4 --tp 4`. For B200 8-GPU: `--ep 8 --tp 8`. Never "simplify" by reducing parallelism dimensions. When in doubt, check SA InferenceX config.
