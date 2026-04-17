---
name: Trace/profiling configs must match expected design exactly
description: Never speculatively modify trace/profiling configurations — they must match the intended benchmark design
type: feedback
---

When capturing traces (torch profiler, nsys, or any profiling), configurations must exactly match the expected/designed benchmark parameters. Do not make speculative changes or "simplify" any parameter.

**Why:** Trace data is used for kernel-level performance analysis. If the config (EP, TP, concurrency, model params, server flags) doesn't match the intended design, the trace is worthless — it captures behavior of the wrong configuration. The user has flagged this as a recurring issue where I reduce EP/TP or change parameters speculatively.

**How to apply:** Before creating or modifying any profiling/trace workflow or script:
1. Verify every parameter matches the SA InferenceX reference config or user's explicit design
2. Never change EP, TP, concurrency, or server launch flags without explicit user approval
3. If unsure about any config value, ask the user — do not guess or "simplify"
4. This applies to all platforms (B200, 355X) and all profiling tools (nsys, torch profiler, ATOM)
