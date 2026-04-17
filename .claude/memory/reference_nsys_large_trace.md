---
name: Nsys large trace analysis methodology
description: How to analyze 2-3GB nsys-rep files that Nsight Systems can't open — SQLite/CSV based approach
type: reference
---

For large nsys-rep files (2-3GB) that Nsight Systems GUI can't open:

**Step 1: Export to SQLite (with optional time range to reduce size)**
```bash
nsys export --type sqlite --timeunit sec --timerange 20,40 -o trace_20s.sqlite trace.nsys-rep
```

**Step 2: Find steady-state decode steps via NVTX markers**
- TRT-LLM: `[Executor] _forward_step N: 0 ctx reqs, 64 gen reqs`
- SGLang: `decode[bs=N]` or layer-wise NVTX markers

**Step 3: Query kernels within decode step time range**

Our script: `scripts/analyze_nsys_sqlite.py trace.sqlite --gpu 0 --per-layer`

Alternative CSV approach:
```bash
nsys stats --report cuda_gpu_trace --format csv -o kernels trace.nsys-rep
grep -i "kernel_name" kernels.csv | head -1  # find target kernel
grep -n "^start_ns," kernels.csv              # find position
sed -n 'start,endp' kernels.csv               # extract context
```
