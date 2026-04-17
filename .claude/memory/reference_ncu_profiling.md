---
name: NCU profiling methodology
description: How to run ncu kernel-level profiling for decode phase — scripts, parameters, kernel patterns, and integration with nsys
type: reference
---

## Scripts

- **`scripts/collect_ncu_trace.sh`** — standalone ncu profiling
- **`scripts/collect_nsys_trace.sh --enable-ncu`** — integrated nsys + ncu pipeline
- **`scripts/find_decode_region.py`** — steady-state decode detection from nsys trace
- **`scripts/validate_ncu_trace.py`** — ncu/nsys cross-validation
- **`scripts/ncu_infer.py`** — offline inference script wrapped by ncu

## Standalone ncu usage

```bash
bash scripts/collect_ncu_trace.sh \
  --model /path/to/model --backend trtllm \
  --tp 1 --ep 1 --isl 64 --osl 8 --warmup-prompts 1 \
  --ncu-set pmsampling --launch-count 140 \
  --result-dir ./results/ncu_test
```

- Phase 1: runs nsys dry-run → `find_decode_region.py` detects steady-state decode → outputs `--launch-skip`
- Phase 2: runs ncu with detected launch-skip
- `--nsys-rep PATH`: reuse existing nsys trace, skip Phase 1 dry-run
- `--skip-dry-run --launch-skip N`: manually set skip value

## Integrated nsys + ncu usage

```bash
bash scripts/collect_nsys_trace.sh \
  --model /path/to/model --mode serve --scenario chat \
  --concurrency 64 --quant fp4 --config throughput \
  --tp 8 --ep 8 --enable-ncu --ncu-set pmsampling --ncu-launch-count 140
```

Runs nsys first, then feeds .nsys-rep to `collect_ncu_trace.sh` with `--nsys-rep`.

## ncu section sets

- `full` — ~7800 metrics, all sections including PM Sampling (slowest, most complete)
- `detailed` — ~900 metrics, compute + memory analysis
- `basic` — ~200 metrics, SpeedOfLight + occupancy
- `pmsampling` — PM Sampling only (SM utilization timeline, fastest)

## Kernel pattern (Qwen3-32B, TRT-LLM, no filter)

14 kernels per transformer layer:

| # | Kernel | Role | Duration (BS=1) |
|---|--------|------|-----------------|
| 0 | nvjet gate_up | MLP GEMM | ~144us |
| 1 | cublasSplitK | GEMM epilogue | ~5us |
| 2 | silu_and_mul_kernel | MLP activation | ~1us |
| 3 | nvjet down_proj | MLP GEMM | ~78us |
| 4 | cublasSplitK | GEMM epilogue | ~3us |
| 5 | FusedAddRMSNormKernel | LayerNorm + residual | ~3us |
| 6 | nvjet QKV | Attention GEMM | ~33us |
| 7 | cublasSplitK | GEMM epilogue | ~2us |
| 8 | fusedQKNormRopeKernel | QK norm + RoPE | ~2us |
| 9 | QKVPreprocessing | KV cache update | ~2us |
| 10 | kernel_mha | Decode attention | ~37us |
| 11 | nvjet O_proj | Attention GEMM | ~28us |
| 12 | cublasSplitK | GEMM epilogue | ~2us |
| 13 | FusedAddRMSNormKernel | LayerNorm + residual | ~3us |

With `-k regex:nvjet|fmha|kernel_mha|cutlass|...` filter: 5 kernels/layer (GEMM + attention only).
Without filter: 14 kernels/layer (includes SiLU, RMSNorm, RoPE, cublasSplitK).

## Decode region detection (`find_decode_region.py`)

Algorithm:
1. Export nsys-rep → sqlite
2. Extract inference kernels matching kernel regex, ordered by launch time
3. Detect repeating layer pattern at middle of trace
4. Group into decode-pass-sized blocks, compute median duration
5. Filter passes within 50% of median as "steady state" (excludes prefill = long, ramp-up = short)
6. Target middle of steady-state region → `--launch-skip`
7. Optional TPOT cross-validation (`--tpot-ms`)

Key insight: prefill and decode use **same kernel names** but vastly different durations (prefill ~2000ms vs decode ~6ms for 64 layers).

## launch-skip calculation

- With `-k` filter: `find_decode_region.py` outputs filtered launch-skip directly
- Without `-k` filter: need to convert filtered skip → unfiltered by querying nsys sqlite for the timestamp of the filtered kernel, then counting ALL kernels before that timestamp

Example (Qwen3-32B, ISL=64, OSL=8, warmup=1):
- Filtered (5 kernels/layer): launch-skip = 21760
- Unfiltered (14 kernels/layer): launch-skip = 60680
- 10 layers = 140 kernels (unfiltered), 50 kernels (filtered)
