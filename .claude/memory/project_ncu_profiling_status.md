---
name: NCU profiling status
description: NCU profiling on B200 SGLang — 17 runs all failed, root cause is attach mode injection timeout for 671B MoE model
type: project
originSessionId: 49e0962a-ee55-4b7c-9023-e07ec969a46b
---
NCU profiling on B200 SGLang: **22+ runs attempted, 0 produced .ncu-rep files**.

## Progress (2026-04-16)

**Breakthrough: `--profile-from-start off` lets server start normally under NCU.**
- Server ready in ~8 min (vs timeout in all previous 17 attempts)
- Warmup c=64 completed successfully (196s, 128 prompts)
- BUT: `--profile-from-start off` requires `cudaProfilerStart()` to trigger profiling, which must be called from the GPU worker process — no mechanism to inject this

**Profiling ON from start (no skip): server never ready after 120 min.**
- NCU replays every matching kernel during init, ~45s each
- 140 kernel × 45s = ~105 min of replay, then init still needs to complete
- Timeout at 120 min, no .ncu-rep produced

## Current Best Approach (Not Yet Tried)

Combine `--profile-from-start off` (server can start) + `cudaProfilerStart()` injection:
1. Start server under NCU with `--profile-from-start off` — server ready in ~8 min
2. Send warmup requests to reach steady state
3. Inject `cudaProfilerStart()` into the SGLang worker process:
   - Option A: `gdb -p <worker_pid> -ex 'call cudaProfilerStart()' -ex quit`
   - Option B: Add `/profile` endpoint to SGLang that calls `torch.cuda.cudart().cudaProfilerStart()`
   - Option C: Use `SIGUSR1` signal handler in worker to trigger profiling
4. Send c=64 benchmark requests (decode kernels get profiled)
5. Inject `cudaProfilerStop()`

## Key Gotcha

Previous NCU processes may survive between runs — cleanup must kill ALL ncu/sglang processes, not just GPU processes (nvidia-smi misses NCU orchestrator).

**Why:** Kernel-level profiling needed for B200 vs MI355X comparison.
**How to apply:** Use `--profile-from-start off` + gdb injection approach. Never re-run profiling ON from start.

Status as of 2026-04-16.
