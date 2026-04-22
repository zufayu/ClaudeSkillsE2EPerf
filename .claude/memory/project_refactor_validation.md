---
name: Refactor end-to-end validation status
description: 2026-04-22 B300 real-machine validation results for refactor/workflow-consolidation; 11 bugs found, 10 fixed (commits f74d62f..4fd116a), 1 deferred to KR-Migration
type: project
---

## Final state (2026-04-22)

**End-to-end B300 SGLang FP4 chat c=4 ran cleanly** through all 3 profiling
phases (run #24771031787, 4 min total, all phases success). Performance within
2% of pre-refactor bench (no regression).

Output 1010 tok/s, TPOT 7.39ms, 286k events, 5125 FMHA samples,
per-layer breakdown FMHA-to-FMHA median 117μs across 60 layers, 26 kernels/layer.

## What got validated (refactor surface tested on real B300)

ci_lib.sh: ci_load_platform · ci_parse_configs · ci_get_image · ci_detect_env_tag ·
ci_result_dir · ci_exec (docker mode) · ci_verify_gpu · ci_fix_ownership.

Profiling pipeline: collect_sglang_trace.sh (capture) · fix_torch_trace_pro.py
(overlap fix) · trace_layer_detail.py (per-layer analysis).

Bench pipeline: sa_bench_sglang.sh (full bench) — 1051 tok/s c=4 chat (run #24758870225).

Pure-logic surface unit-tested on H20: 21/21 assertions across ci_lib.sh
(3 platforms × ci_load_platform, 6 configs × ci_parse_configs, ci_result_dir,
CI_DRY_RUN mode, safe_kill survival).

## Bug list — 10 fixed, 1 deferred

| # | Bug | Fix commit |
|---|---|---|
| 1 | b300.env REPO=/home/zufayu/... wrong (real home is /home/zufa) | `35cf0a2` |
| 2 | platform .env files unconditionally overwrote REPO when re-sourced (bare `VAR=`); workflow steps couldn't override | `35cf0a2` (changed to `: ${VAR:=...}`) |
| 3 | sa_bench_sglang.sh summary.md hardcoded `## B200 ${GPU_COUNT}×GPU` | `fbfd294` (added --platform) |
| 4 | fix_torch_trace_pro.py used bare sys.argv[1] → --help became filename → FileNotFoundError | `54d16dc` (added argparse) |
| 5 | kernel_registry missing 8 critical patterns (splitK_TNT, fmha, flash_fwd, mha_, splitKreduce, fused_silu_and_mul, cublasLt, nvjet_splitK_TNT) — extract_cuda_kernels_torch_trace inline map covers these but registry doesn't | **DEFERRED** to KR-Migration Step 0 (task #9) |
| 6 | compare_traces.py rejected project's own kernel_breakdown_*.csv and per_layer_breakdown_*.csv ('Cannot compare unknown with unknown') | `e104665` (added 2 format detectors) |
| 7 | ci_fix_ownership only called from ci_commit_results — when commit step skipped/failed, container-UID files broke next checkout's `clean: true` (EACCES on rmdir) | `c3ce1ef` (added always-run step in 8 unified workflows) |
| 8 | collect_sglang_trace.sh TAG hardcoded `trace_torch_b200_...` regardless of platform | `6d06aea` (added --platform) |
| 9 | collect_sglang_trace.sh analysis call hardcoded `--platform b200` | `6d06aea` (same fix) |
| 10 | PROFILE_STEPS=1000 hardcoded in collect_sglang_trace.sh — torch profiler buffer swamped CUDA queue → SGLang scheduler watchdog (300s) → SIGQUIT on 671B MoE | `f74d62f` (auto-derive from bench params) |
| 11 | PROFILE_START_STEP=30 too shallow — captured before cudaGraphs warmed up | `f74d62f` (auto-derive total_steps/4 = 25% mark) |

## Auto-derivation formula (collect_sglang_trace.sh)

For chat OSL=1024, num_prompts=CONC*10:
- `total = OSL * num_prompts / concurrency`
- `start_step = total / 4` (enter steady-state region)
- `num_steps = max(100, min(100, total/50))` — floor 100 (operator avg requires
  ≥100 decode samples), cap 100 (proven safe under watchdog/buffer)
- Worked example: chat c=4 → total=10240, start=2560 (~18s in), num_steps=100 (~700ms window)

## Companion fix in ATOM trace script (related, not B300 validation finding)

`scripts/collect_atom_trace.sh` PROFILE_NUM_PROMPTS default was CONC*10 (640 at c=64) —
contradicted both its own comment ("defaults to WARMUP") and help text
("default: conc*2"). Aligned to ATOM CI's CONC*2 standard (commit `4fd116a`).
Trace size 5x smaller, still ≥100 decode samples for stable averaging.

Also added dual-signal trace-flush wait: server log "Profiler stopped." count
(primary, more reliable) + .json→.json.gz file rename (fallback).

## Dispatch flow gotcha (re-encountered)

Newly-pushed workflow files don't register for `workflow_dispatch` API
immediately. First POST returns 404. Workaround: add temporary `push:` trigger
filtered to the workflow file path → push → auto-fires once with empty inputs
(expect failure on `ci_parse_configs ""`) → workflow_dispatch API works for
subsequent runs. Already documented in `feedback_remote_workflow_pitfalls.md` §4.

## Validation methodology — reusable pattern

To validate refactor on a `<platform>` runner without polluting main:
1. Derive `test/<platform>-validation` from `refactor/workflow-consolidation`
2. Fork `<platform>_<framework>_(bench|profiling).yml` to a `<platform>_test_*.yml`
3. Drop the "Commit results" step (no main pollution)
4. Add pre-checkout chown step (workaround for stale container-UID files; not
   needed once bug #7 fix is in main, but useful for clean-room test runs)
5. Add diagnostic step (RAM, GPU, container limits) before Phase 1 to fail fast
   on environment contamination — saved us 2 false-positive OOM diagnoses
   (turned out to be zombie trtllm-bench from same team's old container)
6. Dispatch via `gh api` POST to `/actions/workflows/<file>.yml/dispatches` with
   ref=test/... + inputs JSON (see reference_machine_access.md)

Run took ~3-4 min on B300 (warm container, no model reload). FP4 model (385GB)
cold-load adds ~5 min.

## Test infrastructure cleanup

The `test/b300-validation` branch was used for this validation. After successful
regression run #24771031787, branch can be deleted (work merged into refactor as
the 8 fix commits listed above). Test-only workflow files (b300_test_bench.yml,
b300_test_profiling.yml, b300_diag_processes.yml) only existed on test branch
and disappear with it.
