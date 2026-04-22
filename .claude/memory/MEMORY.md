# Project Memory

## Workflow Rules
- **Always `git pull` before making code changes** — user has multiple places modifying code simultaneously, must sync first to avoid conflicts
- **Commands on one line** — never use backslash line continuation; give single-line commands
- Repo: `/home/kqian/ClaudeSkillsE2EPerf` (GitHub: zufayu/ClaudeSkillsE2EPerf)

## Project: ClaudeSkillsE2EPerf
- B200/H20 benchmark suite for DeepSeek-R1 with SemiAnalysis methodology
- Key scripts: `scripts/sa_bench_b200.sh`, `scripts/sa_bench_h20.sh`
- Dashboard: `docs/index.html` → GitHub Pages at zufayu.github.io/ClaudeSkillsE2EPerf
- Pipeline: `runs/*.json` → `generate_dashboard.py` → `docs/data.js` → `docs/index.html`

## DAR (Draft Acceptance Rate)
- MTP=3 (EP=1) / MTP=1 (EP>1) for latency configs; MTP=0 for throughput configs
- `/perf_metrics` approach does NOT work on rc4 (H20) or rc6.post3 (B200) — no `speculative_decoding` in response
- Working approach: `collect_dar()` via trtllm-bench (standalone LLM API)
- Config naming: fp8-latency = MTP3, fp8-throughput = MTP0
- **B200 DAR values:** chat 80.2%, reasoning 73.8%, summarize 72.9% (per-scenario, from trtllm-bench offline, single concurrency)
- **355X DAR values (per-scenario per-concurrency, from ATOM CI run #23408094038):**
  - chat: 48-55% (high conc ~49%, low conc c=4 ~55%)
  - reasoning: 63-69% (close to B200's 74%!)
  - summarize: 54-64%
  - DAR mainly determined by scenario, concurrency impact ±3% within same scenario
- DAR report JSON keys: `draft_acceptance_rate_percentiles.average` (NOT `avg`), `acceptance_length_percentiles.average`

## import_results.py — Fixed Bugs (2026-03-23)
- **DAR filename parsing:** `dar_report_fp8_mtp3_ep1_chat.json` → scenario is `parts[-1]` (last), NOT `parts[3]`
- **DAR key name:** JSON uses `"average"` not `"avg"` for mean values in `draft_acceptance_rate_percentiles` and `acceptance_length_percentiles`

## SemiAnalysis InferenceX Methodology
- `num_prompts = CONC × 10` (e.g., c=1→10, c=4→40, c=128→1280)
- `warmup = CONC × 2` (from `benchmark_lib.sh: --num-warmups "$((2 * max_concurrency))"`)
- Our `sa_bench_b200.sh` num_prompts matches, but warmup was wrong (fixed 8, should be CONC×2)
- `collect_atom_trace.sh` warmup changed from CONC×10 to CONC×2 (2026-03-31)
- Source: `github.com/SemiAnalysisAI/InferenceX/benchmarks/`

## Dashboard (docs/index.html)
- Scatter tab has 2 subtabs: "Interactive vs Token Throughput" + "Concurrency Scaling"
- Concurrency Scaling: dual Y-axis (left=Throughput circles, right=TPOT triangles), X=concurrency log scale
- Chart.js gotcha: canvas in `display:none` gets 0 dimensions → use `requestAnimationFrame` before building chart after visibility change
- Subtab `.tab` elements must not conflict with main `#tabs > .tab` selector

## TRT-LLM Environments
- H20: rc4 (`/raid/data/models/deepseekr1`) — [SSH access details](reference_h20_ssh.md)
- B200: rc10 (`/SFS-aGqda6ct/models/models--DeepSeek-R1-0528`) — docker `zufa_trt` (1.3.0rc10), driver 580/CUDA 13.0, 8×B200
- B200 SGLang: `zufa_sglang` (v0.5.9-cu130)

## MTP3 vs MTP0 Analysis (2026-03-24)
- Scoreboard: B200 8 : 355X 10 (18 comparable points)
- Report: `reports/mtp3-vs-mtp0-analysis.md` (v8)
- **Root cause of MTP0 baseline gap: TRT-LLM rc6.post2→post3 MoE optimizations (~15%)**
  - SA InferenceX benchmark confirms: B200 rc6.post2 ≈ 355X ATOM (±5%), same config
  - post2→post3 has only 5 PRs, 3 optimize MoE: shared+sparse fusion (#11143), FP8 cubins (#11104), multi-stream fix (#11160)
  - Our B200 (post3) is ~15% faster than SA's B200 (post2), while 355X numbers match SA exactly
- DAR gap varies by scenario — reasoning small (74% vs 65%), chat large (80% vs 49%)
- DAR utilization efficiency low on both platforms (6-53%), bottleneck is serving overhead not DAR

## B200 4-GPU FP4 Benchmark
- Model: `/home/models/DeepSeek-R1-0528-NVFP4-v2`
- Config: 4 GPUs (0,1,2,3), **TP=4, EP=1**, DP=false
- **TP×EP must ≤ N_GPUs** — EP=4 with TP=4 on 4 GPUs = 16 ranks, crashes
- Throughput (mtp0) results: `./results_b200_fp4_mtp0_4gpu`
- Latency (mtp3) results: `./results_b200_fp4_mtp3_4gpu`
- SA reference (4×B200 FP4 summarize c=32): Output Tput/GPU = 369.6
- `--use-chat-template`: only used with MTP (latency) configs, aligned with SA InferenceX

## MI355X Environment
- [MI355X env details](reference_mi355x_env.md) — machine, podman container, model paths, runner setup
- [ATOM CI actual config](project_atom_ci_config.md) — uses hybrid model (MTP-MoE-MXFP4-Attn-PTPC-FP8) + MTP=3, not pure MXFP4
- [MI355X FP4 model comparison](project_mi355x_fp4_models.md) — 4 models independently quantized, different weights/configs/timestamps

## Workflow Rules (cont.)
- [Confirm key configs before running](feedback_confirm_configs.md) — never assume benchmark/profiling params, always confirm with user first

## fetch_competitors.py — Key Fixes
- **CI run selection:** accept `conclusion=failure` (not just `success`) — individual jobs may still succeed
- **DAR injection:** no cross-scenario fallback — only exact (isl,osl,conc) or scenario (isl,osl) match
- **DAR source:** parsed from `[MTP Stats]` lines in CI job logs via `_parse_mtp_stats()`

## Naming Convention
- [Directory & result naming rules](project_naming_convention.md) — env tag, version fields, dashboard columns

## Benchmark Metrics
- [5 standard metrics](feedback_5metrics.md) — Total Tput, Output Tput, TPOT p50, TTFT p50, Interactivity; Per-GPU based on Total Tput

## Pending: FP4 4GPU Table
- [4GPU comparison table](project_fp4_4gpu_table.md) — waiting for SGLang profiling (ratio=0.8), then present full table for user verification before updating report

## Trace Files
- [No upload traces](feedback_no_upload_traces.md) — large .json.gz trace files must NOT be committed to git, only keep locally

## Nsys Large Trace Analysis
- [SQLite/CSV methodology](reference_nsys_large_trace.md) — for 2-3GB nsys-rep files, export to SQLite, query NVTX+kernels; script: `analyze_nsys_sqlite.py`

## Machine Access
- [GitHub Actions runners](reference_machine_access.md) — B200/355X via self-hosted runners, trigger workflows via API from this container

## Workflow Discipline
- [Verify commands before commit](feedback_verify_before_commit.md) — never commit untested commands to workflows, check --help or docs first

## Profiling Config Integrity
- [Trace configs must match design](feedback_trace_config_integrity.md) — never speculatively modify profiling params (EP/TP/concurrency/flags), must match intended benchmark design exactly
- [Never reduce EP/TP values](feedback_ep_tp_values.md) — recurring error: always use SA config EP=4 TP=4 for B200 4-GPU, EP=8 TP=8 for 8-GPU

## Kernel Mapping Methodology
- [B200 vs MI355X kernel map rules](reference_kernel_map_rules.md) — strict timeline alignment (not functional), pass grouping (MOE/MHA/O_proj/EP_AR), overlap annotation, module corrections
- [Kernel map integrity rules](feedback_kernel_map_integrity.md) — no missing operators, preserve original names, mark data source, sum verification

## NCU Profiling
- [NCU profiling methodology](reference_ncu_profiling.md) — scripts, kernel patterns (14/layer unfiltered, 5/layer filtered), decode region detection, --enable-ncu integration
- [NCU profiling status](project_ncu_profiling_status.md) — 17 B200 SGLang runs all failed, attach mode injection timeout for 671B MoE

## CI/Workflow Gotchas
- [No host-level pkill in workflows](feedback_no_host_pkill.md) — podman exec pkill returns 137, use continue-on-error

## B300 Environment
- [B300 env details](reference_b300_env.md) — 8×B300 SXM6 AC 275GB, driver 595, runner `b300-runner`, no dedicated container yet

## Refactor Validation (refactor/workflow-consolidation, 2026-04-22)
- [End-to-end validation status](project_refactor_validation.md) — 11 bugs found via B300 SGLang FP4 test, 10 fixed (commits f74d62f..4fd116a), 1 deferred to KR-Migration (kernel_registry missing 8 patterns). Last good run #24771031787 within 2% of bench baseline.
