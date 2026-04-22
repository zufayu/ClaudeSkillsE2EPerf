# Agent Handoff — Workflow Consolidation Refactor

## Branch

```
refactor/workflow-consolidation
```

Latest commit: `4fd116a`

**HEAD timeline (validation + fix work, 2026-04-22):**
- `4fd116a` fix(atom-trace): default PROFILE_NUM_PROMPTS=CONC*2 + dual-signal flush wait
- `c3ce1ef` fix(workflow): always-run 'Fix file ownership' step in 8 unified workflows
- `e104665` fix(compare_traces): recognize project's own torch trace analysis CSVs
- `54d16dc` fix(profiling): fix_torch_trace_pro.py uses argparse
- `fbfd294` fix(bench): sa_bench_sglang.sh accepts --platform
- `6d06aea` fix(profiling): collect_sglang_trace.sh accepts --platform
- `35cf0a2` fix(platform): env vars now overridable + b300 path corrected + min 100 profile steps
- `f74d62f` fix(profiling): auto-derive PROFILE_STEPS/START_STEP from bench params
- `334c03c` docs: add HANDOFF.md (original)
- `95eabc8` refactor: workflow consolidation + structural error prevention (initial)

See `.claude/memory/project_refactor_validation.md` for the bug list and validation evidence.

**不要 push 到 remote** — main 分支有正在跑的任务。等用户确认后再 merge。

---

## 做了什么

从 222 条 fix commit 中诊断出 3 大反复犯错场景（配置搞混 23 fixes、git 操作 28 fixes、pkill 自杀 9 fixes），按 Hermes Agent 的"结构性防错"原则重构。

### 核心改动

1. **8 个统一 workflow** 替代 ~30 个 copy-paste 的旧 YML  
   - 按 (硬件 × 框架 × bench|profiling) 组织  
   - EP/TP/concurrency/scenario/node/container 全是 workflow_dispatch inputs  
   - 平台路径从 `configs/platforms/*.env` 自动加载，不硬编码

2. **`scripts/ci_lib.sh`** — workflow 执行抽象层  
   - `ci_exec` / `ci_exec_host` / `ci_sync` / `ci_commit_results`  
   - 自动处理 ssh+docker / docker / podman 差异  
   - `ci_detect_env_tag` 自动检测框架版本（TRT→post2/rc10, SGLang→sglang059, ATOM→rocm722）  
   - `ci_result_dir` 自动生成 result 目录路径

3. **`scripts/benchmark_lib.sh`** 扩展  
   - `safe_kill()` 替代 `pkill -f`（9 处已全部替换）  
   - `log()` / `TS()` 消除 15 处重复定义  
   - `preflight_trt/sglang/atom()` 运行前环境检查  
   - `kill_framework_procs()` 按框架清理进程

4. **`scripts/kernel_registry.py`** — 中央 kernel 分类  
   - B200 + MI355X 的 kernel→operator→module→category 四层映射  
   - 支持位置依赖消歧（`classify_by_position()`）  
   - **注意**：只对新代码有效。现有 3 个分析脚本（parse_torch_trace, extract_cuda_kernels, compare_traces）保留内联 map，因为 display names 格式不同。自检时发现强制替换会改变输出格式，已还原。

5. **`scripts/trace_utils.py`** — 共享 trace 工具  
   - 3 个分析脚本已改为 import（trace_layer_detail, analyze_layer_from_trace, decode_kernel_breakdown）

6. **`scripts/sa_bench_trt.sh`** — 合并 b200/h20/h200  
   - 平台差异提取到 `configs/adaptive/*.sh`（compute_adaptive_params 函数）  
   - 加新 GPU = 创建 adaptive config，不需要 copy 整个 950 行脚本

7. **`scripts/dashboard.sh`** — 合并 deploy/refresh/update 三个 wrapper

8. **RULES.md** + CLAUDE.md 强制读取 + Skill 预检步骤 + `/mem` 命令

---

## 你需要做的

### 优先级 1：真机测试新 workflow

在 B200 上测试：
```bash
# GitHub Actions → b200_trt_bench.yml → Run workflow
# Inputs: node=<当天节点>, container=zufa_trt, configs=fp4-throughput, ep=8, tp=8, scenario=chat, concurrency=64
```

对比结果与旧 `b200_fp4_bench.yml` 的 result_dir 命名、文件内容一致。

然后测 SGLang：
```bash
# b200_sglang_bench.yml → framework inputs 自动选 zufa_sglang 容器
```

MI355X 同理。

### 优先级 2：验证通过后删除旧文件

旧 workflow（保留 NCU/diagnostic/ops 类的）：
```
b200_fp4_bench.yml, b200_fp4_ep1_bench.yml, b200_fp4_ep4_bench.yml,
b200_fp4_ep1_tp4_bench.yml, b200_fp4_c4_bench.yml,
b200_fp4_profiling.yml, b200_fp4_c4_profiling.yml,
b200_sglang_fp4_bench.yml, b200_sglang_fp4_ep8_c4_bench.yml,
b200_sglang_fp4_profiling.yml, b200_sglang_fp4_ep8_c4_profiling.yml,
b200_sglang_fp4_nsys_profiling.yml,
b200_sglang_fp4_ep1_tp4_bench.yml,
mi355x_mxfp4_bench.yml, mi355x_mxfp4_ep4_bench.yml,
mi355x_mxfp4_ep8_c4_bench.yml, mi355x_mxfp4_profiling.yml,
mi355x_mxfp4_ep8_c4_profiling.yml, mi355x_mxfp4_rocm722_bench.yml,
b300_sglang_fp4_ep8_c4_bench.yml, b300_sglang_fp4_ep8_c4_profiling.yml,
sglang_fp4_profiling.yml, sglang_vs_atom_bench.yml,
...
```

旧脚本：
```
scripts/sa_bench_h20.sh       # 合并到 sa_bench_trt.sh
scripts/sa_bench_h200.sh      # 合并到 sa_bench_trt.sh
scripts/deploy_dashboard.sh   # 合并到 dashboard.sh
scripts/refresh_dashboard.sh  # 合并到 dashboard.sh
scripts/update_dashboard.sh   # 合并到 dashboard.sh
scripts/run_bench.sh          # 被 remote_agent 取代
scripts/serve.sh              # 被 bench 脚本内嵌
scripts/run_b200_sweep.sh     # 被 workflow dispatch 取代
```

### 优先级 3：渐进迁移 kernel_registry

现有 3 个分析脚本（parse_torch_trace, extract_cuda_kernels, compare_traces）保留内联 kernel map。要迁移需要：
1. 在 kernel_registry 中增加 `display_name` 字段（带模块前缀如 `"qkv_proj: qkv_a_proj_GEMM"`）
2. 统一 category 名称（`"MoE"` vs `"MoE/Expert"`）
3. 逐个脚本迁移并验证输出不变

---

## 已知限制

1. **`safe_kill()` 在 `bash -c 'script'` 模式下会误杀** — 因为 pgrep 匹配到 bash 进程自身的 cmdline。生产环境（workflow step / 脚本文件执行）正常。

2. **新 workflow 需要 `actions/checkout@v4`** — 因为 ci_lib.sh 和 env 文件在 repo 里。B200 runner (jumphost) 之前不用 checkout。需确认 runner 有 git 权限。

3. **`workflow_dispatch` 新文件缓存问题** — GitHub 可能不立即识别新 workflow。首次可加临时 `push:` trigger 让 GitHub 注册（见 `feedback_remote_workflow_pitfalls.md` §4）。

4. **sa_bench_trt.sh 的 `--platform` 参数** — 当前 workflow 都传 `b200`。如果要给 H20/H200 用，需要新建 workflow 文件（`h20_trt_bench.yml`）或加 platform input。

---

## Validation status (2026-04-22)

11 bugs found via end-to-end B300 test (run #24771031787, 4 min, all phases ✅).
**10 fixed in HEAD timeline above. 1 remaining is bug #5 (kernel_registry missing
patterns) — defer to KR-Migration plan, see project_refactor_validation.md.**

Last good run produced: 286k events, 5125 FMHA samples, per-layer breakdown
clean (FMHA-to-FMHA median 117μs, 26 kernels/layer). Output 1010 tok/s and
TPOT 7.39ms within 2% of bench (no profiler regression).

---

## 文件依赖链

```
kernel_registry.py (no deps)     trace_utils.py (no deps)     configs/*.env (data)
         ↑                               ↑                          ↑
    3 analysis scripts            3 analysis scripts          ci_lib.sh
                                                                   ↑
benchmark_lib.sh (no deps)     configs/adaptive/*.sh           workflows
         ↑                          ↑
    sa_bench_*.sh              sa_bench_trt.sh
    collect_*.sh               collect_nsys_trace.sh
```

## 关键规则

读 `RULES.md`。特别注意：
- **R1**: 不要创建新 workflow YML
- **R4**: 禁止 pkill -f，用 safe_kill()
- **R5**: kernel 分类从 kernel_registry 导入（新代码）
- **R6**: result_dir 用 ci_result_dir() 自动生成
