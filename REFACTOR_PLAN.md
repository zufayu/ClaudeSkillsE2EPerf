# ClaudeSkillsE2EPerf 重构计划

## 来源

- 222 条 fix commit 诊断 → `diagnosis.md`
- Hermes Agent 设计原则 → 中央注册表、auto-discovery、结构性防错
- 现有记忆 + 新建 3 条结构化规则

## 原则

> 不依赖 Claude 记住做正确的事，让架构强制执行。
> — 对标 Hermes `get_hermes_home()` 消灭 5 个硬编码 bug

## 三条线并行

### 线 1: Workflow 重构 (已完成设计)

**目标**: 80+ workflow → 8 个 (hw×fw×task)

```
.github/workflows/
  b200_trt_bench.yml         ✅ 已创建
  b200_trt_profiling.yml     ✅ 已创建
  b200_sglang_bench.yml      ✅ 已创建
  b200_sglang_profiling.yml  ✅ 已创建
  mi355x_atom_bench.yml      ✅ 已创建
  mi355x_atom_profiling.yml  ✅ 已创建
  b300_sglang_bench.yml      ✅ 已创建
  b300_sglang_profiling.yml  ✅ 已创建
```

支撑文件:
```
configs/platforms/b200.env      ✅ 已创建
configs/platforms/b300.env      ✅ 已创建
configs/platforms/mi355x.env    ✅ 已创建
scripts/ci_lib.sh               ✅ 已创建 (pkill bug 已修复)
```

**待做**:
- [ ] 验证 workflow 在真实环境可运行
- [ ] 验证通过后删除旧的 ~30 个 bench/profiling workflow

### 线 2: 脚本重构

**Phase A: 共享基础设施 (无行为变更)**

| 改动 | 对应诊断 | 对应规则 |
|------|---------|---------|
| `benchmark_lib.sh` 加 `log()`, `TS()` | 15 处重复定义 | — |
| `benchmark_lib.sh` 加 `safe_kill()` | 9 fixes pkill 自杀 | R4 |
| `benchmark_lib.sh` 加 `preflight_trt/sglang/atom()` | 10 fixes container 环境 | R7 |
| 新建 `scripts/trace_utils.py` | 6 处 `load_trace()` 重复 | — |
| 新建 `scripts/kernel_registry.py` | 6 处 kernel map 重复, 8 fixes | R5 |

**Phase B: Bench 脚本合并**

| 改动 | 行数减少 |
|------|---------|
| `sa_bench_{b200,h20,h200}.sh` → `sa_bench_trt.sh` + `configs/adaptive/*.sh` | -1100 行 |
| `collect_{nsys,sglang_nsys}_trace.sh` → `collect_nsys.sh --framework` | -400 行 |
| `{deploy,refresh,update}_dashboard.sh` → `dashboard.sh --mode` | -300 行 |
| `sa_bench_mi355x.sh` → `sa_bench_atom.sh` (改名) | 0 |

**Phase C: 分析脚本重构 (逐个，每改一个验证)**

将 6 个分析脚本改为 import `kernel_registry` + `trace_utils`，不改功能。

**不动的脚本**:
- `collect_sglang_trace.sh` / `collect_atom_trace.sh` (处理流程差异大)
- `collect_ncu_trace.sh` (15+ 次迭代稳定结果)
- `sa_bench_sglang.sh` (已是单一脚本)
- 所有独立分析工具 (compare_*.py, analyze_*.py 等)

**可清理的脚本** (合并后或确认废弃):
- `sa_bench_h20.sh` (合并后删)
- `sa_bench_h200.sh` (合并后删)
- `run_bench.sh` (被 remote_agent 取代)
- `serve.sh` (被 bench 脚本内嵌)
- `run_b200_sweep.sh` (被 workflow dispatch 取代)

### 线 3: 记忆仓库

**已完成**:
- [x] `diagnosis.md` — Top 3 错误类型 + 触发场景
- [x] `feedback_workflow_architecture.md` — R1-R7 结构化规则
- [x] `feedback_git_sync_rules.md` — 28 fixes 提炼
- [x] `feedback_kernel_classification.md` — 8 fixes 提炼
- [x] `MEMORY.md` 索引重组 (Rules / Feedback / Reference 三层)
- [x] `ci_lib.sh` pkill bug 修复

**待做**:
- [ ] CLAUDE.md 加入强制规则引用段
- [ ] Skills 加入 prerequisite: 读 MEMORY.md
- [ ] benchmark_lib.sh 加入 error trap → 自动捕获新错误

## 实施顺序

```
Week 1: Foundation (当前)
  Day 1: ✅ 诊断 + 记忆仓库
  Day 2: benchmark_lib.sh 扩展 (log/TS/safe_kill/preflight)
  Day 3: trace_utils.py + kernel_registry.py

Week 2: Integration
  Day 4-5: 分析脚本逐个改为 import kernel_registry
  Day 6: sa_bench_trt.sh 合并 (提取 adaptive configs)
  Day 7: collect_nsys.sh 合并 + dashboard.sh 合并

Week 3: Workflow deployment
  Day 8: 在 B200 上测试新 workflow
  Day 9: 在 MI355X 上测试
  Day 10: 删除旧 workflow + 清理废弃脚本

Each phase independently deployable and rollback-safe.
```

## 验证标准

每个 phase 完成后必须验证:
1. `bash -n` 所有 .sh 文件
2. `python3 -c "import ..."` 验证 Python 模块
3. Dry-run 模式 (CI_DRY_RUN=1) 跑一遍完整流程
4. 对比新旧 result_dir 命名一致
5. 对比分析脚本输出不变 (用现有 trace 文件)
