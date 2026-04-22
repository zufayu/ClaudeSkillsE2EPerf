# 必须遵守的规则（违反将导致数据错误或流程中断）

> 来源：222 条 fix commit 诊断提炼。每条规则背后都有 5+ 次真实修复。

## R1: 不要为新配置创建新 workflow 文件

- **触发条件**：需要用不同 EP/TP/concurrency/scenario 跑 bench 或 profiling
- **正确做法**：使用对应 (hw×fw) 的现有 workflow，通过 `workflow_dispatch` inputs 修改参数
- **禁止**：复制旧 workflow YML 并手动修改参数（copy-paste 是 23 条 fix commit 的根因）
- **验证**：`ls .github/workflows/*_bench.yml *_profiling.yml` 确认用的是统一 workflow

## R2: 平台配置不要手动写，从 env 文件读取

- **触发条件**：写 workflow 或脚本中涉及 NODE/REPO/MODEL 路径
- **正确做法**：`source configs/platforms/${platform}.env`，从环境变量读取
- **禁止**：在 workflow 或脚本中硬编码 `/home/ubuntu/zufa/ClaudeSkillsE2EPerf` 或 `/SFS-aGqda6ct/models/...`
- **验证**：`grep -r "home/ubuntu/zufa" .github/workflows/` 应返回空

## R3: Git sync 只用 hard reset

- **触发条件**：workflow 中同步 GPU 节点的 repo
- **正确做法**：`git fetch origin && git reset --hard origin/main`
- **禁止**：`git stash`、`git pull --rebase`、`git merge`（28 条 fix commit）
- **验证**：workflow 中的 sync 步骤调用 `ci_sync()`

## R4: 进程清理禁止 pkill -f

- **触发条件**：清理残留 GPU 进程
- **正确做法**：`safe_kill "pattern"` 或 `pgrep -f "pattern" | grep -v $$ | xargs -r kill -9`
- **禁止**：`pkill -f "pattern"`（匹配自身命令行 → exit 143，9 条 fix commit）
- **验证**：`grep -rn "pkill -f" scripts/*.sh` 应返回 0 结果（benchmark_lib.sh 中已有 safe_kill）

## R5: Kernel 分类从 kernel_registry.py 导入

- **触发条件**：分析脚本中需要 kernel name → operator 映射
- **正确做法**：`from kernel_registry import classify_kernel, get_operator_map`
- **禁止**：在脚本中独立定义 KERNEL_MAP / OPERATOR_MAP / CATEGORY_PATTERNS
- **验证**：新增的分析脚本必须 import kernel_registry

## R6: result_dir 路径自动生成

- **触发条件**：指定 bench/profiling 结果存放目录
- **正确做法**：`ci_result_dir "$PLATFORM" "$QUANT" "$MTP" "$EP" "$TP" "$ENV_TAG"`
- **格式**：`results/{platform}_dsr_{quant}/{platform}_dsr_{quant}_{mtp}_ep{ep}_tp{tp}_{env_tag}[_profiling]`
- **禁止**：手写 result_dir 路径（15 条 data label fix commit）
- **验证**：对比生成路径与 `ls results/` 下现有目录格式一致

## R7: 脚本运行前做 preflight check

- **触发条件**：在容器内执行 bench/profiling 脚本
- **正确做法**：调用 `preflight_trt` / `preflight_sglang` / `preflight_atom`
- **检查项**：`which python3 trtllm-serve nsys ncu`、`python3 -c "import tensorrt_llm"`
- **验证**：脚本启动后前 5 行应输出环境信息

---

## 自动捕获的错误

> 以下由 benchmark_lib.sh 错误陷阱自动追加，人工确认后移入上方规则。

（暂无）
