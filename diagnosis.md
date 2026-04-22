# 诊断报告：Claude 反复犯错的具体场景

> 数据来源：222 条 fix commit 分类统计 + 现有 memory 文件交叉验证

## 错误频率分布

| 类别 | 修复次数 | 占比 |
|------|---------|------|
| Git 操作（rebase/push/permissions） | 28 | 12.6% |
| NCU 参数演进 | 24 | 10.8% |
| **配置搞混（EP/TP/REPO/container）** | **23** | **10.4%** |
| 数据标签 mislabel | 15 | 6.8% |
| Shell/YAML 语法 | 12 | 5.4% |
| 分析逻辑（overlap/breakdown） | 11 | 5.0% |
| Container 环境差异 | 10 | 4.5% |
| pkill 自杀 / 进程管理 | 9 | 4.1% |
| Trace 格式（NVTX/SQLite schema） | 8 | 3.6% |
| Kernel 分类错误 | 8 | 3.6% |
| 其他 | 74 | 33.3% |

## Top 3 错误类型

---

### #1: 配置搞混 — EP/TP/REPO/container 硬编码写错 (23 fixes)

**症状**：
- EP=1 写成 EP=8、EP=8 写成 EP=4
- REPO 路径 `/home/ubuntu/zufa` 写成其他
- 用错 container（该用 zufa_trt 用了 zufa_sglang）
- result_dir 命名不一致

**代表 commit**：
```
d9568dc fix: B200 c4 bench+profiling use EP=8 TP=8 (8-GPU), not EP=4 TP=4
283b2b0 fix: nsys profiling workflow EP=1 -> EP=4
e69dacf Fix ATOM MI355X: EP=1 (TP=8), not EP=8
875d13e fix: SGLang profiling REPO path to /home/ubuntu/zufa
0ad8e84 fix: standardize REPO paths and result dir naming across all workflows
1ffd771 fix: collect_nsys_trace.sh args (--ep not --ep-sizes, --trace-dir not --result-dir)
```

**触发条件**：
1. 新建 workflow 时 copy-paste 另一个 workflow，修改目标参数但忘改配套参数
2. 不同平台/框架的 EP/TP 参数语法不同（TRT: `--ep-sizes N`, SGLang: `--ep N`, ATOM: `--ep` flag）
3. 同一节点上多个 container 同名但版本不同（zufa_trt=post2 vs zufa_trt2=post3）
4. NODE 每天变，copy 旧 workflow 时忘更新

**记忆交叉验证**：`feedback_remote_workflow_pitfalls.md` §3 "EP/TP 参数搞混" 完全吻合（5+ 次修复）

**根因**：80+ workflow 文件各自硬编码 6 个配置值。改一个维度需要检查所有相关值的一致性，人/AI 无法可靠完成。

---

### #2: Git 操作连环 fix — rebase stuck / permission / detached HEAD (28 fixes)

**症状**：
- `git pull --rebase` 在 GPU 节点卡住（有未提交的 profiling 文件）
- Docker container 创建的文件是 root 权限，host 的 git 无法 add
- rebase 中断后 `.git/rebase-merge` 残留，后续所有 git 命令失败
- detached HEAD 状态下 push 失败

**代表 commit**：
```
35c4ba4 fix: B200 ops — force-remove .git/rebase-merge directory
d76bed6 fix: B200 ops — chown root-owned result files before git reset
c321adc fix: B200 workflows — handle detached HEAD and docker file permissions
775c759 fix: profiling workflow — exclude nsys-rep from git push (>100MB)
166d517 fix: MI355X profiling stash before rebase to handle unstaged trace files
```

**触发条件**：
1. 上一次 workflow 失败 → 留下 rebase 中间状态 → 下一次 workflow 启动时 git 损坏
2. Profiling 产生大文件（nsys-rep 几 GB）→ git add 全部 → push 超 100MB 限制失败
3. Container (root) 创建文件 → host (ubuntu) 的 git 无法操作

**演进轨迹**：
```
初始: git pull --rebase
→ 卡在 rebase: git rebase --abort 2>/dev/null
→ .git/rebase-merge 残留: rm -rf .git/rebase-merge
→ detached HEAD: git checkout -B main origin/main --force
→ 权限问题: chown before git add
→ 最终方案: git fetch && git reset --hard origin/main（最暴力但最可靠）
```

经过 28 次修复，最终稳定在 `git fetch origin && git reset --hard origin/main`。新的 `ci_lib.sh:ci_sync()` 直接用这个。

**根因**：Git sync 策略在 80+ workflow 里各自演进，有 4 种不同版本共存。旧策略（stash+rebase）不适合 CI 场景但没统一替换。

---

### #3: pkill -f 自杀 / 进程清理 (9 fixes)

**症状**：
- `pkill -f "sglang.launch_server"` 匹配到自身命令行 → kill 自己 → exit 143/137
- Runner 进程被杀
- GPU 残留进程阻塞下一次运行

**代表 commit**：
```
9564e80 fix: pkill -f kills itself via cmdline match → exit 143
7a8b506 fix: exclude self PID from pkill to avoid killing runner (exit 137)
53c5b8f fix: debug workflow cleanup step pkill -f self-kill
482c67e fix: remove host-level pkill that kills the runner process
a71cff4 fix: replace pkill -f with pgrep+kill in collect_ncu_trace.sh
```

**触发条件**：
1. Cleanup step 用 `pkill -f "pattern"` 清理残留进程
2. 该命令自身的 cmdline 也包含 pattern → 匹配到自己 → kill 自己
3. B300 情况更严重 — Runner 进程就在 GPU 节点上，被 pkill 会导致整个 job 中断

**记忆交叉验证**：`feedback_remote_workflow_pitfalls.md` §1 完全吻合。记忆说"永远不用 pkill -f"。

**根因**：pkill 的行为不直观（匹配自身），且修复方案分散在各脚本。当前 `ci_lib.sh:ci_kill_gpu_procs()` 仍然使用 `pkill -9 -f`，**直接违反记忆规则**。

---

## 未被记忆覆盖的错误模式

### 补充 #4: Kernel 分类错误 (8 fixes)

kernel name → operator 映射写错，导致分析结论错误。

```
1a5636a fix: router_splitK_reduce mis-classified as mla_qkv_a
8be04ce fix: kernel-name priority over module in MI355X classification
07bfac1 fix: correct kernel→module mapping for rocm722 trace
9ee8d43 fix: update B200_KERNEL_MAP for torch profiler full kernel names
```

**触发条件**：
1. 新 ROCm/TRT-LLM 版本改了 kernel 名
2. 同一 kernel regex 匹配到了错误的 operator（优先级问题）
3. 6 个分析脚本各自维护 kernel map → 改了一个忘改其他

**根因**：kernel map 分散在 6 个文件里，没有 central registry。

### 补充 #5: 数据标签 mislabel (15 fixes)

MTP/量化/平台标签搞混，dashboard 显示错误数据。

```
b5deff6 fix: B200 FP8 MTP=0 benchmark results (was mislabeled mtp1)
a600ed7 fix: unify mtp labels with env_tag
e69dacf Fix ATOM MI355X: EP=1 (TP=8), not EP=8
d3c374f fix: correct delta comparison logic and unify mtp labels
```

**触发条件**：
1. import_results.py 的 label 参数手动填错
2. env_tag 格式不统一（`mtp0` vs `[mtp0]` vs 无标签）
3. result_dir 命名含错误的 EP/TP 信息

**根因**：label 手动输入，没有从 result_dir 路径自动推导。

---

## 错误模式 vs 解决方案映射

| 错误模式 | 频率 | 已有记忆覆盖? | 结构性解决方案 |
|---------|------|-------------|--------------|
| 配置搞混 | 23 | ✅ §3 | ✅ workflow inputs + platform env（已设计） |
| Git 操作 | 28 | 部分 | ✅ ci_sync() 统一用 hard reset（已实现） |
| pkill 自杀 | 9 | ✅ §1 | ❌ **ci_lib.sh 仍在用 pkill -f！需立即修复** |
| NCU 参数 | 24 | ✅ profiling pitfalls | ⏸️ NCU 实验中，不动 |
| Kernel 分类 | 8 | ❌ 无记忆 | 🔲 kernel_registry.py（待实现） |
| 数据标签 | 15 | ✅ §5 | 🔲 自动从 result_dir 推导 label（ci_lib.sh） |
| Container 环境 | 10 | ✅ profiling §4 | 🔲 preflight_check() in benchmark_lib.sh |
| Trace 格式 | 8 | ✅ profiling §2 | 保持现状（已稳定） |
| Shell/YAML 语法 | 12 | ❌ | 无法结构性解决，靠经验 |

---

## 立即行动项

1. **修复 ci_lib.sh 的 ci_kill_gpu_procs()** — 当前用 `pkill -9 -f`，直接违反记忆规则
2. **补建 kernel_registry.py** — 消除 6 处 kernel map 重复，防止分类不一致
3. **benchmark_lib.sh 加 preflight check** — 脚本运行前验证 tool/container 环境
