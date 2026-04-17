---
name: Profiling tool chain pitfalls
description: Lessons from ncu/nsys/torch trace profiling on B200 and MI355X. Covers ncu parameter evolution, NVTX schema discovery, roctx marker failures, container environment issues.
type: feedback
originSessionId: ef6a5f82-0e8a-4423-8898-b406f3722dca
---
profiling 工具链（ncu/nsys/torch trace/roctx）的踩坑经验，从 git 历史中提炼。

## 1. ncu 参数演进 (15+ commits, 2 次完全重写)

ncu profiling 脚本被重写了 2 次 (ff98ae1, 07f47fb)，几乎每个参数都踩过坑：

| 尝试 | 问题 | 最终方案 |
|------|------|---------|
| `cudaProfilerStart/Stop` | TRT-LLM 多进程不支持 | 不用 cudaProfiler API |
| `--profile-from-start off` | 部分 backend 不兼容 | 去掉，用 kernel filter 替代 |
| `--force-launch` | ncu 2024.3.1 不支持 | 去掉 |
| `--replay-mode kernel` | 每个 kernel 备份显存，OOM | `--replay-mode application` |
| 离线 bench 模式 | 无法控制 decode steady-state | server 模式 + 并发请求 |
| client 侧 wrap ncu | 只 profile 了 client | server 进程在 ncu 外启动，client 在外 |
| launch-skip/count | 不知道 decode 从第几个 kernel 开始 | nsys dry-run 先跑一遍定位 decode region |
| 无 kernel filter | 抓到 loading phase kernels | 加 `--kernel-regex` 过滤 |

**How to apply:**
- ncu 不要从零写，从现有 `collect_ncu_trace.sh` 模板改
- 新平台/新框架版本先跑 `--help` 确认支持的参数
- 永远用 nsys dry-run 先定位 decode region，再用 ncu 精确抓取

## 2. NVTX/SQLite schema 摸索 (7 commits)

nsys export 的 SQLite 里 NVTX marker 字段名在不同 nsys 版本和不同框架间不一致：

| 框架 | marker 位置 | 发现过程 |
|------|------------|---------|
| TRT-LLM | `StringIds` 表 join | 直接查 OK |
| SGLang | `jsonText` 列 | 试了 `textId` → `binaryData` → 最终 `jsonText`，开了 3 个 debug workflow 才定位 |

**Why:** 7 次 commit (135cee6 → 027fc85)，因为 nsys SQLite schema 没有官方文档，只能 trial-and-error。

**How to apply:**
- 遇到新 nsys 版本或新框架的 trace，第一步用 `sqlite3 file.sqlite ".schema"` dump 完整 schema
- 不要猜列名，先 `SELECT * FROM NVTX_EVENTS LIMIT 5` 看实际数据
- SGLang 的 module markers 用 `jsonText` 列，TRT-LLM 的用 StringIds join

## 3. roctx markers 在 MI355X 上失败 (6 commits)

试图给 ATOM 加 roctx markers 做 per-module 分析，反复失败：

```
5593f7c (加 roctx) → ac55e6f (fix batch size) → a34a212 (fix patch 入口)
→ 5506b9b (fix import path) → da2cfbb (fix spawn workers)
→ 796af6d (fix stdout 污染) → 3fbb964 (放弃 import hook)
→ 5373456 (清理残留 .pth) → 最终被 ATOM --mark-trace 取代 (95d758e)
```

**How to apply:**
- 不要自己给第三方框架加 roctx/nvtx markers，先查框架是否有内置 trace 功能
- ATOM 有 `--mark-trace` 参数，SGLang 有 `SGLANG_TORCH_PROFILER_DIR`
- 多进程框架用 `.pth` import hook 注入代码极不可靠（spawn 后 hook 可能不生效）

## 4. Container 环境差异

不同 container image 的工具链差异导致脚本失败：

| 问题 | commit |
|------|--------|
| rc10 container 缺 .so 文件 | 1faff94, cb07c51 |
| ncu 未安装 | b30080b |
| nsys `--force-overwrite` vs `-f` | 3c95740 |
| python 在 container 外 vs 内运行差异 | 3c95740 |
| `trtllm-serve` 命令语法 `serve serve MODEL` | bfab83f |

**How to apply:**
- 切换 container image 后先跑环境检查：`which nsys ncu python3`、`python3 -c "import tensorrt_llm"`
- 脚本中关键工具调用加 `command -v xxx || { echo "xxx not found"; exit 1; }` 前置检查
- nsys/ncu 参数在不同版本间有差异，用 `nsys --version` 确认后查对应文档
