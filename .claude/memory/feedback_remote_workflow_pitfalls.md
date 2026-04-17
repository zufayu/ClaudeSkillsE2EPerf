---
name: Remote GPU workflow pitfalls
description: Recurring failure patterns in GitHub Actions self-hosted workflows on B200/MI355X/B300 GPU nodes. Covers pkill self-kill, health check, EP/TP params, workflow_dispatch cache, data mislabel.
type: feedback
originSessionId: ef6a5f82-0e8a-4423-8898-b406f3722dca
---
远程 GPU workflow 反复踩坑的模式，从 git 历史 155 条 fix commit 中提炼。

## 1. pkill -f 自杀 (exit 143)

`pkill -f "server_pattern"` 会匹配到自身命令行 → kill 自己 → exit 143。

**Why:** 3 轮 commit 才修完 (9564e80 → a71cff4 → 53c5b8f)，B300 还专门开了 debug workflow 排查。

**How to apply:**
- 永远不用 `pkill -f`，改用 `pgrep -f pattern | grep -v $$ | xargs kill` 或保存 PID 文件
- cleanup 步骤中同理，不能用 `pkill -f` 清理进程

## 2. Health check 不能用 grep

`curl ... | grep -q ok` 在不同框架（TRT-LLM vs SGLang vs ATOM）返回格式不同，grep 会失败。

**Why:** 修了 3 次 (c39bbf8 → b5603ef → a71cff4)。

**How to apply:**
- 统一用 `curl -sf http://host:port/health` 或 `curl -sf http://host:port/v1/models`，只检查 HTTP 200，不 grep body

## 3. EP/TP 参数搞混

多平台多配置下 EP/TP 默认值不同，写 workflow 时经常写错：EP=1 写成 EP=8、EP=8 写成 EP=4 等。

**Why:** 5+ 次修复 (e69dacf, d9568dc, 283b2b0, 159fe9b, 4d74f0c)。ATOM 默认 EP=1 TP=8，B200 有 EP=1/4/8 多种。

**How to apply:**
- Workflow 中 EP/TP 必须显式声明，不依赖脚本默认值
- 提交前用 `grep -n "EP\|TP\|ep_size\|tp_size"` 校验参数一致性
- 目录名本身就含 ep/tp 字段，和实际参数不一致时要立即修正

## 4. GitHub workflow_dispatch 缓存

新建或改名 workflow 后 GitHub 不立即识别 workflow_dispatch trigger，手动 dispatch 会报 "workflow not found"。

**Why:** 10+ commits 绕这个问题 (db7ec87, 041cc41, 789648f, 1948b07, 9b6ff0f...)，用了 rename file / add push trigger / repository_dispatch workaround / trigger refresh 等方法。

**How to apply:**
- 新 workflow 文件先加 `push:` trigger 让 GitHub 注册它，跑一次后再改成 `workflow_dispatch:`
- 或者用 `repository_dispatch` + `gh api` 手动触发来绕过缓存
- 不要反复 commit "trigger refresh" 空提交，这是浪费

## 5. 数据标签 mislabel

MTP/量化/平台 标签在 import/generate 流程中搞混，导致 dashboard 显示错误数据。

**Why:** 4 次修复 (b5deff6, 7994d87, d3c374f, a600ed7)。MTP=0 标成 MTP=1、delta 方向搞反、env_tag 不统一。

**How to apply:**
- import_results.py 运行后必须 spot-check 输出 JSON 的 label 字段
- dashboard deploy 前在本地 `python3 -m http.server` 打开检查数据是否正确
- env_tag 格式统一为 `[mtpN]`，不要出现 mtp1（不存在这个配置）
