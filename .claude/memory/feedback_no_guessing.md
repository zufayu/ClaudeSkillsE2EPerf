---
name: No guessing — use original data
description: Never reconstruct operator shapes/names/types from memory. Always read the verified source (CSV, profiling data, config.json) before citing any technical detail.
type: feedback
---

推论和总结时必须严谨，以原始数据为准，不要凭记忆推测。

**Why:** 用户花了大量时间从 config.json、modeling_deepseek.py、profiling trace 逐一验证了每个算子的 shape、数据类型、量化方法、kernel 名称，并写入了 CSV 需求表。我多次凭记忆重新编表，导致算子类型搞错（如 kv_b_proj 是 batched GEMM 写成了普通 GEMM）、shape 编错、quant 类型混淆。用户已反复纠正。

**How to apply:**
- 涉及算子 shape、类型、kernel 名、量化方法等技术细节时，必须先 Read 原始数据文件（CSV、profiling 数据），不允许从记忆中重构
- 不要自行简化或修改原始算子名称
- 如果没有读过原始数据，明确说"我需要先查看数据"，而不是猜测后输出
- 宁可少说，也不要说错
