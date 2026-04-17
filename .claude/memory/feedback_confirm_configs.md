---
name: Confirm key configs before running
description: Must confirm all key benchmark/profiling parameters with user before executing, never assume values
type: feedback
---

每次跑 benchmark 或 profiling 之前，必须和用户确认所有关键配置参数，不能擅自假设或填写数值。

**Why:** 用户发现我擅自把 profiling mode 的 `--profile-prompts` 设为 128，没有经过确认。配置参数直接影响测试结果的有效性。

**How to apply:** 在生成任何 benchmark/trace 运行命令之前，先列出关键配置（如 TP、EP、concurrency、num_prompts、warmup、profile-prompts、max_model_len 等）让用户确认，不要自己填值。如果不确定某个参数，明确问用户，而不是从其他地方推测。
