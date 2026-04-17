---
name: FP4 4GPU comparison table pending
description: Complete 4GPU FP4 comparison table to add to fp4-b200-vs-mi355x-breakdown.md after 端到端性能总表, ready for user verification
type: project
---

4GPU (TP=4) FP4 对比表，数据完整，待用户验证后更新 report。

**已有数据（全部 ratio=0.8 对齐 SA）：**
- B200 SGLang SA InferenceX: Total=6000.8, Output=2999.7, TPOT=20.05, TTFT=471.1, Inter=49.87
- B200 SGLang Ours-bench (ratio=0.8): Total=6397.3, Output=3197.9, TPOT=19.04, TTFT=403.5, Inter=52.52
- B200 SGLang Ours-profiling (ratio=0.8, WITH_STACK=False): Total=6311.8, Output=3155.2, TPOT=19.26, TTFT=411.5, Inter=51.92
- B200 TRT-LLM post2 Ours-bench: Total=6426.9, Output=3212.7, TPOT=19.4, TTFT=86.1, Inter=51.61
- MI355X ATOM EP4 TP4 rocm711 Ours-bench: Total=4767.3, Output=2383.1, TPOT=26.1, TTFT=101.8, Inter=38.25
- MI355X ATOM EP4 TP4 rocm711 Ours-profiling (roctracer 20M): Total=4435.6, Output=2217.3, TPOT=28.0, TTFT=111.5, Inter=35.77
- MI355X ATOM EP1 TP4 SA CI: Total=4806.8, Output=2402.9, TPOT=25.94, Inter=38.55
- MI355X ATOM EP1 TP4 rocm711 Ours-bench: Total=4906.9, Output=2452.9, TPOT=25.5, TTFT=104.4, Inter=39.28

**Why:** User wants complete verified data before updating report
**How to apply:** Present full table, get user confirmation, then edit report (add after 端到端性能总表)
