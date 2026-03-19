# DeepSeek R1 Benchmark Results (InferenceX-style)
## B200 8×GPU

| Config | Quant | Scenario | EP | CONC | DP | Req/s | Output TPS | Per-GPU TPS | TTFT p50 (ms) | TPOT p50 (ms) | E2E p50 (ms) |
|--------|-------|----------|-----|------|----|-------|------------|-------------|---------------|---------------|--------------|
| throughput | fp8 | chat | 1 | 128 | N | 4.91 | 4518.5 | 564.8 | 536 | 26.9 | 25152 |
