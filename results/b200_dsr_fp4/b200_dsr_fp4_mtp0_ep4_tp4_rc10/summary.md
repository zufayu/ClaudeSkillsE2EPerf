# DeepSeek R1 Benchmark Results (InferenceX-style)
## B200 4×GPU

| Config | Quant | Scenario | EP | CONC | Total Tput | Output Tput | Interac. | TPOT (ms) | TTFT (ms) | DAR |
|--------|-------|----------|-----|------|------------|-------------|----------|-----------|-----------|-----|
| throughput | fp4 | chat | 4 | 4 | 917.4 | 456.5 | 117.90 | 8.5 | 77 | - |
| throughput | fp4 | chat | 4 | 64 | 6552.1 | 3275.3 | 52.57 | 19.0 | 94 | - |
| throughput | fp4 | reasoning | 4 | 4 | 543.4 | 481.9 | 122.41 | 8.2 | 82 | - |
| throughput | fp4 | summarize | 4 | 4 | 3743.4 | 416.4 | 109.15 | 9.2 | 223 | - |
