# DeepSeek R1 Benchmark Results (InferenceX-style)
## B200 1×GPU

| Config | Quant | Scenario | EP | CONC | Total Tput | Output Tput | Interac. | TPOT (ms) | TTFT (ms) | DAR |
|--------|-------|----------|-----|------|------------|-------------|----------|-----------|-----------|-----|
| throughput | fp4 | chat | 1 | 1 | 859.7 | 431.4 | 435.95 | 2.3 | 23 | - |
| throughput | fp4 | chat | 1 | 16 | 6297.0 | 3131.8 | 199.50 | 5.0 | 27 | - |
| throughput | fp4 | chat | 1 | 64 | 14179.1 | 7088.0 | 113.20 | 8.8 | 36 | - |
