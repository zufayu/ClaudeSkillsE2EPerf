# DeepSeek R1 Benchmark Results (SGLang)
## B200 8×GPU

| Config | Quant | Scenario | TP | EP | CONC | Total Tput | Output Tput | Interac. | TPOT (ms) | TTFT (ms) |
|--------|-------|----------|----|----|------|------------|-------------|----------|-----------|-----------|
| throughput | fp4 | chat | 8 | 8 | 4 | 1002.3 | 498.7 | 130.51 | 7.7 | 115 |
| throughput | fp4 | chat | 8 | 8 | 64 | 7998.0 | 3998.1 | 65.62 | 15.2 | 324 |
| throughput | fp4 | reasoning | 8 | 8 | 4 | 576.1 | 510.9 | 130.09 | 7.7 | 112 |
| throughput | fp4 | summarize | 8 | 8 | 4 | 4051.2 | 450.7 | 118.61 | 8.4 | 211 |
