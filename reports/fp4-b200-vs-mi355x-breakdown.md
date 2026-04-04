# DeepSeek-R1 FP4 端到端性能对比：B200 vs MI355X

> **Last updated:** 2026-04-04
> **Model:** DeepSeek-R1-0528, MTP=0, chat 1K/1K, c=64, DP=false
> **5 Metrics:** Total Tput (tok/s), Output Tput (tok/s), Interactivity (tok/s/user), TPOT p50 (ms), TTFT p50 (ms)

| Platform | Quant | EP | TP | Env | Mode | Total Tput | Output Tput | Interac. | TPOT (ms) | TTFT (ms) |
|----------|-------|----|----|-----|------|------------|-------------|----------|-----------|-----------|
| B200 | NVFP4 | 8 | 8 | post2 | bench | 7577.9 | 3788.1 | 60.96 | 16.4 | 72 |
| B200 | NVFP4 | 1 | 8 | post2 | bench | 7602.5 | 3800.4 | 60.61 | 16.5 | 83 |
| B200 | NVFP4 | 8 | 8 | post2 | profiling | — | — | — | — | — |
| MI355X | MXFP4 | 1 | 8 | rocm721 | bench | 6961.7 | 3480.1 | 55.95 | 17.9 | 94.7 |
| MI355X | MXFP4 | 1 | 8 | rocm711 | bench | 6833.0 | 3415.7 | 54.95 | 18.2 | 93.5 |
| MI355X | MXFP4 | 8 | 8 | rocm721 | bench | 6558.0 | 3278.2 | 52.61 | 19.0 | 96.8 |
| MI355X | MXFP4 | 8 | 8 | rocm711 | bench | 6470.5 | 3234.5 | 52.05 | 19.2 | 90.6 |
| MI355X | MXFP4 | 1 | 4 | rocm721 | bench | 5054.8 | 2526.8 | 40.40 | 24.8 | 104.5 |
| MI355X | MXFP4 | 1 | 4 | rocm711 | bench | 4906.9 | 2452.9 | 39.28 | 25.5 | 104.4 |
| MI355X | MXFP4 | 1 | 8 | rocm711 | profiling | 6302.5 | 3150.6 | 50.5 | 19.8 | 117.3 |
| MI355X | MXFP4 | 8 | 8 | rocm711 | profiling | 6014.6 | 3006.6 | 48.3 | 20.7 | 114.6 |
