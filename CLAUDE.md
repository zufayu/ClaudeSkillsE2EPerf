# E2EDeepPerf - Claude Context

## Project Overview
Benchmark comparison dashboard for DeepSeek R1 671B inference across GPU platforms.
- Dashboard: https://zufayu.github.io/ClaudeSkillsE2EPerf/ (password: `deepperf2026`)
- GitHub: `zufayu/ClaudeSkillsE2EPerf`

## Machines
| Machine | Working Dir | Role |
|---------|------------|------|
| H20 (kqian) | `/home/kqian/ClaudeSkillsE2EPerf` | Dashboard deploy, data management |
| B200 (ubuntu) | `/home/ubuntu/zufa/ClaudeSkillsE2EPerf` | NVIDIA benchmark runner |

## Model Paths (B200 Machine)
- **FP8**: `/home/models/models--DeepSeek-R1-0528/` (quant_method: fp8)
- **FP4**: `/home/models/DeepSeek-R1-0528-NVFP4-v2/` (nvidia/DeepSeek-R1-0528-NVFP4-v2)

## Key Scripts
| Script | Purpose |
|--------|---------|
| `scripts/sa_bench_b200.sh` | Main benchmark runner (InferenceX-style) |
| `scripts/import_results.py` | Convert result JSONs to unified run format |
| `scripts/fetch_competitors.py` | Fetch ATOM/ROCm MI355X data from their dashboard |
| `scripts/generate_dashboard.py` | Merge all runs into docs/data.js |
| `scripts/deploy_dashboard.sh` | One-stop: import + fetch + generate + deploy to gh-pages |

## Data Pipeline
```
results_*/result_*.json → import_results.py → runs/*.json
ATOM dashboard data.js  → fetch_competitors.py → runs/atom-*.json
                           runs/*.json → generate_dashboard.py → docs/data.js
                           docs/* → deploy_dashboard.sh → gh-pages
```

## Benchmark Commands

### B200 FP8 mtp0 (no MTP, throughput config)
```bash
bash scripts/sa_bench_b200.sh --model-fp8 /home/models/models--DeepSeek-R1-0528/ --configs fp8-throughput --ep-sizes 1 --result-dir ./results_b200_fp8_ep1
```

### B200 FP8 mtp3 (MTP3, latency config)
```bash
bash scripts/sa_bench_b200.sh --model-fp8 /home/models/models--DeepSeek-R1-0528/ --configs fp8-latency --ep-sizes 1 --result-dir ./results_b200_fp8_mtp3
```

### B200 FP4
```bash
bash scripts/sa_bench_b200.sh --model-fp4 /home/models/DeepSeek-R1-0528-NVFP4-v2/ --configs fp4-throughput --ep-sizes 1 --result-dir ./results_b200_fp4_ep1
```

### Single point test
```bash
bash scripts/sa_bench_b200.sh --model-fp8 /home/models/models--DeepSeek-R1-0528/ --configs fp8-throughput --scenario chat --concurrency 128 --ep-sizes 1 --result-dir ./results_b200_test
```

### Deploy dashboard (from H20 machine)
```bash
bash scripts/deploy_dashboard.sh --fetch-competitors
```

## B200 Configuration
- GPUs: 8×B200 (192GB each, Blackwell SM100)
- TP=8, EP=1, DP Attention=False
- Docker image: `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3`
- SA (SemiAnalysis) uses same config but `rc6.post2` image

## SemiAnalysis Reference (chat 1K/1K c=128)
- Output TPS/GPU: 557.71 (our post2: 564.8, post3: 649.2)
- Image `post3` is ~15% faster than `post2`

## Docker Launch (B200)
```bash
sudo docker run -it --gpus all --ipc host --ulimit memlock=-1 \
  --name zufa_trt --ulimit stack=67108864 \
  -v /home:/home -v /mnt:/mnt -v /data:/data \
  --shm-size=64g -p 8000:8000 \
  --cap-add=SYS_PTRACE \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3
```
- `--cap-add=SYS_PTRACE` required for EP>1 (NVLink MNNVL)
- Use `tmux` inside Docker to survive SSH disconnects

## Known Issues
1. **EP=8 FP8 broken on B200** - DeepGEMM only supports SM90 (Hopper), not SM100 (Blackwell). Use EP=1.
2. **Chat c>=128 OOM** - KV cache lowered to 0.7 in script for c>=128
3. **Summarize not in mtp0 data** - B200 mtp0 (results_b200_fp8_ep1) missing summarize results

## Competitor: ATOM MI355X
- Dashboard: https://rocm.github.io/ATOM/benchmark-dashboard/
- Auto-fetched via `scripts/fetch_competitors.py`
- Uses `env_tag` field for mtp0/mtp3 labeling
- Current data: FP8 mtp0 (23 pts), FP8 mtp3 (17 pts)

## Dashboard Labels
Series use `env_tag` for MTP differentiation:
- `8×B200 DeepSeek-R1-0528 FP8 (TRT-LLM) [mtp0]`
- `8×B200 DeepSeek-R1-0528 FP8 (TRT-LLM) [mtp3]`
- `8×MI355X DeepSeek-R1-0528 FP8 (ATOM) [mtp0]`
- `8×MI355X DeepSeek-R1-0528 FP8 (ATOM) [mtp3]`

## Delta Comparison Logic
- TPS: `(ref - cmp) / cmp` — positive green = Ref higher throughput
- TPOT/TTFT: `(cmp - ref) / ref` — positive green = Ref lower latency (better)
