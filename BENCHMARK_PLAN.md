# B200 Full Benchmark Plan — Dashboard Demo

## Goal
Run complete FP8 + FP4 benchmark sweeps on 8×B200, fetch MI355X competitor data, and deploy the first comparison dashboard at https://zufayu.github.io/ClaudeSkillsE2EPerf/

## Configuration
| Parameter | Value |
|-----------|-------|
| GPUs | 8×B200 (192GB each) |
| Tensor Parallelism | 8 |
| Expert Parallelism | 1 |
| DP Attention | False |
| Image | `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3` |
| Model FP8 | `/home/models/models--DeepSeek-R1-0528` |
| Model FP4 | `/home/models/DeepSeek-R1-0528-NVFP4-v2` |

## Test Matrix

| Quant | Scenarios | Concurrency Sweep | Data Points |
|-------|-----------|-------------------|-------------|
| FP8 | chat (1K/1K), reasoning (1K/8K), summarize (8K/1K) | 1, 4, 8, 16, 32, 64, 128, 256 | 24 |
| FP4 | chat (1K/1K), reasoning (1K/8K), summarize (8K/1K) | 1, 4, 8, 16, 32, 64, 128, 256 | 24 |
| **B200 Total** | | | **48** |
| MI355X (ATOM) | auto-fetched from ATOM dashboard | | ~24+ |

## Steps

### Step 1: B200 Prepare
```bash
cd ~/zufa/ClaudeSkillsE2EPerf && git pull
```

### Step 2: B200 FP8 Full Sweep
```bash
bash scripts/sa_bench_b200.sh --model-fp8 /home/models/models--DeepSeek-R1-0528 --configs fp8-throughput --ep-sizes 1 --result-dir ./results_b200_fp8_ep1
```

### Step 3: B200 FP4 Full Sweep
```bash
bash scripts/sa_bench_b200.sh --model-fp4 /home/models/DeepSeek-R1-0528-NVFP4-v2 --configs fp4-throughput --ep-sizes 1 --result-dir ./results_b200_fp4_ep1
```

### Step 4: Copy Results to Local Machine
```bash
scp -r ubuntu@<B200_IP>:~/zufa/ClaudeSkillsE2EPerf/results_b200_fp8_ep1 ~/ClaudeSkillsE2EPerf/
scp -r ubuntu@<B200_IP>:~/zufa/ClaudeSkillsE2EPerf/results_b200_fp4_ep1 ~/ClaudeSkillsE2EPerf/
```

### Step 5: Import + Fetch Competitors + Deploy Dashboard
```bash
bash scripts/deploy_dashboard.sh --import ./results_b200_fp8_ep1 --platform "8×B200" --framework "TRT-LLM 1.2.0rc6.post3" --quantization FP8 --import ./results_b200_fp4_ep1 --platform "8×B200" --framework "TRT-LLM 1.2.0rc6.post3" --quantization NVFP4 --fetch-competitors
```

## Estimated Time
- Each data point: ~5-10 min
- FP8 full sweep (24 points): ~2-4 hours
- FP4 full sweep (24 points): ~2-4 hours
- Total B200 runtime: ~4-8 hours
- Import + deploy: ~2 min

## Notes
1. **Image**: Using `post3` (15% faster than `post2`). Verified that `post2` matches SA baseline (564.8 vs 557.71 tok/s/gpu, +1.3%)
2. **OOM risk**: chat c=256 may OOM — script records zeros, doesn't block other points. KV cache already lowered to 0.7 for c>=128
3. **MI355X data**: `--fetch-competitors` auto-pulls latest from https://rocm.github.io/ATOM/benchmark-dashboard/
4. **Dashboard**: After deploy, live at https://zufayu.github.io/ClaudeSkillsE2EPerf/
