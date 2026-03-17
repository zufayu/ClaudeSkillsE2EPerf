# MI355X (ROCm) Benchmark Guide

## Goal
Run DeepSeek-R1-0528 benchmarks on 8×MI355X to compare against 8×B200 results on the unified dashboard at https://zufayu.github.io/ClaudeSkillsE2EPerf/

## Hardware
- 8×AMD Instinct MI355X
- ROCm software stack
- Framework: ATOM (vLLM-based)

## ATOM Project Reference
- GitHub: https://github.com/ROCm/ATOM
- Dashboard: https://rocm.github.io/ATOM/benchmark-dashboard/
- CI Benchmarks: https://github.com/ROCm/ATOM/actions/runs/23115155923

## Existing ATOM CI Data (2026-03-15)

The ATOM CI run #23115155923 contains comprehensive DeepSeek-R1-0528 BF16 results on 8×MI355X.

### Chat 1K/1K (DeepSeek-R1-0528 BF16, 8×MI355X)
| CONC | Num Prompts | Output TPS | Out TPS/GPU | Total TPS | TPOT p50 (ms) | TTFT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) |
|------|-------------|-----------|-------------|-----------|---------------|---------------|-------------|--------------|
| 1 | 10 | 79.8 | 10.0 | 159.0 | 10.4 | 60.2 | 10.3 | 10450.7 |
| 2 | 20 | 179.9 | 22.5 | 356.5 | 10.7 | 76.3 | 10.6 | 10350.5 |
| 4 | 40 | 335.2 | 41.9 | 673.6 | 11.3 | 78.7 | 11.1 | 10551.6 |
| 8 | 80 | 628.3 | 78.5 | 1251.9 | 12.0 | 79.7 | 11.6 | 11489.0 |
| 16 | 160 | 1103.9 | 138.0 | 2219.4 | 13.7 | 82.7 | 12.9 | 12972.7 |
| 32 | 320 | 1655.0 | 206.9 | 3304.7 | 18.6 | 76.8 | 16.7 | 17467.7 |
| 64 | 640 | 2743.5 | 342.9 | 5488.3 | 22.5 | 91.1 | 19.0 | 20969.6 |
| 128 | 1280 | 4266.8 | 533.4 | 8543.0 | 29.1 | 123.6 | 23.0 | 26984.9 |
| 256 | 2560 | 5584.8 | 698.1 | 11164.1 | 42.4 | 163.4 | 30.6 | 39665.9 |

### Reasoning 1K/8K (DeepSeek-R1-0528 BF16, 8×MI355X)
| CONC | Num Prompts | Output TPS | Out TPS/GPU | Total TPS | TPOT p50 (ms) | TTFT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) |
|------|-------------|-----------|-------------|-----------|---------------|---------------|-------------|--------------|
| 1 | 10 | 91.8 | 11.5 | 103.1 | 10.7 | 67.2 | 10.7 | 79877.2 |
| 2 | 20 | 180.1 | 22.5 | 202.8 | 11.0 | 77.4 | 10.9 | 81020.4 |
| 4 | 40 | 336.9 | 42.1 | 379.9 | 11.6 | 79.2 | 11.4 | 84977.5 |
| 8 | 80 | 622.9 | 77.9 | 701.0 | 12.3 | 69.9 | 12.1 | 92881.1 |
| 16 | 160 | 1130.9 | 141.4 | 1272.8 | 13.7 | 83.8 | 13.6 | 101727.1 |
| 32 | 320 | 1735.2 | 216.9 | 1950.7 | 17.9 | 77.7 | 17.6 | 133763.8 |

### Summarize 8K/1K (DeepSeek-R1-0528 BF16, 8×MI355X)
| CONC | Num Prompts | Output TPS | Out TPS/GPU | Total TPS | TPOT p50 (ms) | TTFT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) |
|------|-------------|-----------|-------------|-----------|---------------|---------------|-------------|--------------|
| 1 | 10 | 84.3 | 10.5 | 754.5 | 11.0 | 271.0 | 10.9 | 10450.9 |
| 2 | 20 | 142.1 | 17.8 | 1266.0 | 11.6 | 250.7 | 11.2 | 11818.4 |
| 4 | 40 | 296.8 | 37.1 | 2667.8 | 12.4 | 282.3 | 11.6 | 12081.4 |
| 8 | 80 | 534.1 | 66.8 | 4750.9 | 14.0 | 282.5 | 12.2 | 13437.2 |
| 16 | 160 | 851.0 | 106.4 | 7679.8 | 17.8 | 286.1 | 13.8 | 16678.5 |
| 32 | 320 | 1126.2 | 140.8 | 10071.1 | 27.4 | 283.0 | 18.0 | 25629.8 |
| 64 | 640 | 1687.5 | 211.0 | 15210.5 | 37.0 | 359.6 | 21.6 | 34683.7 |
| 128 | 1280 | 2139.6 | 267.4 | 19316.3 | 58.1 | 474.9 | 28.5 | 54007.8 |

### MTP3 Chat 1K/1K (DeepSeek-R1-0528-mtp3 BF16 +MTP3, 8×MI355X)
| CONC | Num Prompts | Output TPS | Out TPS/GPU | TPOT p50 (ms) | ITL p50 (ms) |
|------|-------------|-----------|-------------|---------------|-------------|
| 4 | 40 | 410.7 | 51.3 | 6.3 | 15.0 |
| 8 | 80 | 934.8 | 116.8 | 8.2 | 18.7 |
| 16 | 160 | 1429.1 | 178.6 | 10.8 | 21.6 |
| 32 | 320 | (data cut off) | | | |

## B200 vs MI355X Comparison (Chat 1K/1K c=128)

| Metric | B200 FP8 (post3) | B200 FP8 (post2) | MI355X BF16 | B200(post3) vs MI355X |
|--------|-----------------|-----------------|-------------|----------------------|
| Output TPS | 5193.8 | 4518.5 | 4266.8 | +21.7% |
| Out TPS/GPU | 649.2 | 564.8 | 533.4 | +21.7% |
| TPOT p50 (ms) | 23.4 | 26.9 | 29.1 | -19.6% (lower=better) |
| TTFT p50 (ms) | 474.6 | 536.0 | 123.6 | MI355X wins |
| ITL p50 (ms) | — | — | 23.0 | — |

### SA InferenceX Verified
- B200 FP8 with SA's exact image (rc6.post2): 564.8 tok/s/gpu vs SA's 557.71 → +1.3%, confirms our setup is correct
- B200 FP8 with newer image (rc6.post3): 649.2 tok/s/gpu → +16.4% vs SA

## Data Availability

### What ATOM Dashboard Has (https://rocm.github.io/ATOM/benchmark-dashboard/data.js)
- Only latest c=128 single data point per model
- DeepSeek-R1-0528 BF16: chat c=128 = 4340.6 tok/s (slightly different from CI)
- DeepSeek-R1-0528-mtp3 BF16: chat c=128 = 5065.1 tok/s
- GLM-5 FP8: chat c=128 = 2905.3 tok/s
- **NOT sufficient** for full concurrency sweep comparison

### What ATOM CI Has (GitHub Actions #23115155923)
- Full concurrency sweep: 1, 2, 4, 8, 16, 32, 64, 128, 256
- All 3 scenarios: chat, reasoning, summarize
- MTP3 variant data (partial)
- **Sufficient** for dashboard comparison
- Missing: reasoning c=64/128/256, summarize c=256

### Concurrency Mapping (ATOM vs B200)
| ATOM | B200 | Match |
|------|------|-------|
| 1 | 1 | Yes |
| 2 | — | ATOM only |
| 4 | 4 | Yes |
| 8 | 8 | Yes |
| 16 | 16 | Yes |
| 32 | 32 | Yes |
| 64 | 64 | Yes |
| 128 | 128 | Yes |
| 256 | 256 | Yes |

## Running Benchmarks on MI355X

### If Using ATOM Docker
```bash
# Find latest ATOM docker image
# Check https://github.com/ROCm/ATOM for recommended image

# Example (verify actual image name):
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video \
  --shm-size=64g -v /home:/home -p 8000:8000 \
  <ATOM_DOCKER_IMAGE>
```

### Key Differences from B200 Script
| Aspect | B200 (sa_bench_b200.sh) | MI355X (needs new script) |
|--------|------------------------|--------------------------|
| Framework | TRT-LLM (`trtllm-serve`) | vLLM/ATOM (`vllm serve`) |
| Docker | `nvcr.io/nvidia/tensorrt-llm/...` | ROCm ATOM image |
| GPU flags | `--gpus all` | `--device=/dev/kfd --device=/dev/dri` |
| MoE params | MOE_BACKEND, PIECEWISE_CUDA_GRAPHS | Not applicable |
| GPU monitor | `nvidia-smi` | `rocm-smi` |
| Benchmark client | `benchmark_serving.py` (OpenAI compat) | Same — reusable |
| Result format | `result_*.json` | Same — reusable |

### What's Reusable
- `benchmark_serving.py` client (uses OpenAI-compatible API, works with any vLLM server)
- `import_results.py` (result JSON format is identical)
- `deploy_dashboard.sh` (import + deploy pipeline)
- `generate_dashboard.py` and `dashboard/index.html`

### What Needs New Script
- Server launch logic (vLLM instead of trtllm-serve)
- GPU monitoring (rocm-smi instead of nvidia-smi)
- Adaptive parameters (different tuning knobs for ROCm/vLLM)

### Recommended Approach
Create `scripts/sa_bench_mi355x.sh` based on `sa_bench_b200.sh`, replacing:
1. Server start: `trtllm-serve` → `vllm serve` with ROCm-specific flags
2. GPU monitor: `nvidia-smi` → `rocm-smi`
3. Health check: same (`/health` endpoint)
4. Benchmark client: same (`benchmark_serving.py`)
5. Result collection: same format

## Importing ATOM CI Data (Alternative to Running)

If not running benchmarks yourself, import the CI data directly:

### Option A: Manual Import
Convert the CI summary table data into `runs/atom-mi355x-deepseek-r1-0528.json` matching the unified run format.

### Option B: fetch_competitors.py Enhancement
Modify `scripts/fetch_competitors.py` to also pull from GitHub Actions API:
```bash
gh api repos/ROCm/ATOM/actions/runs/23115155923/artifacts
```

## Dashboard Deploy (After Getting Data)

```bash
# Import B200 results + fetch/import MI355X + deploy
bash scripts/deploy_dashboard.sh \
  --import ./results_b200_fp8_ep1 --platform "8×B200" --framework "TRT-LLM 1.2.0rc6.post3" --quantization FP8 \
  --import ./results_b200_fp4_ep1 --platform "8×B200" --framework "TRT-LLM 1.2.0rc6.post3" --quantization NVFP4 \
  --fetch-competitors
```

Dashboard: https://zufayu.github.io/ClaudeSkillsE2EPerf/
