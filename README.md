# ClaudeSkillsE2EPerf

End-to-end performance benchmarking skills for LLM inference, generated and validated by Claude.

> **First time?** Follow the [Deployment Guide](docs/deployment-guide.md) from zero to results on a fresh machine.

## Supported Platforms

| Platform | GPU | Memory | Quantization | Docker Image | Status |
|----------|-----|--------|-------------|--------------|--------|
| **B200** | 8x NVIDIA B200 | 192GB/GPU | FP4 + FP8 | `release:1.2.0rc4` / `rc6.post3` | Active |
| **H200** | 8x NVIDIA H200 | 141GB/GPU | FP8 only | `release:1.2.0rc4` | Active |
| **H20**  | 8x NVIDIA H20  | 96GB/GPU  | FP8 only | `release:1.2.0rc2` | Done |

## Test Matrix (SemiAnalysis InferenceX-aligned)

| Scenario | ISL | OSL | Use Case |
|----------|-----|-----|----------|
| chat | 1024 | 1024 | Short-in Short-out |
| reasoning | 1024 | 8192 | Short-in Long-out |
| summarize | 8192 | 1024 | Long-in Short-out |

- **Concurrency sweep**: 1, 4, 8, 16, 32, 64, 128, 256 (filterable via `--concurrency`)
- **Output variation**: ±20% (random_range_ratio=0.8)
- **Warmups**: 8 (SA default, overridable via `--num-warmups`)
- **Metrics**: Output TPS, TTFT p50, TPOT p50, E2E latency p50

## Platform Configurations

### B200

| Config | Quant | MTP | DP Attention | MOE Backend | Target |
|--------|-------|-----|-------------|-------------|--------|
| `fp4-throughput` | NVFP4 | No | Auto | TRTLLM/CUTLASS | Max throughput |
| `fp4-latency` | NVFP4 | MTP-3/1 | Auto | TRTLLM/CUTLASS | Min latency |
| `fp8-throughput` | FP8 | No | Auto | TRTLLM/DEEPGEMM | Max throughput |
| `fp8-latency` | FP8 | MTP-3/1 | Auto | TRTLLM/DEEPGEMM | Min latency |

EP sizes: EP=1 (pure TP), EP=8 (pure EP with DP attention). Filterable via `--ep-sizes`.

### Quick Start: Single Data Point

```bash
# Reproduce SA B200 TRT FP8 c=256 EP=8 chat data point
bash scripts/sa_bench_b200.sh \
  --model-fp8 /path/to/DeepSeek-R1-FP8 \
  --configs fp8-throughput \
  --scenario chat \
  --concurrency 256 \
  --ep-sizes 8 \
  --result-dir ./results_b200_fp8_repro
```

All filter flags: `--scenario` (chat/reasoning/summarize/all), `--concurrency` ("256" or "4 128 256"), `--ep-sizes` ("8" or "1 8"), `--num-warmups` (default: 8). See `bash scripts/sa_bench_b200.sh --help` for details.

Auto-adaptive optimizations per scenario/concurrency:
- Piecewise CUDA Graphs for high concurrency
- Dynamic MOE backend: TRTLLM → CUTLASS (FP4+DP) → DEEPGEMM (FP8+DP)
- Delay Batching for FP8 throughput at high concurrency
- KV Cache Fraction: 0.8 default, 0.7 for DEEPGEMM configs

### H200

| Config | Quant | MTP | MOE Backend | Target |
|--------|-------|-----|-------------|--------|
| `fp8-throughput` | FP8 | No | CUTLASS | Max throughput |
| `fp8-latency` | FP8 | MTP-3/1 | CUTLASS | Min latency |

EP sizes: EP=4, EP=8. KV cache fraction 0.75. Max concurrency 128.

### H20

FP8 only, EP=4, max concurrency 64. Uses `trtllm-bench` (not `benchmark_serving.py`).

### Platform Comparison

| Parameter | B200 | H200 | H20 |
|-----------|------|------|-----|
| GPU Memory | 192GB | 141GB | 96GB |
| Quantization | FP4 + FP8 | FP8 | FP8 |
| MOE Backend | TRTLLM/CUTLASS/DEEPGEMM | CUTLASS | TRTLLM |
| KV Cache Fraction | 0.80 | 0.75 | 0.80 |
| Piecewise CUDA Graphs | Yes | No | No |
| Max Concurrency | 256 | 128 | 64 |
| EP Sizes | 1, 8 | 4, 8 | 4 |
| Benchmark Method | `benchmark_serving.py` | `benchmark_serving.py` | `trtllm-bench` |

## Unified Comparison Dashboard

Interactive dashboard for cross-platform benchmark comparison. Compares our TRT-LLM results (B200/H200/H20) against competitor data (e.g., ATOM/ROCm MI355X).

### Data Pipeline

```
Benchmark scripts          Competitor CI
(sa_bench_*.sh)            (ATOM gh-pages)
       |                         |
       v                         v
  results_*/              fetch_competitors.py
  result_*.json                  |
       |                         v
       v                    runs/*.json  <-- unified format
  import_results.py ----------->|
                                v
                       generate_dashboard.py
                                |
                                v
                         dashboard/data.js
                                |
                                v
                       dashboard/index.html
```

### Dashboard Features

- **Throughput vs Latency** scatter plot (InferenceX-style, lower-right = better)
- **Throughput / Latency** line charts by concurrency
- **Data Table** with auto-calculated Delta% between platforms
- Filters: Platform, ISL/OSL scenario, Concurrency

### Dashboard Scripts

| Script | Purpose |
|--------|---------|
| `scripts/import_results.py` | Convert `results_*/result_*.json` → unified `runs/*.json` |
| `scripts/fetch_competitors.py` | Fetch ATOM's latest data from GitHub Pages → `runs/atom-*.json` |
| `scripts/generate_dashboard.py` | Merge all `runs/*.json` → `dashboard/data.js` |

See [Step 10 in Deployment Guide](docs/deployment-guide.md#step-10-generate-comparison-dashboard) for usage.

### Unified Run Format

```json
{
  "run_id": "8xb200-nvfp4-20260316",
  "platform": "8×B200",
  "framework": "TRT-LLM 1.2.0rc4",
  "model": "DeepSeek-R1-0528",
  "quantization": "NVFP4",
  "gpu_count": 8,
  "source": "manual",
  "date": "2026-03-16",
  "results": [
    {"isl": 1024, "osl": 1024, "conc": 1, "output_tps": 114.58, "tpot_p50": 8.66, "ttft_p50": 62.55}
  ]
}
```

### Competitor Data Sources

| Competitor | GPU | Source | Auto-fetch |
|-----------|-----|--------|-----------|
| [ATOM (ROCm)](https://rocm.github.io/ATOM/benchmark-dashboard/) | 8×MI355X | CI nightly | `fetch_competitors.py` |

## Architecture Notes

DeepSeek R1 uses `DeepseekV3ForCausalLM` (`model_type: deepseek_v3`):
- 671B params, 256 routed experts, 8 experts/token
- MLA (Multi-head Latent Attention) with KV LoRA
- Only **PyTorch backend** supported (no TRT engine)

## File Structure

```
.
├── README.md                       # Project overview & technical reference
├── docs/
│   └── deployment-guide.md         # End-to-end operations manual
├── dashboard/                      # Unified comparison dashboard
│   ├── index.html                  # Dashboard page (dark theme, Chart.js)
│   └── data.js                     # Generated data (do not edit manually)
├── runs/                           # Unified benchmark run data
│   ├── 8xb200-nvfp4-*.json        # Our B200 results
│   ├── 8xh20-fp8-*.json           # Our H20 results
│   └── atom-mi355x-*.json         # Competitor data (auto-fetched)
├── scripts/
│   ├── sa_bench_b200.sh            # B200 benchmark suite
│   ├── sa_bench_h200.sh            # H200 benchmark suite
│   ├── sa_bench_h20.sh             # H20 benchmark suite
│   ├── benchmark_lib.sh            # Shared utilities
│   ├── import_results.py           # results_*/ → runs/*.json
│   ├── fetch_competitors.py        # Fetch competitor data → runs/*.json
│   ├── generate_dashboard.py       # runs/ → dashboard/data.js
│   ├── launch_b200_docker.sh       # B200 Docker launcher
│   ├── launch_h200_docker.sh       # H200 Docker launcher
│   ├── gen_dataset.py              # Dataset generator
│   ├── run_bench.sh                # Simple trtllm-bench runner
│   └── serve.sh                    # Model serving script
├── utils/
│   └── bench_serving/              # benchmark_serving.py (from InferenceX)
├── configs/
│   ├── bench_config.yaml           # H20 config
│   ├── bench_config_b200.yaml      # B200 config
│   └── bench_config_h200.yaml      # H200 config
└── results/
    └── deepseek_r1_8xh20_fp8_pytorch.json
```

## License

MIT
