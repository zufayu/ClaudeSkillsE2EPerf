# ClaudeSkillsE2EPerf

End-to-end performance benchmarking skills for LLM inference, generated and validated by Claude.

## Supported Platforms

| Platform | GPU | Memory | Docker Image | Status |
|----------|-----|--------|--------------|--------|
| **B200** | 8x NVIDIA B200 | 192GB/GPU (1.5TB) | `release:1.2.0rc4` | New |
| **H200** | 8x NVIDIA H200 | 141GB/GPU (1.1TB) | `release:1.2.0rc4` | New |
| **H20**  | 8x NVIDIA H20  | 96GB/GPU (768GB)  | `release:1.2.0rc2` | Done |

## B200 Benchmarks (InferenceX/MAX-style)

### Quick Start

```bash
# 1. Launch Docker container
bash scripts/launch_b200_docker.sh --name B200_trtllm

# 2. Attach and run benchmarks
docker attach B200_trtllm
bash /home/kqian/ClaudeSkillsE2EPerf/scripts/sa_bench_b200.sh \
  --model-fp4 /data/amd/DeepSeek-R1-0528-NVFP4-v2 \
  --model-fp8 /data/DeepSeek-R1-FP8 \
  --configs all
```

### Configs

| Config | Quant | MTP | DP Attention | MOE Backend | Target |
|--------|-------|-----|-------------|-------------|--------|
| `fp4-throughput` | NVFP4 | No | Auto | TRTLLM/CUTLASS | Max throughput |
| `fp4-latency` | NVFP4 | MTP-3/1 | Auto | TRTLLM/CUTLASS | Min latency |
| `fp8-throughput` | FP8 | No | Auto | TRTLLM/DEEPGEMM | Max throughput |
| `fp8-latency` | FP8 | MTP-3/1 | Auto | TRTLLM/DEEPGEMM | Min latency |

### Test Matrix

- **Scenarios**: chat (1K/1K), reasoning (1K/8K), summarize (8K/1K)
- **Concurrency**: 1, 4, 8, 16, 32, 64, 128, 256
- **EP sizes**: EP=1 (pure TP), EP=8 (pure EP with DP attention)
- **Output variation**: ±20% (random_range_ratio=0.8)

### Auto-Adaptive Optimizations

The B200 script automatically selects optimal parameters based on scenario/concurrency:

- **Piecewise CUDA Graphs**: Enabled for high-concurrency configs
- **MOE Backend**: TRTLLM → CUTLASS (FP4+DP) → DEEPGEMM (FP8+DP)
- **Delay Batching**: For FP8 throughput at high concurrency
- **KV Cache Fraction**: 0.8 default, adjusted to 0.7 for DEEPGEMM configs
- **CUDA Graph Max Batch Size**: Dynamically scaled with DP attention

### Run Individual Configs

```bash
# FP4 throughput only
bash scripts/sa_bench_b200.sh --model-fp4 /data/DeepSeek-R1-NVFP4-v2 --configs fp4-throughput

# FP8 latency only, single EP
bash scripts/sa_bench_b200.sh --model-fp8 /data/DeepSeek-R1-FP8 --configs fp8-latency --ep-sizes "1"
```

## H200 Benchmarks (InferenceX/MAX-style)

### Quick Start

```bash
# 1. Launch Docker container
bash scripts/launch_h200_docker.sh --name H200_trtllm

# 2. Attach and run benchmarks
docker attach H200_trtllm
bash /home/kqian/ClaudeSkillsE2EPerf/scripts/sa_bench_h200.sh \
  --model /data/DeepSeek-R1-FP8 \
  --configs all
```

### Configs

| Config | Quant | MTP | MOE Backend | Target |
|--------|-------|-----|-------------|--------|
| `fp8-throughput` | FP8 | No | CUTLASS | Max throughput |
| `fp8-latency` | FP8 | MTP-3/1 | CUTLASS | Min latency |

### H200 vs B200 Differences

| Parameter | H200 | B200 |
|-----------|------|------|
| GPU Memory | 141GB | 192GB |
| Quantization | FP8 only | FP4 + FP8 |
| MOE Backend | Always CUTLASS | TRTLLM/CUTLASS/DEEPGEMM |
| KV Cache Fraction | 0.75 | 0.80 |
| CUDA Graph Max BS | Fixed 128 | Dynamic |
| Piecewise CUDA Graphs | No | Yes (high concurrency) |
| Delay Batching | No | Yes (FP8 throughput) |
| Max Concurrency | 128 | 256 |
| EP Sizes | EP=4, EP=8 | EP=1, EP=8 |
| MTP (DP mode) | `max_batch_size=CONC/TP` | `CONC/4` or `CONC/8` |

### NVIDIA Reference Performance (8×H200)

| Mode | ISL/OSL | Config | Metric |
|------|---------|--------|--------|
| Min Latency | 1K/2K | MTP-3, EP=4, batch=1 | **158 TPS/user** |
| Max Throughput | 1K/2K | EP=8, batch=128 | **11,489 TPS** (1,436/GPU) |

### Run Individual Configs

```bash
# FP8 throughput only, EP=8
bash scripts/sa_bench_h200.sh --model /data/DeepSeek-R1-FP8 --configs fp8-throughput --ep-sizes "8"

# FP8 latency only
bash scripts/sa_bench_h200.sh --model /data/DeepSeek-R1-FP8 --configs fp8-latency
```

## H20 Benchmarks (Legacy)

### Quick Start

```bash
docker run --rm -it --gpus all --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home:/home -v /raid:/raid \
  -p 8888:8888 \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2

# Inside container:
bash /home/kqian/ClaudeSkillsE2EPerf/scripts/sa_bench_h20.sh \
  --model /raid/data/models/deepseekr1/
```

### H20 Results (Baseline)

**Config:** 8x H20, TP=8, FP8, PyTorch backend, max_batch_size=8

| Metric | Value |
|--------|-------|
| Output Throughput | 246.0 tokens/s |
| Per GPU Throughput | 30.8 tokens/s/GPU |
| Avg Latency | 16.2s |

## Architecture Notes

DeepSeek R1 uses `DeepseekV3ForCausalLM` (`model_type: deepseek_v3`):
- 671B params, 256 routed experts, 8 experts/token
- MLA (Multi-head Latent Attention) with KV LoRA
- Only **PyTorch backend** supported (no TRT engine)

## File Structure

```
.
├── README.md
├── scripts/
│   ├── sa_bench_b200.sh        # B200 benchmark suite (InferenceX-style)
│   ├── sa_bench_h200.sh        # H200 benchmark suite (InferenceX-style)
│   ├── sa_bench_h20.sh         # H20 benchmark suite
│   ├── benchmark_lib.sh        # Shared utilities (server, GPU monitor, benchmark client)
│   ├── launch_b200_docker.sh   # B200 Docker launcher
│   ├── launch_h200_docker.sh   # H200 Docker launcher
│   ├── gen_dataset.py          # Dataset generator
│   ├── run_bench.sh            # Simple trtllm-bench runner
│   └── serve.sh                # Model serving script
├── utils/
│   └── bench_serving/          # benchmark_serving.py (from InferenceX)
├── configs/
│   ├── bench_config.yaml       # H20 config
│   ├── bench_config_b200.yaml  # B200 config
│   └── bench_config_h200.yaml  # H200 config
└── results/
    └── deepseek_r1_8xh20_fp8_pytorch.json
```

## Benchmark Method Comparison

| Aspect | H20 (sa_bench_h20.sh) | H200/B200 (sa_bench_h200/b200.sh) |
|--------|----------------------|-----------------------------------|
| **Engine** | `trtllm-bench` direct | `trtllm-serve` + `benchmark_serving.py` |
| **Benchmark** | Internal throughput test | OpenAI API load test (realistic) |
| **Metrics** | Throughput only | Throughput + TTFT + TPOT + E2E latency |
| **Concurrency** | Batch size control | Request-level concurrency |
| **Output Len** | Fixed | ±20% random variation |

## License

MIT
