# ClaudeSkillsE2EPerf

End-to-end performance benchmarking skills for LLM inference, generated and validated by Claude.

## DeepSeek R1 671B on TensorRT-LLM (H20)

Deploy and benchmark DeepSeek R1 671B (FP8) on NVIDIA H20 GPUs using TensorRT-LLM.

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| GPU       | 8x NVIDIA H20 (96GB each, 768GB total) |
| Model     | DeepSeek R1 671B MoE (FP8 quantized) |
| Docker    | `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2` |

## Quick Start

### 1. Launch Docker Container

```bash
docker run --rm -it \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /raid/data/models:/models \
  -v /raid/data/trtllm_workspace:/workspace \
  -p 8000:8000 \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2
```

### 2. Serve the Model (OpenAI-Compatible API)

```bash
trtllm-serve serve \
  /models/deepseekr1/ \
  --backend pytorch \
  --tp_size 8 \
  --max_batch_size 8 \
  --max_seq_len 4096 \
  --kv_cache_free_gpu_memory_fraction 0.85 \
  --trust_remote_code \
  --reasoning_parser deepseek-r1 \
  --host 0.0.0.0 \
  --port 8000
```

### 3. Test the API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseekr1",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 256
  }'
```

### 4. Run Benchmark

```bash
# Generate benchmark dataset
python3 scripts/gen_dataset.py \
  --tokenizer /models/deepseekr1/ \
  --num_requests 50 \
  --output_tokens 128

# Run throughput benchmark
bash scripts/run_bench.sh /models/deepseekr1/ 8
```

## Architecture Notes

DeepSeek R1 uses `DeepseekV3ForCausalLM` architecture (`model_type: deepseek_v3`):
- 671B parameters, 256 routed experts, 8 experts per token
- MLA (Multi-head Latent Attention) with KV LoRA
- FP8 block-wise quantization (128x128)
- 61 hidden layers, 128 attention heads

### Backend Compatibility (TRT-LLM 1.2.0rc2)

| Backend | Support | Notes |
|---------|---------|-------|
| **PyTorch** | Supported | Full support via `_torch` path. Uses DeepGEMM, FlashMLA, FP8 MLA, CUDA Graphs |
| **TensorRT Engine** | Not supported | `MODEL_MAP` only has `DeepseekV2ForCausalLM`. Need TRT-LLM >= 1.3.x |

> For TRT engine backend, upgrade to `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6` or later.

## Benchmark Results

**Config:** 8x H20, TP=8, FP8, PyTorch backend, max_batch_size=8, max_seq_len=4096

| Metric | Value |
|--------|-------|
| Request Throughput | 1.92 req/s |
| Output Throughput | **246.0 tokens/s** |
| Total Token Throughput | 278.1 tokens/s |
| Per GPU Output Throughput | 30.8 tokens/s/GPU |
| Per User Output Throughput | 9.3 tokens/s/user |
| Avg Request Latency | 16.2s |
| P50 Latency | 18.1s |
| P99 Latency | 26.0s |

*Test: 50 requests, avg input 17 tokens, output 128 tokens*

## Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--tp_size` | Tensor parallelism (number of GPUs) | 8 for 8x H20 |
| `--max_batch_size` | Max concurrent requests | 4-16 |
| `--max_seq_len` | Max total sequence length | 4096 (start small, scale up) |
| `--kv_cache_free_gpu_memory_fraction` | GPU memory for KV cache | 0.80-0.90 |
| `--reasoning_parser` | Thinking chain parser | `deepseek-r1` |

## File Structure

```
.
├── README.md
├── scripts/
│   ├── gen_dataset.py       # Generate benchmark dataset
│   ├── run_bench.sh         # One-click benchmark runner
│   └── serve.sh             # One-click model serving
└── configs/
    └── bench_config.yaml    # Benchmark configuration
```

## Troubleshooting

### "DeepseekV3ForCausalLM is not supported in TRT-LLM yet"
This happens when using `--backend tensorrt`. DeepSeek R1 (V3 arch) only supports `--backend pytorch` in TRT-LLM 1.2.0rc2.

### Low KV cache allocation
If you see `Allocated 0.14 GiB for max tokens in paged KV cache`, increase `--kv_cache_free_gpu_memory_fraction` or reduce `--max_batch_size`.

### UCX version warning
`UCP API version is incompatible: required >= 1.20, actual 1.19.0` — This is a harmless warning and does not affect performance.

## License

MIT
