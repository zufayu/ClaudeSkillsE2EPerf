# ClaudeSkillsE2EPerf

End-to-end performance benchmarking skills for LLM inference, generated and validated by Claude.

## Supported Platforms

| Platform | GPU | Memory | Framework | Quantization | Status |
|----------|-----|--------|-----------|-------------|--------|
| **B200** | 8x NVIDIA B200 | 192GB/GPU | TRT-LLM | FP4 + FP8 | Active |
| **H200** | 8x NVIDIA H200 | 141GB/GPU | TRT-LLM | FP8 only | Active |
| **H20**  | 8x NVIDIA H20  | 96GB/GPU  | TRT-LLM | FP8 only | Done |
| **MI355X** | 8x AMD MI355X | 288GB/GPU | ATOM | FP8 (block-scale) | Active |

## Test Matrix (SemiAnalysis InferenceX-aligned)

| Scenario | ISL | OSL | Use Case |
|----------|-----|-----|----------|
| chat | 1024 | 1024 | Short-in Short-out |
| reasoning | 1024 | 8192 | Short-in Long-out |
| summarize | 8192 | 1024 | Long-in Short-out |

- **Concurrency sweep**: 1, 4, 8, 16, 32, 64, 128, 256 (filterable via `--concurrency`)
- **Output variation**: ±20% (random_range_ratio=0.8)
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

### MI355X

| Config | Quant | KV Cache | MTP | Target |
|--------|-------|----------|-----|--------|
| `fp8-throughput` | FP8 (block-scale e4m3) | FP8 | No | Max throughput |
| `fp8-latency` | FP8 | FP8 | MTP-3 | Min latency |
| `bf16-throughput` | BF16 | FP8 | No | Max throughput |
| `bf16-latency` | BF16 | FP8 | MTP-3 | Min latency |

Framework: ATOM (lightweight vLLM from ROCm). TP=8, EP=1. Server: `atom.entrypoints.openai_server`. Benchmark client: `atom.benchmarks.benchmark_serving`.

### Platform Comparison

| Parameter | B200 | H200 | H20 | MI355X |
|-----------|------|------|-----|--------|
| GPU Memory | 192GB | 141GB | 96GB | 288GB |
| Framework | TRT-LLM | TRT-LLM | TRT-LLM | ATOM |
| Quantization | FP4 + FP8 | FP8 | FP8 | FP8 + BF16 |
| KV Cache | BF16 | BF16 | BF16 | FP8 |
| Max Concurrency | 256 | 128 | 64 | 256 |
| EP Sizes | 1, 8 | 4, 8 | 4 | 1 |
| Benchmark Method | `benchmark_serving.py` | `benchmark_serving.py` | `trtllm-bench` | `atom.benchmarks.benchmark_serving` |

## Quick Start

### Single Data Point

```bash
# B200: Reproduce SA FP8 c=256 EP=8 chat
bash scripts/sa_bench_b200.sh \
  --model-fp8 /data/DeepSeek-R1-FP8 \
  --configs fp8-throughput \
  --scenario chat --concurrency 256 --ep-sizes 8

# MI355X: Reproduce ATOM FP8 c=128 chat
bash scripts/sa_bench_mi355x.sh \
  --model-fp8 /data/DeepSeek-R1-0528 \
  --configs fp8-throughput \
  --scenario chat --concurrency 128
```

### Filter Flags

| Flag | Values | Default |
|------|--------|---------|
| `--scenario` | `chat`, `reasoning`, `summarize`, `all` | `all` |
| `--concurrency` | Space-separated, e.g. `"128"` or `"4 128 256"` | `1 4 8 16 32 64 128 256` |
| `--ep-sizes` | Space-separated, e.g. `"8"` (B200/H200 only) | `"1 8"` |

---

## Deployment Guide

Step-by-step for running benchmarks on a fresh machine.

### Prerequisites

- Bare-metal server with GPUs (B200/H200/H20 or MI355X)
- Ubuntu 22.04 or 24.04
- NVIDIA Driver 570+ (NVIDIA) or ROCm 6.x+ (AMD)
- Root or sudo access, internet access
- High-speed storage (NVMe) — model weights are 350-650GB

### Step 1: Install Docker + GPU Toolkit

**NVIDIA:**
```bash
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

**AMD (MI355X):**
```bash
# ROCm driver should already be installed; verify with rocm-smi
# Docker with ROCm uses --device=/dev/kfd --device=/dev/dri --group-add video
```

### Step 2: Pull Container Image

```bash
# B200 / H200
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4

# H20
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2

# MI355X (ATOM)
docker pull rocm/atom:rocm7.1.1-ubuntu24.04-pytorch2.9-atom0.1.1-MI350x
```

### Step 3: Download Model Weights

```bash
pip install huggingface_hub

# FP4 (~350GB, B200 only)
huggingface-cli download nvidia/DeepSeek-R1-0528-NVFP4-v2 \
  --local-dir /data/models/DeepSeek-R1-NVFP4-v2

# BF16+FP8 (~650GB, all platforms)
huggingface-cli download deepseek-ai/DeepSeek-R1-0528 \
  --local-dir /data/DeepSeek-R1-0528
```

### Step 4: Clone & Launch

```bash
git clone https://github.com/zufayu/ClaudeSkillsE2EPerf.git ~/ClaudeSkillsE2EPerf
```

**B200:**
```bash
bash ~/ClaudeSkillsE2EPerf/scripts/launch_b200_docker.sh --name B200_bench
docker attach B200_bench
```

**MI355X:**
```bash
bash ~/ClaudeSkillsE2EPerf/scripts/launch_mi355x_docker.sh --name MI355X_bench
docker attach MI355X_bench
```

### Step 5: Install Dependencies (Inside Container)

```bash
# NVIDIA containers
pip install aiohttp tqdm

# ATOM container: dependencies pre-installed
```

### Step 6: Run Benchmarks

**B200:**
```bash
# Full suite (~24-48 hours)
bash ~/ClaudeSkillsE2EPerf/scripts/sa_bench_b200.sh \
  --model-fp4 /data/models/DeepSeek-R1-NVFP4-v2 \
  --model-fp8 /data/models/DeepSeek-R1-FP8 \
  --configs all

# Quick validation (~1-2 hours)
bash ~/ClaudeSkillsE2EPerf/scripts/sa_bench_b200.sh \
  --model-fp4 /data/models/DeepSeek-R1-NVFP4-v2 \
  --configs fp4-throughput --ep-sizes "1"
```

**MI355X:**
```bash
# FP8 throughput full sweep
bash ~/ClaudeSkillsE2EPerf/scripts/sa_bench_mi355x.sh \
  --model-fp8 /data/DeepSeek-R1-0528 \
  --configs fp8-throughput

# Single point (chat c=128)
bash ~/ClaudeSkillsE2EPerf/scripts/sa_bench_mi355x.sh \
  --model-fp8 /data/DeepSeek-R1-0528 \
  --configs fp8-throughput --scenario chat --concurrency 128
```

> **Tip**: Use `tmux` for long benchmarks: `tmux new -s bench`, then Ctrl+B D to detach.

### Step 7: View Results

```bash
cat ./results_b200/summary.md     # B200
cat ./results_mi355x/summary.md   # MI355X

# Individual JSON results
python3 -c "import json; print(json.dumps(json.load(open('results_mi355x/result_fp8_throughput_chat_c128.json')), indent=2))"

# GPU monitoring data
head -20 ./results_mi355x/gpu_fp8_throughput_chat_c128.csv

# Server logs (debugging)
tail -50 ./results_mi355x/server_fp8_throughput_chat_c128.log
```

### Summary Table Columns

| Column | Meaning |
|--------|---------|
| Output TPS | Total output tokens per second |
| Out TPS/GPU | Output TPS / GPU count |
| Interactivity | Output TPS / Concurrency |
| TTFT p50 | Time To First Token (median) |
| TPOT p50 | Time Per Output Token (median) |
| E2E p50 | End-to-end latency (median) |

### Step 8: Export & Dashboard

```bash
# Pack results
tar czf ~/bench_results_$(date +%Y%m%d).tar.gz ./results_mi355x/

# Import into unified dashboard
python3 scripts/import_results.py \
  --results-dir ./results_mi355x \
  --platform "8×MI355X" --framework "ATOM 0.1.1" --quantization FP8

# Fetch competitor data & regenerate dashboard
python3 scripts/fetch_competitors.py
python3 scripts/generate_dashboard.py

# View locally
cd ~/ClaudeSkillsE2EPerf/docs && python3 -m http.server 8899
```

### Environment Comparison (--env-tag)

同一平台、相同配置但不同环境（Docker 版本、分支、运行日期等）的对比：

```bash
# 环境 A: 旧 Docker / 旧分支
python3 scripts/import_results.py \
  --results-dir ./results_b200_old \
  --platform "8×B200" --framework "TRT-LLM" --quantization FP8 \
  --env-tag "docker-v1"

# 环境 B: 新 Docker / 新分支
python3 scripts/import_results.py \
  --results-dir ./results_b200_new \
  --platform "8×B200" --framework "TRT-LLM" --quantization FP8 \
  --env-tag "docker-v2"

# 重新生成看板
python3 scripts/generate_dashboard.py
```

`--env-tag` 会在 series_key 中追加标记，生成独立的对比系列：
- `8×B200 DeepSeek-R1-0528 FP8 (TRT-LLM) [docker-v1]`
- `8×B200 DeepSeek-R1-0528 FP8 (TRT-LLM) [docker-v2]`

看板的 Apple-to-Apple 对比表、柱状图、Delta 列自动生效。

`--env-tag` 可以是任何字符串，例如：
| 场景 | 示例 |
|------|------|
| Docker 版本对比 | `--env-tag "trtllm-1.2.0rc4"` vs `--env-tag "trtllm-1.3.0"` |
| 分支对比 | `--env-tag "main"` vs `--env-tag "fix-kv-cache"` |
| 日期对比 | `--env-tag "0317"` vs `--env-tag "0320"` |
| Commit 对比 | `--env-tag "abc1234"` vs `--env-tag "def5678"` |

> 不加 `--env-tag` 时，同平台同配置的多次导入会合并为同一个 series（去重取最新）。

### Time Estimates

| Platform | Scope | Approximate Time |
|----------|-------|-----------------|
| B200 | Full suite (FP4+FP8, all configs) | 24-48 hours |
| B200 | Single config, single EP | 1-2 hours |
| MI355X | Full FP8 throughput sweep | 4-8 hours |
| MI355X | Single test point | 5-15 minutes |
| H200 | Full suite | 12-24 hours |

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named 'aiohttp'` | `pip install aiohttp tqdm` |
| Server startup timeout (>15 min) | Normal for 671B model; check `server_*.log` |
| `CUDA out of memory` | Lower concurrency; check GPU memory utilization |
| `cudagraph capture sizes must be less than max_num_seqs` | ATOM: ensure `--max-num-seqs` >= max cudagraph size (default 512) |
| `Server process died` | Check `server_*.log` last 50 lines |
| Docker can't see GPUs | NVIDIA: reinstall container toolkit; AMD: check `--device=/dev/kfd --device=/dev/dri` |

---

## Unified Comparison Dashboard

Interactive dashboard for cross-platform benchmark comparison at https://zufayu.github.io/ClaudeSkillsE2EPerf/

### Data Pipeline

```
Benchmark scripts          Competitor CI
(sa_bench_*.sh)            (ATOM gh-pages)
       |                         |
       v                         v
  results_*/              fetch_competitors.py
  result_*.json                  |
       |                         v
       v                    runs/*.json
  import_results.py ----------->|
                                v
                       generate_dashboard.py
                                |
                                v
                           docs/data.js
                                |
                                v
                         docs/index.html
```

### Dashboard Features

- **Performance** tab: bar chart with metric toggle (Output TPS / Total TPS / TPOT / TTFT) + Apple-to-Apple comparison table with Delta%
- **Throughput vs Latency** scatter plots (TPOT vs TPS, Interactive vs TPS/GPU, TTFT vs TPS)
- **Trends** tab: line charts by concurrency with series selector
- **Data & Trace** tab: full data table with expandable detail rows + CSV export
- **Apple-to-Apple** toggle: filter to only matching configs across platforms
- Heat coloring (TPOT/TTFT), in-cell data bars, expandable detail rows
- Filters: Platform, ISL/OSL scenario, Concurrency

### Dashboard Scripts

| Script | Purpose |
|--------|---------|
| `scripts/import_results.py` | Convert `results_*/result_*.json` → unified `runs/*.json` |
| `scripts/fetch_competitors.py` | Fetch ATOM data from GitHub Pages → `runs/atom-*.json` |
| `scripts/generate_dashboard.py` | Merge all `runs/*.json` → `docs/data.js` |

### Unified Run Format

```json
{
  "run_id": "8xmi355x-fp8-20260317",
  "platform": "8×MI355X",
  "framework": "ATOM 0.1.1",
  "model": "DeepSeek-R1-0528",
  "quantization": "FP8",
  "gpu_count": 8,
  "source": "manual",
  "date": "2026-03-17",
  "results": [
    {"isl": 1024, "osl": 1024, "conc": 128, "output_tps": 4320.8, "tpot_p50": 28.88, "ttft_p50": 97.3}
  ]
}
```

## Architecture Notes

DeepSeek R1 uses `DeepseekV3ForCausalLM` (`model_type: deepseek_v3`):
- 671B params, 256 routed experts, 8 experts/token
- MLA (Multi-head Latent Attention) with KV LoRA
- FP8 block-scale quantization: `quant_method=fp8, fmt=e4m3, block_size=[128,128]`

## File Structure

```
.
├── README.md                       # This file
├── docs/                           # Unified comparison dashboard (GitHub Pages)
│   ├── index.html
│   └── data.js
├── runs/                           # Unified benchmark run data
├── scripts/
│   ├── sa_bench_b200.sh            # B200 benchmark suite
│   ├── sa_bench_h200.sh            # H200 benchmark suite
│   ├── sa_bench_h20.sh             # H20 benchmark suite
│   ├── sa_bench_mi355x.sh          # MI355X benchmark suite (ATOM)
│   ├── benchmark_lib.sh            # Shared utilities
│   ├── launch_b200_docker.sh       # B200 Docker launcher
│   ├── launch_h200_docker.sh       # H200 Docker launcher
│   ├── launch_mi355x_docker.sh     # MI355X Docker launcher
│   ├── import_results.py           # results_*/ → runs/*.json
│   ├── fetch_competitors.py        # Fetch competitor data
│   ├── generate_dashboard.py       # runs/ → docs/data.js
│   └── setup_claude_memory.sh      # Claude Code memory persistence
├── configs/
│   ├── bench_config.yaml           # H20 config
│   ├── bench_config_b200.yaml      # B200 config
│   ├── bench_config_h200.yaml      # H200 config
│   └── bench_config_mi355x.yaml    # MI355X config
├── utils/
│   └── bench_serving/              # benchmark_serving.py (InferenceX)
└── results/
    └── deepseek_r1_8xh20_fp8_pytorch.json
```

## Claude Code 记忆持久化

在 Docker 容器内使用 Claude Code 时，记忆默认存在容器内部，容器重建后会丢失。本脚本将记忆目录 symlink 到宿主机挂载的 home 目录，实现持久化。

### 首次设置（容器内执行）

```bash
bash ~/ClaudeSkillsE2EPerf/scripts/setup_claude_memory.sh
```

脚本会自动检测挂载的 home 目录（如 `/home/zufayu`），将记忆存储到 `~/.claude_memory/`。

### 容器重建后恢复

```bash
bash ~/ClaudeSkillsE2EPerf/scripts/setup_claude_memory.sh
```

同一条命令，会检测到已有记忆文件并重新建立 symlink，之前的记忆全部恢复。

### 手动指定挂载路径

```bash
bash ~/ClaudeSkillsE2EPerf/scripts/setup_claude_memory.sh /home/your_username
```

### 原理

```
容器内 Claude 写入:
  /root/.claude/projects/-/memory/  (symlink)
         ↓
  /home/zufayu/.claude_memory/      (宿主机磁盘)
         ↓
  容器重建后文件仍在，重新 symlink 即可恢复
```

## License

MIT
