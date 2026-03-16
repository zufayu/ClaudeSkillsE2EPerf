# Deployment Guide: DeepSeek R1 Benchmark on Fresh GPU Machines

Step-by-step guide for running the ClaudeSkillsE2EPerf benchmark suite on a brand-new machine with nothing pre-installed.

Covers B200, H200, and H20. Pick the section that matches your hardware.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Install Docker + NVIDIA Container Toolkit](#step-1-install-docker--nvidia-container-toolkit)
- [Step 2: Pull the TRT-LLM Container Image](#step-2-pull-the-trt-llm-container-image)
- [Step 3: Download Model Weights](#step-3-download-model-weights)
- [Step 4: Clone This Repository](#step-4-clone-this-repository)
- [Step 5: Launch Docker Container](#step-5-launch-docker-container)
- [Step 6: Install Benchmark Dependencies (Inside Container)](#step-6-install-benchmark-dependencies-inside-container)
- [Step 7: Run Benchmark](#step-7-run-benchmark)
- [Step 8: View Results](#step-8-view-results)
- [Step 9: Export Results](#step-9-export-results)
- [Step 10: Generate Comparison Dashboard](#step-10-generate-comparison-dashboard)
- [Time Estimates](#time-estimates)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- A bare-metal server with NVIDIA GPUs (B200, H200, or H20)
- Ubuntu 22.04 or 24.04
- NVIDIA Driver 570+ already installed (verify with `nvidia-smi`)
- Root or sudo access
- Internet access (to pull container images and download model weights)
- High-speed storage recommended (NVMe RAID) — model weights are 350GB-650GB

---

## Step 1: Install Docker + NVIDIA Container Toolkit

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker
sudo usermod -aG docker $USER  # optional: run docker without sudo

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

You should see all 8 GPUs listed. If not, check that the NVIDIA driver is installed and `nvidia-smi` works on the host.

---

## Step 2: Pull the TRT-LLM Container Image

```bash
# B200 / H200: use 1.2.0rc4
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4

# H20: use 1.2.0rc2
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2
```

> If you get authentication errors, log in to NGC first:
> ```bash
> docker login nvcr.io
> # Username: $oauthtoken
> # Password: <your NGC API Key from https://ngc.nvidia.com/setup>
> ```

---

## Step 3: Download Model Weights

```bash
# Install huggingface-cli on the host
pip install huggingface_hub

# --- Option A: FP4 model (~350GB, B200 only) ---
huggingface-cli download nvidia/DeepSeek-R1-0528-NVFP4-v2 \
  --local-dir /data/models/DeepSeek-R1-NVFP4-v2

# --- Option B: FP8 model (~650GB, all platforms) ---
huggingface-cli download deepseek-ai/DeepSeek-R1 \
  --local-dir /data/models/DeepSeek-R1-FP8
```

Tips:
- FP4 is smaller and faster to download; sufficient for B200 benchmarking
- FP8 is needed for H200 and H20 (they don't support FP4)
- For B200 with `--configs all`, you need both FP4 and FP8
- Use `--local-dir` on a fast filesystem (NVMe), not NFS

---

## Step 4: Clone This Repository

```bash
git clone https://github.com/zufayu/ClaudeSkillsE2EPerf.git ~/ClaudeSkillsE2EPerf
```

---

## Step 5: Launch Docker Container

### B200

```bash
bash ~/ClaudeSkillsE2EPerf/scripts/launch_b200_docker.sh --name B200_bench
docker attach B200_bench
```

### H200

```bash
bash ~/ClaudeSkillsE2EPerf/scripts/launch_h200_docker.sh --name H200_bench
docker attach H200_bench
```

### H20

```bash
docker run -itd --rm \
  --gpus all \
  --ulimit memlock=-1:-1 \
  --shm-size 8G \
  --net host \
  -v /home:/home \
  -v /data:/data \
  --name H20_bench \
  --entrypoint /bin/bash \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2
docker attach H20_bench
```

After attaching, press **Enter** to get a shell prompt. Verify GPUs:

```bash
nvidia-smi   # should show all 8 GPUs
```

---

## Step 6: Install Benchmark Dependencies (Inside Container)

The TRT-LLM container has most dependencies pre-installed. You only need to add:

```bash
pip install aiohttp tqdm
```

These are required by `benchmark_serving.py` (the load-testing client).

---

## Step 7: Run Benchmark

> Replace `~/ClaudeSkillsE2EPerf` with the actual path if different (the `/home` mount makes it visible inside the container).

### B200

```bash
# Full suite (FP4+FP8, all configs) — ~24-48 hours
bash ~/ClaudeSkillsE2EPerf/scripts/sa_bench_b200.sh \
  --model-fp4 /data/models/DeepSeek-R1-NVFP4-v2 \
  --model-fp8 /data/models/DeepSeek-R1-FP8 \
  --configs all

# Quick validation (~1-2 hours): FP4 throughput, EP=1 only
bash ~/ClaudeSkillsE2EPerf/scripts/sa_bench_b200.sh \
  --model-fp4 /data/models/DeepSeek-R1-NVFP4-v2 \
  --configs fp4-throughput \
  --ep-sizes "1"

# FP4 only (both throughput and latency)
bash ~/ClaudeSkillsE2EPerf/scripts/sa_bench_b200.sh \
  --model-fp4 /data/models/DeepSeek-R1-NVFP4-v2 \
  --configs all

# FP8 latency only
bash ~/ClaudeSkillsE2EPerf/scripts/sa_bench_b200.sh \
  --model-fp8 /data/models/DeepSeek-R1-FP8 \
  --configs fp8-latency
```

### H200

```bash
# Full suite — ~12-24 hours
bash ~/ClaudeSkillsE2EPerf/scripts/sa_bench_h200.sh \
  --model /data/models/DeepSeek-R1-FP8 \
  --configs all

# Quick validation (~1-2 hours): throughput only, EP=8
bash ~/ClaudeSkillsE2EPerf/scripts/sa_bench_h200.sh \
  --model /data/models/DeepSeek-R1-FP8 \
  --configs fp8-throughput \
  --ep-sizes "8"
```

### H20

```bash
# Full suite (3 configs × 2 scenarios × 6 concurrency levels)
bash ~/ClaudeSkillsE2EPerf/scripts/sa_bench_h20.sh \
  --model /data/models/DeepSeek-R1-FP8

# Single config
bash ~/ClaudeSkillsE2EPerf/scripts/sa_bench_h20.sh \
  --model /data/models/DeepSeek-R1-FP8 \
  --configs trt-throughput
```

> **Tip**: Use `screen` or `tmux` before running long benchmarks so you can detach without killing the process:
> ```bash
> apt-get update && apt-get install -y tmux
> tmux new -s bench
> # run benchmark...
> # Ctrl+B, D to detach
> # tmux attach -t bench to reattach
> ```

---

## Step 8: View Results

After completion, the script prints a summary table automatically. You can also view it manually:

```bash
# Summary table
cat ./results_b200/summary.md    # B200
cat ./results_h200/summary.md    # H200
cat ./results_sa_bench/summary.md  # H20

# Individual test point results (JSON)
ls ./results_b200/result_*.json

# Example: view one result
python3 -c "import json; print(json.dumps(json.load(open('results_b200/result_fp4_throughput_chat_ep1_c64.json')), indent=2))"

# GPU monitoring data (power, temp, utilization)
head -20 ./results_b200/gpu_fp4_throughput_chat_ep1_c64.csv

# Server logs (for debugging startup or OOM issues)
tail -50 ./results_b200/server_fp4_throughput_chat_ep1_c64.log
```

### Understanding the Summary Table

| Column | Meaning |
|--------|---------|
| Config | throughput or latency |
| Quant | fp4 or fp8 |
| Scenario | chat (1K/1K), reasoning (1K/8K), summarize (8K/1K) |
| EP | Expert Parallelism size |
| CONC | Concurrent request count |
| DP | DP Attention enabled (Y/N) |
| Req/s | Request throughput |
| Output TPS | Total output tokens per second |
| Per-GPU TPS | Output TPS / 8 |
| TTFT p50 | Time To First Token (median) |
| TPOT p50 | Time Per Output Token (median) |
| E2E p50 | End-to-end latency (median) |

---

## Step 9: Export Results

```bash
# Pack everything into a tarball
tar czf ~/b200_bench_results_$(date +%Y%m%d).tar.gz ./results_b200/

# Results are already visible on the host via the /home mount
# No need to docker cp
```

To detach from the container without stopping it: press **Ctrl+P** then **Ctrl+Q**.

To stop and remove the container: `docker stop B200_bench`

---

## Step 10: Generate Comparison Dashboard

After benchmarks finish, import results into the unified dashboard to compare against competitors (e.g., ATOM MI355X).

### Import Your Results

```bash
# B200 example
python3 ~/ClaudeSkillsE2EPerf/scripts/import_results.py \
  --results-dir ./results_b200 \
  --platform "8×B200" \
  --framework "TRT-LLM 1.2.0rc4" \
  --quantization NVFP4

# H200 example
python3 ~/ClaudeSkillsE2EPerf/scripts/import_results.py \
  --results-dir ./results_h200 \
  --platform "8×H200" \
  --framework "TRT-LLM 1.2.0rc4" \
  --quantization FP8

# H20 example
python3 ~/ClaudeSkillsE2EPerf/scripts/import_results.py \
  --results-dir ./results_sa_bench \
  --platform "8×H20" \
  --framework "TRT-LLM 1.2.0rc2" \
  --quantization FP8
```

This creates a unified JSON file in `runs/` (e.g., `runs/8xb200-nvfp4-20260316.json`).

### Fetch Competitor Data

```bash
python3 ~/ClaudeSkillsE2EPerf/scripts/fetch_competitors.py
```

This fetches the latest ATOM (ROCm MI355X) benchmark data from their public GitHub Pages dashboard and saves it to `runs/atom-*.json`.

### Generate Dashboard

```bash
python3 ~/ClaudeSkillsE2EPerf/scripts/generate_dashboard.py
```

This merges all `runs/*.json` files into `dashboard/data.js`.

### View Dashboard

```bash
cd ~/ClaudeSkillsE2EPerf/dashboard
python3 -m http.server 8899
# Open http://localhost:8899 in browser
```

Or deploy to GitHub Pages for team access.

### Refresh Data After New Runs

```bash
# Re-import (overwrites previous run for same platform/date)
python3 scripts/import_results.py --results-dir ./results_b200 \
  --platform "8×B200" --framework "TRT-LLM 1.2.0rc4" --quantization NVFP4

# Refresh competitor data
python3 scripts/fetch_competitors.py

# Regenerate dashboard
python3 scripts/generate_dashboard.py
```

---

## Time Estimates

| Platform | Scope | Approximate Time |
|----------|-------|-----------------|
| B200 | Full suite (all configs, FP4+FP8) | 24-48 hours |
| B200 | Single config, single EP | 1-2 hours |
| B200 | Single test point (1 concurrency) | 5-15 minutes |
| H200 | Full suite (all configs) | 12-24 hours |
| H200 | Single config, single EP | 1-2 hours |
| H20 | Full suite (3 configs) | 8-16 hours |

Most time is spent on server startup (~5-10 min per restart for 671B model). The benchmark itself runs in 1-5 minutes per concurrency point.

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `No module named 'aiohttp'` | Missing pip package | `pip install aiohttp tqdm` |
| `Cannot find benchmark_serving.py` | Script can't locate the file | Add `--bench-serving-dir ~/ClaudeSkillsE2EPerf/utils` |
| Server startup timeout (>15 min) | Normal for 671B model first load | Wait; check `server_*.log` for progress |
| `CUDA out of memory` during server start | Model too large for config | Lower concurrency; use `--ep-sizes "1"` only; check KV cache fraction |
| `Server process died` | OOM or config incompatibility | Check `server_*.log` last 50 lines for the real error |
| `trtllm-serve: command not found` | Wrong container image | Ensure you're using `release:1.2.0rc4` (B200/H200) or `1.2.0rc2` (H20) |
| Docker can't see GPUs | Missing NVIDIA Container Toolkit | Re-run Step 1, verify with `docker run --gpus all ... nvidia-smi` |
| `permission denied` on scripts | Not executable | `chmod +x ~/ClaudeSkillsE2EPerf/scripts/*.sh` |
| `UCP API version incompatible` warning | UCX version mismatch | Harmless warning, ignore it |
| `DeepseekV3ForCausalLM not supported` | Using `--backend tensorrt` | DeepSeek R1 only supports `--backend pytorch` |
| Only have FP4 model, no FP8 | Want to run on H200/H20 | H200/H20 require FP8; download the FP8 model |
| Benchmark runs but 0 TPS | Server crashed mid-benchmark | Check `server_*.log`; reduce concurrency |



> For platform-specific technical details (configs, EP sizes, adaptive optimizations), see the [Platform Comparison table in README](../README.md#platform-comparison).
