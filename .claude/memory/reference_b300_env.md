---
name: B300 environment
description: B300 machine access, GPU specs, container setup, model paths, NCU
type: reference
originSessionId: 49e0962a-ee55-4b7c-9023-e07ec969a46b
---
## Machine
- Hostname: `smcb300-ccs-aus-j13-21`
- User: `zufayu`
- Home: `/home/zufa`
- Disk: 6.9TB NVMe, ~5TB free

## GPUs
- 8× NVIDIA B300 SXM6 AC, **275GB** VRAM each
- Driver: 595.45.04
- CUDA: 13.1

## GitHub Actions Runner
- Name: `b300-runner`
- Labels: `[self-hosted, Linux, X64, b300]`
- Install path: `/home/zufa/actions-runner/`
- Start: `nohup ./run.sh > runner.log 2>&1 &`

## Container
- Name: `zufa_sglang`
- Image: `lmsysorg/sglang:v0.5.9-cu130`
- SGLang: 0.5.9
- Mounts: `/home:/home`, `/models:/models`
- Workdir: `/home/zufa`
- gdb: installed
- Created: 2026-04-17
- **注意**: B300 直接在本机 docker exec（不需要 ssh 跳板），和 B200 不同

## Model
- Path: `/models/DeepSeek-R1-0528-NVFP4-v2` (385GB)
- 从 HuggingFace `nvidia/DeepSeek-R1-0528-NVFP4-v2` 下载

## NCU (Nsight Compute)
- Host path: `/opt/nvidia/nsight-compute/2025.4.1/ncu`
- Version: 2025.4.1 (比 B200 的 2025.3.1 更新)
- 也在 `/usr/local/cuda-13.1/bin/ncu`
- **不在容器内** — 需要从 host 跑或 mount 进去

## 与 B200 的差异
| | B200 | B300 |
|---|---|---|
| VRAM | 183GB | **275GB** |
| Driver | 580 | **595** |
| NCU | 2025.3.1 | **2025.4.1** |
| 访问方式 | `ssh hungry-hippo-fin-03-2` | **直接本机** |
| 模型路径 | `/SFS-aGqda6ct/models/...` | `/models/...` |
| Workflow 模式 | `ssh $NODE "docker exec ..."` | `docker exec ...` |

## Existing Containers (as of 2026-04-17)
- `zufa_sglang` — lmsysorg/sglang:v0.5.9-cu130 (ours)
- `veergopu_2.8_tables` — nvcr.io/nvidia/pytorch:25.10-py3 (other user)
- `veergopu_2.8_tables_jax` — nvcr.io/nvidia/jax:25.10-py3 (other user)

## Access
- Runner on local machine — workflows run directly, no SSH needed
- User accesses via VS Code tunnel
