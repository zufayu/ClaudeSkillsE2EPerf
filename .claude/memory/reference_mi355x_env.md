---
name: MI355X environment
description: MI355X GPU machine access, container, model paths, and runner setup
type: reference
originSessionId: 49e0962a-ee55-4b7c-9023-e07ec969a46b
---
## MI355X Machine
- VS Code tunnel: `my-gpu-41`
- Machine name: `mi355-gpu-41`
- GPUs: 8× AMD Instinct MI355X (gfx950)
- No sudo/root access
- Uses **podman** (not docker)

## Container
- Name: `zufa_atom`
- Image: `rocm/atom-dev:latest` (nightly_202604151530, upgraded 2026-04-16)
- ROCm: 7.2.2, PyTorch: 2.10.0+rocm7.2.2, Python: 3.12.3
- ATOM: 0.1.3.dev71+gb9d5f8fa3 (at /app/ATOM)
- vLLM: not installed (native image, not OOT variant)
- Base: rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12
- Previous image: `rocm/atom:rocm7.1.1-ubuntu24.04-pytorch2.9-atom0.1.1-MI350x` (Jan 12)

## Paths
- Repo: `/shared/amdgpu/home/zufa_yu_qle/ClaudeSkillsE2EPerf`
- Models: `/shared/data/amd_int/models/`
  - BF16: `DeepSeek-R1-0528`
  - MXFP4 (no MTP): `DeepSeek-R1-0528-MXFP4`
  - MXFP4+MTP: `DeepSeek-R1-0528-MTP-MoE-MXFP4-Attn-PTPC-FP8`

## GitHub Actions Runner
- Runner name: `mi355x-runner`, labels: `self-hosted,mi355x`
- Installed at: `~/actions-runner` (no systemd service, run via `nohup ./run.sh &`)
- Workflow: `.github/workflows/test_mi355x.yml`
- Commands use `podman exec zufa_atom ...` (not docker)
