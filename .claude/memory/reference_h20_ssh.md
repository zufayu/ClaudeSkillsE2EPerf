---
name: H20 SSH access
description: How to access the H20 host and run GPU workloads — SSH with sshpass, sudo for docker
type: reference
---

H20 host: `dl-server-h20`, user `kqian`, password `123123`
- SSH: `sshpass -p 123123 ssh kqian@dl-server-h20 "command"`
- Docker needs sudo: `echo 123123 | sudo -S docker exec ...`
- This container (82611ca61bfa) is a docker on H20 but has NO GPU access (CUDA 13.0 toolkit vs host driver 570/CUDA 12.8 mismatch)
- For GPU workloads, either SSH to host and use existing containers, or run inside `zufa_ncu_trt` (rc4) / `zufa_ncu_sglang` containers
- 8× NVIDIA H20 GPUs, 98GB each
- Profiling model: Qwen3-32B at `/raid/data/models/Qwen3-32B` (fits single H20 GPU)
- DeepSeek-R1 at `/raid/data/models/deepseekr1` (needs TP≥8)
