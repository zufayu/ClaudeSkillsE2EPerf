---
name: Directory and result naming convention
description: Agreed naming rules for benchmark result directories, env tags, and result JSON version fields
type: project
---

## Result Directory Naming

Pattern: `{platform}_{model}_{quant}_{mtp}_{ep}_{tp}_{env}[_{mode}]`

- **env**: primary software version tag, **always required** (never omit — avoids retroactive renaming when a second docker version is added later)
  - MI355X: ROCm version → `rocm721`, `rocm711`
  - B200: TRT-LLM version → `post3`, `post2`
- **mode**: optional, default=bench → `profiling`
- Never use personal container names (e.g., `zufa_atom2`) in directory names
- **ncu results** go under `_profiling/ncu/` subdirectory (not separate `_ncu` dir)
- **Concurrency NOT in directory name** — different concurrency data (c4, c64, etc.) coexist in the same directory. Filenames distinguish concurrency (e.g., `result_*_c4.json`, `decode_breakdown_c4.xlsx`). One-level dir, not `_c4/` subdirs.

Examples:
```
mi355x_dsr_mxfp4_mtp0_ep1_tp8_rocm721/
mi355x_dsr_mxfp4_mtp0_ep1_tp8_rocm721_profiling/
mi355x_dsr_mxfp4_mtp0_ep1_tp8_rocm721_profiling/ncu/    ← ncu results here
b200_dsr_fp4_mtp0_ep8_tp8_post2/
b200_dsr_fp8_mtp3_ep1_tp8_post3/
```

## Result JSON Version Fields

Bench scripts should inject into result JSON:
- `container_image`: original docker/podman image name (from `docker inspect --format '{{.Config.Image}}'`)
- `rocm_version`: from `cat /opt/rocm/.info/version`
- `trtllm_version`: from `pip show tensorrt-llm | grep Version`
- `atom_version`: from `pip show atom-svc | grep Version`

**Why:** Different docker versions = different kernel optimizations, must coexist for comparison. Directory env tag for quick distinction, JSON for full traceability.

## Dashboard Column Split (agreed, not yet implemented)

| Column | Content | Example |
|--------|---------|---------|
| Model | model + quant | `DeepSeek-R1 NVFP4` |
| Platform | GPU | `8×B200` |
| Config | EP/TP/MTP | `EP1 TP8 MTP0` |
| Env | software version | `TRT-LLM rc6.post3` / `ROCm 7.2.1` |
| Mode | bench/profiling/ci | `bench` |
