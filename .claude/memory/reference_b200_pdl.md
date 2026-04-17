---
name: B200 PDL (Programmatic Dependent Launch)
description: NVIDIA PDL mechanism enabling same-stream kernel overlap by overlapping tail of kernel A with preamble of kernel B. Hopper+ feature. Key to understanding B200 overlap savings.
type: reference
---

# PDL — Programmatic Dependent Launch

## What It Is

CUDA feature (sm_90+ / Hopper onwards) that breaks same-stream serial execution constraint. Allows secondary kernel's preamble to overlap with primary kernel's tail on the same stream.

## Mechanism

```
Traditional same-stream:
  Kernel A:  |==========[====tail====]|
  Kernel B:                             |---launch---[=========]|

PDL same-stream:
  Kernel A:  |==========[====tail====]|
  Kernel B:            |---preamble---|--wait--|[====dependent====]|
                       ↑ B's CTAs start on SMs freed by A's finishing CTAs
```

## Key APIs

- **Primary kernel**: `cudaTriggerProgrammaticLaunchCompletion()` — "my CTA's critical output is done, dependents can start scheduling"
- **Secondary kernel**: `cudaGridDependencySynchronize()` — "do preamble first (zero buffers, load constants, init LDS), then wait for primary to fully complete"
- **Host launch attribute**: `cudaLaunchAttributeProgrammaticStreamSerialization` on secondary kernel
- **PTX**: `griddepcontrol.launch_dependents` / `griddepcontrol.wait`

## What It Saves

| Overhead eliminated | Mechanism | Per-boundary |
|---|---|---|
| Kernel launch latency | B dispatch overlaps A tail | ~3-5μs |
| Tail effect (wave quantization) | A's last-wave idle SMs run B's CTAs | variable |
| Preamble overlap | B's init runs on freed SMs while A finishes | ~1-3μs |

Typical saving: ~2μs per kernel boundary.

## Impact on DeepSeek-V3 B200 Analysis

- 29 kernels/layer → 28 boundaries × ~2μs ≈ ~27μs PDL savings (out of 66μs total overlap)
- Remaining ~39μs comes from dual-stream cross-stream parallelism
- PDL savings scale with kernel count — DeepSeek-V3's MLA decomposition creates many small kernels (10 for attention alone), amplifying PDL benefit
- Standard dense Transformer (~10 kernels): PDL saves only ~18μs
- MI355X/ROCm has NO equivalent mechanism — `hipGridDependencySynchronize` does not exist

## cuBLAS / CUTLASS Integration

- cuBLAS: PDL supported in some kernels for sm_90+ (CUDA 13.2+)
- CUTLASS: `launch_with_pdl` parameter, `grid_dependency_control.h` header
- Used in production inference (e.g., nvjet GEMM chains)

## References

- CUDA Programming Guide Section 4.5
- PTX ISA Section 9.7.13.13 (`griddepcontrol`)
- CUTLASS `include/cutlass/arch/grid_dependency_control.h`
