---
name: B200 PDL (Programmatic Dependent Launch)
description: NVIDIA PDL mechanism enabling same-stream kernel overlap by overlapping tail of kernel A with preamble of kernel B. Hopper+ feature. Key to understanding B200 overlap savings.
type: reference
originSessionId: ef6a5f82-0e8a-4423-8898-b406f3722dca
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

## Impact on DeepSeek-V3 B200 Analysis (verified from trace)

Total single-stream PDL overlap: **~34μs/layer** (out of ~65μs total overlap), broken down:

| Component | μs | Mechanism |
|-----------|---:|-----------|
| EP_AR ∥ qkv_a (cross-layer pipeline) | ~22 (unstable, 7-35μs) | moefinalize tail overlaps qkv_a preamble. **NOT real compute savings** — qkv_a duration is inflated 1:1 by EP_AR wait time (r=1.000 correlation). qkv_a actual compute = ~15μs. |
| Kernel boundary PDL (~14 boundaries) | ~12 (stable) | ~0.1-2.5μs per boundary, saves kernel launch overhead |

**Key finding**: EP_AR∥qkv_a overlap looks large (~22μs avg) but is mostly EP_AR allreduce wait time stuffed into qkv_a's PDL preamble. Without PDL, qkv_a would be ~15μs (not 31.7μs). The true PDL benefit for MI355X gap analysis is the **~12μs boundary savings** (structural, no AMD equivalent).

- MI355X/ROCm has NO equivalent mechanism — `hipGridDependencySynchronize` does not exist
- MI355X can partially compensate via kernel fusion (fewer boundaries = less launch overhead to hide)

## cuBLAS / CUTLASS Integration

- cuBLAS: PDL supported in some kernels for sm_90+ (CUDA 13.2+)
- CUTLASS: `launch_with_pdl` parameter, `grid_dependency_control.h` header
- Used in production inference (e.g., nvjet GEMM chains)

## References

- CUDA Programming Guide Section 4.5
- PTX ISA Section 9.7.13.13 (`griddepcontrol`)
- CUTLASS `include/cutlass/arch/grid_dependency_control.h`
