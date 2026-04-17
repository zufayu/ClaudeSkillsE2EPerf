---
name: ATOM CI actual configuration
description: ATOM official CI uses hybrid model (MTP-MoE-MXFP4-Attn-PTPC-FP8) with MTP=3, not pure MXFP4
type: project
originSessionId: 49e0962a-ee55-4b7c-9023-e07ec969a46b
---
ATOM CI uses hybrid model `DeepSeek-R1-0528-MTP-MoE-MXFP4-Attn-PTPC-FP8`, NOT pure MXFP4.

- **Attention**: FP8 PTPC (Tensile GEMM), not BF16 (ASM)
- **MTP**: `--method mtp --num-speculative-tokens 3` enabled
- **MoE**: MXFP4
- **MTP head (layer 61)**: FP8 PTPC (explicit `--layer_quant_scheme 'model.layers.61.*' ptpc_fp8`)

**Why:** Our MI355X profiling CI was using pure MXFP4 without MTP — misaligned with ATOM official CI. Kernel behavior differs: FP8 PTPC attention uses different GEMM kernels (Tensile vs ASM), and MTP adds speculative head computation.

**How to apply:** When comparing against ATOM CI results or aligning profiling configs, use the hybrid model path and enable MTP=3. Pure MXFP4 benchmarks are still valid for cross-model comparison but don't represent ATOM's official configuration.

Discovery date: 2026-04-15. Updated 2026-04-16: All 4 FP4 model variants are independently quantized with different weights — see `project_mi355x_fp4_models.md` for full comparison.
