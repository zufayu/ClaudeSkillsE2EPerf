---
name: MI355X FP4 model comparison results
description: Full comparison of 4 DeepSeek-R1-0528 FP4 model variants on MI355X — all independently quantized with different weights, configs, and timestamps
type: project
originSessionId: 49e0962a-ee55-4b7c-9023-e07ec969a46b
---
All 4 DeepSeek-R1-0528 FP4 model variants on MI355X are **independently quantized** — different timestamps, different owners, different shard counts, different sizes, and different quantization configs.

## Model Summary

| Model | Short Name | Date | Owner | Shards | Size | Quark Ver | Attn | MTP head (L61) |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-R1-0528-MXFP4 | pure_mxfp4 | Mar 18 | zejun_chen | 82 | 376G | 0.11 | MXFP4 (all self_attn excluded) | excluded (re:model.layers.61.*) |
| DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8 | moe_mixed | Mar 20 | lingpeng_jin | 77 | 355G | 0.12+ | FP8 PTPC | FP8 PTPC |
| DeepSeek-R1-0528-MTP-MoE-MXFP4-Attn-PTPC-FP8 | mtp_mixed (ATOM CI) | Mar 23 | lingpeng_jin | 76 | 350G | 0.11.1 | FP8 PTPC | FP8 PTPC |
| DeepSeek-R1-0528-MXFP4-MTP-MoEFP4 | mxfp4_mtp (latest) | Apr 13 | lingpeng_jin | 76 | 350G | 0.11.1 | FP8 PTPC | MXFP4 (inherits global) |

## Key Differences Between mtp_mixed (ATOM CI) and mxfp4_mtp (latest)

Both use Quark 0.11.1, same shard count (76), same size (350G), same attention quantization (FP8 PTPC).

**Only difference:** MTP head (layer 61) quantization:
- **mtp_mixed**: `--layer_quant_scheme 'model.layers.61.*' ptpc_fp8` → FP8 PTPC
- **mxfp4_mtp**: No layer 61 override → inherits global MXFP4

The latest model (mxfp4_mtp) removed the FP8 special treatment for MTP head, letting it use MXFP4 like the MoE layers.

## pure_mxfp4 vs the rest

- Only model by zejun_chen (others by lingpeng_jin)
- Uses Quark 0.11 (oldest version)
- Excludes ALL self_attn layers individually (layers 0-60, listed one by one)
- No FP8 anywhere — pure MXFP4 for everything not excluded
- Largest: 82 shards, 376G

**Why:** Understanding which model was used in which benchmark is critical for interpreting results. ATOM CI uses mtp_mixed; the latest model (mxfp4_mtp) has a subtly different MTP head quantization.

**How to apply:** When running MI355X benchmarks, always specify which model path is being used. Results across models are not directly comparable due to different quantization strategies.

Verified 2026-04-16 via workflow `mi355x_fp4_full_compare.yml` (run 24486506271).
