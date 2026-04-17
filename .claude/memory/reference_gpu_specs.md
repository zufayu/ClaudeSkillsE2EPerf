---
name: GPU Specs Reference (InferenceX)
description: Data center GPU hardware specs for LLM inference performance analysis — HBM BW, TFLOPS, TDP, interconnect. Covers B200, H200, H100, H20, MI355X, MI325X, MI300X.
type: reference
---

# GPU Specifications Reference — LLM Inference Performance Analysis

> Source: [SemiAnalysis InferenceX GPU Specs](https://inferencex.semianalysis.com/gpu-specs), AMD/NVIDIA official datasheets.
> Collected: 2026-04-03

## Master Comparison Table

| Spec | **B200** | **H200** | **H100 SXM** | **H20** | **MI355X** | **MI325X** | **MI300X** |
|------|----------|----------|-------------|---------|------------|------------|------------|
| **Vendor** | NVIDIA | NVIDIA | NVIDIA | NVIDIA | AMD | AMD | AMD |
| **Architecture** | Blackwell (SM100) | Hopper (SM90) | Hopper (SM90) | Hopper (SM90) | CDNA 4 | CDNA 3 | CDNA 3 |
| **Process** | TSMC 4NP | TSMC 4N | TSMC 4N | TSMC 4N | TSMC N3P (XCD) + N6 (IOD) | 5nm + 6nm | 5nm + 6nm |
| **Transistors** | 208B | 80B | 80B | 80B | 185B | ~80B | ~80B |
| **HBM Type** | HBM3e | HBM3e | HBM3 | HBM3 | HBM3e | HBM3e | HBM3 |
| **HBM Capacity** | **192 GB** | **141 GB** | 80 GB | 96 GB | **288 GB** | 256 GB | 192 GB |
| **HBM Bandwidth** | **8.0 TB/s** | **4.8 TB/s** | 3.35 TB/s | 4.0 TB/s | **8.0 TB/s** | 6.0 TB/s | 5.3 TB/s |
| **FP4 Dense** | **9,000 TFLOPS** | N/A | N/A | N/A | **10,060 TFLOPS** | N/A | N/A |
| **FP4 Sparse** | 18,000 TFLOPS | N/A | N/A | N/A | 20,120 TFLOPS | N/A | N/A |
| **FP8 Dense** | **4,500 TFLOPS** | 3,958 TFLOPS | 3,958 TFLOPS | 296 TFLOPS | **5,030 TFLOPS** | 2,615 TFLOPS | 2,615 TFLOPS |
| **FP8 Sparse** | 9,000 TFLOPS | — | — | — | 10,060 TFLOPS | ~5,230 TFLOPS | ~5,230 TFLOPS |
| **FP16/BF16 Dense** | 2,250 TFLOPS | 1,979 TFLOPS | 1,979 TFLOPS | 148 TFLOPS | 2,515 TFLOPS | 1,307 TFLOPS | 1,307 TFLOPS |
| **TF32 Dense** | 1,125 TFLOPS | 989 TFLOPS | 989 TFLOPS | 74 TFLOPS | — | 654 TFLOPS | 654 TFLOPS |
| **FP32** | 80 TFLOPS | 67 TFLOPS | 67 TFLOPS | 44 TFLOPS | — | 163 TFLOPS | 163 TFLOPS |
| **FP64** | 40 TFLOPS | 34 TFLOPS | 34 TFLOPS | 1 TFLOPS | 78.6 TFLOPS | 81.7 TFLOPS | 81.7 TFLOPS |
| **TDP** | **1,000W** | 700W | 700W | 400W | **1,400W** | 1,000W | 750W |
| **Interconnect** | NVLink 5 | NVLink 4 | NVLink 4 | NVLink 4 | Infinity Fabric 4 | Infinity Fabric 4 | Infinity Fabric 3 |
| **Inter-GPU BW** | **1,800 GB/s** | 900 GB/s | 900 GB/s | 900 GB/s | **1,075 GB/s** | ~1,024 GB/s | ~1,024 GB/s |
| **PCIe** | Gen6 x16 | Gen5 x16 | Gen5 x16 | Gen5 x16 | Gen5 x16 | Gen5 x16 | Gen5 x16 |
| **Cooling** | Liquid | Air/Liquid | Air/Liquid | Air | Liquid | Air (OAM) | Air (OAM) |
| **Form Factor** | HGX/DGX | HGX/DGX | HGX/DGX | HGX | OAM (UBB) | OAM (UBB) | OAM (UBB) |

## Key Ratios for Performance Analysis

### HBM Bandwidth (decode 瓶颈指标)

| 对比 | 比值 | 含义 |
|------|------|------|
| B200 / H200 | 8.0/4.8 = **1.67x** | Blackwell BW 优势 |
| B200 / H100 | 8.0/3.35 = **2.39x** | 两代 BW 差距 |
| MI355X / MI325X | 8.0/6.0 = **1.33x** | CDNA4 vs CDNA3 |
| MI355X / MI300X | 8.0/5.3 = **1.51x** | |
| **B200 / MI355X** | **8.0/8.0 = 1.0x** | **BW 相同 — decode GEMM 理论平手** |
| H200 / MI325X | 4.8/6.0 = **0.80x** | MI325X BW 领先 20% |
| H20 / H100 | 4.0/3.35 = **1.19x** | H20 BW 反而高于 H100 |

### FP4 Compute (prefill/高 batch 瓶颈指标)

| 对比 | 比值 |
|------|------|
| MI355X / B200 (FP4 Dense) | 10,060/9,000 = **1.12x** |
| MI355X / B200 (FP4 Sparse) | 20,120/18,000 = **1.12x** |

### FP8 Compute

| 对比 | 比值 |
|------|------|
| MI355X / B200 (FP8 Dense) | 5,030/4,500 = **1.12x** |
| B200 / H200 | 4,500/3,958 = **1.14x** |
| MI355X / MI325X | 5,030/2,615 = **1.92x** |

### Inter-GPU Bandwidth (EP/TP 通信瓶颈指标)

| 对比 | 比值 | 含义 |
|------|------|------|
| B200 NVLink5 / MI355X IF4 | 1,800/1,075 = **1.67x** | B200 inter-GPU 通信显著快 |
| B200 NVLink5 / H200 NVLink4 | 1,800/900 = **2.0x** | NVLink5 翻倍 |
| MI355X IF4 / MI325X IF4 | 1,075/1,024 ≈ **1.05x** | 几乎持平 |

### 能效比 (Performance per Watt)

| GPU | FP8 Dense / TDP | HBM BW / TDP |
|-----|----------------|--------------|
| **B200** | 4,500/1,000 = **4.5 TFLOPS/W** | 8.0/1,000 = **8.0 GB/s/W** |
| **MI355X** | 5,030/1,400 = **3.6 TFLOPS/W** | 8.0/1,400 = **5.7 GB/s/W** |
| H200 | 3,958/700 = **5.7 TFLOPS/W** | 4.8/700 = **6.9 GB/s/W** |
| MI325X | 2,615/1,000 = **2.6 TFLOPS/W** | 6.0/1,000 = **6.0 GB/s/W** |
| H100 | 3,958/700 = **5.7 TFLOPS/W** | 3.35/700 = **4.8 GB/s/W** |

## Decode Bound Analysis Template

对于 decode (M≈batch_per_expert, 通常 1-4)，GEMM 是 **memory bandwidth bound**：

```
Decode time per layer ≈ weight_bytes_per_GPU / achieved_HBM_BW

weight_bytes_per_GPU = total_model_weight / num_GPUs

Achieved BW 通常为 peak 的 75-95%（取决于 GEMM kernel 质量）
```

B200 vs MI355X 的 decode GEMM 理论等价条件：
- 相同 GPU 数 → 相同 per-GPU 权重
- 相同 peak HBM BW (8 TB/s) → 相同理论 decode 时间
- 差异只来自 kernel 利用率

## Prefill / High-Batch Bound Analysis Template

对于 prefill (M >> 1) 或高 batch decode，GEMM 是 **compute bound**：

```
Prefill time ≈ FLOPs / achieved_TFLOPS

FLOPs = 2 × M × K × N (per GEMM)

Achieved TFLOPS 通常为 peak 的 40-70%（取决于 GEMM shape 和 kernel）
```

## Data Sources

- [SemiAnalysis InferenceX](https://inferencex.semianalysis.com/)
- [AMD MI355X Official](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)
- [AMD MI325X Official](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
- [AMD MI300X Official](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [NVIDIA B200 Datasheet](https://www.nvidia.com/en-us/data-center/dgx-b200/)
- [NVIDIA H200 Official](https://www.nvidia.com/en-us/data-center/h200/)
- [NVIDIA H100 Official](https://www.nvidia.com/en-us/data-center/h100/)
- [Tom's Hardware MI355X](https://www.tomshardware.com/pc-components/gpus/amd-announces-mi350x-and-mi355x-ai-gpus-claims-up-to-4x-generational-gain-up-to-35x-faster-inference-performance)
- [Oracle OCI MI355X](https://blogs.oracle.com/cloud-infrastructure/amd-instinct-mi355x-on-oci-performance-technical-details)
- [WCCFTech MI350 Launch](https://wccftech.com/amd-instinct-mi350-mi355x-launched-3nm-185-billion-transistors-288-gb-hbm3e-fp4-fp6-2-2x-faster-blackwell-b200/)

## Notes

- B200 有 **180GB** (datasheet) 和 **192GB** (部分 SKU/实测) 两种说法。本项目实测用的 B200 为 192GB 版本。
- MI355X 的 HBM BW 在 AMD 官方文档中为 **8 TB/s**。部分第三方引用 10 TB/s，可能是 8-GPU 平台级或含 Infinity Cache 加速的数值。保守用 8 TB/s。
- H100/H200 的 FP8 TFLOPS (3,958) 含 Transformer Engine sparsity 时可翻倍。表中为 dense 值。
- MI355X 的 FP6 与 FP4 共享同一物理电路，TFLOPS 相同。
- 所有 TFLOPS 均为 peak theoretical，实际 kernel 利用率需用 trace 数据验证。
