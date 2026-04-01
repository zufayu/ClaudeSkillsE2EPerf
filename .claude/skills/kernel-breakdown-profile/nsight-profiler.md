# Nsight Profiler Reference for LLM Inference

Nsight Systems (nsys) and Nsight Compute (ncu) usage guide adapted for TRT-LLM inference workloads on DeepSeek R1 671B.

## Nsight Systems (nsys) -- System-Level Tracing

### Profile Command

```bash
nsys profile \
  -o output_trace -f true \
  -t 'cuda,nvtx,python-gil' \
  -c cudaProfilerApi \
  --cuda-graph-trace node \
  -e TLLM_PROFILE_RECORD_GC=1,TLLM_LLMAPI_ENABLE_NVTX=1 \
  --trace-fork-before-exec=true \
  <your_command>
```

**Key flags for TRT-LLM**:
- `--trace-fork-before-exec=true`: Required for multi-process TRT-LLM (MPI-based TP/EP).
- `--cuda-graph-trace node`: Trace individual kernels within CUDA graphs (critical -- without this, CUDA graph launches appear as single opaque events).
- `-c cudaProfilerApi`: Use `TLLM_PROFILE_START_STOP` env var to control profiling window (e.g., `TLLM_PROFILE_START_STOP=100-150` profiles iterations 100-150).
- `-t 'cuda,nvtx,python-gil'`: Capture CUDA API, NVTX markers (layer/iteration boundaries), and Python GIL events.

### Export & Analysis

```bash
# Export to SQLite for programmatic analysis
nsys export --type sqlite -o trace.sqlite trace.nsys-rep

# Export kernel CSV
nsys stats --report cuda_gpu_trace --format csv -o kernels trace.nsys-rep

# Automated analysis (project script)
bash scripts/analyze_nsys_trace.sh --trace trace.nsys-rep --top 30
```

### SQLite Schema (Common Tables)

| Table | Key Columns | Purpose |
|-------|------------|---------|
| `CUPTI_ACTIVITY_KIND_KERNEL` | shortName, demangledName, start, end, duration | GPU kernel events |
| `StringIds` | id, value | String lookup (kernel names stored as integer IDs) |
| `NVTX_EVENTS` | text, start, end, duration | NVTX markers (layers, iterations) |

### Useful Queries

```sql
-- Top 10 kernels by total time
SELECT shortName, COUNT(*) as count,
       SUM(end-start)/1e6 as total_ms,
       AVG(end-start)/1e3 as avg_us
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY shortName
ORDER BY total_ms DESC
LIMIT 10;

-- Category breakdown (example for MoE)
SELECT CASE
  WHEN shortName LIKE '%moe%' OR shortName LIKE '%expert%' THEN 'MoE'
  WHEN shortName LIKE '%fmha%' THEN 'Attention'
  WHEN shortName LIKE '%nccl%' THEN 'NCCL'
  WHEN shortName LIKE '%nvjet%' OR shortName LIKE '%gemm%' THEN 'GEMM'
  ELSE 'Other'
END AS category,
SUM(end-start)/1e6 as total_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY category
ORDER BY total_ms DESC;
```

**Note**: When `shortName` stores integer IDs (not strings), JOIN with `StringIds` table. The `analyze_nsys_trace.sh` script handles this automatically.

---

## Nsight Compute (ncu) -- Per-Kernel Analysis

### Profile Command

```bash
# Discovery: profile all kernels in a window
ncu --target-processes all \
    --set full \
    --graph-profiling node \
    --launch-skip 50 --launch-count 200 \
    -o output_report \
    <your_command>

# Targeted: profile specific kernel
ncu --target-processes all \
    --set full \
    --graph-profiling node \
    --kernel-name "bmm_E2m1" \
    --launch-skip 10 --launch-count 3 \
    -o output_report \
    <your_command>
```

**Key flags**:
- `--target-processes all`: Profile all processes (required for MPI-based TP).
- `--set full`: Collect all metrics (DRAM throughput, SM throughput, occupancy, roofline data).
- `--graph-profiling node`: Profile individual kernels inside CUDA graphs.
- `--kernel-name`: Regex filter for targeted profiling.
- `--launch-skip / --launch-count`: Skip warmup launches, then profile N invocations.

### Automated Analysis

```bash
# Use project script
bash scripts/ncu_kernel_analysis.sh \
  --model /home/models/models--DeepSeek-R1-0528 \
  --mode discovery --scenario chat --concurrency 32

# Or targeted
bash scripts/ncu_kernel_analysis.sh \
  --model /home/models/models--DeepSeek-R1-0528 \
  --mode targeted --kernel-name "fmhaSm100" \
  --scenario chat --concurrency 32
```

### Export Metrics

```bash
# Raw CSV export
ncu --import report.ncu-rep --page raw --csv > metrics.csv

# Specific metrics
ncu --import report.ncu-rep --csv \
  --metrics "sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active" \
  > selected_metrics.csv
```

### Key Metrics

| Metric | ncu Name | Interpretation |
|--------|----------|---------------|
| **DRAM Throughput** | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | % of peak memory bandwidth used. >60% = memory-bound |
| **SM Throughput** | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | % of peak compute used. >50% = compute-bound |
| **Achieved Occupancy** | `sm__warps_active.avg.pct_of_peak_sustained_active` | % of max concurrent warps. <25% = latency-bound |
| **L2 Hit Rate** | `lts__t_sector_hit_rate.pct` | L2 cache effectiveness |
| **Tensor Core Utilization** | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | Tensor core usage (HMMA) |

### Diagnosis Decision Tree

```
DRAM throughput > 60%?
  YES -> Memory-bound
    -> Reduce data movement, kernel fusion, lower precision
    -> Check if L2 hit rate is low (data doesn't fit in cache)
  NO -> SM throughput > 50%?
    YES -> Compute-bound
      -> Optimize arithmetic intensity, verify tensor core usage
      -> Check if occupancy is reasonable (>50%)
    NO -> Achieved occupancy < 25%?
      YES -> Latency-bound (low occupancy)
        -> Increase block size, reduce register usage
        -> Reduce shared memory per block
      NO -> Under-utilized / Launch overhead
        -> Consider kernel fusion
        -> Check for excessive small kernel launches
```

---

## Theoretical Hardware Specs

### Memory Bandwidth

| GPU | Memory | Theoretical BW | Practical Peak (~85%) |
|-----|--------|----------------|----------------------|
| B200 SXM | 192 GB HBM3e | 8,000 GB/s | ~6,800 GB/s |
| H200 SXM | 141 GB HBM3e | 4,800 GB/s | ~4,080 GB/s |
| H100 SXM | 80 GB HBM3 | 3,350 GB/s | ~2,850 GB/s |
| A100 80GB SXM | 80 GB HBM2e | 2,039 GB/s | ~1,730 GB/s |
| MI355X | 288 GB HBM3e | 8,000 GB/s | ~6,800 GB/s |
| MI325X | 256 GB HBM3e | 5,300 GB/s | ~4,500 GB/s |

### Compute (BF16 Tensor Core)

| GPU | BF16 TFLOPS | FP8 TFLOPS | FP4 TFLOPS |
|-----|------------|------------|------------|
| B200 SXM | 2,250 | 4,500 | 9,000 |
| H200 SXM | 989 | 1,979 | N/A |
| H100 SXM | 989 | 1,979 | N/A |
| MI355X | ~2,300 (est.) | ~4,600 (est.) | ~9,200 (est.) |

### Compute Capability

| GPU | Compute Cap | Architecture |
|-----|------------|-------------|
| B200 | sm_100 | Blackwell |
| H200/H100 | sm_90 | Hopper |
| A100 | sm_80 | Ampere |
| MI355X | gfx950 | CDNA4 |

---

## Bandwidth Efficiency Calculation

For memory-bound kernels, calculate achieved bandwidth efficiency:

```python
# Total bytes transferred
bytes_read = input_elements * dtype_size
bytes_written = output_elements * dtype_size
total_bytes = bytes_read + bytes_written

# Achieved bandwidth
achieved_bw_gbps = (total_bytes / 1e9) / (kernel_time_sec)

# Efficiency
efficiency = achieved_bw_gbps / theoretical_bw_gbps * 100
```

**Targets**:
- Memory-bound kernels (RMSNorm, RoPE, elementwise): >= 60% efficiency is good
- GEMM kernels: Check TFLOPS instead of bandwidth
- Fused kernels (userbuffers_rmsnorm): >= 40% is acceptable (communication overhead)

---

## Common Pitfalls

1. **CUDA Graph opacity**: Without `--cuda-graph-trace node` (nsys) or `--graph-profiling node` (ncu), CUDA graph launches appear as single events -- you see zero kernel-level data.
2. **Multi-process profiling**: TRT-LLM uses MPI for TP/EP. Use `--target-processes all` (ncu) and `--trace-fork-before-exec=true` (nsys).
3. **Profiling overhead**: ncu replays each kernel multiple times to collect metrics. A 10-second workload can take 30+ minutes with `--set full`. Use `--launch-count` to limit.
4. **TLLM_PROFILE_START_STOP**: Controls which iterations are profiled. Set appropriately (e.g., `100-150`) to skip warmup and CUDA graph capture.
5. **StringIds in SQLite**: nsys stores kernel names as integer IDs referencing a `StringIds` table. Always JOIN when querying programmatically.
6. **SM100 vs SM90 kernels**: B200 (SM100) has different kernel implementations than H100 (SM90). Verify the correct kernel variant is running (e.g., `fmhaSm100f` not `fmhaSm90`).
