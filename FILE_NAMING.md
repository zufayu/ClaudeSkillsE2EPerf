# File Naming Convention

> Standard naming rules for all artifact files in `results/`. Goal: one glance
> tells you platform, role, scenario, batch size, and (when relevant) trace
> sub-window. No file name should require parsing > 6 fields.

## Directory hierarchy

```
results/<platform>_<model>_<quant>/<run-config-tag>/<file>
```

Where:

| field | regex | examples | rule |
|---|---|---|---|
| `<platform>` | `b200|b300|mi355x|h20` | `b200` | physical GPU family. one of fixed set. |
| `<model>` | `dsr|dsr_v2|llama|qwen` | `dsr` (DeepSeek-R1/V3 default), `llama` | base name only, no version suffix in dir |
| `<quant>` | `fp4|fp8|mxfp4|bf16` | `fp4`, `mxfp4` | quantization scheme |
| `<run-config-tag>` | `<plat>_<mod>_<q>_mtp<N>_ep<N>_tp<N>_<env>[_profiling]` | `b300_dsr_fp4_mtp0_ep8_tp8_sglang059_profiling` | full run identifier; `_profiling` suffix marks runs with torch profiler enabled |

`<env>` is `<framework><version>` joined: `sglang059` (sglang 0.5.9), `rocm722`
(ROCm 7.2.2 ATOM), `trtllmrc10`, etc.

## File name format

```
<role>_<scenario>_c<concurrency>[_<extra>].<ext>
```

| field | regex | examples | rule |
|---|---|---|---|
| `<role>` | one of the canonical role list (below) | `decode_breakdown`, `kernel_breakdown` | mandatory |
| `<scenario>` | `chat|reasoning|summarize` | `chat` | mandatory for any benchmark/profiling output |
| `c<concurrency>` | `c\d+` | `c4`, `c64`, `c640` | mandatory |
| `<extra>` | varies by role (see below) | `_step30-1030`, `_ep1`, `_full` | optional, role-specific |
| `<ext>` | `csv|xlsx|log|md|json` | `.csv`, `.log` | mandatory |

## Canonical roles

| role | what it is | extra fields | typical extensions |
|---|---|---|---|
| `decode_breakdown` | per-kernel per-layer aggregated decode stats (the R8d primary output) | none | `.csv` `.xlsx` |
| `prefill_breakdown` | prefill-phase kernel stats | none | `.xlsx` |
| `kernel_breakdown` | flat kernel listing from extract_cuda_kernels (all kernels in profile window) | `_step<X>-<Y>` (trace step window) | `.csv` `.log` |
| `per_layer_breakdown` | per-layer kernel breakdown from same script | `_step<X>-<Y>` | `.csv` `.log` |
| `nsys_kernel_breakdown` | nsys-derived kernel summary | none | `.csv` |
| `gpu` | GPU-side bench output (gpu_<bench-mode>_<scenario>) | bench mode | `.csv` |
| `result` | benchmark JSON output (latency, throughput) | bench mode | `.json` |
| `server` | server stdout log | bench mode (or `_<TAG>`) | `.log` |
| `summary` | human-readable run summary | none | `.md` |
| `trace_torch` | PyTorch profiler trace (raw + analyzed forms) | `_full`, `.fix.json.gz`, `_serialized.json.gz` | `.json.gz` `.log` |
| `dispatch_inputs` | workflow_dispatch context (added 2026-04-25) | none | `.json` |

## Examples (canonical good)

```
results/b300_dsr_fp4/b300_dsr_fp4_mtp0_ep8_tp8_sglang059_profiling/
    decode_breakdown_chat_c4.csv               # R8d primary, MI355X-comparable schema
    decode_breakdown_chat_c4.xlsx              # same data, xlsx
    server_chat_c4.log                         # server stdout
    summary.md                                 # human readable
    dispatch_inputs.json                       # added 2026-04-25
    trace_torch_chat_c4_step30-1030.json.gz   # raw torch trace
    kernel_breakdown_chat_c4_step30-1030.csv  # extract_cuda_kernels output
```

## Forbidden patterns (current violations to fix)

| pattern | why bad | example |
|---|---|---|
| Missing `c<N>` suffix | can't tell concurrency from filename | `decode_breakdown.xlsx` |
| Trace tag duplicated in derivative file | redundant, name explodes to 80+ chars | `kernel_breakdown_trace_torch_b200_sglang_dsr1_fp4_sglang059_chat_ep1_tp4_c64_step30-1030.csv` (use `kernel_breakdown_chat_c64_step30-1030.csv`; the dir already encodes `b200_sglang_dsr_fp4_sglang059`) |
| Mixed naming for same role across env | hard to grep | rocm711 has `decode_breakdown.xlsx`; rocm722 has `decode_breakdown_c64.xlsx` |
| Non-canonical role names | breaks tooling | `layer_kernel_avg.csv`, `layer_walltime.csv` (deprecated; use `decode_breakdown` or `per_layer_breakdown`) |
| `gpu_` prefix without scenario | can't tell what bench config | `gpu_fp4.csv` (use `gpu_fp4_throughput_chat_c64.csv`) |

## Migration policy

- New files MUST follow this spec.
- Legacy files (pre-2026-04-25) NOT auto-renamed; they're frozen historical artifacts.
- When a legacy result dir is regenerated, the new run uses canonical names and overwrites the legacy ones.
- A `validate_naming.py` linter can be added later to fail PR if new files violate.

## Why this spec exists

Audit on 2026-04-25 found 4 distinct naming conventions in `results/`:
- Old `b200_dsr_fp4_mtp0_ep4_tp4_post2/`: short — `gpu_fp4_throughput_chat_ep4_c64.csv`
- Mid `_sglang059/`: medium — `gpu_sglang_fp4_throughput_chat_tp4_ep4_c64.csv`
- New profiling `_sglang059_profiling/`: super long — `kernel_breakdown_trace_torch_b200_sglang_dsr1_fp4_sglang059_chat_ep1_tp4_c64_step30-1030.csv` (80+ chars duplicating dir info)
- MI355X rocm711 vs rocm722: `decode_breakdown.xlsx` vs `decode_breakdown_c64.xlsx` (concurrency suffix inconsistently applied)

Without a canonical rule each workflow makes its own. This doc is the rule.

## Initial violation scan (snapshot 2026-04-25)

97 files found violating one or more rules across `results/` (after spec
draft). Per migration policy these are NOT auto-renamed:

| category | count | example | action |
|---|---|---|---|
| `no_concurrency_suffix` | 2 | `mi355x_dsr_mxfp4_mtp0_ep4_tp4_rocm711_profiling/decode_breakdown.xlsx` | frozen (rocm711 legacy) |
| `long_redundant_name` (>70 chars) | 31 | `decode_walltime_trace_torch_mi355x_atom_dsr1_mxfp4_rocm722_chat_ep4_tp4_c64_full.csv` | kept (no-rename per user; collect_atom_trace.sh's TAG form) |
| `unknown_role` | 59 | `perf_metrics_fp8_latency_chat_ep1_c64.json`, `ncu_decode.log` | many one-offs; consider extending canonical role list as needed |
| `legacy_role_name` (layer_kernel_avg / layer_walltime) | 5 | `b200_dsr_fp4_mtp0_ep4_tp4_sglang059_profiling/layer_walltime.csv` | frozen (deprecated by trace_layer_detail.py R8d output) |
| `no_scenario` | 0 | — | ✅ |

**New generator compliance** (going forward, not retro-fix):
- ✅ `decode_kernel_breakdown.py` writes `decode_breakdown_c<N>.{xlsx,csv}` — partial: missing `_chat` scenario in name (could add as next-step polish)
- ✅ `trace_layer_detail.py` writes to `STEADY/decode_breakdown_c<N>.csv` — same partial
- ⚠️ `collect_sglang_trace.sh` writes `kernel_breakdown_<TAG>.csv` where `<TAG>` repeats dir info — kept by user request, NOT compliant
- ✅ `dispatch_inputs.json` (added 2026-04-25) — compliant
