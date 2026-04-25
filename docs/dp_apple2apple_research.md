# SA CI DP-attention behavior: B200/B300 vs MI355X

Researched 2026-04-25, source-of-truth = `~/InferenceX` repo (HEAD).

## TL;DR

- **B300 has no dedicated single-node SA CI scripts.** Only multi-node `dsr1-fp{4,8}-b300-dynamo-trt` exists. For single-node B300 testing we must either reuse B200 scripts (same TRT codepath) or accept "no SA baseline".
- **DP-attention is a TRT-only knob in SA CI.** The `enable_attention_dp` YAML toggle is only flipped by `dsr1_*_b200_trt*.sh` scripts. **All sglang scripts (B200 single-node + MI355X single-node) hardcode `data-parallel-size=1`**, so DP-on-vs-off cannot be A/B'd in sglang at SA-CI parity.
- **MI355X ATOM scripts require `DP_ATTENTION` as env but never read it** — ATOM decides DP internally based on model/topology. There is no user-facing DP toggle on ATOM.
- **The user's "tp=1, ep=4" MI355X plan is NOT in SA CI matrix.** SA CI MI355X ATOM only enumerates `tp:{4,8}, ep:1`. Closest SA CI cell to compare: B200 fp4-trt `tp:4 ep:4 dp-attn:true conc:256` — but model SKU (V2 nvfp4 vs amd mxfp4) and concurrency don't align with the user's c=4/64 plan.

## DP-attention matrix (SA CI as of HEAD)

### TRT B200 — DP toggled per cell in `nvidia-master.yaml`

| Bench | ISL/OSL | DP-on cells | DP-off cells |
|---|---|---|---|
| `dsr1-fp4-b200-trt` | 1k/1k | `tp4 ep4 c=256`, `tp8 ep8 c=128-256` | `tp4 c=4-16`, `tp8 c=4`, `tp8 ep8 c=64` |
| `dsr1-fp4-b200-trt` | 1k/8k | `tp4 ep4 c=256`, `tp8 ep8 c=128-256` | `tp4 c=4-16`, `tp8 c=4`, `tp8 ep8 c=64` |
| `dsr1-fp4-b200-trt` | 8k/1k | `tp4 ep4 c=256`, `tp8 ep8 c=128-256` | `tp4 c=4-32`, `tp4 ep4 c=32`, `tp8 c=4` |
| `dsr1-fp4-b200-trt-mtp` | 1k/1k | `tp4 ep4 c=256`, `tp8 ep8 c=32-64` | `tp4 c=4-8`, `tp8 c=4,128`, `tp8 ep8 c=32-128` |
| `dsr1-fp8-b200-trt` | 1k/8k | `tp8 ep8 c=256` (only) | `tp8 c=4-128` |
| `dsr1-fp8-b200-trt` | 1k/1k, 8k/1k | (none) | `tp4/tp8 various` |
| `dsr1-fp8-b200-trt-mtp` | 1k/1k | `tp8 ep8 c=256` | `tp8 c=4-128` |
| `dsr1-fp8-b200-trt-mtp` | 1k/8k | `tp8 ep8 c=128-256` | `tp8 c=4-64` |
| `dsr1-fp8-b200-trt-mtp` | 8k/1k | (none) | `tp8 c=4-256` |

**Pattern:** DP is only flipped on at *high concurrency with EP enabled*. SA CI never tests DP at low concurrency — the assumption is DP-attn is purely a throughput feature.

### What flipping `DP_ATTENTION=true` actually changes (TRT scripts)

From `dsr1_fp4_b200_trt.sh`:
```
DP=true:  MOE_BACKEND=CUTLASS  CUDA_GRAPH_MAX_BATCH_SIZE=max(CONC, CONC/4)
DP=false: MOE_BACKEND=TRTLLM   CUDA_GRAPH_MAX_BATCH_SIZE=CONC
```
From `dsr1_fp4_b200_trt_mtp.sh` (and fp8 mtp similar):
```
DP=true:  MOE_BACKEND=CUTLASS (fp4) or DEEPGEMM (fp8)
          CUDA_GRAPH_MAX_BATCH_SIZE=CONC/4 (fp4) or CONC/8 (fp8)
          MTP=1   (vs MTP=3 default)
          + attention_dp_config block: batching_wait_iters=0, enable_balance=true, timeout_iters=60
DP=false: MOE_BACKEND=TRTLLM  CUDA_GRAPH_MAX_BATCH_SIZE=CONC  MTP=3
```
DP=true is therefore not "just toggle DP" — it co-changes MOE backend, graph capture size, MTP layer count, and adds a tuning block. **Comparing DP-on vs DP-off naively conflates four variables.**

### sglang B200 — DP NOT toggleable

- `dsr1_fp4_b200.sh`, `dsr1_fp8_b200.sh`, `dsr1_fp8_b200_mtp.sh`: all hardcode `--data-parallel-size=1`
- `dsr1-fp8-b200-sglang` matrix: `tp:8 ep:1` and `tp:4 ep:1` only — no DP cell
- `dsr1-fp8-b200-sglang-mtp` matrix: `tp:8 ep:1` only — uses EAGLE speculative (not MTP heads), TP=8 hardcoded

### MI355X ATOM — DP NOT toggleable

- `dsr1_fp4_mi355x_atom.sh`, `dsr1_fp8_mi355x_atom.sh`, `*_mtp.sh`: declare `DP_ATTENTION` as required env but never reference it in script body
- ATOM determines DP from model topology automatically; the env var is purely for matrix-row tagging
- `dsr1-fp4-mi355x-atom` matrix: `tp:{4,8} ep:1` only
- `dsr1-fp8-mi355x-atom` matrix: `tp:8 ep:1` only

### MI355X sglang single-node — DP NOT toggleable

- `dsr1_fp{4,8}_mi355x.sh`: no `--data-parallel-size` flag at all
- Matrix: `tp:{4,8}` only

### MI355X sglang **disagg** (multi-node) — DP IS per-stage

- `dsr1_fp{4,8}_mi355x_sglang-disagg.sh`: `PREFILL_DP_ATTN`, `DECODE_DP_ATTN` are independent toggles
- e.g. `dsr1-fp8-mi355x-sglang-disagg`: prefill `tp:8 ep:1 dp-attn:false`, decode `tp:8 ep:8 dp-attn:true` (high-throughput recipe)
- This is the only place DP-on/off can be A/B'd within the AMD CI matrix, and it requires multi-node

## Apple-to-apple cells for tomorrow's 12-point MI355X plan

User plan: MI355X tp=1, ep=4, c={4, 64}, 3 ISL/OSL × MTP{on,off} = 12 points.

**No SA CI cell maps directly.** The closest pairs, ranked by similarity:

| User's MI355X cell | Closest SA CI cell | Caveat |
|---|---|---|
| MI355X tp1 ep4 mxfp4 c=4 | B200 `dsr1-fp4-b200-trt` `tp4 c=4` | Different TP (1 vs 4), different model SKU |
| MI355X tp1 ep4 mxfp4 c=64 | B200 `dsr1-fp4-b200-trt` `tp8 ep8 c=64` (DP=off) | Different TP/EP topology |
| MI355X tp1 ep4 mxfp4-MTP c=4 | B200 `dsr1-fp4-b200-trt-mtp` `tp4 c=4-8 mtp` | Different TP, MTP heads count differs (B200 MTP=3, MI355X ATOM MTP via `--method mtp` unspecified count) |
| MI355X tp1 ep4 mxfp4-MTP c=64 | B200 `dsr1-fp4-b200-trt-mtp` `tp4 ep4 c=8-64 mtp` (DP=off variants only) | OK-ish at c=8/16/32 but B200 doesn't have c=64 at this cell |

**Better-aligned alternative** if the user wants comparable SA points: switch the MI355X plan to `tp=4, ep=1` (the SA-CI MI355X ATOM cell that already exists) and compare directly against `dsr1-fp4-mi355x-atom` historical runs. Then a single SA dispatch reproduces the baseline.

## Recommendations for tomorrow

1. **Run the user's 12-point plan as a custom config**, not a SA-CI replica. Document it as "non-SA topology — used to probe ATOM behavior at minimum-TP / max-EP".
2. **Skip ci_commit_results** — workflow currently force-resets stray local edits (saw exit 128 in today's smoke test) and the user explicitly said "不上传污染main".
3. **For DP-on-vs-off study**, the only clean A/B is **B200 TRT at high concurrency with EP enabled** (e.g. `dsr1-fp4-b200-trt tp:8 ep:8 c:128`, where the matrix has both `dp-attn:true` and adjacent `dp-attn:false` rows — but note the four-variable confound above).
4. **For sglang DP-on-vs-off**, no SA-CI baseline exists in single-node — would need to add a `--data-parallel-size>1` cell to either MI355X or B200 sglang scripts ourselves (out-of-scope for tonight).
5. **B300 single-node** has no SA reference — any B300 run we do is original work, not "alignment".

## Source citations

- TRT B200 scripts: `~/InferenceX/benchmarks/single_node/dsr1_fp{4,8}_b200_trt{,_mtp}.sh`
- sglang B200 scripts: `~/InferenceX/benchmarks/single_node/dsr1_fp{4,8}_b200{,_mtp}.sh`
- ATOM MI355X scripts: `~/InferenceX/benchmarks/single_node/dsr1_fp{4,8}_mi355x_atom{,_mtp}.sh`
- sglang MI355X single-node: `~/InferenceX/benchmarks/single_node/dsr1_fp{4,8}_mi355x.sh`
- sglang MI355X disagg: `~/InferenceX/benchmarks/multi_node/dsr1_fp{4,8}_mi355x_sglang-disagg.sh`
- Matrices: `~/InferenceX/.github/configs/{nvidia,amd}-master.yaml`
