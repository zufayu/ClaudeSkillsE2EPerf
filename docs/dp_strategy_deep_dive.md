# DP-attention strategy: per-framework config, mechanism, verification, A/B

Researched 2026-04-28. Source-of-truth = `~/InferenceX` HEAD + framework source under `~/{TensorRT-LLM,ATOM}` and sglang public github.

**Supersedes** `docs/dp_apple2apple_research.md` (which contained an incorrect claim that ATOM auto-detects DP from topology — see ATOM section below for the correction).

---

## TL;DR

| Framework | Real DP-attn knob | SA-CI exercises it? | Auto co-changes inside framework? |
|---|---|---|---|
| **TRT-LLM** | YAML `enable_attention_dp: true` + `attention_dp_config.{enable_balance, batching_wait_iters, timeout_iters}` | Yes (B200 trt + trt-mtp scripts toggle per cell) | **No** — MOE_BACKEND / CUDA_GRAPH_MAX_BATCH_SIZE / MTP co-changes happen in SA shell scripts, not TRT-LLM |
| **SGLang** | CLI `--enable-dp-attention` + `--data-parallel-size N` (must be paired) | Single-node: **no** (all hardcode dp=1). Disagg multi-node: yes (`PREFILL_DP_ATTN`/`DECODE_DP_ATTN`) | **Yes** — `chunked_prefill_size /= dp_size`, `schedule_conservativeness *= 0.3`, `disable_piecewise_cuda_graph=True` |
| **ATOM** | CLI `--enable-dp-attention` (real flag, exists) | **NO** — ATOM CI scripts never pass it. The `DP_ATTENTION` env var is metadata-only | When passed: flattens TP→DP (tp_size becomes 1, dp_size becomes N) |

---

## 1. TRT-LLM (B200, B300)

### 1.1 Config surface

YAML in file passed via `--extra_llm_api_options`:

| Key | Type | Default | Source |
|---|---|---|---|
| `enable_attention_dp` | bool | false | `tensorrt_llm/llmapi/llm_args.py:2218` |
| `attention_dp_config.enable_balance` | bool | false | `llm_args.py:475` |
| `attention_dp_config.batching_wait_iters` | int | 10 | `llm_args.py:479` |
| `attention_dp_config.timeout_iters` | int | 50 | `llm_args.py:477` |
| `kv_cache_config.attention_dp_events_gather_period_ms` | int | 5 | `llm_args.py:1887` |

CLI: only `--extra_llm_api_options PATH.yml` on `trtllm-serve` (`serve.py:652`). No dedicated `TRTLLM_DP_*` env var. SA scripts read shell `$DP_ATTENTION` and inline it into the YAML they generate.

### 1.2 Config-takes-effect mechanism

1. `trtllm-serve` calls `update_llm_args_with_extra_dict` → `AttentionDpConfig(**dict)` (`llm_args.py:3431, 3454`).
2. `TorchLlmArgs.validate_and_init_parallel_config` builds `_ParallelConfig(enable_attention_dp=...)` → `Mapping`.
3. `Mapping.dp_size` returns `tp_size if enable_attention_dp else 1` (`mapping.py:241`).
4. **Attention layer** (`_torch/modules/attention.py:433-450`):
   ```python
   if mapping.enable_attention_dp:
       dp_size = tp_size; tp_size = 1
   ```
   QKV and o_proj built with `tp_size=1` → **attention weights are replicated** per rank, not sharded.
5. **KV cache** (`resource_manager.py:823`): `tp_size = 1 if enable_attention_dp else mapping.tp_size`.
6. **Scheduler** (`py_executor.py:334-339, 2619-2627`): consumes `attention_dp_config.{enable_balance, batching_wait_iters, timeout_iters}` for cross-rank ctx-request balancing.

**Key insight — co-changes are NOT auto-wired in TRT-LLM:**

The 4-variable confound from `dp_apple2apple_research.md` (MOE_BACKEND, CUDA_GRAPH_MAX_BATCH_SIZE, MTP, attention_dp_config) is **entirely in the SA shell scripts**. Verified:

- `dsr1_fp4_b200_trt.sh:30-32`: `if DP_ATTENTION=true → MOE_BACKEND=CUTLASS` (else TRTLLM)
- `dsr1_fp8_b200_trt_mtp.sh:26`: same pattern → DEEPGEMM
- CUDA_GRAPH_MAX_BATCH_SIZE: shell-level `$(( CONC < 4 ? CONC : CONC / 4 ))` formula, not TRT-LLM
- MTP=3→1 flip: shell-level
- `kv_cache_free_gpu_memory_fraction`: 0.7 vs 0.8 also shell-level

→ **A clean A/B is possible** by writing YAML manually and pinning all four variables identical across DP-on/DP-off.

### 1.3 Runtime verification

Most reliable signals (from server stdout/log):

```bash
# 1. Config-load warning (most reliable single signal)
grep "Overriding attention_dp_config" $LOG

# 2. Feature status (fires only when MTP+DP combo)
grep "Attention DP is enabled" $LOG  # _util.py:584-587

# 3. Per-rank request-count divergence (DP routes per rank)
grep "currank_total_requests" $LOG | head -20

# 4. Effective MoE backend
grep -iE "moe.*backend|cutlass|deepgemm|trtllm" $LOG | head -5

# 5. CUDA graph dimensions
grep -iE "max_batch_size|cuda_graph" $LOG | head -10

# 6. Inspect generated YAML directly
cat $WORKSPACE/dsr1-fp8-mtp.yml | grep -A5 attention_dp
```

No HTTP `/metrics` field exposes DP state. `print_iter_log: true` (in YAML) gives per-iter request distribution which is the runtime smoking gun.

Failure modes:
- Pydantic raises `ValueError` on malformed `attention_dp_config` (`llm_args.py:484`).
- Per-iter CUDA graph allgather (`cuda_graph_runner.py:250-260`) silently falls back to eager when ranks disagree on batch — visible only in iter log timing spikes, no warning.
- `_pad_attention_dp_dummy_request` (`py_executor.py:2747`) prevents zero-active-request deadlock under DP.

### 1.4 A/B methodology

**Pure DP-attn isolation** (recommended for any cross-cell comparison):

```yaml
# DP-on cell
enable_attention_dp: true
attention_dp_config:
  enable_balance: false        # disable scheduler delay confounder
  batching_wait_iters: 0
  timeout_iters: 50
moe_config:
  backend: TRTLLM              # PIN — don't auto-flip to CUTLASS
cuda_graph_config:
  max_batch_size: <SAME_AS_OFF>  # PIN — don't divide by 4
speculative_config:
  num_nextn_predict_layers: 0   # PIN if studying without MTP
kv_cache_config:
  free_gpu_memory_fraction: 0.8 # PIN
```

```yaml
# DP-off cell — same YAML except:
enable_attention_dp: false
# (drop attention_dp_config block)
```

**Caveat**: at CONC>=256 fp8, native script's MOE_BACKEND varies. Pinning to TRTLLM may underperform vs SA's auto-CUTLASS choice — document the choice.

**Target metric**: decode TPOT (inter-token latency) at CONC>=64. DP-attn primarily reduces attention all-reduce volume in decode; prefill may be neutral or slightly negative.

**Sample sizes**: SA uses `--num-prompts $((CONC*10))`. For rigorous A/B, use 20x and discard first 5% (cold KV cache).

**Confounders**:
- KV-cache reuse: pin `enable_block_reuse: false` for both cells.
- `enable_balance=true` adds prefill latency on DP-on side — disable for compute-isolation A/B.
- MTP draft acceptance varies with batch size — fix `max_batch_size` AND `num_nextn_predict_layers` together.

---

## 2. SGLang (B200, B300, MI355X)

### 2.1 Config surface

**Two ORTHOGONAL concepts** — most-confused thing in sglang:

| Knob | Meaning | GPU count |
|---|---|---|
| `--data-parallel-size N` (alias `--dp-size`) WITHOUT `--enable-dp-attention` | **Replica DP**: N independent full-model copies, each owning `tp_size` GPUs and its own KV cache. Round-robin / shortest-queue routing. | `N * tp_size` |
| `--data-parallel-size N` PAIRED WITH `--enable-dp-attention` | **Attention DP**: ONE TP group of `tp_size` GPUs, subdivided so each "DP rank" handles a different batch slice ONLY at attention layers. FFN/MoE still uses full TP width. | `tp_size` (no extra GPUs) |

Other knobs (newer sglang):

| Flag | Effect |
|---|---|
| `--enable-dp-lm-head` | Shards LM head across DP ranks; requires `enable_dp_attention=True`. SA `models.yaml` `dp_flags` always includes it for DeepSeek. |
| `--moe-dense-tp-size 1` | Restricts dense (non-MoE) TP width under DP-attn. SA disagg uses 1. |
| `--moe-dp-size N` | Independent expert-layer DP, separate from attention DP. |
| `--moe-a2a-backend mori` | RDMA all-to-all for MoE dispatch. AMD-specific. SA disagg uses it. |
| `--deepep-mode normal` | DeepEP MoE mode. `auto` is **rejected** when `enable_dp_attention=True`. |

ROCm-specific env vars (set in SA `env.sh`): `SGLANG_USE_AITER=1`, `SGLANG_MORI_*`, `MORI_MAX_DISPATCH_TOKENS_{PREFILL,DECODE}`, `SGLANG_ROCM_FUSED_DECODE_MLA`, `SGLANG_USE_AITER_UNIFIED_ATTN`.

### 2.2 Config-takes-effect mechanism

1. `server_args.py` parse: when `enable_dp_attention=True`, **automatic side-effects fire immediately**:
   - `chunked_prefill_size //= dp_size`
   - `schedule_conservativeness *= 0.3`
   - `disable_piecewise_cuda_graph = True`
2. `data_parallel_controller.py:85-90` branch:
   - `enable_dp_attention=True` → `launch_dp_attention_schedulers()` spawns **one** TP process group, all `tp_size` ranks share one nccl port.
   - `enable_dp_attention=False, dp_size>1` → `launch_dp_schedulers()` spawns N independent TP groups with separate nccl ports (replica DP).
3. `scheduler.py:195-201` (`compute_dp_attention_world_info`): each TP rank computes `attn_tp_size = tp_size // dp_size`, `dp_rank = tp_rank // attn_tp_size`. Example: tp=8, dp=8 → attn_tp_size=1, every GPU is its own DP rank.
4. `dp_attention.py:41-74` (`initialize_dp_attention`): creates `_ATTN_TP_GROUP` GroupCoordinator partitioning the tp_group into sub-groups of size `attn_tp_size`. Sub-groups handle attention's intra-layer TP comm; full tp_group remains for FFN/MoE.
5. `deepseek_v2.py:1032, 1291`: model reads `enable_dp_attention` from `global_server_args_dict`, sets `enable_tp = not enable_dp_attention` for attention projection.
6. **ROCm**: `aiter_backend.py:208,237,255` has separate code path — if `enable_dp_attention=True` AND `num_head` collapses to 128 with `tp=8 dp=8`, persist MLA kernel is disabled.

**Hard validation errors** (no silent fallback):
- `dp_size > 1` + multi-node + no `enable_dp_attention` → assert
- `enable_dp_attention=True` + `dp_size==1` → assert "Please set a dp-size > 1"
- `tp_size % dp_size != 0` → assert
- `enable_dp_lm_head=True` without `enable_dp_attention` → assert
- `deepep_mode='auto'` + `enable_dp_attention` → assert

### 2.3 Runtime verification

```bash
# 1. Canonical: query effective config
curl -s http://localhost:$PORT/get_server_info | python3 -c "
import json,sys
d = json.load(sys.stdin)
for k in ['dp_size', 'enable_dp_attention', 'tp_size', 'ep_size',
          'chunked_prefill_size', 'enable_dp_lm_head', 'attention_backend']:
    print(f'{k}: {d.get(k)}')
"

# 2. chunked_prefill_size sanity: should equal original / dp_size when DP-attn on
# (e.g., set --chunked-prefill-size 8192 with dp=8 → effective 1024)

# 3. Process names: per-rank includes dp_rank (only when DP-attn active)
ps aux | grep "sglang::scheduler" | grep -oE "DP[0-9]+ TP[0-9]+"

# 4. Startup log
grep -E "DP attention is enabled|Launch DP" $LOG
# DP-attn: "DP attention is enabled. The chunked prefill size is adjusted to X"
# Replica DP: "Launch DP{N} starting at GPU #{M}" (one per replica)

# 5. ROCm: confirm aiter
grep -iE "aiter|attention backend" $LOG | head -5
```

**Memory pattern:**
- Pure TP (tp=8, dp=1): each GPU = 1/8 of weight, full KV pool distributed across 8.
- DP-attn (tp=8, dp=8): each GPU = 1/8 of weight (FFN unchanged); KV cache **per-rank only** (each DP rank caches only its own request slice). `cuda_graph_max_bs` should be sized per-rank batch.
- Replica DP (dp=N, no DP-attn): each replica has independent full-size KV. Total KV = N× single.

### 2.4 A/B methodology

**Single-node sglang DP-on vs DP-off** — SA CI does NOT test this. To roll our own:

```bash
# Baseline DP-off (tp=8)
python3 -m sglang.launch_server \
  --model-path $MODEL --tensor-parallel-size 8 --data-parallel-size 1 \
  --attention-backend aiter --ep-size 8 ...

# Treatment DP-on (same 8 GPUs, no extra hardware)
python3 -m sglang.launch_server \
  --model-path $MODEL --tensor-parallel-size 8 --data-parallel-size 8 \
  --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1 \
  --deepep-mode normal --ep-size 8 \
  --chunked-prefill-size 8192 \   # SA's DP-on formula: MORI_MAX_DISPATCH * PREFILL_TP
  --attention-backend aiter ...
```

**Confounders**:
1. `chunked_prefill_size` auto-divides — pre-scale baseline if comparing at same effective per-rank chunk.
2. `disable_piecewise_cuda_graph=True` forced under DP-attn — baseline gets piecewise graphs by default. Disable for both: `--disable-piecewise-cuda-graph`.
3. `schedule_conservativeness` auto×0.3 — if studying scheduler, normalize manually.
4. MoE dispatch backend differs — pin `--moe-a2a-backend` and `--ep-size` identical.

**Target metric**: ITL (inter-token latency) at concurrency where KV is stressed; throughput at saturated CONC.

**Disagg DP A/B** — feasible (SA matrix already has it), but requires multi-node. SA's most-tested config: decode-only DP-attn (DP-off prefill, DP-on decode).

---

## 3. ATOM (MI355X)

### 3.1 Config surface

| Knob | Source | Default |
|---|---|---|
| `--enable-dp-attention` CLI | `arg_utils.py:132`, `EngineArgs.enable_dp_attention` | False |
| `Config.enable_dp_attention` | `config.py:764` (also in `compute_hash`) | False |
| `--data-parallel-size / -dp` | `arg_utils.py:69` → `ParallelConfig.data_parallel_size` | 1 |
| `--enable-expert-parallel` | `arg_utils.py:120` → `Config.enable_expert_parallel` | False |
| Env `ATOM_DP_SIZE` | `envs.py:26-30` | 1 |
| Env `ATOM_DP_RANK` / `ATOM_DP_RANK_LOCAL` | `envs.py` | 0 |
| Env `ATOM_DP_MASTER_IP` / `ATOM_DP_MASTER_PORT` | `envs.py` | 127.0.0.1 / 29500 |

### 3.2 Config-takes-effect mechanism

**CRITICAL CORRECTION** to `dp_apple2apple_research.md` claim "ATOM determines DP from model topology automatically":

→ **FALSE.** No topology-driven detection. `enable_dp_attention` defaults to False and stays False unless `--enable-dp-attention` CLI flag is passed.

`DP_ATTENTION` (bare, no `ATOM_` prefix) is **NOT in `atom/utils/envs.py`** — it has zero effect on ATOM. SA CI scripts:
1. Declare `DP_ATTENTION` as required env in `check_env_vars`
2. Echo it at startup
3. **Never pass it to** `python3 -m atom.entrypoints.openai_server`
4. `benchmark_lib.sh:633` writes it into result `meta_env.json` purely as a metadata tag
5. `benchmark-tmpl.yml:139` uses it in `RESULT_FILENAME=..._dpa$DP_ATTENTION_...` to avoid file collisions when sweeping

When `--enable-dp-attention` IS passed, `CoreManager.__init__()` (`engine_core_mgr.py:32-40`):
```python
self.local_engine_count = config.tensor_parallel_size * config.parallel_config.data_parallel_size
config.parallel_config.data_parallel_size = self.local_engine_count
config.tensor_parallel_size = 1
```
**Flattens TP→DP**: `-tp 8 --enable-dp-attention` becomes 8 single-GPU DP ranks, each doing attention independently. Logs: `"Enable dp attention, using N data parallel ranks"`.

DSR1 ATOM CI never enables this → all DSR1 MI355X ATOM runs use **pure TP attention with NCCL AllReduce**.

aiter integration (`topK.py:31-33`): `if dp_size>1 and mori-available and config.enable_dp_attention: return False`. aiter does NOT have its own DP knob; it consumes `data_parallel_size`/`data_parallel_rank` from ATOM config via `init_dist_env()` (`model_runner.py:529-535`).

### 3.3 Runtime verification

```bash
# 1. Definitive check: is DP-attn flag actually passed?
cat /proc/$(pgrep -f atom.entrypoints.openai_server)/cmdline | tr '\0' ' ' | grep -o 'enable-dp-attention'
# In ATOM CI: empty (= DP-attn OFF)

# 2. Log signature for DP-attn ON
grep "Enable dp attention" $LOG       # present only if --enable-dp-attention
grep "ModelRunner rank=" $LOG | head  # always; check dp_rank_local for all ranks

# 3. EngineCore count
grep "EngineCore fully initialized" $LOG | wc -l
# = local_engine_count = tp * dp_size_after_flatten
# Pure TP=8: usually 1 EngineCore (managing 8 ranks)
# DP-attn flatten of tp=8: 8 EngineCores

# 4. Result-file label mismatch warning
# meta_env.json may show dp_attention=true (from CI env) while ATOM was actually pure-TP
cat *.json | jq .dp_attention
# Reflects DP_ATTENTION env, NOT actual ATOM runtime state — DON'T trust it for ATOM
```

**No** `/config` HTTP endpoint exists in `atom.entrypoints.openai_server`. Endpoints: `/v1/{chat,completions,models}`, `/health`, `/start_profile`, `/stop_profile`. Server log is the only source of truth.

### 3.4 A/B methodology

**Single-node ATOM DP-attn A/B** — requires a CI script change. Currently impossible to A/B from outside without modifying the launch line. Minimum diff:

```bash
# Current ATOM CI (DP-attn always OFF)
python3 -m atom.entrypoints.openai_server -tp 8 ...

# DP-attn ON variant (would require new CI script or override hook)
python3 -m atom.entrypoints.openai_server -tp 8 --enable-dp-attention ...
```

When enabled with `-tp 8`: ATOM flattens to 8 single-GPU DP ranks. Memory layout changes drastically:
- Off (TP=8): each GPU = 1/8 attention weights
- On (DP=8): each GPU = full attention weights replicated

→ **Total VRAM goes UP** under DP-attn on ATOM (replicated weights). Verify before benchmarking — may not even fit.

---

## 4. Cross-framework A/B comparison

### 4.1 Question to answer first

What are you actually comparing? Pick ONE:

| Question | Required setup |
|---|---|
| "Does DP-attn help on this hardware regardless of framework?" | DP-on vs DP-off **within each framework**, then compare deltas across frameworks |
| "Which framework has the best DP-attn implementation?" | DP-on cells across frameworks at matched topology |
| "Which framework is fastest overall?" | Each framework's own optimal config (no DP-attn alignment needed) |

The first two require **DP-on cells exist for all compared frameworks**. Today on MI355X this means **adding** DP-attn to ATOM CI (currently absent) and **adding** DP-attn to single-node sglang CI (currently absent — only disagg has it).

### 4.2 Matched-topology recipe

For a fair "DP-attn helps?" comparison on MI355X 8×GPU:

| Variable | TRT-LLM (B200) | SGLang (MI355X) | ATOM (MI355X) |
|---|---|---|---|
| Hardware | 8×B200 | 8×MI355X | 8×MI355X |
| Model | DSR1-fp4-NVFP4 | DSR1-fp4-mxfp4 | DSR1-fp4-mxfp4 |
| TP | 8 | 8 | 8 |
| DP-attn ON config | `enable_attention_dp: true` + pinned moe/graph/mtp | `--data-parallel-size 8 --enable-dp-attention --enable-dp-lm-head` | `--enable-dp-attention` (TP→DP flatten) |
| DP-attn OFF config | drop `enable_attention_dp` | drop both flags, keep `--data-parallel-size 1` | (default — drop the flag) |
| Verification | `Overriding attention_dp_config` log | `/get_server_info`.enable_dp_attention | `Enable dp attention` log + cmdline grep |

**Cross-framework caveats** that no normalization can erase:
- **Model SKU differs**: NVFP4 (B200 native) vs mxfp4 (AMD/atom). Even at "same accuracy", quantization details affect compute pattern.
- **MoE backend**: TRT-LLM uses CUTLASS/TRTLLM/DEEPGEMM; sglang ROCm uses MORI; ATOM uses aiter. These have different all-to-all latency profiles.
- **CUDA graph implementation**: TRT-LLM has piecewise-with-padding; sglang has piecewise (disabled under DP-attn); ATOM has its own graph manager.
- **KV-cache layout**: paged-attention details differ per framework.

→ Cross-framework numbers are **directional, not apples-to-apples**. Always report absolute throughput and DP-on/DP-off ratio separately per framework. Don't normalize across frameworks unless you've controlled for model SKU and MoE backend.

### 4.3 Suggested first experiment

To validate the methodology before any sweep:

1. **Pick TRT-LLM single config** (B200, DSR1-fp4, tp=8 ep=8 conc=128). Run DP-on vs DP-off with all 4 confounders pinned. Verify the verification checklist passes (Overriding warning, per-rank divergence, etc.). Expect: DP-on faster on decode TPOT, slightly slower on prefill.
2. **Add DP-attn to one MI355X sglang script** (modify `dsr1_fp4_mi355x.sh` to take `DP_ATTENTION=true` and pass `--enable-dp-attention --data-parallel-size $TP --enable-dp-lm-head`). Run same A/B at matched topology.
3. Compare: does DP-attn improvement on B200 trt match DP-attn improvement on MI355X sglang? Same direction? Same magnitude? If wildly different → framework or hardware effect. If similar → DP-attn benefit is robust.
4. ATOM is **last** — needs CI script change to even produce a DP-on cell. Defer until steps 1-3 establish the methodology works.

---

## Source citations

**TRT-LLM**:
- Config schema: `/home/kqian/TensorRT-LLM/tensorrt_llm/llmapi/llm_args.py` (lines 471, 475, 479, 484, 1887, 2218, 2476-2487, 3431, 3454, 3461)
- Attention layer: `/home/kqian/TensorRT-LLM/tensorrt_llm/_torch/modules/attention.py:433-450`
- KV mgr: `/home/kqian/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/resource_manager.py:304, 823`
- Scheduler: `/home/kqian/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/py_executor.py:334-339, 876-886, 2619-2627, 2747-2788`
- CUDA graph: `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py:250-260`
- DSV3 model: `tensorrt_llm/_torch/models/modeling_deepseekv3.py:237`
- MTP: `tensorrt_llm/_torch/speculative/mtp.py:1090`

**SGLang** (public github paths; not local):
- `python/sglang/srt/server_args.py` — ServerArgs schema, _handle_data_parallelism
- `python/sglang/srt/managers/data_parallel_controller.py:85-90, 174` — launch branch
- `python/sglang/srt/managers/scheduler.py:195-201, 304, 1979` — DP world info
- `python/sglang/srt/layers/dp_attention.py:41-74` — initialize_dp_attention
- `python/sglang/srt/models/deepseek_v2.py:1032, 1291` — model integration
- `python/sglang/srt/layers/attention/aiter_backend.py:208, 237, 255, 274` — ROCm DP path
- `python/sglang/srt/layers/attention/attention_registry.py` — backend registry

**ATOM**:
- CLI: `/home/kqian/ATOM/atom/model_engine/arg_utils.py:69, 120, 132`
- Config: `/home/kqian/ATOM/atom/config.py:663-671, 764, 894`
- Env vars: `/home/kqian/ATOM/atom/utils/envs.py:26-30`
- TP→DP flatten: `/home/kqian/ATOM/atom/model_engine/engine_core_mgr.py:32-40`
- Model runner: `/home/kqian/ATOM/atom/model_engine/model_runner.py:468, 503-504, 529-535`
- MoE config: `/home/kqian/ATOM/atom/model_ops/moe.py:101-114`
- aiter integration: `/home/kqian/ATOM/atom/model_ops/topK.py:31-33`
- Docs: `/home/kqian/ATOM/docs/{distributed_guide,configuration_guide}.md`

**SA CI**:
- TRT scripts: `/home/kqian/InferenceX/benchmarks/single_node/dsr1_fp{4,8}_b200_trt{,_mtp}.sh`
- sglang single-node: `/home/kqian/InferenceX/benchmarks/single_node/dsr1_fp{4,8}_{b200,mi355x}{,_mtp}.sh`
- sglang disagg: `/home/kqian/InferenceX/benchmarks/multi_node/dsr1_fp{4,8}_mi355x_sglang-disagg.sh`
- ATOM scripts: `/home/kqian/InferenceX/benchmarks/single_node/dsr1_fp{4,8}_mi355x_atom{,_mtp}.sh`
- Helpers: `/home/kqian/InferenceX/benchmarks/.../{benchmark_lib.sh,amd_utils/server.sh,env.sh}`
- Models config: `/home/kqian/InferenceX/.github/configs/models.yaml`
- Matrix: `/home/kqian/InferenceX/.github/configs/{nvidia,amd}-master.yaml`
- Workflow tmpl: `/home/kqian/InferenceX/.github/workflows/benchmark-tmpl.yml:139`
