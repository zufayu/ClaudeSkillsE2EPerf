# Tomorrow's MI355X 12-point refresh — dispatch plan

User intent (2026-04-25 night): refresh MI355X data on the **latest ATOM image** (or last-known-good), 12 points total, **without uploading results to main** (avoid pollution).

## 12-point grid

- **Quant**: mxfp4 only (atom-mtp uses MXFP4 model `amd/DeepSeek-R1-0528-MXFP4` for MTP, MXFP4-Preview for non-MTP)
- **TP=1, EP=4** (custom; not in SA CI matrix — see `dp_apple2apple_research.md`)
- **Scenarios**: 3 ISL/OSL pairs — chat (1k/1k), reasoning (1k/8k), summarize (8k/1k) [`sa_bench_mi355x.sh` scenario aliases]
- **Concurrency**: c=4 and c=64
- **MTP**: on / off
- = 3 × 2 × 2 = **12 runs**

| # | Scenario | CONC | MTP | configs |
|--:|---|--:|---|---|
| 1 | chat | 4 | off | mxfp4-throughput |
| 2 | chat | 4 | on | mxfp4-latency |
| 3 | chat | 64 | off | mxfp4-throughput |
| 4 | chat | 64 | on | mxfp4-latency |
| 5 | reasoning | 4 | off | mxfp4-throughput |
| 6 | reasoning | 4 | on | mxfp4-latency |
| 7 | reasoning | 64 | off | mxfp4-throughput |
| 8 | reasoning | 64 | on | mxfp4-latency |
| 9 | summarize | 4 | off | mxfp4-throughput |
| 10 | summarize | 4 | on | mxfp4-latency |
| 11 | summarize | 64 | off | mxfp4-throughput |
| 12 | summarize | 64 | on | mxfp4-latency |

> ✅ Verified: `mi355x_atom_bench.yml:11-12` choices `mxfp4-throughput, mxfp4-latency` map cleanly to MTP-off / MTP-on. No new config string needed.

## Required workflow change BEFORE dispatch (no-upload safeguard)

`mi355x_atom_bench.yml` currently always runs `Commit results` step which calls `ci_commit_results` → pushes to main. To honor "不上传污染main", add a boolean input gating the commit step:

```yaml
# add to inputs:
commit_results:
  description: "Push results to main (uncheck for local-only runs)"
  type: boolean
  default: true

# change Commit results step:
- name: Commit results
  if: always() && inputs.commit_results
  ...
```

**Do NOT make this change tonight** — wait for user to confirm tomorrow. The change is small and reversible.

## Image selection

Per `feedback_docker_recreate_not_inplace_update.md` (2026-04-25 update):
1. Try `atom-dev:latest` (or daily-built tag) first
2. If broken → roll back to most recent known-good tag
3. **Never** debug on an old image lying around

Workflow input `container` only chooses between `zufa_atom` (ROCm7.1.1) and `zufa_atom2` (ROCm7.2.1). Image tag itself is resolved by `ci_get_image` from platform config — verify it pulls latest before dispatching.

## Dispatch commands (DO NOT RUN TONIGHT)

After workflow patch + `commit_results=false`:

```bash
# Example for run #1 (chat c=4 MTP-off)
gh workflow run mi355x_atom_bench.yml \
  -f container=zufa_atom2 \
  -f configs=mxfp4-throughput \
  -f ep=4 -f tp=1 \
  -f scenario=chat -f concurrency=4 \
  -f commit_results=false

# Loop for all 12 — pseudocode:
for SCEN in chat reasoning summarize; do
  for C in 4 64; do
    for MTP_CFG in mxfp4-throughput mxfp4-throughput-mtp; do
      gh workflow run mi355x_atom_bench.yml \
        -f container=zufa_atom2 -f configs=$MTP_CFG \
        -f ep=4 -f tp=1 -f scenario=$SCEN -f concurrency=$C \
        -f commit_results=false
    done
  done
done
```

**Concurrency caveat**: MI355X node has slot-a/slot-b parallel TP=4 architecture (per `project_mi355x_parallel_slots.md`). With TP=1 EP=4, each run uses 4 GPUs → can run 2 in parallel by slot. Workflow currently doesn't expose slot selection — sequential dispatch will queue them.

## Outstanding open questions for tomorrow

### BLOCKER 1 — MTP config name was wrong in my earlier draft

`sa_bench_mi355x.sh:286-287` shows MTP-on is selected by config_name=="latency", **not** by a `*-mtp` suffix:
- MTP **off** → `--configs mxfp4-throughput`
- MTP **on**  → `--configs mxfp4-latency`  (sets MTP_LAYERS=3, no other deltas in `compute_mi355x_params`)

So the workflow's existing `configs` choice (`mxfp4-throughput`/`mxfp4-latency`) already covers both. **No workflow patch needed for MTP.**

### BLOCKER 2 — `tp=1, ep=4` is NOT representable in current pipeline

`sa_bench_mi355x.sh:70` exposes `--ep` as a **pure flag** (no value):
```bash
--ep)  EXPERT_PARALLEL="true"; shift 1 ;;
```
which only flips on `--enable-expert-parallel` for ATOM. ATOM then uses EP size = TP by default. The wrapper has no `--ep-size N` knob.

With `--tp 1 --ep`, the resulting world size is **1 GPU**, not 4. To genuinely run "tp=1, ep=4" we need either:
- (a) Script extension to pass `--ep-size 4` through to ATOM (need to confirm ATOM supports separating EP world size from TP), OR
- (b) User clarification — possible they meant `tp=4, ep=1` (the SA-CI default for MI355X fp4) or `tp=4, ep=4` (=4 GPUs, full TP+EP)

**Do not dispatch until this is resolved.** Wrong interpretation costs ~12 × ~10 min runs and produces non-comparable data.

### Other open items

3. **Decide image policy**: latest atom-dev or pin to a known-good tag for reproducibility
4. **Read tomorrow** ~ `dp_apple2apple_research.md` for context on why tp=1/ep=4 is non-SA-CI

## Tonight's status

- Research complete (`dp_apple2apple_research.md`)
- Plan drafted (this file)
- **No benchmarks dispatched** — blocked on tp/ep interpretation ambiguity (BLOCKER 2)
- **No workflow patches applied** — `commit_results=false` would be a 5-line change but moot until BLOCKER 2 is resolved
- Memory left untouched for these in-flight items (per "don't save in-progress task state")
