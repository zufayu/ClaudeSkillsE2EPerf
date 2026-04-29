#!/usr/bin/env bash
# =============================================================================
# CI Library — Platform-aware execution abstraction
#
# Source this on the runner (jumphost or GPU node).
# All functions read env vars: NODE, CONTAINER, REPO, EXEC_MODE
# which come from platform env files + workflow inputs.
#
# Usage in workflow steps:
#   source scripts/ci_lib.sh
#   ci_load_platform b200
#   export NODE="${{ inputs.node }}" CONTAINER="${{ inputs.container }}"
#   ci_sync
#   ci_exec "nvidia-smi"
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Platform loading
# -----------------------------------------------------------------------------

ci_load_platform() {
  local platform=$1
  local dir
  dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../configs/platforms" && pwd)"
  if [[ ! -f "$dir/${platform}.env" ]]; then
    echo "ERROR: Platform config not found: $dir/${platform}.env"
    return 1
  fi
  source "$dir/${platform}.env"
}

# -----------------------------------------------------------------------------
# Execution abstraction
# -----------------------------------------------------------------------------

# Execute command inside container (handles ssh+docker / docker / podman)
ci_exec() {
  local cmd="$1"
  case "${EXEC_MODE:-}" in
    ssh+docker)
      ssh "$NODE" "docker exec $CONTAINER bash -c '${cmd}'"
      ;;
    docker)
      docker exec "$CONTAINER" bash -c "${cmd}"
      ;;
    podman)
      podman exec "$CONTAINER" bash -c "${cmd}"
      ;;
    *)
      echo "ERROR: Unknown EXEC_MODE='${EXEC_MODE:-}'"
      return 1
      ;;
  esac
}

# Execute command on the host (not in container)
ci_exec_host() {
  local cmd="$1"
  case "${EXEC_MODE:-}" in
    ssh+docker)
      ssh "$NODE" "${cmd}"
      ;;
    docker|podman)
      eval "${cmd}"
      ;;
    *)
      echo "ERROR: Unknown EXEC_MODE='${EXEC_MODE:-}'"
      return 1
      ;;
  esac
}

# Execute container inspect (runtime-aware)
ci_inspect() {
  local fmt="$1"
  case "${EXEC_MODE:-}" in
    ssh+docker)
      ssh "$NODE" "docker inspect $CONTAINER --format '${fmt}'" 2>/dev/null || echo unknown
      ;;
    docker)
      docker inspect "$CONTAINER" --format "${fmt}" 2>/dev/null || echo unknown
      ;;
    podman)
      podman inspect "$CONTAINER" --format "${fmt}" 2>/dev/null || echo unknown
      ;;
  esac
}

# -----------------------------------------------------------------------------
# Git operations
# -----------------------------------------------------------------------------

# Sync repo to origin/main (hard reset — the only reliable strategy)
# `checkout -B main` ensures we stay on main branch (not detached HEAD), so
# subsequent commits via ci_commit_results land on main and can be pushed.
ci_sync() {
  # Clean any leftover rebase state from a prior interrupted ci_commit_results
  # — `git pull --rebase` writes .git/rebase-merge and refuses to retry until
  # cleared, so any subsequent workflow gets stuck at fatal "rebase in progress".
  ci_exec_host "cd $REPO && (git rebase --abort 2>/dev/null; rm -rf .git/rebase-merge .git/rebase-apply 2>/dev/null; true)"
  ci_exec_host "cd $REPO && git fetch origin && git checkout -B main origin/main"
  ci_exec_host "cd $REPO && git log --oneline -3"
}

# -----------------------------------------------------------------------------
# GPU / container checks
# -----------------------------------------------------------------------------

ci_verify_gpu() {
  ci_exec "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -8"
}

# Get container image name → sets IMAGE env var
ci_get_image() {
  IMAGE=$(ci_inspect '{{.Config.Image}}')
  export IMAGE
  echo "IMAGE=$IMAGE"
}

# Print image provenance (digest + ID + size + framework commits) so every
# bench/profiling artifact has reproducible "what was running" metadata.
# Output is plain text suitable for prepending to server logs or saving as
# its own file. Adds ~2s; safe to call after ci_get_image().
ci_log_image_provenance() {
  echo "=== Image provenance ==="
  echo "Container:  ${CONTAINER:-<unset>}"
  echo "Image:      ${IMAGE:-<unset>}"
  ci_inspect '{{.Image}}'           | sed 's/^/Image ID:   /'
  ci_inspect '{{.Created}}'         | sed 's/^/Container created: /'
  # Image-level metadata (RepoDigests, Created, Size). Wrapped to tolerate
  # podman/docker inspect format diff and missing fields.
  ci_exec_host "docker image inspect ${IMAGE:-} --format '{{index .RepoDigests 0}}' 2>/dev/null \
                || podman image inspect ${IMAGE:-} --format '{{index .RepoDigests 0}}' 2>/dev/null \
                || echo '(digest unavailable)'" | sed 's/^/Digest:     /'
  ci_exec_host "docker image inspect ${IMAGE:-} --format '{{.Size}}' 2>/dev/null \
                || podman image inspect ${IMAGE:-} --format '{{.Size}}' 2>/dev/null \
                || echo unknown" | awk '{ if ($1+0>0) printf "Size:       %.1f GB\n", $1/1024/1024/1024; else print "Size:       " $1 }'
  # Per-framework commit fingerprints (best-effort; missing dirs print "n/a")
  case "${IMAGE:-}" in
    *atom*)
      ci_exec "cd /app/atom 2>/dev/null && echo \"ATOM commit:  \$(git rev-parse HEAD 2>/dev/null) (\$(git log -1 --format=%cd --date=short 2>/dev/null))\" || echo 'ATOM commit:  n/a (no /app/atom)'"
      ci_exec "cd /app/aiter-test 2>/dev/null && echo \"aiter commit: \$(git rev-parse HEAD 2>/dev/null)\" || echo 'aiter commit: n/a (no /app/aiter-test)'"
      ;;
    *sglang*)
      ci_exec "python3 -c 'import sglang; print(\"sglang ver:  \", sglang.__version__)' 2>/dev/null || echo 'sglang ver:   n/a'"
      ci_exec "python3 -c 'import torch; print(\"torch ver:   \", torch.__version__)' 2>/dev/null || echo 'torch ver:    n/a'"
      ;;
  esac
  echo "========================="
}

# Kill residual GPU processes (MI355X needs this between runs)
# CRITICAL: Never use pkill -f — it matches its own cmdline and kills itself (exit 143).
#   See: 9564e80, 7a8b506, 53c5b8f, 482c67e, a71cff4 (5 fix commits for this bug)
#   Use pgrep + grep -v $$ + xargs kill instead.
ci_kill_gpu_procs() {
  case "${EXEC_MODE:-}" in
    podman)
      podman exec "$CONTAINER" bash -c \
        'for pat in "python" "vllm" "atom.*serve" "model_executor"; do pgrep -f "$pat" 2>/dev/null | grep -v $$ | xargs -r kill -9 2>/dev/null; done; true'
      pgrep -f "python.*vllm\|python.*atom" 2>/dev/null | grep -v $$ | xargs -r kill -9 2>/dev/null || true
      sleep 5
      ;;
    ssh+docker|docker)
      ci_exec 'for pat in "sglang" "trtllm"; do pgrep -f "$pat" 2>/dev/null | grep -v $$ | xargs -r kill -9 2>/dev/null; done; true'
      sleep 3
      ;;
  esac
}

# -----------------------------------------------------------------------------
# Version / env_tag detection
# -----------------------------------------------------------------------------

# Detect env_tag (includes framework abbreviation)
# TRT → post2/post3/rc10,  SGLang → sglang059,  ATOM → rocm722
ci_detect_env_tag() {
  local framework=$1
  case "$framework" in
    trt)
      local ver
      ver=$(ci_exec "pip show tensorrt-llm 2>/dev/null | grep ^Version | cut -d' ' -f2" 2>/dev/null || true)
      if [[ -z "$ver" ]]; then
        ver=$(echo "${IMAGE:-}" | grep -oP 'release:\K[0-9a-z.]+' || echo unknown)
      fi
      case "$ver" in
        *post2*) echo "post2" ;;
        *post3*) echo "post3" ;;
        *rc10*)  echo "rc10" ;;
        *)       echo "$(echo "$ver" | sed 's/[^a-zA-Z0-9]/_/g')" ;;
      esac
      ;;
    sglang)
      local ver
      ver=$(ci_exec "python3 -c 'import sglang; print(sglang.__version__)'" 2>/dev/null || echo "0.0.0")
      echo "sglang$(echo "$ver" | tr -d '.')"
      ;;
    atom)
      local ver
      ver=$(ci_exec "cat /opt/rocm/.info/version 2>/dev/null || echo unknown" 2>/dev/null | tr -d '\n')
      echo "rocm$(echo "$ver" | tr -d '.')"
      ;;
    *)
      echo "unknown"
      ;;
  esac
}

# -----------------------------------------------------------------------------
# Config parsing & result dir
# -----------------------------------------------------------------------------

# Parse configs string → sets QUANT, MODE, and MTP env vars
# e.g. "fp4-throughput" → QUANT=fp4, MODE=throughput, MTP=mtp0
#      "mxfp4-latency"  → QUANT=mxfp4, MODE=latency, MTP=mtp3
ci_parse_configs() {
  local configs=$1
  # Handle multi-word quant types like mxfp4
  MODE="${configs##*-}"
  QUANT="${configs%-*}"
  case "$MODE" in
    throughput) MTP="mtp0" ;;
    latency)    MTP="mtp3" ;;
    *)          echo "ERROR: Unknown mode '$MODE' in configs '$configs'"; return 1 ;;
  esac
  export QUANT MODE MTP
}

# Generate result directory path
ci_result_dir() {
  local platform=$1 quant=$2 mtp=$3 ep=$4 tp=$5 env_tag=$6 suffix=${7:-}
  echo "results/${platform}_dsr_${quant}/${platform}_dsr_${quant}_${mtp}_ep${ep}_tp${tp}_${env_tag}${suffix}"
}

# -----------------------------------------------------------------------------
# Result commit
# -----------------------------------------------------------------------------

# Fix file ownership (container uid != host uid)
ci_fix_ownership() {
  local result_dir=$1
  ci_exec "chown -R \$(stat -c %u:%g $REPO/.git) $REPO/${result_dir} 2>/dev/null; true" 2>/dev/null || true
}

# Commit results and push
ci_commit_results() {
  local result_dir=$1
  local message=$2

  ci_fix_ownership "$result_dir"

  ci_exec_host "cd $REPO && git config user.email '$CI_USER_EMAIL' && git config user.name '$CI_USER_NAME'"

  # Stage result files (exclude large trace binaries). Stash any tracked-but-unstaged
  # modifications first so `git pull --rebase` doesn't refuse with "unstaged changes"
  # — bench writes summary.md / regression_report.txt which are tracked.
  ci_exec_host "cd $REPO && git reset HEAD 2>/dev/null; true"
  ci_exec_host "cd $REPO && git stash push --keep-index -m 'ci-pre-commit-stash' -- ${result_dir} 2>/dev/null || true"
  ci_exec_host "cd $REPO && for ext in json log yml md csv xlsx txt; do git add -f ${result_dir}/*.\$ext 2>/dev/null || true; done"
  ci_exec_host "cd $REPO && git diff --cached --name-only"

  ci_exec_host "cd $REPO && if ! git diff --cached --quiet; then git commit -m '${message}

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>' && git pull --rebase origin main && git push; else echo 'Nothing to commit'; fi"
  ci_exec_host "cd $REPO && git stash drop 2>/dev/null || true"
}

# Snapshot result_dir to shared SFS so data survives node reallocation.
# /home is per-node virtiofs (NOT shared); /SFS-aGqda6ct/dynamo is shared and writable.
# Keeps last few snapshots; safe to call multiple times in same run.
ci_snapshot_results() {
  local result_dir=$1
  local archive_root="${SFS_ARCHIVE_ROOT:-/SFS-aGqda6ct/dynamo/zufayu_e2e_results}"
  local run_tag="${GITHUB_RUN_ID:-local}_${GITHUB_RUN_ATTEMPT:-1}"
  local dest="$archive_root/$run_tag"
  ci_exec_host "mkdir -p '$dest' && rsync -a --exclude='*.gz' --exclude='*serialized*' '$REPO/$result_dir/' '$dest/' && echo 'Snapshot saved: $dest' && ls -lt '$dest' | head -10"
}

# -----------------------------------------------------------------------------
# Container recreate (cross-platform, framework-aware)
# -----------------------------------------------------------------------------
# Used by recreate_container.yml. Centralized here so that:
#   1) bench/profile workflows can also recreate before runs (one-stop).
#   2) Adding a new platform/framework = update one file.
#
# See feedback_workflow_consolidation.md and feedback_docker_recreate_not_inplace_update.md.

# Print platform-specific docker/podman run flags.
# Usage: FLAGS=$(ci_runtime_flags); $RUNTIME run -d --name $C $FLAGS $IMAGE sleep infinity
ci_runtime_flags() {
  case "${PLATFORM:-}" in
    mi355x)
      # Combined ROCm flags from mi355x_recreate_zufa_atom_baseline.yml (most complete).
      printf '%s ' \
        --security-opt seccomp=unconfined \
        --ipc=host \
        --network=host \
        --group-add keep-groups \
        --cap-add CAP_SYS_PTRACE \
        --device /dev/kfd \
        --device /dev/dri \
        --ulimit nproc=4194304:4194304 \
        --ulimit nofile=1048576:1048576 \
        -v /shared:/shared
      ;;
    b200)
      # From setup_b200.yml. SFS path is platform-pinned.
      printf '%s ' \
        --gpus all --ipc host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --shm-size=64g --cap-add=SYS_PTRACE \
        -v /home:/home -v /mnt:/mnt -v /data:/data \
        -v /SFS-aGqda6ct:/SFS-aGqda6ct
      ;;
    b300)
      # From b300_env_setup.yml.
      printf '%s ' \
        --gpus all --ipc=host --net=host \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        -v /home/zufa:/home/zufa \
        -v /SFS-aGqda6ct:/SFS-aGqda6ct \
        -v /home:/home \
        -w /home/zufa
      ;;
    *)
      echo "ERROR: ci_runtime_flags: unknown PLATFORM='${PLATFORM:-}'" >&2
      return 1
      ;;
  esac
}

# Resolve (platform, framework, image_tag) → image ref.
# Edit this table when new images are released.
# Usage: IMAGE=$(ci_resolve_image atom latest)
ci_resolve_image() {
  local framework=$1 image_tag=$2
  case "${PLATFORM:-}/${framework}/${image_tag}" in
    mi355x/atom/latest)      echo "docker.io/rocm/atom-dev:latest" ;;
    mi355x/atom/known-good)  echo "docker.io/rocm/atom-dev:nightly_202604201537" ;;
    b200/trt/latest)         echo "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10" ;;
    b200/trt/known-good)     echo "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10" ;;
    b200/sglang/latest)      echo "lmsysorg/sglang:v0.5.9-cu130" ;;
    b200/sglang/known-good)  echo "lmsysorg/sglang:v0.5.9-cu130" ;;
    b300/trt/latest)         echo "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10" ;;
    b300/trt/known-good)     echo "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10" ;;
    b300/sglang/latest)      echo "lmsysorg/sglang:v0.5.9-cu130" ;;
    b300/sglang/known-good)  echo "lmsysorg/sglang:v0.5.9-cu130" ;;
    *)
      echo "ERROR: ci_resolve_image: no mapping for PLATFORM=${PLATFORM:-} framework=${framework} image_tag=${image_tag}" >&2
      return 1
      ;;
  esac
}

# Recreate a container: pull (if requested) + rm -f + run -d sleep infinity.
# Inputs: $1=container name, $2=image ref, $3=pull (true/false, default true).
# Reads $PLATFORM and $EXEC_MODE from ci_load_platform.
ci_recreate_container() {
  local container=$1 image=$2 pull=${3:-true}
  local runtime
  case "${EXEC_MODE:-}" in
    podman)     runtime=podman ;;
    docker)     runtime=docker ;;
    ssh+docker) runtime=docker ;;
    *) echo "ERROR: ci_recreate_container: unknown EXEC_MODE='${EXEC_MODE:-}'" >&2; return 1 ;;
  esac

  local flags
  flags=$(ci_runtime_flags) || return 1

  if [[ "$pull" == "true" ]]; then
    echo "=== Pulling $image ==="
    ci_exec_host "$runtime pull '$image'"
  else
    echo "=== Skipping pull (pull=false) — using locally-cached $image ==="
  fi

  echo "=== Removing existing container '$container' (if any) ==="
  ci_exec_host "$runtime rm -f '$container' 2>/dev/null || true"

  echo "=== Creating fresh '$container' from $image ==="
  ci_exec_host "$runtime run -d --name '$container' $flags '$image' sleep infinity"
  sleep 3

  echo "=== Verify container is up ==="
  ci_exec_host "$runtime ps --filter name=^${container}\$ --format '{{.Names}} {{.Status}} {{.Image}}'"
}

# -----------------------------------------------------------------------------
# Dry-run mode (for local testing)
# -----------------------------------------------------------------------------

# When CI_DRY_RUN=1, print commands instead of executing
if [[ "${CI_DRY_RUN:-0}" == "1" ]]; then
  ci_exec()      { echo "[DRY-RUN ci_exec] $1"; }
  ci_exec_host() { echo "[DRY-RUN ci_exec_host] $1"; }
  ci_inspect()   { echo "[DRY-RUN ci_inspect] $1"; echo "dry-run-image:latest"; }
fi
