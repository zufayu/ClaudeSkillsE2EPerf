#!/usr/bin/env bash
# Default (no-op) adaptive params for platforms without specific tuning.
# Sets safe defaults; override by creating configs/adaptive/{platform}_trt.sh

compute_adaptive_params() {
    local quant=${1:-fp8} isl=${2:-1024} osl=${3:-1024} conc=${4:-64}
    local dp_attn=${5:-false} ep_size=${6:-1} has_mtp=${7:-false}

    MOE_BACKEND="CUTLASS"
    PIECEWISE_CUDA_GRAPHS="false"
    CUDA_GRAPH_MAX_BATCH_SIZE=$conc
    KV_CACHE_FREE_MEM_FRACTION=0.8
    DELAY_BATCHING="false"
    MTP_LAYERS=0

    if [[ "$has_mtp" == "true" ]]; then
        MTP_LAYERS=3
    fi
}
