#!/usr/bin/env bash
# H200 TRT-LLM adaptive server parameters
# Renamed from compute_h200_params to compute_adaptive_params for unified interface
compute_adaptive_params() {
    local isl=$1 osl=$2 conc=$3 dp_attn=$4 has_mtp=$5

    # --- H200 defaults (from InferenceX dsr1_fp8_h200_trt*.sh) ---
    MOE_BACKEND="CUTLASS"
    CUDA_GRAPH_MAX_BATCH_SIZE=128
    KV_CACHE_FREE_MEM_FRACTION=0.75
    MTP_LAYERS=0
    MAX_BATCH_SIZE=$conc
    ALLOC_CONF_OVERRIDE=""

    if [[ "$has_mtp" == "true" ]]; then
        # MTP-3 default, MTP-1 when DP attention
        if [[ "$dp_attn" == "true" ]]; then
            MTP_LAYERS=1
            MAX_BATCH_SIZE=$(( conc / TP ))
            [[ $MAX_BATCH_SIZE -lt 1 ]] && MAX_BATCH_SIZE=1
        else
            MTP_LAYERS=3
            MAX_BATCH_SIZE=$conc
        fi

        # ISL=8192 + DP needs CUDA alloc config to avoid OOM
        if [[ "$isl" == "8192" && "$dp_attn" == "true" ]]; then
            ALLOC_CONF_OVERRIDE="max_split_size_mb:8192"
        fi
    fi
}
