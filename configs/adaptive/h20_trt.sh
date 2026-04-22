#!/usr/bin/env bash
# H20 TRT-LLM adaptive server parameters
compute_adaptive_params() {
    local isl=$1 osl=$2 conc=$3 dp_attn=$4 ep_size=$5 has_mtp=$6

    # --- Defaults for H20 ---
    MOE_BACKEND="CUTLASS"
    PIECEWISE_CUDA_GRAPHS="false"
    CUDA_GRAPH_MAX_BATCH_SIZE=$conc
    KV_CACHE_FREE_MEM_FRACTION=0.8
    DELAY_BATCHING="false"
    MTP_LAYERS=0

    if [[ "$has_mtp" == "true" ]]; then
        MTP_LAYERS=3
    fi

    # H20 FP8 throughput (no MTP)
    if [[ "$has_mtp" == "false" ]]; then
        if [[ "$dp_attn" == "true" ]]; then
            MOE_BACKEND="CUTLASS"
            CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc < 4 ? conc : conc / 4 ))
        fi
        # Enable piecewise CUDA graphs for high concurrency chat
        if [[ "$isl" == "1024" && "$osl" == "1024" && $conc -ge 32 ]]; then
            PIECEWISE_CUDA_GRAPHS="true"
        fi
        # Reasoning at high conc: reduce KV fraction to avoid OOM
        if [[ "$osl" == "8192" && $conc -ge 32 ]]; then
            KV_CACHE_FREE_MEM_FRACTION=0.7
        fi
        # Summarize (8K input): needs more headroom
        if [[ "$isl" == "8192" && $conc -ge 32 ]]; then
            PIECEWISE_CUDA_GRAPHS="true"
            KV_CACHE_FREE_MEM_FRACTION=0.75
        fi

    # H20 FP8 latency (MTP)
    else
        if [[ "$dp_attn" == "true" ]]; then
            MTP_LAYERS=1
            MOE_BACKEND="CUTLASS"
            CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc < 4 ? conc : conc / 4 ))
            KV_CACHE_FREE_MEM_FRACTION=0.7
        fi
        # Piecewise for medium concurrency
        if [[ "$isl" == "1024" && "$osl" == "1024" ]]; then
            if [[ $conc -ge 16 && $conc -le 64 ]]; then
                PIECEWISE_CUDA_GRAPHS="true"
            fi
        fi
    fi
}
