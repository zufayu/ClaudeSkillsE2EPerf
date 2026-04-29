#!/usr/bin/env bash
# Seeded from b200_trt.sh (Blackwell shares MoE/CUDA-graph tuning).
# B300 cross-check 2026-04-29: TRTLLM MoE backend +21-41% vs CUTLASS on
# GPT-OSS-120B FP4 TP=1 chat — same defaults as B200 apply.
# Re-tune here if any B300-specific point regresses vs B200.

compute_adaptive_params() {
    local quant=$1 isl=$2 osl=$3 conc=$4 dp_attn=$5 ep_size=$6 has_mtp=$7

    # --- Defaults ---
    MOE_BACKEND="TRTLLM"
    PIECEWISE_CUDA_GRAPHS="false"
    CUDA_GRAPH_MAX_BATCH_SIZE=$conc
    KV_CACHE_FREE_MEM_FRACTION=0.8
    DELAY_BATCHING="false"
    MTP_LAYERS=0
    ENABLE_CONFIGURABLE_MOE_FLAG=""

    if [[ "$has_mtp" == "true" ]]; then
        MTP_LAYERS=3
    fi

    # --- FP4 logic (from dsr1_fp4_b200_trt.sh / dsr1_fp4_b200_trt_mtp.sh) ---
    if [[ "$quant" == "fp4" ]]; then
        if [[ "$has_mtp" == "false" ]]; then
            # fp4-throughput: from dsr1_fp4_b200_trt.sh
            if [[ "$dp_attn" == "true" ]]; then
                MOE_BACKEND="CUTLASS"
                CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc < 4 ? conc : conc / 4 ))
            fi
            # Piecewise CUDA graphs for EP=8 + 1k/1k
            if [[ "$isl" == "1024" && "$osl" == "1024" ]]; then
                if [[ "$TP" == "8" && "$ep_size" == "8" ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                fi
            fi
        else
            # fp4-latency: from dsr1_fp4_b200_trt_mtp.sh
            if [[ "$dp_attn" == "true" ]]; then
                CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc < 4 ? conc : conc / 4 ))
                MOE_BACKEND="CUTLASS"
                MTP_LAYERS=1
            fi
            # Piecewise for specific conc values
            if [[ "$isl" == "1024" && "$osl" == "1024" ]]; then
                if [[ $conc == 32 || $conc == 64 ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                elif [[ $conc == 128 && "$dp_attn" == "false" ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                fi
            elif [[ "$isl" == "1024" && "$osl" == "8192" ]]; then
                if [[ $conc == 64 ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                fi
            fi
        fi

    # --- FP8 logic (from dsr1_fp8_b200_trt.sh / dsr1_fp8_b200_trt_mtp.sh) ---
    elif [[ "$quant" == "fp8" ]]; then
        if [[ "$has_mtp" == "false" ]]; then
            # fp8-throughput: from dsr1_fp8_b200_trt.sh
            if [[ "$isl" == "1024" && "$osl" == "1024" ]]; then
                if [[ $conc -ge 128 ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                    DELAY_BATCHING="true"
                    KV_CACHE_FREE_MEM_FRACTION=0.7
                elif [[ $conc -ge 64 ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                    DELAY_BATCHING="true"
                fi
            elif [[ "$isl" == "1024" && "$osl" == "8192" ]]; then
                if [[ $conc -ge 256 ]]; then
                    CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc / 8 ))
                    MOE_BACKEND="DEEPGEMM"
                    KV_CACHE_FREE_MEM_FRACTION=0.7
                elif [[ $conc -ge 128 ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                fi
            elif [[ "$isl" == "8192" && "$osl" == "1024" ]]; then
                if [[ $conc -ge 64 ]]; then
                    PIECEWISE_CUDA_GRAPHS="true"
                fi
                if [[ "$TP" == "4" ]]; then
                    KV_CACHE_FREE_MEM_FRACTION=0.75
                fi
            fi
            if [[ "$dp_attn" == "true" ]]; then
                MOE_BACKEND="CUTLASS"
            fi
        else
            # fp8-latency: from dsr1_fp8_b200_trt_mtp.sh
            PIECEWISE_CUDA_GRAPHS="true"
            if [[ "$dp_attn" == "true" ]]; then
                MOE_BACKEND="DEEPGEMM"
                PIECEWISE_CUDA_GRAPHS="false"
                CUDA_GRAPH_MAX_BATCH_SIZE=$(( conc < 8 ? conc : conc / 8 ))
                KV_CACHE_FREE_MEM_FRACTION=0.7
                ENABLE_CONFIGURABLE_MOE_FLAG="1"
                MTP_LAYERS=1
            fi
            # Disable PW CUDA for narrow conc
            if [[ "$isl" == "1024" && "$osl" == "1024" ]]; then
                if [[ $conc -le 4 ]]; then
                    PIECEWISE_CUDA_GRAPHS="false"
                fi
            elif [[ "$isl" == "1024" && "$osl" == "8192" ]]; then
                if [[ $conc -le 8 ]]; then
                    PIECEWISE_CUDA_GRAPHS="false"
                fi
            elif [[ "$isl" == "8192" && "$osl" == "1024" ]]; then
                if [[ $conc -le 16 ]]; then
                    PIECEWISE_CUDA_GRAPHS="false"
                fi
            fi
        fi
    fi
}
