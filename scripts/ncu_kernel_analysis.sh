#!/usr/bin/env bash
# =============================================================================
# Nsight Compute (ncu) Deep Kernel Analysis for TRT-LLM Inference
#
# Two modes:
#   targeted:   Profile a specific kernel by name (3 invocations)
#   discovery:  Profile all kernels in a window, rank by duration
#
# Outputs CSV with: kernel_name, duration_us, dram_pct, sm_pct, occupancy_pct
# and prints a diagnosis (memory-bound / compute-bound / latency-bound).
#
# Usage:
#   # Targeted: profile MoE GEMM kernel
#   bash scripts/ncu_kernel_analysis.sh \
#     --model /home/models/models--DeepSeek-R1-0528 \
#     --mode targeted --kernel-name "bmm_E2m1" \
#     --scenario chat --concurrency 32
#
#   # Discovery: find top 10 bottleneck kernels
#   bash scripts/ncu_kernel_analysis.sh \
#     --model /home/models/models--DeepSeek-R1-0528 \
#     --mode discovery --scenario chat --concurrency 32
#
# Prerequisites:
#   - NVIDIA GPU with ncu installed (Nsight Compute 2023.1+)
#   - trtllm-bench available
#   - Root or CAP_SYS_ADMIN for full metrics
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ======================== Defaults ============================================
MODEL=""
MODE="discovery"
KERNEL_NAME=""
SCENARIO="chat"
CONCURRENCY=32
QUANT="fp8"
CONFIG="throughput"
TP=8
EP=1
OUTPUT_DIR="./ncu_reports"
LAUNCH_SKIP=50
LAUNCH_COUNT=200
TARGETED_SKIP=10
TARGETED_COUNT=3
NUM_REQUESTS=100
ITER_RANGE="50-100"

# ======================== Argument Parsing ====================================
usage() {
    cat <<EOF
Usage: bash $(basename "$0") [options]

Nsight Compute deep kernel analysis for TRT-LLM inference.

Required:
  --model PATH            Model path

Options:
  --mode MODE             targeted | discovery [default: discovery]
  --kernel-name PATTERN   Kernel name/regex for targeted mode (required if targeted)
  --scenario SCENARIO     chat | reasoning | summarize [default: chat]
  --concurrency N         Request concurrency [default: 32]
  --quant QUANT           fp4 | fp8 [default: fp8]
  --config CONFIG         throughput | latency [default: throughput]
  --tp N                  Tensor parallelism [default: 8]
  --ep N                  Expert parallelism [default: 1]
  --output-dir DIR        Output directory [default: ./ncu_reports]
  --launch-skip N         Kernels to skip in discovery mode [default: 50]
  --launch-count N        Kernels to capture in discovery mode [default: 200]
  --num-requests N        Benchmark requests [default: 100]
  -h, --help              Show this help

Examples:
  # Discovery: find top bottleneck kernels
  bash $(basename "$0") --model /path/to/model --mode discovery

  # Targeted: deep-dive on specific MoE kernel
  bash $(basename "$0") --model /path/to/model \\
    --mode targeted --kernel-name "bmm_E2m1"

  # Targeted: analyze attention kernel
  bash $(basename "$0") --model /path/to/model \\
    --mode targeted --kernel-name "fmha"
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)          MODEL="$2"; shift 2 ;;
        --mode)           MODE="$2"; shift 2 ;;
        --kernel-name)    KERNEL_NAME="$2"; shift 2 ;;
        --scenario)       SCENARIO="$2"; shift 2 ;;
        --concurrency)    CONCURRENCY="$2"; shift 2 ;;
        --quant)          QUANT="$2"; shift 2 ;;
        --config)         CONFIG="$2"; shift 2 ;;
        --tp)             TP="$2"; shift 2 ;;
        --ep)             EP="$2"; shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
        --launch-skip)    LAUNCH_SKIP="$2"; shift 2 ;;
        --launch-count)   LAUNCH_COUNT="$2"; shift 2 ;;
        --num-requests)   NUM_REQUESTS="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *)                echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ======================== Validation ==========================================
if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required"; usage
fi

if [[ "$MODE" == "targeted" && -z "$KERNEL_NAME" ]]; then
    echo "ERROR: --kernel-name is required for targeted mode"; exit 1
fi

case "$SCENARIO" in
    chat)      ISL=1024; OSL=1024 ;;
    reasoning) ISL=1024; OSL=8192 ;;
    summarize) ISL=8192; OSL=1024 ;;
    *) echo "ERROR: --scenario must be chat, reasoning, or summarize"; exit 1 ;;
esac

if ! command -v ncu &>/dev/null; then
    echo "ERROR: ncu (Nsight Compute) not found. Install NVIDIA Nsight Compute."
    exit 1
fi

# ======================== Utilities ===========================================
TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(TS)] $*"; }

mkdir -p "$OUTPUT_DIR"

TAG="ncu_${QUANT}_${CONFIG}_${SCENARIO}_tp${TP}_ep${EP}_c${CONCURRENCY}"

log "============================================================"
log "  Nsight Compute Kernel Analysis"
log "============================================================"
log "  Model:       $MODEL"
log "  Mode:        $MODE"
log "  Scenario:    $SCENARIO (ISL=$ISL, OSL=$OSL)"
log "  Concurrency: $CONCURRENCY"
log "  Quant:       $QUANT"
log "  Config:      $CONFIG"
log "  TP/EP:       $TP / $EP"
log "  Output:      $OUTPUT_DIR"
if [[ "$MODE" == "targeted" ]]; then
    log "  Kernel:      $KERNEL_NAME"
    log "  Skip/Count:  $TARGETED_SKIP / $TARGETED_COUNT"
else
    log "  Skip/Count:  $LAUNCH_SKIP / $LAUNCH_COUNT"
fi
log "============================================================"

# ======================== Generate Dataset ====================================
DATASET="$OUTPUT_DIR/dataset_${TAG}.jsonl"
if [[ -f "$SCRIPT_DIR/gen_dataset.py" ]]; then
    log "Generating dataset..."
    python3 "$SCRIPT_DIR/gen_dataset.py" \
        --tokenizer "$MODEL" \
        --fixed_input_len "$ISL" \
        --output_tokens "$OSL" \
        --num_requests "$NUM_REQUESTS" \
        --input_mode random \
        --output "$DATASET"
else
    log "WARN: gen_dataset.py not found, assuming dataset exists"
fi

# ======================== Build Config YAML ===================================
CONFIG_YAML="$OUTPUT_DIR/config_${TAG}.yml"
cat > "$CONFIG_YAML" << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: $CONCURRENCY
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.8
    enable_block_reuse: false
stream_interval: 10
moe_config:
    backend: TRTLLM
EOF

if [[ "$CONFIG" == "latency" ]]; then
    cat >> "$CONFIG_YAML" << EOF
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: 3
EOF
fi

# ======================== Build ncu Command ===================================
NCU_REP="$OUTPUT_DIR/${TAG}"

ncu_cmd=(
    ncu --target-processes all
    --set full
    --graph-profiling node
)

if [[ "$MODE" == "targeted" ]]; then
    NCU_REP="${NCU_REP}_${KERNEL_NAME//[^a-zA-Z0-9_]/_}"
    ncu_cmd+=(
        --kernel-name "$KERNEL_NAME"
        --launch-skip "$TARGETED_SKIP"
        --launch-count "$TARGETED_COUNT"
    )
else
    ncu_cmd+=(
        --launch-skip "$LAUNCH_SKIP"
        --launch-count "$LAUNCH_COUNT"
    )
fi

ncu_cmd+=(-o "$NCU_REP")

# ======================== Build Benchmark Command =============================
MAX_SEQ_LEN=8192
[[ $((ISL + OSL)) -gt 2048 ]] && MAX_SEQ_LEN=10240

bench_cmd=(
    trtllm-bench --model "$MODEL" --model_path "$MODEL"
    throughput --backend pytorch
    --extra_llm_api_options "$CONFIG_YAML"
    --max_seq_len "$MAX_SEQ_LEN"
    --tp "$TP" --ep "$EP"
    --dataset "$DATASET"
    --concurrency "$CONCURRENCY"
    --num_requests "$NUM_REQUESTS"
    --warmup 0
)

# ======================== Run ncu =============================================
log "Running ncu profiling..."
log "  ncu cmd: ${ncu_cmd[*]}"
log "  bench cmd: ${bench_cmd[*]}"

TLLM_PROFILE_START_STOP="$ITER_RANGE" \
    "${ncu_cmd[@]}" "${bench_cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/ncu_${TAG}.log"

RC=${PIPESTATUS[0]}
if [[ $RC -ne 0 ]]; then
    log "ERROR: ncu profiling failed (rc=$RC)"
    exit $RC
fi

log "ncu profiling complete: ${NCU_REP}.ncu-rep"

# ======================== Extract Metrics =====================================
log "Extracting metrics..."

CSV_FILE="$OUTPUT_DIR/${TAG}_metrics.csv"

python3 - "$NCU_REP.ncu-rep" "$CSV_FILE" <<'PYEOF'
import csv
import subprocess
import sys
import re

ncu_rep = sys.argv[1]
csv_out = sys.argv[2]

# Export raw metrics via ncu CLI
result = subprocess.run(
    ["ncu", "--import", ncu_rep, "--page", "raw", "--csv"],
    capture_output=True, text=True, timeout=120
)

if result.returncode != 0:
    print(f"ERROR: ncu --import failed: {result.stderr[:200]}", file=sys.stderr)
    sys.exit(1)

lines = result.stdout.strip().split("\n")
if len(lines) < 2:
    print("ERROR: No data in ncu report", file=sys.stderr)
    sys.exit(1)

reader = csv.DictReader(lines)

# Collect per-kernel metrics
kernels = {}
for row in reader:
    kname = row.get("Kernel Name", row.get("kernel_name", "unknown"))
    # Clean up kernel name (take short form)
    short = kname.split("(")[0].split("<")[0].strip()
    if len(short) > 80:
        short = short[:77] + "..."

    duration = 0
    dram_pct = 0
    sm_pct = 0
    occupancy = 0

    for key, val in row.items():
        kl = key.lower()
        try:
            fval = float(val.replace(",", "").replace("%", "")) if val else 0
        except (ValueError, AttributeError):
            fval = 0

        if "duration" in kl and ("nsecond" in kl or "usecond" in kl):
            if "nsecond" in kl:
                duration = fval / 1000  # ns -> us
            else:
                duration = fval
        elif "dram" in kl and "throughput" in kl and "pct" in kl:
            dram_pct = max(dram_pct, fval)
        elif ("sm__throughput" in kl or ("sm" in kl and "throughput" in kl)) and "pct" in kl:
            sm_pct = max(sm_pct, fval)
        elif "occupancy" in kl and ("achieved" in kl or "active" in kl):
            occupancy = max(occupancy, fval)

    if short not in kernels:
        kernels[short] = {
            "count": 0, "total_us": 0,
            "dram_pct_sum": 0, "sm_pct_sum": 0, "occ_sum": 0,
        }
    kernels[short]["count"] += 1
    kernels[short]["total_us"] += duration
    kernels[short]["dram_pct_sum"] += dram_pct
    kernels[short]["sm_pct_sum"] += sm_pct
    kernels[short]["occ_sum"] += occupancy

# Write summary CSV
with open(csv_out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kernel", "count", "total_us", "avg_us", "dram_pct", "sm_pct", "occupancy_pct", "diagnosis"])

    total_time = sum(k["total_us"] for k in kernels.values())

    for kname, m in sorted(kernels.items(), key=lambda x: -x[1]["total_us"]):
        count = m["count"]
        avg_us = m["total_us"] / count if count else 0
        dram = m["dram_pct_sum"] / count if count else 0
        sm = m["sm_pct_sum"] / count if count else 0
        occ = m["occ_sum"] / count if count else 0
        pct = m["total_us"] / total_time * 100 if total_time else 0

        # Diagnosis
        if dram > 60:
            diag = "memory-bound"
        elif sm > 50:
            diag = "compute-bound"
        elif occ < 25:
            diag = "latency-bound (low occupancy)"
        else:
            diag = "balanced"

        w.writerow([kname, count, f"{m['total_us']:.1f}", f"{avg_us:.1f}",
                     f"{dram:.1f}", f"{sm:.1f}", f"{occ:.1f}", diag])

# Print summary table
print()
print(f"{'Kernel':<60} {'Count':>5} {'Total(us)':>10} {'Avg(us)':>9} {'DRAM%':>6} {'SM%':>5} {'Occ%':>5} {'Diagnosis'}")
print("-" * 120)

for kname, m in sorted(kernels.items(), key=lambda x: -x[1]["total_us"])[:15]:
    count = m["count"]
    avg_us = m["total_us"] / count if count else 0
    dram = m["dram_pct_sum"] / count if count else 0
    sm = m["sm_pct_sum"] / count if count else 0
    occ = m["occ_sum"] / count if count else 0
    pct = m["total_us"] / total_time * 100 if total_time else 0

    if dram > 60:
        diag = "MEM-BOUND"
    elif sm > 50:
        diag = "COMPUTE"
    elif occ < 25:
        diag = "LOW-OCC"
    else:
        diag = "balanced"

    label = kname[:58] if len(kname) <= 58 else kname[:55] + "..."
    print(f"{label:<60} {count:>5} {m['total_us']:>10.1f} {avg_us:>9.1f} {dram:>5.1f}% {sm:>4.1f}% {occ:>4.1f}% {diag}")

print()
print(f"Metrics CSV: {csv_out}")
PYEOF

log "============================================================"
log "  NCU ANALYSIS COMPLETE"
log "============================================================"
log "  Report:  ${NCU_REP}.ncu-rep"
log "  Metrics: $CSV_FILE"
log "  Log:     $OUTPUT_DIR/ncu_${TAG}.log"
log "============================================================"
log ""
log "Next steps:"
log "  # View in GUI:"
log "  ncu --import ${NCU_REP}.ncu-rep"
log ""
log "  # Compare baseline vs optimized:"
log "  ncu --import baseline.ncu-rep --page details --csv > baseline.csv"
log "  ncu --import optimized.ncu-rep --page details --csv > optimized.csv"
