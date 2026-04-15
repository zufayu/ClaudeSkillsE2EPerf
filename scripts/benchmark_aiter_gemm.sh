#!/usr/bin/env bash
# benchmark_aiter_gemm.sh — Benchmark + tune aiter GEMM operators on MI355X
#
# Usage:
#   bash scripts/benchmark_aiter_gemm.sh --action benchmark [--result-dir ./results]
#   bash scripts/benchmark_aiter_gemm.sh --action tune [--result-dir ./results]
#   bash scripts/benchmark_aiter_gemm.sh --action benchmark-tuned --tuned-ptpc /path/to/tuned_ptpc.csv --tuned-batched /path/to/tuned_batched.csv [--result-dir ./results]
#   bash scripts/benchmark_aiter_gemm.sh --action all [--result-dir ./results]
#
# Target operators (DeepSeek-R1 decode, BS=64):
#   q_b_proj  : PTPC GEMM   M=64 K=1536 N=6144  (profiled 11.8μs)
#   o_proj    : PTPC GEMM   M=64 K=4096 N=7168  (profiled 21.4μs)
#   k_up      : Batched GEMM B=32 M=64 K=512 N=128 (profiled 5.5μs)
#   uv        : Batched GEMM B=32 M=64 K=512 N=128 (profiled 6.6μs)

set -euo pipefail

ACTION="benchmark"
RESULT_DIR="./results"
TUNED_PTPC=""
TUNED_BATCHED=""
WARMUP=20
ITERS=100

while [[ $# -gt 0 ]]; do
    case "$1" in
        --action)       ACTION="$2"; shift 2 ;;
        --result-dir)   RESULT_DIR="$2"; shift 2 ;;
        --tuned-ptpc)   TUNED_PTPC="$2"; shift 2 ;;
        --tuned-batched) TUNED_BATCHED="$2"; shift 2 ;;
        --warmup)       WARMUP="$2"; shift 2 ;;
        --iters)        ITERS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULT_DIR"

echo "============================================================"
echo "  aiter GEMM Benchmark"
echo "  Action: $ACTION"
echo "  Result dir: $RESULT_DIR"
echo "============================================================"

# ── Discovery ──────────────────────────────────────────────────
discover() {
    echo ""
    echo "=== Discovery ==="
    python3 -c "
import aiter
print(f'aiter version: {getattr(aiter, \"__version__\", \"unknown\")}')
print(f'aiter path: {aiter.__path__[0]}')
import torch
print(f'torch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
"
    AITER_ROOT=$(python3 -c "import aiter; print(aiter.__path__[0])")
    AITER_REPO=$(dirname "$AITER_ROOT")
    echo "AITER_ROOT=$AITER_ROOT"
    echo "AITER_REPO=$AITER_REPO"

    echo ""
    echo "=== Available GEMM functions ==="
    python3 -c "
import aiter
for name in sorted(dir(aiter)):
    if 'gemm' in name.lower():
        obj = getattr(aiter, name, None)
        doc = ''
        if hasattr(obj, '__doc__') and obj.__doc__:
            doc = obj.__doc__.split(chr(10))[0][:80]
        print(f'  {name}: {type(obj).__name__} — {doc}')
"

    echo ""
    echo "=== Tune scripts ==="
    ls -la "$AITER_REPO/csrc/ck_gemm_a8w8_bpreshuffle/"*tune* 2>/dev/null || echo "  no PTPC tune scripts"
    ls -la "$AITER_REPO/csrc/ck_batched_gemm_a8w8/"*tune* 2>/dev/null || echo "  no batched tune scripts"
}

# ── Benchmark ──────────────────────────────────────────────────
run_benchmark() {
    local TAG="${1:-baseline}"
    local EXTRA_ENV="${2:-}"
    echo ""
    echo "=== Benchmark ($TAG, warmup=$WARMUP, iters=$ITERS) ==="

    # Set tuned config env vars if provided
    if [ -n "$EXTRA_ENV" ]; then
        echo "  Env: $EXTRA_ENV"
        eval "export $EXTRA_ENV"
    fi

    python3 - "$RESULT_DIR" "$TAG" "$WARMUP" "$ITERS" << 'PYEOF'
import sys
import os
import csv
import time
import torch

RESULT_DIR = sys.argv[1]
TAG = sys.argv[2]
WARMUP = int(sys.argv[3])
ITERS = int(sys.argv[4])

# FP8 dtype for MI300/MI355 (fnuz variant)
FP8 = torch.float8_e4m3fnuz

def benchmark_gemm(name, M, N, K, warmup=WARMUP, iters=ITERS):
    """Benchmark aiter PTPC (per-token-per-channel) GEMM a8w8."""
    import aiter

    # A: [M, K] fp8, B: [N, K] fp8 (note: B is [N,K] not [K,N] for aiter)
    A = torch.randn(M, K, device="cuda").to(FP8)
    B = torch.randn(N, K, device="cuda").to(FP8)
    # Per-token scale for A: [M, 1], per-channel scale for B: [1, N]
    scale_a = torch.ones(M, 1, device="cuda", dtype=torch.float32)
    scale_b = torch.ones(1, N, device="cuda", dtype=torch.float32)

    # Try different API signatures
    result = None
    api_used = "unknown"

    # Strategy 1: aiter.gemm_a8w8 with bias=None
    try:
        result = aiter.gemm_a8w8(A, B, scale_a, scale_b, None)
        api_used = "aiter.gemm_a8w8(A,B,sa,sb,bias=None)"
    except Exception as e1:
        # Strategy 2: without bias
        try:
            result = aiter.gemm_a8w8(A, B, scale_a, scale_b)
            api_used = "aiter.gemm_a8w8(A,B,sa,sb)"
        except Exception as e2:
            print(f"  ERROR {name}: gemm_a8w8 failed")
            print(f"    Attempt 1: {e1}")
            print(f"    Attempt 2: {e2}")
            return None

    print(f"  {name}: API={api_used}, output shape={result.shape}, dtype={result.dtype}")

    # Warmup
    torch.cuda.synchronize()
    for _ in range(warmup):
        try:
            aiter.gemm_a8w8(A, B, scale_a, scale_b, None)
        except:
            aiter.gemm_a8w8(A, B, scale_a, scale_b)
    torch.cuda.synchronize()

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    torch.cuda.synchronize()
    for i in range(iters):
        start_events[i].record()
        try:
            aiter.gemm_a8w8(A, B, scale_a, scale_b, None)
        except:
            aiter.gemm_a8w8(A, B, scale_a, scale_b)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [start_events[i].elapsed_time(end_events[i]) * 1000 for i in range(iters)]  # μs
    avg_us = sum(times) / len(times)
    min_us = min(times)
    max_us = max(times)
    print(f"  {name}: M={M} N={N} K={K} → avg={avg_us:.1f}μs min={min_us:.1f}μs max={max_us:.1f}μs")
    return {"op": name, "type": "ptpc", "M": M, "N": N, "K": K, "B": "", "avg_us": f"{avg_us:.1f}", "min_us": f"{min_us:.1f}", "max_us": f"{max_us:.1f}"}


def benchmark_batched_gemm(name, B_count, M, N, K, warmup=WARMUP, iters=ITERS):
    """Benchmark aiter batched GEMM a8w8."""
    import aiter

    # A: [B, M, K] fp8, B_mat: [B, N, K] fp8
    A = torch.randn(B_count, M, K, device="cuda").to(FP8)
    B_mat = torch.randn(B_count, N, K, device="cuda").to(FP8)
    # Scales
    scale_a = torch.ones(B_count, M, 1, device="cuda", dtype=torch.float32)
    scale_b = torch.ones(B_count, 1, N, device="cuda", dtype=torch.float32)

    result = None
    api_used = "unknown"

    # Strategy 1: aiter.batched_gemm_a8w8 with bias=None
    try:
        result = aiter.batched_gemm_a8w8(A, B_mat, scale_a, scale_b, None)
        api_used = "aiter.batched_gemm_a8w8(A,B,sa,sb,bias=None)"
    except Exception as e1:
        # Strategy 2: without bias
        try:
            result = aiter.batched_gemm_a8w8(A, B_mat, scale_a, scale_b)
            api_used = "aiter.batched_gemm_a8w8(A,B,sa,sb)"
        except Exception as e2:
            # Strategy 3: 2D scales
            try:
                sa2 = torch.ones(M, 1, device="cuda", dtype=torch.float32)
                sb2 = torch.ones(1, N, device="cuda", dtype=torch.float32)
                result = aiter.batched_gemm_a8w8(A, B_mat, sa2, sb2)
                api_used = "aiter.batched_gemm_a8w8(A,B,sa2d,sb2d)"
            except Exception as e3:
                print(f"  ERROR {name}: batched_gemm_a8w8 failed")
                print(f"    Attempt 1: {e1}")
                print(f"    Attempt 2: {e2}")
                print(f"    Attempt 3: {e3}")
                return None

    print(f"  {name}: API={api_used}, output shape={result.shape}, dtype={result.dtype}")

    # Warmup
    torch.cuda.synchronize()
    for _ in range(warmup):
        try:
            aiter.batched_gemm_a8w8(A, B_mat, scale_a, scale_b, None)
        except:
            try:
                aiter.batched_gemm_a8w8(A, B_mat, scale_a, scale_b)
            except:
                sa2 = torch.ones(M, 1, device="cuda", dtype=torch.float32)
                sb2 = torch.ones(1, N, device="cuda", dtype=torch.float32)
                aiter.batched_gemm_a8w8(A, B_mat, sa2, sb2)
    torch.cuda.synchronize()

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    torch.cuda.synchronize()
    for i in range(iters):
        start_events[i].record()
        try:
            aiter.batched_gemm_a8w8(A, B_mat, scale_a, scale_b, None)
        except:
            try:
                aiter.batched_gemm_a8w8(A, B_mat, scale_a, scale_b)
            except:
                sa2 = torch.ones(M, 1, device="cuda", dtype=torch.float32)
                sb2 = torch.ones(1, N, device="cuda", dtype=torch.float32)
                aiter.batched_gemm_a8w8(A, B_mat, sa2, sb2)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [start_events[i].elapsed_time(end_events[i]) * 1000 for i in range(iters)]  # μs
    avg_us = sum(times) / len(times)
    min_us = min(times)
    max_us = max(times)
    print(f"  {name}: B={B_count} M={M} N={N} K={K} → avg={avg_us:.1f}μs min={min_us:.1f}μs max={max_us:.1f}μs")
    return {"op": name, "type": "batched", "M": M, "N": N, "K": K, "B": B_count, "avg_us": f"{avg_us:.1f}", "min_us": f"{min_us:.1f}", "max_us": f"{max_us:.1f}"}


# ── Main ──
print(f"Tag: {TAG}")
print(f"Warmup: {WARMUP}, Iters: {ITERS}")
print(f"FP8 dtype: {FP8}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

results = []

# PTPC GEMMs
print("--- PTPC GEMM (aiter.gemm_a8w8) ---")
r = benchmark_gemm("q_b_proj", M=64, N=6144, K=1536)
if r: results.append(r)

r = benchmark_gemm("o_proj", M=64, N=7168, K=4096)
if r: results.append(r)

# Batched GEMMs
print()
print("--- Batched GEMM (aiter.batched_gemm_a8w8) ---")
r = benchmark_batched_gemm("k_up", B_count=32, M=64, N=128, K=512)
if r: results.append(r)

r = benchmark_batched_gemm("uv", B_count=32, M=64, N=128, K=512)
if r: results.append(r)

# Save CSV
csv_path = os.path.join(RESULT_DIR, f"aiter_gemm_{TAG}.csv")
if results:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["op", "type", "M", "N", "K", "B", "avg_us", "min_us", "max_us"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Print summary table
    print()
    print(f"{'Op':<12} {'Type':<8} {'Shape':<28} {'Avg(μs)':>8} {'Min(μs)':>8} {'Max(μs)':>8}")
    print("-" * 80)
    for r in results:
        if r['B']:
            shape = f"B={r['B']} M={r['M']} N={r['N']} K={r['K']}"
        else:
            shape = f"M={r['M']} N={r['N']} K={r['K']}"
        print(f"{r['op']:<12} {r['type']:<8} {shape:<28} {r['avg_us']:>8} {r['min_us']:>8} {r['max_us']:>8}")
else:
    print("\nNo results collected — all benchmarks failed")
    sys.exit(1)

PYEOF
}

# ── Tune ───────────────────────────────────────────────────────
run_tune() {
    echo ""
    echo "=== Tune ==="

    AITER_ROOT=$(python3 -c "import aiter; print(aiter.__path__[0])")
    AITER_REPO=$(dirname "$AITER_ROOT")

    # Create untuned PTPC CSV
    UNTUNED_PTPC="$RESULT_DIR/untuned_ptpc.csv"
    cat > "$UNTUNED_PTPC" << 'EOF'
M,N,K,q_dtype_w
64,6144,1536,torch.float8_e4m3fnuz
64,7168,4096,torch.float8_e4m3fnuz
EOF
    echo "  Created $UNTUNED_PTPC"
    cat "$UNTUNED_PTPC"

    # Create untuned batched CSV
    UNTUNED_BATCHED="$RESULT_DIR/untuned_batched.csv"
    cat > "$UNTUNED_BATCHED" << 'EOF'
B,M,N,K
32,64,128,512
EOF
    echo "  Created $UNTUNED_BATCHED"
    cat "$UNTUNED_BATCHED"

    TUNED_PTPC_OUT="$RESULT_DIR/tuned_ptpc.csv"
    TUNED_BATCHED_OUT="$RESULT_DIR/tuned_batched.csv"

    # Tune PTPC GEMM
    PTPC_TUNE="$AITER_REPO/csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py"
    if [ -f "$PTPC_TUNE" ]; then
        echo ""
        echo "--- Tuning PTPC GEMM ---"
        python3 "$PTPC_TUNE" -i "$UNTUNED_PTPC" -o "$TUNED_PTPC_OUT" 2>&1
        echo ""
        echo "  Tuned PTPC result:"
        cat "$TUNED_PTPC_OUT" 2>/dev/null || echo "  (no output)"
    else
        echo "  PTPC tune script not found: $PTPC_TUNE"
    fi

    # Tune batched GEMM
    BATCHED_TUNE="$AITER_REPO/csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py"
    if [ -f "$BATCHED_TUNE" ]; then
        echo ""
        echo "--- Tuning Batched GEMM ---"
        python3 "$BATCHED_TUNE" -i "$UNTUNED_BATCHED" -o "$TUNED_BATCHED_OUT" 2>&1
        echo ""
        echo "  Tuned batched result:"
        cat "$TUNED_BATCHED_OUT" 2>/dev/null || echo "  (no output)"
    else
        echo "  Batched tune script not found: $BATCHED_TUNE"
    fi
}

# ── Compare ────────────────────────────────────────────────────
compare_results() {
    echo ""
    echo "=== Comparison: Baseline vs Tuned ==="
    python3 - "$RESULT_DIR" << 'PYEOF'
import sys, csv, os

RESULT_DIR = sys.argv[1]
baseline_path = os.path.join(RESULT_DIR, "aiter_gemm_baseline.csv")
tuned_path = os.path.join(RESULT_DIR, "aiter_gemm_tuned.csv")

if not os.path.exists(baseline_path):
    print("  No baseline CSV found")
    sys.exit(0)
if not os.path.exists(tuned_path):
    print("  No tuned CSV found")
    sys.exit(0)

def load_csv(path):
    with open(path) as f:
        return {row["op"]: row for row in csv.DictReader(f)}

baseline = load_csv(baseline_path)
tuned = load_csv(tuned_path)

print(f"{'Op':<12} {'Baseline(μs)':>12} {'Tuned(μs)':>10} {'Speedup':>8} {'Profiled(μs)':>12}")
print("-" * 60)

profiled = {"q_b_proj": 11.8, "o_proj": 21.4, "k_up": 5.5, "uv": 6.6}

for op in ["q_b_proj", "o_proj", "k_up", "uv"]:
    b_us = float(baseline[op]["avg_us"]) if op in baseline else 0
    t_us = float(tuned[op]["avg_us"]) if op in tuned else 0
    p_us = profiled.get(op, 0)
    speedup = f"{b_us / t_us:.2f}x" if t_us > 0 else "N/A"
    print(f"{op:<12} {b_us:>12.1f} {t_us:>10.1f} {speedup:>8} {p_us:>12.1f}")

# Save comparison
comp_path = os.path.join(RESULT_DIR, "aiter_gemm_comparison.csv")
with open(comp_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["op", "baseline_us", "tuned_us", "speedup", "profiled_us"])
    for op in ["q_b_proj", "o_proj", "k_up", "uv"]:
        b_us = float(baseline[op]["avg_us"]) if op in baseline else 0
        t_us = float(tuned[op]["avg_us"]) if op in tuned else 0
        p_us = profiled.get(op, 0)
        speedup = f"{b_us / t_us:.2f}" if t_us > 0 else ""
        w.writerow([op, f"{b_us:.1f}", f"{t_us:.1f}", speedup, f"{p_us:.1f}"])
print(f"\nComparison saved to {comp_path}")

PYEOF
}

# ── Main dispatch ──────────────────────────────────────────────
case "$ACTION" in
    discover)
        discover
        ;;
    benchmark|benchmark-baseline)
        discover
        run_benchmark "baseline"
        ;;
    tune)
        run_tune
        ;;
    benchmark-tuned)
        if [ -z "$TUNED_PTPC" ] || [ -z "$TUNED_BATCHED" ]; then
            # Default to result dir paths from tune step
            TUNED_PTPC="$RESULT_DIR/tuned_ptpc.csv"
            TUNED_BATCHED="$RESULT_DIR/tuned_batched.csv"
        fi
        if [ ! -f "$TUNED_PTPC" ] && [ ! -f "$TUNED_BATCHED" ]; then
            echo "ERROR: No tuned CSVs found. Run --action tune first."
            exit 1
        fi
        TUNE_ENV=""
        [ -f "$TUNED_PTPC" ] && TUNE_ENV="AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE=$TUNED_PTPC"
        [ -f "$TUNED_BATCHED" ] && TUNE_ENV="$TUNE_ENV AITER_CONFIG_A8W8_BATCHED_GEMM=$TUNED_BATCHED"
        echo "Tuned config env: $TUNE_ENV"
        run_benchmark "tuned" "$TUNE_ENV"
        ;;
    all)
        discover
        run_benchmark "baseline"
        run_tune
        # Re-benchmark with tuned configs
        TUNED_PTPC="$RESULT_DIR/tuned_ptpc.csv"
        TUNED_BATCHED="$RESULT_DIR/tuned_batched.csv"
        TUNE_ENV=""
        [ -f "$TUNED_PTPC" ] && TUNE_ENV="AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE=$TUNED_PTPC"
        [ -f "$TUNED_BATCHED" ] && TUNE_ENV="$TUNE_ENV AITER_CONFIG_A8W8_BATCHED_GEMM=$TUNED_BATCHED"
        run_benchmark "tuned" "$TUNE_ENV"
        compare_results
        ;;
    *)
        echo "Unknown action: $ACTION"
        echo "Valid: discover, benchmark, benchmark-baseline, tune, benchmark-tuned, all"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  Done ($ACTION)"
echo "============================================================"
