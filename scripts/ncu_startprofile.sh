#!/usr/bin/env bash
# =============================================================================
# NCU profiling via SGLang /start_profile + /stop_profile HTTP API
#
# Instead of CUDA_INJECTION64_PATH (which slows startup 10x),
# NCU wraps the server with --profile-from-start off. The server
# starts at normal speed. After warmup, we trigger profiling via
# SGLang's built-in HTTP API, which calls cudaProfilerStart/Stop
# inside the worker processes.
#
# Key fix: after /stop_profile, we wait for NCU to serialize,
# then SIGTERM the server gracefully so NCU can write .ncu-rep.
# =============================================================================

set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Defaults ──
MODEL=""
TP=8
CONCURRENCY=64
SCENARIO="chat"
PORT=8888
NCU_SET="pmsampling"
LAUNCH_COUNT=5
WAIT_AFTER_STOP=120
OUTPUT_DIR="/tmp"
REPORT_NAME="ncu_decode"

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)            MODEL="$2"; shift 2 ;;
        --tp)               TP="$2"; shift 2 ;;
        --concurrency)      CONCURRENCY="$2"; shift 2 ;;
        --scenario)         SCENARIO="$2"; shift 2 ;;
        --port)             PORT="$2"; shift 2 ;;
        --ncu-set)          NCU_SET="$2"; shift 2 ;;
        --launch-count)     LAUNCH_COUNT="$2"; shift 2 ;;
        --wait-after-stop)  WAIT_AFTER_STOP="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --report-name)      REPORT_NAME="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

[[ -z "$MODEL" ]] && { echo "ERROR: --model required"; exit 1; }

# ── Scenario params ──
case "$SCENARIO" in
    chat)      ISL=1024; OSL=1024 ;;
    reasoning) ISL=1024; OSL=8192 ;;
    summarize) ISL=8192; OSL=1024 ;;
    *) echo "ERROR: unknown scenario $SCENARIO"; exit 1 ;;
esac

WARMUP_PROMPTS=$((CONCURRENCY * 2))
# Use minimal benchmark under NCU profiling — kernel replay makes each prompt ~28x slower
BENCH_PROMPTS=10
NCU_OUTPUT="$OUTPUT_DIR/$REPORT_NAME"

log "============================================================"
log "  NCU via /start_profile"
log "============================================================"
log "  Model:       $MODEL"
log "  TP=$TP  Scenario=$SCENARIO (ISL=$ISL OSL=$OSL)"
log "  Concurrency: $CONCURRENCY"
log "  NCU set:     $NCU_SET"
log "  Launch-count: $LAUNCH_COUNT"
log "  Wait after:  ${WAIT_AFTER_STOP}s"
log "  Output:      $NCU_OUTPUT"
log "============================================================"

# ── Cleanup ──
log "Cleaning up leftover processes..."
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
pkill -9 -f "sglang" 2>/dev/null || true
fuser -k -9 "${PORT}/tcp" 2>/dev/null || true
sleep 3

# ── Build NCU command ──
NCU_OPTS="--profile-from-start off --target-processes all"
NCU_OPTS="$NCU_OPTS --launch-count $LAUNCH_COUNT"
NCU_OPTS="$NCU_OPTS -f -o $NCU_OUTPUT"

if [[ "$NCU_SET" == "pmsampling" ]]; then
    NCU_OPTS="$NCU_OPTS --section PmSampling --section PmSampling_WarpStates"
else
    NCU_OPTS="$NCU_OPTS --set $NCU_SET"
fi

# Server command — disable cuda graph and autotune for faster startup
SERVER_CMD="python3 -m sglang.launch_server \
  --model-path $MODEL --tp $TP \
  --mem-fraction-static 0.85 --chunked-prefill-size 16384 \
  --kv-cache-dtype fp8_e4m3 --max-running-requests 256 \
  --disable-cuda-graph --disable-flashinfer-autotune \
  --port $PORT --host 0.0.0.0"

log "NCU command: ncu $NCU_OPTS $SERVER_CMD"

# ── Phase 1: Launch NCU wrapping server ──
log ""
log "Phase 1: Starting NCU + SGLang server..."
ncu $NCU_OPTS $SERVER_CMD > "$OUTPUT_DIR/ncu_server.log" 2>&1 &
NCU_PID=$!
log "NCU PID=$NCU_PID"

# ── Phase 2: Wait for server ready ──
log ""
log "Phase 2: Waiting for server health..."
WAIT=0
TIMEOUT=1200
while true; do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        log "Server ready after ${WAIT}s"
        break
    fi
    if ! kill -0 $NCU_PID 2>/dev/null; then
        log "ERROR: NCU/server process died. Log:"
        tail -30 "$OUTPUT_DIR/ncu_server.log" 2>/dev/null
        exit 1
    fi
    sleep 10
    WAIT=$((WAIT + 10))
    if [[ $WAIT -ge $TIMEOUT ]]; then
        log "ERROR: Server not ready after ${TIMEOUT}s"
        tail -30 "$OUTPUT_DIR/ncu_server.log" 2>/dev/null
        kill $NCU_PID 2>/dev/null || true
        exit 1
    fi
    [[ $((WAIT % 60)) -eq 0 ]] && log "  Still waiting... (${WAIT}s)"
done

# ── Phase 3: Warmup ──
log ""
log "Phase 3: Warmup ($WARMUP_PROMPTS prompts, c=$CONCURRENCY)..."
python3 -m sglang.bench_serving \
    --model "$MODEL" --port "$PORT" --backend vllm \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 0.8 --num-prompts "$WARMUP_PROMPTS" \
    --max-concurrency "$CONCURRENCY" --warmup-requests 0 \
    --output-file /tmp/ncu_warmup.jsonl 2>&1 || log "WARNING: warmup returned non-zero"
log "Warmup complete"

# ── Phase 4: Start profiling ──
log ""
log "Phase 4: Triggering /start_profile..."
START_RESP=$(curl -sf --max-time 30 "http://localhost:${PORT}/start_profile" 2>&1 || echo "FAILED")
log "/start_profile response: $START_RESP"

if [[ "$START_RESP" == "FAILED" ]]; then
    log "ERROR: /start_profile failed"
    kill $NCU_PID 2>/dev/null || true
    exit 1
fi

# ── Phase 5: Benchmark ──
log ""
log "Phase 5: Benchmark ($BENCH_PROMPTS prompts, c=$CONCURRENCY)..."
BENCH_START=$(date +%s)
python3 -m sglang.bench_serving \
    --model "$MODEL" --port "$PORT" --backend vllm \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 0.8 --num-prompts "$BENCH_PROMPTS" \
    --max-concurrency "$CONCURRENCY" --warmup-requests 0 \
    --output-file /tmp/ncu_bench.jsonl 2>&1 || log "WARNING: benchmark returned non-zero"
BENCH_END=$(date +%s)
log "Benchmark done ($(( BENCH_END - BENCH_START ))s)"

# ── Phase 6: Stop profiling ──
log ""
log "Phase 6: Triggering /stop_profile (may take minutes for NCU replay)..."
STOP_START=$(date +%s)
STOP_RESP=$(curl -sf --max-time 3600 "http://localhost:${PORT}/stop_profile" 2>&1 || echo "TIMEOUT_OR_FAILED")
STOP_END=$(date +%s)
log "/stop_profile response: $STOP_RESP (took $(( STOP_END - STOP_START ))s)"

# ── Phase 7: Wait for NCU to serialize ──
log ""
log "Phase 7: Waiting ${WAIT_AFTER_STOP}s for NCU to serialize data..."

# Check periodically if .ncu-rep appears
for i in $(seq 1 $WAIT_AFTER_STOP); do
    if ls "$OUTPUT_DIR/${REPORT_NAME}"*.ncu-rep 2>/dev/null | head -1 > /dev/null 2>&1; then
        log ".ncu-rep appeared after ${i}s!"
        break
    fi
    sleep 1
    [[ $((i % 30)) -eq 0 ]] && log "  Still waiting for .ncu-rep... (${i}s)"
done

# ── Phase 8: Graceful shutdown ──
log ""
log "Phase 8: Graceful shutdown..."

# Check for .ncu-rep BEFORE killing anything
log "Files before shutdown:"
ls -lh "$OUTPUT_DIR/${REPORT_NAME}"*.ncu-rep 2>/dev/null || echo "  No .ncu-rep yet"
find "$OUTPUT_DIR" -name "*.ncu-rep" -ls 2>/dev/null || echo "  None in $OUTPUT_DIR"

# Send SIGTERM to NCU (NCU forwards to server)
log "Sending SIGTERM to NCU PID=$NCU_PID..."
kill -TERM $NCU_PID 2>/dev/null || true

# Wait for NCU to exit — it should write .ncu-rep before exiting
log "Waiting for NCU to exit (max 300s)..."
NCU_EXITED=false
for i in $(seq 1 60); do
    if ! kill -0 $NCU_PID 2>/dev/null; then
        wait $NCU_PID 2>/dev/null
        NCU_EXIT=$?
        log "NCU exited with code $NCU_EXIT after $((i * 5))s"
        NCU_EXITED=true
        break
    fi
    sleep 5
    [[ $((i % 6)) -eq 0 ]] && log "  NCU still running... ($((i * 5))s)"
done

if [[ "$NCU_EXITED" != "true" ]]; then
    log "NCU did not exit after 300s, force killing..."
    kill -9 $NCU_PID 2>/dev/null || true
    sleep 3
fi

# Cleanup any remaining sglang processes
pkill -9 -f "sglang" 2>/dev/null || true
sleep 2

# ── Results ──
log ""
log "============================================================"
log "  Results"
log "============================================================"

FOUND_REP=false
for f in "$OUTPUT_DIR/${REPORT_NAME}"*.ncu-rep; do
    if [[ -f "$f" ]]; then
        SIZE=$(du -h "$f" | cut -f1)
        log "  FOUND: $f ($SIZE)"
        FOUND_REP=true
    fi
done

# Also check /tmp broadly
find /tmp -name "*.ncu-rep" -newer "$OUTPUT_DIR/ncu_server.log" -ls 2>/dev/null | while read line; do
    log "  Also in /tmp: $line"
done

if [[ "$FOUND_REP" == "true" ]]; then
    log ""
    log "  SUCCESS! .ncu-rep file generated."
else
    log ""
    log "  FAILED: No .ncu-rep file found."
    log ""
    log "  NCU server log (last 40 lines):"
    tail -40 "$OUTPUT_DIR/ncu_server.log" 2>/dev/null
fi

log "============================================================"
