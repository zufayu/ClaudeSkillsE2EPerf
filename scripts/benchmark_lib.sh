#!/usr/bin/env bash
# =============================================================================
# Shared Benchmark Utilities for ClaudeSkillsE2EPerf
#
# Ported from InferenceX/InferenceMAX benchmark_lib.sh
# Provides: logging, process management, GPU monitoring, server readiness,
#           benchmark_serving client, preflight environment checks
#
# Source this at the top of any bench/collect script:
#   source "$SCRIPT_DIR/benchmark_lib.sh"
# =============================================================================

# --------------------------------
# Error trapping — auto-capture failures to .last_error
# Scripts can opt-in by calling enable_error_trap after sourcing.
# --------------------------------

enable_error_trap() {
  local repo_dir
  repo_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." 2>/dev/null && pwd || pwd)"
  local trap_file="${repo_dir}/.last_error"
  trap 'echo "[$(date)] FAILED: $BASH_COMMAND (exit $?, script=${BASH_SOURCE[0]:-unknown})" >> "'"$trap_file"'"' ERR
}

# --------------------------------
# Logging (eliminates 15 duplicate definitions across scripts)
# --------------------------------

# Guard: only define if not already defined by the sourcing script
if ! declare -f TS >/dev/null 2>&1; then
  TS() { date '+%Y-%m-%d %H:%M:%S'; }
fi
if ! declare -f log >/dev/null 2>&1; then
  log() { echo "[$(TS)] $*"; }
fi

# --------------------------------
# Safe process management
# CRITICAL: Never use pkill -f — it matches its own cmdline → exit 143
#   Evidence: 9 fix commits (9564e80, 7a8b506, 53c5b8f, 482c67e, a71cff4...)
#   Rule: R4 in feedback_workflow_architecture.md
# --------------------------------

# Kill processes matching pattern, excluding self and own process tree.
# Uses a temp file to avoid the pattern appearing in the pgrep cmdline.
safe_kill() {
  local pattern="$1"
  local signal="${2:--9}"
  local tmpf
  tmpf=$(mktemp /tmp/safe_kill_XXXXXX)
  echo "$pattern" > "$tmpf"
  local pids=""
  # Read pattern from file so it doesn't appear in our cmdline
  pids="$(pgrep -f "$(cat "$tmpf")" 2>/dev/null || true)"
  rm -f "$tmpf"
  if [[ -z "$pids" ]]; then return 0; fi
  # Exclude own PID and parent PID
  local my_pid=$$ parent_pid=$PPID
  pids="$(echo "$pids" | grep -vE "^(${my_pid}|${parent_pid})$" || true)"
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs -r kill "$signal" 2>/dev/null || true
    log "Killed processes matching '$pattern': $(echo $pids | tr '\n' ' ')"
  fi
}

# Kill all server processes for a given framework
kill_framework_procs() {
  local framework="$1"
  case "$framework" in
    trt)
      safe_kill "trtllm-serve"
      safe_kill "trtllm-bench"
      ;;
    sglang)
      safe_kill "sglang.launch_server"
      safe_kill "sglang.bench_serving"
      ;;
    atom|vllm)
      safe_kill "atom.entrypoints"
      safe_kill "vllm.entrypoints"
      safe_kill "model_executor"
      ;;
    *)
      log "WARN: Unknown framework '$framework' for kill_framework_procs"
      ;;
  esac
}

# --------------------------------
# Preflight environment checks
# Evidence: 10 fix commits for container env differences
#   (bfab83f, cb07c51, 3c95740, 0f3bbc0, 712d43d...)
#   Rule: R7 in feedback_workflow_architecture.md
# --------------------------------

# Check that a tool exists, exit if not
require_tool() {
  local tool="$1"
  if ! command -v "$tool" &>/dev/null; then
    log "ERROR: Required tool '$tool' not found in PATH"
    log "  PATH=$PATH"
    return 1
  fi
}

# Preflight for TRT-LLM environment
preflight_trt() {
  log "Preflight: TRT-LLM environment"
  require_tool python3
  require_tool trtllm-serve || log "WARN: trtllm-serve not found (may not be needed for bench mode)"
  python3 -c "import tensorrt_llm; print(f'  TRT-LLM {tensorrt_llm.__version__}')" 2>/dev/null || \
    log "WARN: tensorrt_llm import failed"
  python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null || \
    log "WARN: torch import failed"
}

# Preflight for SGLang environment
preflight_sglang() {
  log "Preflight: SGLang environment"
  require_tool python3
  python3 -c "import sglang; print(f'  SGLang {sglang.__version__}')" 2>/dev/null || \
    { log "ERROR: sglang import failed"; return 1; }
  python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null || \
    log "WARN: torch import failed"
}

# Preflight for ATOM/vLLM environment
preflight_atom() {
  log "Preflight: ATOM/vLLM environment"
  require_tool python3
  python3 -c "import vllm; print(f'  vLLM {vllm.__version__}')" 2>/dev/null || \
    log "WARN: vllm import failed (trying atom...)"
  python3 -c "
try:
    from atom import __version__; print(f'  ATOM {__version__}')
except (ImportError, AttributeError):
    # atom.__version__ doesn't exist in some versions (see fix ec2eaca)
    import importlib; m = importlib.import_module('atom'); print('  ATOM (version unknown)')
" 2>/dev/null || log "WARN: atom import failed"
}

# --------------------------------
# GPU monitoring helpers
# --------------------------------

GPU_MONITOR_PID=""
GPU_METRICS_CSV="${GPU_METRICS_CSV:-/tmp/gpu_metrics.csv}"

start_gpu_monitor() {
    local output="$GPU_METRICS_CSV"
    local interval=2

    while [[ $# -gt 0 ]]; do
        case $1 in
            --output)   output="$2"; shift 2 ;;
            --interval) interval="$2"; shift 2 ;;
            *)          shift ;;
        esac
    done

    GPU_METRICS_CSV="$output"

    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.current.sm,utilization.gpu,memory.used \
            --format=csv -l "$interval" > "$output" 2>/dev/null &
        GPU_MONITOR_PID=$!
        echo "[GPU Monitor] Started (PID=$GPU_MONITOR_PID, interval=${interval}s, output=$output)"
    else
        echo "[GPU Monitor] nvidia-smi not found, skipping"
        return 0
    fi
}

stop_gpu_monitor() {
    if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        echo "[GPU Monitor] Stopped (PID=$GPU_MONITOR_PID)"
        if [[ -f "$GPU_METRICS_CSV" ]]; then
            local lines
            lines=$(wc -l < "$GPU_METRICS_CSV")
            echo "[GPU Monitor] Collected $lines rows -> $GPU_METRICS_CSV"
        fi
    fi
    GPU_MONITOR_PID=""
}

# --------------------------------
# Environment variable checker
# --------------------------------

check_env_vars() {
    local missing_vars=()
    for var_name in "$@"; do
        if [[ -z "${!var_name}" ]]; then
            missing_vars+=("$var_name")
        fi
    done
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        echo "ERROR: Required environment variables not set:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        exit 1
    fi
}

# --------------------------------
# Server lifecycle helpers
# --------------------------------

# Wait for trtllm-serve (or any OpenAI-compatible server) to become healthy.
# Parameters:
#   --port:           Server port (required)
#   --server-log:     Path to server log file (required)
#   --server-pid:     Server process PID (required)
#   --max-wait:       Maximum wait in seconds (default: 900 = 15min)
#   --sleep-interval: Poll interval in seconds (default: 5)
wait_for_server_ready() {
    local port="" server_log="" server_pid=""
    local max_wait=900 sleep_interval=5

    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)           port="$2"; shift 2 ;;
            --server-log)     server_log="$2"; shift 2 ;;
            --server-pid)     server_pid="$2"; shift 2 ;;
            --max-wait)       max_wait="$2"; shift 2 ;;
            --sleep-interval) sleep_interval="$2"; shift 2 ;;
            *)                echo "Unknown parameter: $1"; return 1 ;;
        esac
    done

    if [[ -z "$port" || -z "$server_log" || -z "$server_pid" ]]; then
        echo "ERROR: --port, --server-log, and --server-pid are all required"
        return 1
    fi

    # Wait for log file to appear
    local elapsed=0
    while [[ ! -f "$server_log" ]]; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "ERROR: Server died before creating log file"
            return 1
        fi
        sleep 1
        elapsed=$((elapsed + 1))
        if [[ $elapsed -ge $max_wait ]]; then
            echo "ERROR: Log file never appeared after ${max_wait}s"
            return 1
        fi
    done

    # Tail logs while waiting
    tail -f -n +1 "$server_log" &
    local TAIL_PID=$!

    echo "[Server] Waiting for health on port $port (max ${max_wait}s)..."
    while ! curl --output /dev/null --silent --fail "http://0.0.0.0:${port}/health" 2>/dev/null; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "ERROR: Server process died"
            kill "$TAIL_PID" 2>/dev/null
            echo "--- Last 30 lines of server log ---"
            tail -30 "$server_log"
            return 1
        fi
        sleep "$sleep_interval"
        elapsed=$((elapsed + sleep_interval))
        if [[ $elapsed -ge $max_wait ]]; then
            echo "ERROR: Server not ready after ${max_wait}s"
            kill "$TAIL_PID" 2>/dev/null
            echo "--- Last 30 lines of server log ---"
            tail -30 "$server_log"
            return 1
        fi
    done

    kill "$TAIL_PID" 2>/dev/null
    echo "[Server] Ready (took ~${elapsed}s)"
}

# Kill any running trtllm-serve processes
kill_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[Server] Stopping PID=$SERVER_PID..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    safe_kill "trtllm-serve"
    sleep 3
}

# --------------------------------
# Benchmark client (benchmark_serving.py)
# --------------------------------

# Run benchmark_serving.py against a running OpenAI-compatible server.
# Requires benchmark_serving.py to be available (from InferenceX utils/).
#
# Parameters:
#   --model:              Model name/path (required)
#   --port:               Server port (required)
#   --backend:            Backend type: openai|vllm (required)
#   --input-len:          Input sequence length (required)
#   --output-len:         Output sequence length (required)
#   --random-range-ratio: Random range ratio for len variation (required)
#   --num-prompts:        Number of prompts (required)
#   --max-concurrency:    Max concurrent requests (required)
#   --result-filename:    Result filename without .json ext (required)
#   --result-dir:         Result directory (required)
#   --use-chat-template:  Optional flag
#   --bench-serving-dir:  Directory containing benchmark_serving.py (auto-detected)
run_benchmark_serving() {
    local model="" port="" backend="openai"
    local input_len="" output_len="" random_range_ratio=""
    local num_prompts="" max_concurrency=""
    local result_filename="" result_dir=""
    local workspace_dir=""
    local use_chat_template=false
    local num_warmups=""
    local -a metadata_args=()

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)              model="$2"; shift 2 ;;
            --port)               port="$2"; shift 2 ;;
            --backend)            backend="$2"; shift 2 ;;
            --input-len)          input_len="$2"; shift 2 ;;
            --output-len)         output_len="$2"; shift 2 ;;
            --random-range-ratio) random_range_ratio="$2"; shift 2 ;;
            --num-prompts)        num_prompts="$2"; shift 2 ;;
            --max-concurrency)    max_concurrency="$2"; shift 2 ;;
            --num-warmups)        num_warmups="$2"; shift 2 ;;
            --result-filename)    result_filename="$2"; shift 2 ;;
            --result-dir)         result_dir="$2"; shift 2 ;;
            --bench-serving-dir)  workspace_dir="$2"; shift 2 ;;
            --use-chat-template)  use_chat_template=true; shift ;;
            --metadata)           shift; while [[ $# -gt 0 && "$1" != --* ]]; do metadata_args+=("$1"); shift; done ;;
            *)                    echo "Unknown parameter: $1"; return 1 ;;
        esac
    done

    # Validate required params
    for v in model port input_len output_len random_range_ratio num_prompts max_concurrency result_filename result_dir; do
        if [[ -z "${!v}" ]]; then
            echo "ERROR: --${v//_/-} is required"
            return 1
        fi
    done

    # Auto-detect benchmark_serving.py location
    if [[ -z "$workspace_dir" ]]; then
        local script_dir
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        # Check common locations
        for candidate in \
            "$script_dir/../utils/bench_serving" \
            "$script_dir/bench_serving" \
            "/home/kqian/InferenceX/utils/bench_serving" \
            "$(pwd)/utils/bench_serving"; do
            if [[ -f "$candidate/benchmark_serving.py" ]]; then
                workspace_dir="$(dirname "$candidate")"
                break
            fi
        done
        if [[ -z "$workspace_dir" ]]; then
            workspace_dir="$(pwd)"
        fi
    fi

    local bench_py=""
    for candidate in \
        "$workspace_dir/utils/bench_serving/benchmark_serving.py" \
        "$workspace_dir/bench_serving/benchmark_serving.py" \
        "$workspace_dir/benchmark_serving.py"; do
        if [[ -f "$candidate" ]]; then
            bench_py="$candidate"
            break
        fi
    done

    if [[ -z "$bench_py" ]]; then
        echo "ERROR: Cannot find benchmark_serving.py"
        echo "  Searched in: $workspace_dir"
        echo "  Set --bench-serving-dir or copy benchmark_serving.py to utils/bench_serving/"
        return 1
    fi

    mkdir -p "$result_dir"

    local benchmark_cmd=(
        python3 "$bench_py"
        --model "$model"
        --backend "$backend"
        --base-url "http://0.0.0.0:$port"
        --dataset-name random
        --random-input-len "$input_len"
        --random-output-len "$output_len"
        --random-range-ratio "$random_range_ratio"
        --num-prompts "$num_prompts"
        --max-concurrency "$max_concurrency"
        --request-rate inf
        --ignore-eos
        --save-result
        --num-warmups "${num_warmups:-$(( max_concurrency * 2 ))}"
        --percentile-metrics 'ttft,tpot,itl,e2el'
        --result-dir "$result_dir"
        --result-filename "${result_filename}.json"
    )

    if [[ "$use_chat_template" == true ]]; then
        benchmark_cmd+=(--use-chat-template)
    fi

    if [[ ${#metadata_args[@]} -gt 0 ]]; then
        benchmark_cmd+=(--metadata "${metadata_args[@]}")
    fi

    echo "[Benchmark] Running: ${benchmark_cmd[*]}"
    set -x
    "${benchmark_cmd[@]}"
    local rc=$?
    set +x
    return $rc
}
