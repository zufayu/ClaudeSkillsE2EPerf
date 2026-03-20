#!/usr/bin/env bash
# Wrapper: calls trim_logs.py for smart log trimming
# Usage: bash scripts/trim_logs.sh [--all] [--force] [dir1 dir2 ...]
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/trim_logs.py" "$@"
