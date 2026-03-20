#!/usr/bin/env bash
# =============================================================================
# Trim server logs to head+tail for repo storage.
#
# For each server_*.log in a results directory, creates a .trimmed.log with:
#   - First HEAD_LINES lines (startup, config, model loading)
#   - Last TAIL_LINES lines (shutdown, final stats)
#   - Separator showing how many lines were omitted
#
# Usage:
#   bash scripts/trim_logs.sh results_b200_fp8_mtp0_ep1/
#   bash scripts/trim_logs.sh results_b200_fp8_mtp3_ep1/ --head 120 --tail 60
#   bash scripts/trim_logs.sh --all   # process all results_*/ dirs
#
# Typical sizes: 400MB raw → 1.8MB trimmed (for ~80 log files)
# =============================================================================

set -euo pipefail

HEAD_LINES=100
TAIL_LINES=50
DIRS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --head)       HEAD_LINES="$2"; shift 2 ;;
        --tail)       TAIL_LINES="$2"; shift 2 ;;
        --all)
            for d in results_*/; do
                [[ -d "$d" ]] && DIRS+=("$d")
            done
            shift ;;
        -h|--help)
            echo "Usage: bash scripts/trim_logs.sh <results_dir> [--head N] [--tail N]"
            echo "       bash scripts/trim_logs.sh --all"
            exit 0 ;;
        *)  DIRS+=("$1"); shift ;;
    esac
done

if [[ ${#DIRS[@]} -eq 0 ]]; then
    echo "ERROR: No directories specified. Use --all or pass directory paths."
    exit 1
fi

total_created=0
total_skipped=0

for dir in "${DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "WARN: $dir not found, skipping"
        continue
    fi

    logs=("$dir"/server_*.log)
    if [[ ! -f "${logs[0]}" ]]; then
        echo "  $dir: no server_*.log files"
        continue
    fi

    created=0
    skipped=0
    for log in "${logs[@]}"; do
        # Skip if already a trimmed log
        [[ "$log" == *.trimmed.log ]] && continue

        trimmed="${log%.log}.trimmed.log"

        # Skip if trimmed version already exists and is newer than source
        if [[ -f "$trimmed" && "$trimmed" -nt "$log" ]]; then
            ((skipped++)) || true
            continue
        fi

        total_lines=$(wc -l < "$log")
        keep=$((HEAD_LINES + TAIL_LINES))

        if [[ $total_lines -le $keep ]]; then
            # Small file — just copy
            cp "$log" "$trimmed"
        else
            omitted=$((total_lines - keep))
            {
                head -n "$HEAD_LINES" "$log"
                echo ""
                echo "=== TRIMMED: $omitted lines omitted (of $total_lines total) ==="
                echo "=== Showing first $HEAD_LINES + last $TAIL_LINES lines ==="
                echo ""
                tail -n "$TAIL_LINES" "$log"
            } > "$trimmed"
        fi
        ((created++)) || true
    done

    echo "  $dir: ${created} trimmed, ${skipped} unchanged"
    total_created=$((total_created + created))
    total_skipped=$((total_skipped + skipped))
done

echo ""
echo "Total: ${total_created} logs trimmed, ${total_skipped} unchanged"
