#!/usr/bin/env python3
"""Smart-trim server logs for repo storage.

For each server_*.log, creates a .trimmed.log with:
  - Deduplicated startup lines (8-worker warnings collapsed to 1)
  - Server config, model loading, and readiness info
  - Errors and warnings (non-repetitive)
  - Last few meaningful lines (shutdown, final stats)
  - Repetitive iteration logs collapsed to first + last + count

Usage:
    python scripts/trim_logs.py results_b200_fp8_mtp0_ep1/
    python scripts/trim_logs.py --all          # all results_*/ dirs
    python scripts/trim_logs.py --all --force   # re-trim even if .trimmed.log exists
"""

import argparse
import glob
import os
import re
import sys


# Patterns for lines that repeat across workers (keep only first occurrence)
DEDUP_PATTERNS = [
    re.compile(r"FutureWarning: The pynvml package is deprecated"),
    re.compile(r"Skipping import of cpp extensions due to incompatible torch"),
    re.compile(r"Multiple distributions found for package"),
    re.compile(r"transformers version .* is incompatible with nvidia-modelopt"),
    re.compile(r'\[TensorRT-LLM\] TensorRT LLM version:'),
    re.compile(r'Field name "schema" in "ResponseFormat"'),
    re.compile(r"import pynvml"),
]

# Repetitive runtime lines (collapse to count + first + last)
REPETITIVE_PATTERNS = [
    re.compile(r"\[TRT-LLM\] \[RANK 0\] \[I\] iter = \d+"),
    re.compile(r"num_scheduled_requests:.*num_ctx_requests.*num_generation_tokens"),
]

# Important lines to always keep
IMPORTANT_PATTERNS = [
    re.compile(r"\[TRT-LLM\].*\[E\]"),          # Errors
    re.compile(r"\[TRT-LLM\].*\[W\]"),           # Warnings (from rank 0)
    re.compile(r"Using LLM with"),                # Backend info
    re.compile(r"start MpiSession"),              # Multi-GPU setup
    re.compile(r"pre-quantized checkpoint"),      # Quantization
    re.compile(r"Orchestrator"),                  # Orchestrator mode
    re.compile(r"Loading model|loading weights|Loaded model", re.I),
    re.compile(r"Server (started|ready|listening)", re.I),
    re.compile(r"Uvicorn running"),
    re.compile(r"Shutting down"),
    re.compile(r"application shutdown"),
    re.compile(r"health.*ready|server.*ready", re.I),
    re.compile(r"kv.?cache", re.I),
    re.compile(r"mtp_layers|speculative|draft", re.I),
    re.compile(r"ep_size|tensor.parallel|pipeline.parallel", re.I),
    re.compile(r"max_model_len|max_num_seq", re.I),
    re.compile(r"RuntimeError|Exception|Traceback|Error:", re.I),
    re.compile(r"CUDA|GPU|device", re.I),
    re.compile(r"Memory|mem_fraction|free_mem", re.I),
]


def is_dedup_line(line):
    """Check if this line should be deduplicated (only keep first occurrence)."""
    for pat in DEDUP_PATTERNS:
        if pat.search(line):
            return pat.pattern
    return None


def is_repetitive(line):
    """Check if this is a repetitive runtime line."""
    for pat in REPETITIVE_PATTERNS:
        if pat.search(line):
            return True
    return False


def is_important(line):
    """Check if this line should always be kept."""
    for pat in IMPORTANT_PATTERNS:
        if pat.search(line):
            return True
    return False


def smart_trim(lines):
    """Smart-trim log lines: dedup, collapse repetitive, keep important."""
    output = []
    seen_dedup = set()        # Track which dedup patterns we've already kept
    rep_buffer = []           # Buffer for repetitive lines
    rep_count = 0

    def flush_repetitive():
        """Flush accumulated repetitive lines as summary."""
        nonlocal rep_buffer, rep_count
        if not rep_buffer:
            return
        if rep_count <= 3:
            output.extend(rep_buffer)
        else:
            output.append(rep_buffer[0])
            output.append(f"  ... [{rep_count - 2} similar iteration lines omitted] ...\n")
            output.append(rep_buffer[-1])
        rep_buffer = []
        rep_count = 0

    for line in lines:
        # Check if it's a dedup candidate
        dedup_key = is_dedup_line(line)
        if dedup_key:
            if dedup_key not in seen_dedup:
                seen_dedup.add(dedup_key)
                flush_repetitive()
                output.append(line)
            # else: skip duplicate
            continue

        # Check if it's a repetitive runtime line
        if is_repetitive(line):
            rep_buffer.append(line)
            rep_count += 1
            continue

        # Non-repetitive line: flush any pending repetitive buffer
        flush_repetitive()
        output.append(line)

    # Flush any remaining
    flush_repetitive()

    return output


def trim_log(log_path, force=False):
    """Trim a single log file. Returns True if file was created/updated."""
    trimmed_path = log_path.rsplit(".log", 1)[0] + ".trimmed.log"

    if not force and os.path.exists(trimmed_path):
        if os.path.getmtime(trimmed_path) > os.path.getmtime(log_path):
            return False  # Already up to date

    with open(log_path, "r", errors="replace") as f:
        lines = f.readlines()

    trimmed = smart_trim(lines)

    with open(trimmed_path, "w") as f:
        f.write(f"=== Trimmed log: {len(lines)} → {len(trimmed)} lines ===\n")
        f.write(f"=== Source: {os.path.basename(log_path)} ===\n\n")
        f.writelines(trimmed)

    return True


def main():
    parser = argparse.ArgumentParser(description="Smart-trim server logs")
    parser.add_argument("dirs", nargs="*", help="Result directories to process")
    parser.add_argument("--all", action="store_true", help="Process all results_*/ directories")
    parser.add_argument("--force", action="store_true", help="Re-trim even if .trimmed.log exists")
    args = parser.parse_args()

    dirs = list(args.dirs)
    if args.all:
        dirs.extend(sorted(glob.glob("results_*/")))

    if not dirs:
        print("ERROR: No directories specified. Use --all or pass directory paths.")
        sys.exit(1)

    total_created = 0
    total_skipped = 0

    for d in dirs:
        d = d.rstrip("/")
        if not os.path.isdir(d):
            print(f"WARN: {d} not found, skipping")
            continue

        logs = sorted(glob.glob(os.path.join(d, "server_*.log")))
        # Exclude .trimmed.log files
        logs = [l for l in logs if not l.endswith(".trimmed.log")]

        if not logs:
            print(f"  {d}: no server_*.log files")
            continue

        created = 0
        skipped = 0
        for log in logs:
            if trim_log(log, force=args.force):
                created += 1
            else:
                skipped += 1

        print(f"  {d}: {created} trimmed, {skipped} unchanged")
        total_created += created
        total_skipped += skipped

    print(f"\nTotal: {total_created} logs trimmed, {total_skipped} unchanged")


if __name__ == "__main__":
    main()
