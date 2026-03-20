#!/usr/bin/env python3
"""Precision-trim server logs for repo storage.

For each server_*.log, extracts only the useful lines (~30) instead of
the full 16000+ line raw log. Extracts:
  - Reproduce commands from companion result_*.json
  - LLM Args runtime config (formatted as key=value pairs)
  - Memory profiling (model weights, peak, KV cache)
  - Runtime limits, attention features, MoE backend
  - Errors and warnings
  - Server startup/shutdown markers

Usage:
    python scripts/trim_logs.py results_b200_fp8_mtp0_ep1/
    python scripts/trim_logs.py --all          # all results_*/ dirs
    python scripts/trim_logs.py --all --force   # re-trim even if .trimmed.log exists
"""

import argparse
import glob
import json
import os
import re
import sys


# Whitelist patterns to extract from raw logs.
# Each entry: (compiled regex, section label, first_only flag)
WHITELIST = [
    (re.compile(r"LLM Args:"), "llm_args", False),
    (re.compile(r"Memory used after loading model weights"), "memory", False),
    (re.compile(r"Peak memory during memory usage profiling"), "memory", False),
    (re.compile(r"Estimated max memory in KV cache"), "memory", False),
    (re.compile(r"Setting PyTorch memory fraction"), "memory", False),
    (re.compile(r"max_seq_len=.*max_num_requests=.*max_batch_size="), "runtime", False),
    (re.compile(r"ATTENTION RUNTIME FEATURES", re.I), "runtime", False),
    (re.compile(r"AttentionRuntimeFeatures\("), "runtime", False),
    (re.compile(r"NVLinkOneSided.*Allocating workspace"), "comms", True),
    (re.compile(r"DeepGemmFusedMoE selects"), "moe", False),
    (re.compile(r"Application startup complete"), "status", False),
    (re.compile(r"Shutting down"), "status", False),
    (re.compile(r"\[E\]|Error|Exception|Traceback", re.I), "errors", False),
    (re.compile(r"Falling back to greedy decoding for MTP"), "warnings", False),
    (re.compile(r"Failed to send object"), "errors", False),
]


def extract_key_lines(lines):
    """Extract key lines from raw log using whitelist patterns.

    Returns dict mapping section labels to lists of extracted lines.
    """
    sections = {}
    seen_first_only = set()
    llm_args_next = False  # Flag to capture the line after "LLM Args:"

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # If previous line was "LLM Args:", capture this line as the args dump
        if llm_args_next:
            sections.setdefault("llm_args_dump", []).append(stripped)
            llm_args_next = False
            continue

        for pat, section, first_only in WHITELIST:
            if pat.search(stripped):
                # Skip if first_only and we've already seen it
                if first_only:
                    key = pat.pattern
                    if key in seen_first_only:
                        continue
                    seen_first_only.add(key)

                sections.setdefault(section, []).append(stripped)

                # If this is the "LLM Args:" header, capture next line too
                if section == "llm_args" and stripped.endswith("LLM Args:"):
                    llm_args_next = True
                break

    return sections


def format_llm_args(args_line):
    """Parse a long LLM Args dump line into readable key=value pairs.

    Splits on top-level space-separated key=value boundaries, handling
    nested parentheses and brackets.
    """
    pairs = []
    current = []
    depth = 0  # Track nesting depth for (), [], {}

    for char in args_line:
        if char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            current.append(char)
        elif char == " " and depth == 0:
            token = "".join(current).strip()
            if token:
                pairs.append(token)
            current = []
        else:
            current.append(char)

    # Don't forget the last token
    token = "".join(current).strip()
    if token:
        pairs.append(token)

    # Format as indented key=value lines
    formatted = []
    for pair in pairs:
        if "=" in pair:
            formatted.append(f"    {pair}")
        elif pair:
            # Continuation or standalone value
            if formatted:
                formatted[-1] += f" {pair}"
            else:
                formatted.append(f"    {pair}")

    return "\n".join(formatted)


def extract_reproduce_info(result_dir, tag):
    """Extract reproduce commands from companion result_*.json."""
    result_path = os.path.join(result_dir, f"result_{tag}.json")
    if not os.path.exists(result_path):
        return None

    try:
        with open(result_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    info = []
    info.append("=" * 70)
    info.append("REPRODUCE INFO (from result JSON)")
    info.append("=" * 70)

    if data.get("docker_image"):
        info.append(f"Docker Image: {data['docker_image']}")

    if data.get("server_cmd"):
        info.append("")
        info.append("Server Command:")
        info.append(f"  {data['server_cmd']}")

    if data.get("config_yaml"):
        info.append("")
        info.append("Extra LLM API Config (YAML):")
        for line in data["config_yaml"].strip().splitlines():
            info.append(f"  {line}")

    if data.get("benchmark_cmd"):
        info.append("")
        info.append("Benchmark Command:")
        info.append(f"  {data['benchmark_cmd']}")

    # Key metrics summary
    info.append("")
    info.append("Key Config:")
    for key in ["model_id", "max_model_len", "tensor_parallel_size", "ep_size",
                 "dp_attention", "moe_backend", "mtp_layers",
                 "piecewise_cuda_graphs", "kv_cache_free_mem_fraction",
                 "max_concurrency", "num_prompts"]:
        if key in data:
            info.append(f"  {key}: {data[key]}")

    # Outcome
    info.append("")
    completed = data.get("completed", 0)
    num_prompts = data.get("num_prompts", 0)
    out_tps = data.get("output_throughput", 0)
    if completed and num_prompts and completed < num_prompts:
        info.append(f"RESULT: PARTIAL - {completed}/{num_prompts} completed")
    elif completed:
        info.append(f"RESULT: OK - {completed}/{num_prompts} completed, "
                     f"output_throughput={out_tps:.1f} tok/s")
    else:
        info.append("RESULT: NO COMPLETIONS (likely failed)")

    if out_tps:
        info.append(f"  median_ttft_ms: {data.get('median_ttft_ms', 'N/A')}")
        info.append(f"  median_tpot_ms: {data.get('median_tpot_ms', 'N/A')}")

    info.append("=" * 70)
    return "\n".join(info) + "\n"


def format_output(sections, source_name, total_lines):
    """Format extracted sections into the final trimmed output."""
    out = []

    # Server Runtime section
    has_runtime = any(k in sections for k in
                      ["llm_args", "llm_args_dump", "runtime", "comms", "moe"])
    if has_runtime:
        out.append("[SERVER RUNTIME]")

        # LLM Args (formatted)
        if "llm_args_dump" in sections:
            out.append("  LLM Args:")
            for dump in sections["llm_args_dump"]:
                out.append(format_llm_args(dump))

        # Runtime limits
        for line in sections.get("runtime", []):
            # Extract the relevant part after the log prefix
            m = re.search(r"(max_seq_len=.*)", line)
            if m:
                out.append(f"  Runtime: {m.group(1)}")
            elif "AttentionRuntimeFeatures" in line:
                m = re.search(r"(AttentionRuntimeFeatures\(.*\))", line)
                if m:
                    out.append(f"  Attention: {m.group(1)}")
                else:
                    out.append(f"  Attention: {line}")
            else:
                out.append(f"  {line}")

        # Communication
        for line in sections.get("comms", []):
            m = re.search(r"(NVLinkOneSided.*)", line)
            if m:
                out.append(f"  Communication: {m.group(1)}")
            else:
                out.append(f"  Communication: {line}")

        # MoE
        for line in sections.get("moe", []):
            m = re.search(r"(DeepGemmFusedMoE.*)", line)
            if m:
                out.append(f"  MoE: {m.group(1)}")
            else:
                out.append(f"  MoE: {line}")

        out.append("")

    # Memory section
    if "memory" in sections:
        out.append("[MEMORY]")
        for line in sections["memory"]:
            # Extract meaningful part after log prefix
            m = re.search(r"Memory used after loading model weights \(inside torch\).*?: ([\d.]+\s+\w+)", line)
            if m:
                out.append(f"  Model weights (torch): {m.group(1)}")
                continue
            m = re.search(r"Memory used after loading model weights \(outside torch\).*?: ([\d.]+\s+\w+)", line)
            if m:
                out.append(f"  Model weights (non-torch): {m.group(1)}")
                continue
            m = re.search(r"Peak memory.*?: ([\d.]+\s+\w+).*?KV cache.*?: ([\d.]+\s+\w+).*?fraction.*?([\d.]+)", line)
            if m:
                out.append(f"  Peak memory: {m.group(1)}, KV cache: {m.group(2)} (fraction={m.group(3)})")
                continue
            m = re.search(r"Estimated max memory in KV cache\s*:\s*([\d.]+\s+\w+)", line)
            if m:
                out.append(f"  Estimated KV cache: {m.group(1)}")
                continue
            m = re.search(r"Setting PyTorch memory fraction to ([\d.]+)\s*\(([\d.]+\s+\w+)\)", line)
            if m:
                out.append(f"  PyTorch memory fraction: {m.group(1)} ({m.group(2)})")
                continue
            # Fallback: include the line as-is
            out.append(f"  {line}")
        out.append("")

    # Errors section (only if any) — deduplicate across ranks
    if "errors" in sections:
        out.append("[ERRORS]")
        seen_errors = set()
        for line in sections["errors"]:
            # Skip stack trace noise: hex addresses, C++ template symbols
            if re.match(r"^\d+\s+0x[0-9a-f]+\s", line):
                continue
            if "std::_Function_handler" in line or "std::vector<c10::IValue" in line:
                continue
            if line.startswith("^^^"):
                continue
            # Normalize: strip rank number and timestamp for dedup
            normalized = re.sub(r"\[RANK \d+\]", "[RANK *]", line)
            normalized = re.sub(r"\[\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2}\]", "[*]", normalized)
            if normalized in seen_errors:
                continue
            seen_errors.add(normalized)
            out.append(f"  {line}")
        out.append("")

    # Warnings section (only if any)
    if "warnings" in sections:
        out.append("[WARNINGS]")
        for line in sections["warnings"]:
            out.append(f"  {line}")
        out.append("")

    # Status section
    if "status" in sections:
        out.append("[STATUS]")
        for line in sections["status"]:
            # Extract timestamp if present
            m = re.search(r"\[(\d{2}:\d{2}:\d{2})\]", line)
            ts = f" ({m.group(1)})" if m else ""
            if "startup complete" in line.lower():
                out.append(f"  Server startup: OK{ts}")
            elif "shutting down" in line.lower():
                out.append(f"  Server shutdown:{ts}")
            else:
                out.append(f"  {line}")
        out.append("")

    output_lines = len(out)
    header = f"=== Trimmed: {total_lines} -> {output_lines} lines ===\n"
    header += f"=== Source: {source_name} ===\n"

    return header, "\n".join(out) + "\n"


def tag_from_log_name(log_name):
    """Extract tag from log filename.

    server_fp8_latency_chat_ep8_c1_dp.log -> fp8_latency_chat_ep8_c1_dp
    server_fp8_latency_chat_ep8_c1_dp.trimmed.log -> fp8_latency_chat_ep8_c1_dp
    """
    base = os.path.basename(log_name)
    base = base.replace(".trimmed.log", "").replace(".log", "")
    if base.startswith("server_"):
        base = base[len("server_"):]
    return base


def trim_log(log_path, force=False):
    """Trim a single log file. Returns True if file was created/updated."""
    trimmed_path = log_path.rsplit(".log", 1)[0] + ".trimmed.log"

    if not force and os.path.exists(trimmed_path):
        if os.path.getmtime(trimmed_path) > os.path.getmtime(log_path):
            return False  # Already up to date

    with open(log_path, "r", errors="replace") as f:
        lines = f.readlines()

    sections = extract_key_lines(lines)

    result_dir = os.path.dirname(log_path)
    tag = tag_from_log_name(log_path)
    reproduce_info = extract_reproduce_info(result_dir, tag)

    header, body = format_output(sections, os.path.basename(log_path), len(lines))

    with open(trimmed_path, "w") as f:
        f.write(header)
        f.write("\n")
        if reproduce_info:
            f.write(reproduce_info)
            f.write("\n")
        f.write(body)

    return True


def retrim_existing(trimmed_path, force=False):
    """Re-trim an existing .trimmed.log that has no raw log source.

    Returns True if file was updated.
    """
    with open(trimmed_path, "r", errors="replace") as f:
        lines = f.readlines()

    # Check if this is already a precision-trimmed file
    content = "".join(lines[:5])
    if not force and "[SERVER RUNTIME]" in content:
        return False  # Already precision-trimmed

    sections = extract_key_lines(lines)

    result_dir = os.path.dirname(trimmed_path)
    tag = tag_from_log_name(trimmed_path)
    reproduce_info = extract_reproduce_info(result_dir, tag)

    header, body = format_output(
        sections, f"{os.path.basename(trimmed_path)} (re-trimmed)", len(lines))

    with open(trimmed_path, "w") as f:
        f.write(header)
        f.write("\n")
        if reproduce_info:
            f.write(reproduce_info)
            f.write("\n")
        f.write(body)

    return True


def main():
    parser = argparse.ArgumentParser(description="Precision-trim server logs")
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
    total_retrimmed = 0

    for d in dirs:
        d = d.rstrip("/")
        if not os.path.isdir(d):
            print(f"WARN: {d} not found, skipping")
            continue

        # Find raw logs (excluding .trimmed.log)
        all_logs = sorted(glob.glob(os.path.join(d, "server_*.log")))
        raw_logs = [l for l in all_logs if not l.endswith(".trimmed.log")]
        trimmed_logs = [l for l in all_logs if l.endswith(".trimmed.log")]

        created = 0
        skipped = 0
        retrimmed = 0

        # Process raw logs -> create trimmed versions
        for log in raw_logs:
            if trim_log(log, force=args.force):
                created += 1
            else:
                skipped += 1

        # Process orphan trimmed logs (no raw source) -> re-trim in place
        raw_tags = {tag_from_log_name(l) for l in raw_logs}
        for tl in trimmed_logs:
            tag = tag_from_log_name(tl)
            if tag not in raw_tags:
                # This trimmed log has no raw source, re-trim it
                if retrim_existing(tl, force=args.force):
                    retrimmed += 1
                else:
                    skipped += 1

        if raw_logs or trimmed_logs:
            parts = []
            if created:
                parts.append(f"{created} trimmed")
            if retrimmed:
                parts.append(f"{retrimmed} re-trimmed")
            if skipped:
                parts.append(f"{skipped} unchanged")
            print(f"  {d}: {', '.join(parts) if parts else 'nothing to do'}")

        total_created += created
        total_skipped += skipped
        total_retrimmed += retrimmed

    print(f"\nTotal: {total_created} trimmed, {total_retrimmed} re-trimmed, "
          f"{total_skipped} unchanged")


if __name__ == "__main__":
    main()
