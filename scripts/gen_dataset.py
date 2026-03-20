#!/usr/bin/env python3
"""Generate benchmark dataset for trtllm-bench throughput tests."""

import argparse
import json
import random
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark dataset for TRT-LLM benchmarking")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to HuggingFace tokenizer or model directory")
    parser.add_argument("--num_requests", type=int, default=50,
                        help="Total number of requests to generate")
    parser.add_argument("--output_tokens", type=int, default=128,
                        help="Number of output tokens per request")
    parser.add_argument("--output", type=str,
                        default="benchmark_dataset.json",
                        help="Output JSONL file path")
    parser.add_argument("--input_mode", type=str, default="synthetic",
                        choices=["synthetic", "fixed_len", "random"],
                        help="Input generation mode (random: random tokens for realistic DAR)")
    parser.add_argument("--fixed_input_len", type=int, default=512,
                        help="Fixed input length when input_mode=fixed_len")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True)

    prompts = [
        "Please explain the theory of general relativity in detail, covering its key principles and implications.",
        "Write a comprehensive analysis of the economic impacts of artificial intelligence on global markets.",
        "Describe the process of photosynthesis at a molecular level, including all major steps and energy transfers.",
        "What are the main differences between classical and quantum computing? Provide detailed technical comparisons.",
        "Explain the history and evolution of programming languages from Assembly to modern high-level languages.",
        "Discuss the ethical implications of genetic engineering in human medicine and agriculture.",
        "Analyze the causes and effects of climate change, including proposed mitigation strategies.",
        "Write a detailed overview of machine learning algorithms, comparing supervised and unsupervised approaches.",
        "Explain how neural networks work, from basic perceptrons to modern transformer architectures.",
        "Describe the key events and turning points of World War II and their lasting global impacts.",
    ]

    dataset = []
    for i in range(args.num_requests):
        if args.input_mode == "synthetic":
            prompt = prompts[i % len(prompts)]
            input_ids = tokenizer.encode(prompt)
        elif args.input_mode == "fixed_len":
            # Generate fixed-length input by repeating tokens
            base = tokenizer.encode("Repeat this text for benchmarking. ")
            input_ids = (base * (args.fixed_input_len // len(base) + 1)
                         )[:args.fixed_input_len]
        else:  # random
            # Random tokens from vocab for realistic DAR measurement
            vocab_size = tokenizer.vocab_size
            input_ids = [random.randint(100, vocab_size - 1)
                         for _ in range(args.fixed_input_len)]

        dataset.append({
            "task_id": i,
            "input_ids": input_ids,
            "output_tokens": args.output_tokens,
        })

    with open(args.output, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    avg_input = sum(len(d["input_ids"]) for d in dataset) / len(dataset)
    print(f"Dataset written to {args.output}")
    print(f"  Requests: {len(dataset)}")
    print(f"  Avg input tokens: {avg_input:.0f}")
    print(f"  Output tokens: {args.output_tokens}")


if __name__ == "__main__":
    main()
