#!/usr/bin/env python3
"""
Tokenize instruction/Q&A data for SFT → data/instruct.bin

Downloads instruction-following datasets and tokenizes them into the raw
int32 binary format. Conversations are formatted as:

    <s>User: {question}\nAssistant: {answer}</s>

This teaches the model to respond to questions and follow instructions.

Usage:
    pip install datasets sentencepiece
    python scripts/tokenize_instruct.py --tokenizer data/tokenizer.model --output data/instruct.bin

Available datasets (use --dataset flag):
    yahma/alpaca-cleaned          52K simple instructions (fast, good starting point)
    teknium/OpenHermes-2.5        1M mixed instructions (best quality, slower to download)
    iamtarun/python_code_instructions_18k_alpaca  18K code instructions
"""

import argparse
import numpy as np
import os

DATASET_CONFIGS = {
    "yahma/alpaca-cleaned": {
        "input_field": "input",
        "instruction_field": "instruction",
        "output_field": "output",
        "format": "alpaca",
    },
    "teknium/OpenHermes-2.5": {
        "format": "openhermes",
    },
    "iamtarun/python_code_instructions_18k_alpaca": {
        "input_field": "input",
        "instruction_field": "instruction",
        "output_field": "output",
        "format": "alpaca",
    },
}

def format_alpaca(example, cfg):
    instruction = example.get(cfg["instruction_field"], "")
    inp = example.get(cfg["input_field"], "")
    output = example.get(cfg["output_field"], "")

    if inp and inp.strip():
        user_msg = f"{instruction}\n{inp}"
    else:
        user_msg = instruction

    return f"User: {user_msg}\nAssistant: {output}"

def format_openhermes(example):
    convos = example.get("conversations", [])
    if not convos:
        return ""

    parts = []
    for turn in convos:
        role = turn.get("from", "")
        value = turn.get("value", "")
        if role == "human":
            parts.append(f"User: {value}")
        elif role == "gpt":
            parts.append(f"Assistant: {value}")
    return "\n".join(parts)

def main():
    parser = argparse.ArgumentParser(description="Tokenize instruction data for SFT")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to SentencePiece .model file (32k vocab)")
    parser.add_argument("--output", type=str, default="data/instruct.bin",
                        help="Output binary file path")
    parser.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned",
                        help="HuggingFace dataset name")
    parser.add_argument("--max_examples", type=int, default=0,
                        help="Max examples to process (0 = all)")
    args = parser.parse_args()

    from sentencepiece import SentencePieceProcessor
    from datasets import load_dataset

    print(f"Loading tokenizer: {args.tokenizer}")
    sp = SentencePieceProcessor(model_file=args.tokenizer)
    vocab_size = sp.get_piece_size()
    print(f"  Vocab size: {vocab_size}")
    assert vocab_size == 32000, f"Expected 32000 vocab, got {vocab_size}"

    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    print(f"  {len(ds):,} examples")

    cfg = DATASET_CONFIGS.get(args.dataset, {"format": "alpaca",
                                               "instruction_field": "instruction",
                                               "input_field": "input",
                                               "output_field": "output"})

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_tokens = []
    skipped = 0
    max_ex = args.max_examples if args.max_examples > 0 else len(ds)

    for i, example in enumerate(ds):
        if i >= max_ex:
            break

        if cfg["format"] == "openhermes":
            text = format_openhermes(example)
        else:
            text = format_alpaca(example, cfg)

        if not text or len(text) < 20:
            skipped += 1
            continue

        tokens = [bos_id] + sp.encode(text) + [eos_id]
        all_tokens.extend(tokens)

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,} / {max_ex:,} examples ({len(all_tokens):,} tokens)")

    arr = np.array(all_tokens, dtype=np.int32)
    arr.tofile(args.output)

    n_used = min(len(ds), max_ex) - skipped
    file_size = os.path.getsize(args.output)
    print(f"\nDone! Wrote {len(all_tokens):,} tokens from {n_used:,} examples to {args.output}")
    print(f"  File size: {file_size / 1e6:.1f} MB")
    print(f"  Skipped: {skipped} empty examples")
    print(f"  Avg tokens per example: {len(all_tokens) // max(n_used, 1)}")

if __name__ == "__main__":
    main()
