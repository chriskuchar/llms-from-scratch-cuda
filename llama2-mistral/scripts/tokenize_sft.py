#!/usr/bin/env python3
"""
Tokenize Glaive function-calling v2 dataset → data/sft_train.bin

Converts the Glaive function-calling conversations into the raw int32
binary format that DataLoader expects. Each conversation is tokenized
as a flat sequence with an EOS token between examples.

Usage:
    pip install datasets sentencepiece
    python scripts/tokenize_sft.py --tokenizer data/tokenizer.model --output data/sft_train.bin

Dataset: glaiveai/glaive-function-calling-v2 (Apache 2.0 license)
"""

import argparse
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Tokenize SFT function-calling data")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to SentencePiece .model file (32k vocab)")
    parser.add_argument("--output", type=str, default="data/sft_train.bin",
                        help="Output binary file path")
    parser.add_argument("--dataset", type=str, default="glaiveai/glaive-function-calling-v2",
                        help="HuggingFace dataset name")
    args = parser.parse_args()

    from sentencepiece import SentencePieceProcessor
    from datasets import load_dataset

    print(f"Loading tokenizer: {args.tokenizer}")
    sp = SentencePieceProcessor(model_file=args.tokenizer)
    vocab_size = sp.get_piece_size()
    print(f"  Vocab size: {vocab_size}")
    assert vocab_size == 32000, f"Expected 32000 vocab, got {vocab_size}"

    eos_id = sp.eos_id()
    print(f"  EOS token ID: {eos_id}")

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    print(f"  {len(ds):,} examples")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_tokens = []
    skipped = 0

    for i, example in enumerate(ds):
        # Glaive v2 has a single text field with the full conversation
        # Format: SYSTEM: ... \nUSER: ... \nASSISTANT: ...
        text = ""
        if "system" in example and example["system"]:
            text += example["system"] + "\n"
        if "chat" in example and example["chat"]:
            text += example["chat"]
        elif "text" in example and example["text"]:
            text = example["text"]

        if not text or len(text) < 20:
            skipped += 1
            continue

        tokens = sp.encode(text)
        tokens.append(eos_id)  # separator between examples
        all_tokens.extend(tokens)

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,} examples ({len(all_tokens):,} tokens)")

    arr = np.array(all_tokens, dtype=np.int32)
    arr.tofile(args.output)

    file_size = os.path.getsize(args.output)
    print(f"\nDone! Wrote {len(all_tokens):,} tokens to {args.output}")
    print(f"  File size: {file_size / 1e6:.1f} MB")
    print(f"  Skipped: {skipped} empty examples")
    print(f"  Avg tokens per example: {len(all_tokens) // (len(ds) - skipped)}")

if __name__ == "__main__":
    main()
