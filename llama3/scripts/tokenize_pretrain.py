#!/usr/bin/env python3
"""
Tokenize pretraining data from Common Corpus → data/pretrain.bin

Downloads a subset of Common Corpus (public domain + open license)
and tokenizes it into the raw int32 binary format that DataLoader expects.

Usage:
    pip install datasets sentencepiece
    python scripts/tokenize_pretrain.py --tokenizer data/tokenizer.model --output data/pretrain.bin --max_tokens 1000000000

You need a SentencePiece tokenizer with 32k vocab (LLaMA-compatible).
Download one from HuggingFace or train your own.
"""

import argparse
import numpy as np
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Tokenize pretraining data")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to SentencePiece .model file (32k vocab)")
    parser.add_argument("--output", type=str, default="data/pretrain.bin",
                        help="Output binary file path")
    parser.add_argument("--max_tokens", type=int, default=1_000_000_000,
                        help="Stop after this many tokens (default: 1B)")
    parser.add_argument("--dataset", type=str, default="PleIAs/common_corpus",
                        help="HuggingFace dataset name")
    parser.add_argument("--text_field", type=str, default="text",
                        help="Name of the text field in the dataset")
    args = parser.parse_args()

    from sentencepiece import SentencePieceProcessor
    from datasets import load_dataset

    print(f"Loading tokenizer: {args.tokenizer}")
    sp = SentencePieceProcessor(model_file=args.tokenizer)
    vocab_size = sp.get_piece_size()
    print(f"  Vocab size: {vocab_size}")
    assert vocab_size == 32000, f"Expected 32000 vocab, got {vocab_size}"

    print(f"Streaming dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train", streaming=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total_tokens = 0
    buffer = []
    flush_size = 1_000_000  # write to disk every 1M tokens

    with open(args.output, "wb") as f:
        for i, example in enumerate(ds):
            text = example.get(args.text_field, "")
            if not text or len(text) < 50:
                continue

            tokens = sp.encode(text)
            buffer.extend(tokens)

            if len(buffer) >= flush_size:
                arr = np.array(buffer, dtype=np.int32)
                arr.tofile(f)
                total_tokens += len(buffer)
                buffer = []

                if i % 10000 == 0:
                    print(f"  {total_tokens:,} tokens ({total_tokens/1e9:.2f}B) from {i:,} documents")

                if total_tokens >= args.max_tokens:
                    print(f"Reached {args.max_tokens:,} token limit.")
                    break

        if buffer:
            arr = np.array(buffer, dtype=np.int32)
            arr.tofile(f)
            total_tokens += len(buffer)

    file_size = os.path.getsize(args.output)
    print(f"\nDone! Wrote {total_tokens:,} tokens to {args.output}")
    print(f"  File size: {file_size / 1e9:.2f} GB")
    print(f"  Tokens per param: {total_tokens / 125e6:.1f}x (125M model)")

if __name__ == "__main__":
    main()
