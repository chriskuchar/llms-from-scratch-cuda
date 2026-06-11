#!/usr/bin/env python3
"""
Tokenize Python code data for code continue-pretraining → data/code.bin

Downloads Python source code from bigcode/starcoderdata and tokenizes it
into the raw int32 binary format that DataLoader expects.

Usage:
    pip install datasets sentencepiece
    python scripts/tokenize_code.py --tokenizer data/tokenizer.model --output data/code.bin --max_tokens 200000000
"""

import argparse
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Tokenize Python code data")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to SentencePiece .model file (32k vocab)")
    parser.add_argument("--output", type=str, default="data/code.bin",
                        help="Output binary file path")
    parser.add_argument("--max_tokens", type=int, default=200_000_000,
                        help="Stop after this many tokens (default: 200M)")
    parser.add_argument("--dataset", type=str, default="bigcode/starcoderdata",
                        help="HuggingFace dataset name")
    parser.add_argument("--lang", type=str, default="python",
                        help="Programming language to filter (default: python)")
    args = parser.parse_args()

    from sentencepiece import SentencePieceProcessor
    from datasets import load_dataset

    print(f"Loading tokenizer: {args.tokenizer}")
    sp = SentencePieceProcessor(model_file=args.tokenizer)
    vocab_size = sp.get_piece_size()
    print(f"  Vocab size: {vocab_size}")
    assert vocab_size == 32000, f"Expected 32000 vocab, got {vocab_size}"

    eos_id = sp.eos_id()

    print(f"Streaming dataset: {args.dataset} (lang={args.lang})")
    ds = load_dataset(args.dataset, data_dir=args.lang, split="train", streaming=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total_tokens = 0
    buffer = []
    flush_size = 1_000_000
    n_files = 0

    with open(args.output, "wb") as f:
        for i, example in enumerate(ds):
            content = example.get("content", "")
            if not content or len(content) < 50:
                continue

            # skip files that are mostly non-code (data files, configs, etc.)
            if content.count("\n") < 3:
                continue

            tokens = sp.encode(content)
            tokens.append(eos_id)  # separate files with EOS
            buffer.extend(tokens)
            n_files += 1

            if len(buffer) >= flush_size:
                arr = np.array(buffer, dtype=np.int32)
                arr.tofile(f)
                total_tokens += len(buffer)
                buffer = []

                print(f"  {total_tokens:,} tokens ({total_tokens/1e6:.0f}M) from {n_files:,} files")

                if total_tokens >= args.max_tokens:
                    print(f"Reached {args.max_tokens:,} token limit.")
                    break

        if buffer:
            arr = np.array(buffer, dtype=np.int32)
            arr.tofile(f)
            total_tokens += len(buffer)

    file_size = os.path.getsize(args.output)
    print(f"\nDone! Wrote {total_tokens:,} tokens from {n_files:,} Python files to {args.output}")
    print(f"  File size: {file_size / 1e6:.0f} MB")

if __name__ == "__main__":
    main()
