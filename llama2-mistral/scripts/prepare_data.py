#!/usr/bin/env python3
"""
Tokenize a text file into a binary file of int32 token IDs.
Output format: raw int32s, no header — ready for the C++ DataLoader.

Usage:
    python scripts/prepare_data.py --input data/train.txt --output data/train.bin

The script uses the HuggingFace tokenizer for codellama/CodeLlama-7b-hf
(same 32000 vocab as our model). You can swap for any sentencepiece tokenizer.

Install deps:
    pip install transformers sentencepiece
"""

import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Tokenize text → binary int32 file")
    parser.add_argument("--input",  required=True, help="path to input .txt file")
    parser.add_argument("--output", required=True, help="path to output .bin file")
    parser.add_argument("--tokenizer", default="codellama/CodeLlama-7b-hf",
                        help="HuggingFace tokenizer name (default: CodeLlama 32k vocab)")
    args = parser.parse_args()

    # --- Step 1: Load tokenizer ---
    print(f"Loading tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # --- Step 2: Read input text ---
    print(f"Reading: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  {len(text):,} characters")

    # --- Step 3: Tokenize ---
    print("Tokenizing...")
    token_ids = tokenizer.encode(text)
    tokens = np.array(token_ids, dtype=np.int32)
    print(f"  {len(tokens):,} tokens")

    # --- Step 4: Write binary file ---
    print(f"Writing: {args.output}")
    tokens.tofile(args.output)
    file_size_mb = len(tokens) * 4 / (1024 * 1024)
    print(f"  {file_size_mb:.1f} MB ({len(tokens):,} tokens × 4 bytes)")

    print("Done!")

if __name__ == "__main__":
    main()
