# LLaMA 3-Style LLM from Scratch in CUDA/C++

A **LLaMA 3-style** language model built on the exact same hand-written CUDA kernels as [`../llama2-mistral`](../llama2-mistral). LLaMA 3 is *not* a new architecture — it's LLaMA 2's block with a bigger tokenizer, a higher RoPE base, and a longer context window. So this model reuses the shared [`../cuda/`](../cuda) library unchanged and only differs in config.

## What's different from LLaMA 2

Three values, all in [`src/config.h`](src/config.h) — nothing in the kernels changes:

| Knob | LLaMA 2 / Mistral | LLaMA 3 (here) | Why |
|---|---|---|---|
| **Vocab / tokenizer** | 32,000 (SentencePiece) | **128,256** (tiktoken-style BPE) | ~15% fewer tokens per string; better multilingual + code |
| **RoPE base θ** | 10,000 | **500,000** | keeps positions distinguishable over long context |
| **Context length** | 512–4K | **8,192** (3.1 extends to 128K) | long-context training |

Everything else — GQA (12 q / 4 kv), RMSNorm, SwiGLU, pre-norm, fp16 mixed precision, AdamW — is identical to the LLaMA 2 model and runs on the same kernels.

## Architecture

| Component | Detail |
|---|---|
| Parameters | ~272M (128K vocab makes embedding + output ~197M of it) |
| Layers | 12 |
| Embedding dim | 768 |
| Attention | Grouped-Query (12 query heads, 4 KV heads) |
| MLP | SwiGLU (768 → 2048 → 768) |
| Normalization | RMSNorm (pre-norm) |
| Position encoding | RoPE (θ = 500,000) |
| Vocab | 128,256 (tiktoken-style BPE) |
| Context length | 8,192 |

> **Scale note:** at `n_embd = 768`, the 128K vocab makes the embedding/output layer dominate the parameter count (~197M of ~272M). That's the honest cost of the LLaMA 3 tokenizer at this small size — it's exactly why the LLaMA 2 model here keeps a 32K vocab. For a serious LLaMA-3-class run you'd scale `n_embd`/layers up so the vocab isn't the whole model.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The `CMakeLists.txt` compiles the shared kernels in `../cuda/` into a static library and links the `train` executable. Requirements: an NVIDIA GPU (Compute Capability 7.0+), CUDA Toolkit 11.0+, CMake 3.18+.

## Pretrain

LLaMA 3 uses a **128K tiktoken-style BPE tokenizer**, so the only change to the data pipeline versus LLaMA 2 is the tokenizer you point at (it must be a 128K-vocab model, e.g. a LLaMA 3 tokenizer). The model reads the same binary `int32` token files.

```bash
pip install sentencepiece datasets

# 1. Tokenize with a 128K (LLaMA 3) tokenizer  → data/pretrain.bin
python scripts/tokenize_pretrain.py \
  --tokenizer data/tokenizer_llama3.model \
  --output data/pretrain.bin \
  --max_tokens 500000000

# 2. Pretrain (fp16 mixed precision). Long context (8K) is memory-heavy on 12GB,
#    so start at the default seq len and raise it once it's stable.
mkdir -p checkpoints
stdbuf -oL ./build/train data/pretrain.bin \
  --fp16 --steps 5000 --batch 4 --accum 16 --save 500 2>&1 | tee pretrain.log
```

To train at the full 8K context (drop the micro-batch to fit 12GB):

```bash
stdbuf -oL ./build/train data/pretrain.bin \
  --fp16 --seq 8192 --batch 1 --accum 64 --steps 5000 --save 500 2>&1 | tee pretrain_8k.log
```

### Command-Line Options
| Flag | Default | Description |
|---|---|---|
| `--fp16` | off | Enable fp16 mixed precision (tensor cores) |
| `--steps N` | 30000 | Total optimizer steps |
| `--batch N` | 4 | Micro-batch size |
| `--seq N` | 512 | Sequence length (≤ `max_seq_len` = 8192) |
| `--accum N` | 16 | Gradient accumulation steps |
| `--lr F` | 3e-4 | Peak learning rate |
| `--warmup N` | 2000 | LR warmup steps |
| `--save N` | 2000 | Checkpoint interval |
| `--checkpoint PATH` | none | Load weights from file |

## Documentation

The math behind every component is written up in the shared [`../documentation/`](../documentation):

- **[`algorithm_explanations/`](../documentation/algorithm_explanations)** — attention/GQA, RoPE, RMSNorm, SwiGLU, softmax, cross-entropy, embeddings, matmul, AdamW + LR warmup, KV cache, Flash Attention.
- **[`backprop_explanations/`](../documentation/backprop_explanations)** — gradient derivations for every kernel plus a full end-to-end backward pass.
- **[`training_instructions/`](../documentation/training_instructions)** — pretraining setup and a breakdown of every training parameter.

## Kernels & Engineering

This model shares the kernel library and all the mixed-precision engineering with the LLaMA 2 model — see [`../llama2-mistral/README.md`](../llama2-mistral/README.md) for the full kernel table and engineering write-up (CAS-based `atomicAdd` for half types, fp32 embedding gradients, GPU-side gradient-norm sanitization, fp32-in-optimizer gradient clipping, and the numerically stable mixed-precision backward pass).

## License

MIT
