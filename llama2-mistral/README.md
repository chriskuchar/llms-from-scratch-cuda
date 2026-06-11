# LLaMA 2 / Mistral-Style LLM from Scratch in CUDA/C++

A 125M parameter **LLaMA 2 / Mistral-style** language model implemented entirely from scratch — no PyTorch, no TensorFlow, no frameworks. Every layer, every gradient, and every GPU kernel is hand-written in pure CUDA and C++.

**~4,000 lines of code. Zero dependencies beyond CUDA and cuBLAS.**

This README is about *how the model is built* — the architecture and the CUDA kernels behind it.

## Architecture

A **LLaMA 2 / Mistral-style** decoder: pre-norm transformer blocks with grouped-query attention, rotary position embeddings, RMSNorm, and a SwiGLU MLP. The defining LLaMA-2-era choices here are **grouped-query attention** and a **32K SentencePiece vocab** — LLaMA 3 keeps the same block structure but moves to a 128K tiktoken vocab and a higher RoPE base, which at this scale would let the embedding/output layer dominate the parameter count, so the 32K vocab is the deliberate choice for a 125M model.

```
Token IDs → [Embedding] → x
                          ↓
              ┌───────────────────────┐
              │   × 12 Transformer    │
              │   Blocks              │
              │                       │
              │   RMSNorm → GQA Attn  │
              │       ↓    (12q/4kv)  │
              │   + Residual          │
              │       ↓               │
              │   RMSNorm → SwiGLU    │
              │       ↓    MLP        │
              │   + Residual          │
              └───────────────────────┘
                          ↓
              RMSNorm → Linear → Logits → Cross-Entropy Loss
```

| Component | Detail |
|---|---|
| Parameters | 124.7M |
| Layers | 12 |
| Embedding dim | 768 |
| Attention | Grouped-Query (12 query heads, 4 KV heads) |
| MLP | SwiGLU (768 → 2048 → 768) |
| Normalization | RMSNorm (pre-norm) |
| Position encoding | RoPE (Rotary Position Embeddings) |
| Vocab | 32,000 (SentencePiece, LLaMA-compatible) |
| Context length | 512 tokens (extensible to 4096+ with Flash Attention) |

### How close is this to LLaMA 3?

Structurally, **almost identical** — LLaMA 3's gains came from data and scale (~15T tokens), not a new architecture. The same GQA + RMSNorm + SwiGLU + pre-norm decoder you see here *is* the LLaMA 3 block. Turning this into a LLaMA-3-style model is essentially three changes:

1. **Tokenizer:** 32K SentencePiece → 128K tiktoken-style BPE (resize the embedding + output projection).
2. **RoPE base:** θ = 10,000 → 500,000 (a one-line constant in `rope.cu`) for long-context stability.
3. **Context length:** extend from 512 toward 8K / 128K.

Everything else — the kernels, the attention, the norms, the MLP — is already LLaMA-3-compatible. The 32K vocab is kept on purpose: at 125M params a 128K vocab would make the embedding/output layer dominate the model.

## CUDA Kernels

Every operation runs on the GPU via hand-written CUDA kernels — forward *and* backward. No cuDNN, no CUTLASS; just raw CUDA and cuBLAS for the matrix multiplies.

| Kernel | File | Lines | What It Does |
|---|---|---|---|
| Matrix Multiply | `matmul.cu` | 120 | fp32 + fp16 (tensor core) matmul via cuBLAS, forward + backward |
| Attention | `attention.cu` | 738 | Full GQA: score, softmax, value, all 8 backward sub-kernels |
| SwiGLU MLP | `swiglu.cu` | 237 | Gate/up projection, SiLU activation, down projection |
| RoPE | `rope.cu` | 216 | Rotary position embeddings, forward + inverse backward |
| RMSNorm | `rmsnorm.cu` | 181 | Root mean square normalization with learnable scale |
| Cross-Entropy | `crossentropy.cu` | 152 | Fused softmax + log-loss forward, combined backward |
| Embedding | `embedding.cu` | 127 | Lookup forward, scatter-add backward (fp32 for stability) |
| AdamW | `adamw.cu` | 218 | Mixed-precision optimizer + GPU-side gradient norm/sanitize |
| Softmax | `softmax.cu` | 119 | Numerically stable softmax (max-subtract trick) |
| Residual | `residual.cu` | 80 | Skip connections forward + backward split |

All kernels have both **fp32 and fp16 overloads** — the fp16 path reads/writes `__half` but accumulates in fp32 internally to maintain numerical stability.

## fp16 Mixed Precision

A full mixed-precision pipeline, not just casting weights to fp16:

- **fp16 weights** — half the memory, enables tensor cores (2x theoretical throughput)
- **fp32 master weights** — optimizer updates in full precision, then synced to fp16
- **fp32 embedding gradients** — scatter-add accumulation overflows fp16 for high-frequency tokens
- **GPU-side gradient norm** — parallel reduction kernel, sanitizes inf/nan in-place
- **fp32 gradient clipping inside optimizer** — clip scale applied in fp32 after converting from fp16, preventing precision loss
- **fp32 loss, rrms, optimizer state** — critical numerics stay full precision

## How It's Wired Together

The hand-written CUDA kernels live in the **shared** [`../cuda/`](../cuda) library at the repo root — they're config-agnostic (every kernel takes its dimensions as arguments), so the `llama3` model reuses the exact same kernels. This model folder holds only the C++ host code and its build:

```
llms-from-scratch-cuda/
├── cuda/                    # Shared hand-written CUDA kernels (used by every model)
│   ├── kernels.cuh          # Declarations + atomicAddHalf/atomicAddBf16 helpers
│   ├── tensor.h             # GPU memory management (Tensor, HalfParameterBlock)
│   ├── matmul.cu            # Matrix multiply (cublasSgemm + cublasGemmEx)
│   ├── attention.cu         # Grouped-Query Attention (score, softmax, value, backward)
│   ├── flash_attention.cu   # Fused tiled flash-attention kernels
│   ├── swiglu.cu            # SwiGLU MLP (gate, up, SiLU, down)
│   ├── rope.cu              # Rotary Position Embeddings
│   ├── rmsnorm.cu           # RMS Normalization
│   ├── crossentropy.cu      # Cross-entropy loss (fused softmax + log-loss)
│   ├── embedding.cu         # Token embedding lookup + scatter-add backward
│   ├── adamw.cu             # AdamW optimizer + gradient norm kernels
│   ├── softmax.cu           # Standalone softmax (for attention scores)
│   └── residual.cu          # Skip connections
└── llama2-mistral/          # ← this model
    ├── src/                 # C++ host code
    │   ├── config.h         # Model + training hyperparameters
    │   ├── model.h          # Model struct (fp32 + fp16)
    │   ├── model.cpp        # Build, forward, backward, optimizer step
    │   ├── main.cpp         # Training loop entry point
    │   └── dataloader.h/cpp # Binary token file reader
    ├── scripts/             # Data tokenization
    └── CMakeLists.txt       # Build (compiles ../cuda into a static kernels lib)
```

The C++ side (`model.cpp`) owns the parameter/activation memory pools and chains the kernels into a full forward and backward pass; the kernels in `../cuda/` do all the actual math on the GPU.

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Requirements: an NVIDIA GPU (Compute Capability 7.0+), CUDA Toolkit 11.0+, and CMake 3.18+.

## Documentation

Building this from scratch meant deriving and writing up the math behind every component. The [`documentation/`](../documentation) folder contains the notes that back the implementation:

- **[`algorithm_explanations/`](../documentation/algorithm_explanations)** — how each building block works: attention/GQA, RoPE, RMSNorm, SwiGLU, softmax, cross-entropy, embeddings, matmul, AdamW + LR warmup, KV cache, and Flash Attention.
- **[`backprop_explanations/`](../documentation/backprop_explanations)** — the gradient derivations for every kernel (the chain rule, attention, softmax, RMSNorm, RoPE, SwiGLU, matmul, embedding, cross-entropy, residual, AdamW) plus a full end-to-end backward pass walkthrough.
- **[`training_instructions/`](../documentation/training_instructions)** — pretraining setup and a breakdown of every training parameter (batch size, gradient accumulation, learning rate, warmup, etc.).

## Engineering Highlights

The parts that required non-trivial engineering:

- **CAS-based atomicAddHalf** — CUDA doesn't provide `atomicAdd` for `__half` on all architectures. Implemented a portable compare-and-swap loop with 4-byte alignment handling.

- **fp32 embedding gradients** — Discovered that scatter-add for high-frequency tokens overflows fp16 (53 inf values out of 124M). Solved by keeping `dwte` in fp32 with a mixed fp32-input/fp16-output backward kernel.

- **GPU-side gradient norm with sanitization** — Parallel reduction kernel that computes L2 norm across fp32 + fp16 gradient buffers in one pass, replacing inf/nan with zero in-place. Eliminates 280MB/step of GPU→CPU transfers that killed throughput.

- **fp32 gradient clipping inside optimizer** — Clipping in fp16 space zeros out small gradients (clip_scale ≈ 1e-7 underflows). Moving clip application into the AdamW kernel where it operates in fp32 preserves all gradient information.

- **Numerically stable mixed-precision backward pass** — All fp16 kernels read `__half`, accumulate in `float`, write `__half`. Attention softmax backward, RMSNorm backward, and cross-entropy backward all maintain fp32 internal precision.

## Also By Me

**[RFX-Fuse](https://github.com/chriskuchar/RFX-Fuse)** — GPU-accelerated Random Forests engine in CUDA/C++. Published on [PyPI](https://pypi.org/project/rfx-fuse/) and [arXiv](https://arxiv.org/html/2603.13234v1). Scales to 25M+ samples with GPU acceleration, QLoRA compression, and explainable similarity.

## License

MIT
