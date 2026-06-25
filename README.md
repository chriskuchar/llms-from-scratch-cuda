# llms-from-scratch-cuda

A collection of large language models implemented **entirely from scratch in C++/CUDA** — no PyTorch, no TensorFlow, no cuDNN, no CUTLASS. Every layer, every gradient, and every GPU kernel is hand-written. The only external dependency is cuBLAS for the matrix multiplies.

Think of it as **`llm.c` in spirit — training LLMs from scratch with hand-written GPU kernels — but in C++/CUDA and spanning the modern LLaMA/Qwen/Gemma family** (plus MoE, Mamba, BERT, and text diffusion), rather than a single architecture. Where `llm.c` is deliberately plain C, here the host code is C++17 and the kernels are CUDA C++, which buys real abstractions: **one templated kernel serves both fp32 and bf16**, and the memory pools are managed types rather than raw buffers.

It's *not* an inference engine like `llama.cpp` (which is inference-only, with CUDA as one backend) and *not* a PyTorch teaching repo like `nanoGPT` or `rasbt/LLMs-from-scratch` — this is **training from scratch, on the metal**.

The kernels are config-agnostic, so all models in this repo share **one** hand-written kernel library ([`cuda/`](cuda)) and differ only in their host-side config.

## Models

| Model | Style | What's notable | Status |
|---|---|---|---|
| [`llama2-mistral/`](llama2-mistral) | LLaMA 2 / Mistral | 125M params, GQA + RoPE + RMSNorm + SwiGLU, 32K SentencePiece vocab, fp16 mixed precision | **working, trains end-to-end** |
| [`llama3/`](llama3) | LLaMA 3 | same block + 128K tokenizer, RoPE θ = 500,000, 8K context (~272M params) | **builds + passes overfit test**, reuses shared kernels |

LLaMA 3 is intentionally a small diff from LLaMA 2 — same decoder block, just a bigger tokenizer, a higher RoPE base, and a longer context — which is why both build on the identical kernels in `cuda/`.

## Roadmap

Planned architectures, each a small, well-scoped addition on top of the shared `cuda/` kernels. Folders will be added as work starts on each.

- **Qwen2 / Qwen2.5** — LLaMA block + **QKV bias** (the one new kernel), 151K vocab, RoPE θ = 1,000,000, weight tying.
- **Qwen3** — drops QKV bias, adds **QK-norm** (reuses the RMSNorm kernel).
- **Gemma 2** — GeGLU, logit soft-capping, sandwich norm, sliding-window attention.
- **Mixtral (MoE)** — sparse **Mixture-of-Experts** MLP + top-k router (stepping stone to DeepSeek/Kimi).
- **Mamba (SSM)** — **no attention**: a selective state-space scan (parallel prefix-scan kernel).
- **BERT** — **encoder-only**: bidirectional attention + masked-LM objective, LayerNorm.
- **Gemma Diffusion** — **non-autoregressive** text generation: bidirectional attention + masked-diffusion objective with iterative sampling.

Most are small diffs (QKV bias, QK-norm, GeGLU, MoE routing); the standouts are **Mamba** (non-attention scan) and **Gemma Diffusion** (non-autoregressive generation), which introduce genuinely new mechanisms.

## Coming soon

- **Flash Attention (CUDA)** — fused, tiled attention kernel (`cuda/flash_attention.cu`): O(N) memory, no materialized T×T score matrix, longer context on the same GPU.
- **KV cache + inference (CUDA)** — autoregressive generation from trained checkpoints with a cached key/value path for fast decoding.

## Benchmarks

### Pretraining result — LLaMA 2 (`llama2-mistral`) learns end-to-end (loss 10.5 → ~4.0)

A bf16 pretraining run on a single **RTX 3060 (12GB)** takes the model from random init into coherent-text territory. Trained on a **~298M-token** subset of **Common Corpus** (`PleIAs/common_corpus` — multilingual public-domain text: books, news, legal/encyclopedic passages, and OCR'd historical documents across English, French, Spanish, German, and more). 124.7M params, bf16 mixed precision, B=16 × T=512 × accum=16 (effective batch **131,072 tokens/step**), activation memory ~1.87 GB. Full log: [`llama2-mistral/results/pretrain_298m.log`](llama2-mistral/results/pretrain_298m.log).

```
step     1 | loss 10.50 | lr 1.5e-07    <- random init (≈ ln 32000)
step  1000 | loss  5.35 | lr 1.5e-04
step  2000 | loss  6.01 | lr 3.0e-04    <- LR warmup peak
step  3000 | loss  4.74 | lr 3.0e-04
step  4000 | loss  4.31 | lr 3.0e-04
step  5000 | loss  4.55 | lr 2.9e-04
step  6000 | loss  4.15 | lr 2.9e-04
step  6370 | loss  4.26 | lr 2.8e-04
```

- **Loss** falls from 10.50 (random init, ≈ ln 32000) to a **running average of ~4.0**. Per-step loss is a single-micro-batch readout, so it's noisy (best micro-batches dip to **~0.6–2.1** on low-entropy passages — boilerplate, repeated phrasing, predictable text); the running average is the signal.
- **Throughput** holds steady around **~3,500 tok/s** at the larger B=16 effective batch on the consumer GPU.
- **Stable in bf16** — no divergence across 6,000+ steps, thanks to fp32 master weights / Adam state and correct gradient accumulation (`beta=1` weight-grad accumulation across the 16 micro-batches).
- **Checkpoint resume verified** — the run was resumed mid-schedule from a step-1300 checkpoint: both weights and the Adam `m`/`v` state were restored with **no loss spike**, and the LR schedule continued through warmup to the 3e-4 peak.

*(~298M tokens for 124.7M params is ~2.4× tokens/param — below Chinchilla-optimal — so by step 6370 the run has seen ~2.8 epochs and the average flattens near ~4.0; full language quality needs a larger token budget. Numbers demonstrate correctness, stability, and throughput, not final model quality.)*

**Why ~4.0 (and why that's harder than it looks).** Cross-entropy loss is only comparable *within the same dataset and tokenizer* — a loss of 4.0 here is **not** worse than the ~1.5–2.5 a model of this size would reach on a toy corpus like TinyStories. Common Corpus is multilingual, mixed-register, OCR-noisy public-domain text, so its cross-entropy floor is simply much higher: the model is splitting 124.7M parameters across several languages and document types instead of memorizing one simple register. Driving loss from 10.50 → ~4.0 on data this varied is a *stronger* signal that the pipeline (attention, RoPE, optimizer, gradient accumulation, bf16 numerics) is learning real structure than the same drop on uniform children's-story prose would be. The trade-off: a small model on hard, noisy data won't emit fluent single-language text — for coherent generated samples you'd pretrain on something clean and narrow like TinyStories, which gives a lower, prettier number on an easier benchmark, not a better-engineered model.

### Correctness — single-batch overfit test

The standard sanity check that backprop is implemented correctly: train on a **single fixed batch** and confirm the model can memorize it (loss → ~0). With orders of magnitude more parameters than tokens, a correct transformer drives loss to near zero in a few hundred steps; if it stalls, gradients are wrong somewhere.

Make a one-batch dataset and run it:

```bash
cd llama2-mistral
# One batch needs B*T + 1 tokens (the +1 is the next-token target shift).
# The .bin is raw int32 tokens (4 bytes each), so at B=1, T=512:
#   (1*512 + 1) tokens * 4 bytes = 2052 bytes
head -c 2052 data/pretrain.bin > data/overfit.bin

cd build
./train ../data/overfit.bin --bf16 --batch 1 --steps 2000 \
        --accum 1 --lr 3e-4 --warmup 20 --save 1000000
```

Expected output (loss collapses to ~0 — the model memorizes the batch):

```
step     1 | loss 10.5058 | lr 1.50e-05
step    20 | loss  5.0903 | lr 3.00e-04
step    30 | loss  1.3280 | lr 3.00e-04
step    50 | loss  0.0596 | lr 3.00e-04
step   100 | loss  0.0029 | lr 3.00e-04
step   200 | loss  0.0001 | lr 3.00e-04
```

**Both models pass.** `llama2-mistral` (32K vocab) starts at ~10.5 and `llama3` (128K vocab) starts at ~11.8 (≈ ln of each vocab size); both collapse to ~0 within a few hundred steps. This confirms the full hand-written backward pass — attention/RoPE/RMSNorm/SwiGLU gradients, gradient accumulation, and per-layer activation checkpointing — is numerically correct end to end, across both configs and both precisions (`fp32` and `bf16` give the same curve — drop the `--bf16` flag to test the fp32 path).

### Planned

More measurements coming (single RTX 3060, 12GB):

- **Throughput** — training tokens/sec per model and precision (fp32 vs fp16 vs bf16).
- **Memory** — peak VRAM vs sequence length, standard attention vs Flash Attention.
- **Flash Attention** — speedup and max context length with the fused kernel on vs off.
- **Inference** — decode tokens/sec with the KV cache.

## Repo layout

```
llms-from-scratch-cuda/
├── cuda/                 # Shared hand-written CUDA kernels (matmul, attention,
│                         #   flash-attention, rope, rmsnorm, swiglu, softmax,
│                         #   crossentropy, embedding, adamw, residual) + tensor.h
├── documentation/        # Shared write-ups: algorithm + backprop derivations,
│                         #   training instructions, roadmaps
├── llama2-mistral/       # LLaMA 2 / Mistral-style model (src/, scripts/, CMakeLists)
└── llama3/               # LLaMA 3-style model (src/, scripts/, CMakeLists)
```

Each model has its own `README.md`, build (`cmake .. && make`), and training commands.

## Quick start

```bash
cd llama2-mistral
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Then follow that model's README for tokenizing data and launching pretraining. (`llama3` builds the same way and reuses the shared kernels.)

Requirements: an NVIDIA GPU (Compute Capability 7.0+), CUDA Toolkit 11.0+, CMake 3.18+.

## Documentation

The math behind every component is derived and written up in [`documentation/`](documentation):

- **[`algorithm_explanations/`](documentation/algorithm_explanations)** — how each block works (attention/GQA, RoPE, RMSNorm, SwiGLU, softmax, cross-entropy, embeddings, matmul, AdamW + LR warmup, KV cache, Flash Attention).
- **[`backprop_explanations/`](documentation/backprop_explanations)** — gradient derivations for every kernel plus a full end-to-end backward pass.
- **[`training_instructions/`](documentation/training_instructions)** — pretraining setup, a breakdown of every training parameter, and a [data sources guide](documentation/training_instructions/data_sources.md) (what to pretrain/fine-tune on, where to find it, and how to tokenize it).

## Also By Me

**[RFX-Fuse](https://github.com/chriskuchar/RFX-Fuse)** — Breiman and Cutler's Random Forests Compressed as a Unified Learning and Similarity Engine. Extended with native explainable similarity Published on [PyPI](https://pypi.org/project/rfx-fuse/) and [arXiv](https://arxiv.org/html/2603.13234v1). Scales to 25M+ samples with GPU acceleration.

## License

MIT
