# llms-from-scratch-cuda

A collection of large language models implemented **entirely from scratch in C++/CUDA** — no PyTorch, no TensorFlow, no cuDNN, no CUTLASS. Every layer, every gradient, and every GPU kernel is hand-written. The only external dependency is cuBLAS for the matrix multiplies.

Think of it as **`llm.c` in spirit — training LLMs from scratch with hand-written GPU kernels — but in C++/CUDA and spanning the modern LLaMA/Qwen/Gemma family** (plus MoE, Mamba, BERT, and text diffusion), rather than a single architecture. Where `llm.c` is deliberately plain C, here the host code is C++17 and the kernels are CUDA C++, which buys real abstractions: **one templated kernel serves both fp32 and bf16**, and the memory pools are managed types rather than raw buffers.

It's *not* an inference engine like `llama.cpp` (which is inference-only, with CUDA as one backend) and *not* a PyTorch teaching repo like `nanoGPT` or `rasbt/LLMs-from-scratch` — this is **training from scratch, on the metal**.

The kernels are config-agnostic, so all models in this repo share **one** hand-written kernel library ([`cuda/`](cuda)) and differ only in their host-side config.

## Models

| Model | Style | What's notable | Status |
|---|---|---|---|
| [`llama2-mistral/`](llama2-mistral) | LLaMA 2 / Mistral | 125M params, GQA + RoPE + RMSNorm + SwiGLU, 32K SentencePiece vocab, fp16 mixed precision | **working, trains end-to-end** |
| [`llama3/`](llama3) | LLaMA 3 | same block + 128K tokenizer, RoPE θ = 500,000, 8K context (~272M params) | **config-ready**, reuses shared kernels |

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

### Initial result — LLaMA 2 (`llama2-mistral`) pretraining works

A first bf16 pretraining run on a single **RTX 3060 (12GB)** confirms the model learns end-to-end. Trained on a **~10.1M-token** slice of Common Corpus (`data/pretrain_test.bin`). 124.7M params, bf16 mixed precision, B=4 × T=512 × accum=16 (effective batch 32,768 tokens/step), activation memory ~467 MB.

```
step     1 | loss 10.50 | lr 1.5e-07 | 4452 tok/s
step   250 | loss  9.55 | lr 3.8e-05
step   500 | loss  7.81 | lr 7.5e-05
step   750 | loss  8.09 | lr 1.1e-04
step  1000 | loss  6.70 | lr 1.5e-04
step  1250 | loss  7.88 | lr 1.9e-04
step  1500 | loss  7.12 | lr 2.3e-04
step  1750 | loss  5.27 | lr 2.6e-04
step  2000 | loss  7.26 | lr 3.0e-04
```

- **Loss** falls from 10.50 (random init, ≈ ln 32000) into the **6–7 range by ~2,000 steps**, with the best micro-batches dipping to **~4.4–5.6**. Per-step loss is a single-micro-batch readout, so it's noisy; the trend is the signal.
- **Throughput** holds steady around **~4,700 tok/s** on the consumer GPU.
- **Stable in bf16** — no divergence, thanks to fp32 master weights / Adam state and correct gradient accumulation (`beta=1` weight-grad accumulation across the 16 micro-batches).

*(Smoke run on the ~10.1M-token slice — it sees several epochs, so it starts memorizing; full pretraining uses the 500M-token set. Numbers are illustrative of correctness + speed, not final model quality.)*

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
- **[`training_instructions/`](documentation/training_instructions)** — pretraining setup and a breakdown of every training parameter.

## Also By Me

**[RFX-Fuse](https://github.com/chriskuchar/RFX-Fuse)** — Breiman and Cutler's Random Forests Compressed as a Unified Learning and Similarity Engine. Extended with native explainable similarity Published on [PyPI](https://pypi.org/project/rfx-fuse/) and [arXiv](https://arxiv.org/html/2603.13234v1). Scales to 25M+ samples with GPU acceleration.

## License

MIT
