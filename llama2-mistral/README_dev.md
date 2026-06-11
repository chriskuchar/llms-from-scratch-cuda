# LLaMA-Style LLM from Scratch in CUDA/C++

A 125M parameter large language model built entirely from scratch — no PyTorch, no TensorFlow, no frameworks. Pure CUDA kernels and C++, trained on a single RTX 3060.

## What This Is

A ground-up implementation of a modern LLM training pipeline, including:

- **10 custom CUDA kernels** — every GPU operation hand-written, not called from a library
- **LLaMA architecture** — the same design used by Meta's LLaMA, Mistral, Qwen, and most open-source LLMs
- **fp16 mixed precision training** — tensor core acceleration with fp32 master weights and numerically stable gradient handling
- **Full training loop** — AdamW optimizer, gradient clipping, learning rate scheduling, checkpointing

**4,096 lines of code. Zero dependencies beyond CUDA and cuBLAS.**

## Architecture

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

## CUDA Kernels

Every operation runs on the GPU via hand-written CUDA kernels. No cuDNN, no CUTLASS — just raw CUDA and cuBLAS for matrix multiplies.

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

Full mixed-precision training pipeline, not just casting weights to fp16:

- **fp16 weights** — half the memory, enables tensor cores (2x theoretical throughput)
- **fp32 master weights** — optimizer updates in full precision, then syncs to fp16
- **fp32 embedding gradients** — scatter-add accumulation overflows fp16 for high-frequency tokens
- **GPU-side gradient norm** — parallel reduction kernel, sanitizes inf/nan in-place
- **fp32 gradient clipping inside optimizer** — clip scale applied in fp32 after converting from fp16, preventing precision loss
- **fp32 loss, rrms, optimizer state** — critical numerics stay full precision

## Training Results

First working pretraining run — 2,000 steps on 500M tokens of Common Corpus (public domain text), fp16 mixed precision, RTX 3060 12GB:

```
step     1 | loss 10.50 | lr 1.5e-07 | 3765 tok/s
step   200 | loss 10.01 | lr 3.0e-05
step   400 | loss  9.53 | lr 6.0e-05
step   600 | loss  8.64 | lr 9.0e-05
step   800 | loss  8.77 | lr 1.2e-04
step  1000 | loss  7.60 | lr 1.5e-04
step  1200 | loss  7.51 | lr 1.8e-04
step  1400 | loss  7.94 | lr 2.1e-04
step  1600 | loss  8.00 | lr 2.4e-04
step  1800 | loss  7.61 | lr 2.7e-04
step  2000 | loss  7.47 | lr 3.0e-04
```

~4,600 tokens/second on a consumer GPU. Loss drops from 10.50 (random init, ≈ ln 32000) into the 6–7 range over 2,000 steps — the best mini-batches already dip to ~5–6 as the model starts picking up language. Per-step loss is noisy because it's a single micro-batch readout, not a running average.


## Project Structure

```
llm_from_scratch/
├── cuda/                    # All CUDA kernels
│   ├── kernels.cuh          # Declarations + atomicAddHalf helper
│   ├── matmul.cu            # Matrix multiply (cublasSgemm + cublasGemmEx)
│   ├── attention.cu         # Grouped-Query Attention (score, softmax, value, backward)
│   ├── swiglu.cu            # SwiGLU MLP (gate, up, SiLU, down)
│   ├── rope.cu              # Rotary Position Embeddings
│   ├── rmsnorm.cu           # RMS Normalization
│   ├── crossentropy.cu      # Cross-entropy loss (fused softmax + log-loss)
│   ├── embedding.cu         # Token embedding lookup + scatter-add backward
│   ├── adamw.cu             # AdamW optimizer + gradient norm kernels
│   ├── softmax.cu           # Standalone softmax (for attention scores)
│   └── residual.cu          # Skip connections
├── src/                     # C++ host code
│   ├── config.h             # Model + training hyperparameters
│   ├── model.h              # Model struct (fp32 + fp16)
│   ├── model.cpp            # Build, forward, backward, optimizer step
│   ├── main.cpp             # Training loop, arg parsing, checkpointing
│   ├── tensor.h             # GPU memory management (Tensor, HalfParameterBlock)
│   ├── dataloader.h/cpp     # Binary token file reader
├── scripts/                 # Data preparation
│   ├── tokenize_pretrain.py # Download + tokenize pretraining data
│   ├── tokenize_sft.py      # Tokenize SFT instruction data
│   └── prepare_data.py      # Generic text → binary tokenizer
├── data/                    # Training data (binary int32 token files)
├── checkpoints/             # Saved model weights (fp32)
└── CMakeLists.txt           # Build system
```

## Build & Run

### Requirements
- NVIDIA GPU (Compute Capability 7.0+, i.e. Volta/Turing/Ampere)
- CUDA Toolkit 11.0+
- CMake 3.18+
- Python 3.8+ with `sentencepiece`, `datasets`, `transformers` (for data prep only)

### Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Training Pipeline

Three stages take JrLAM from random weights to a model that answers questions.

#### Stage 1: Pretrain — Learn Language
Trains on raw text from [Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) (public domain books and documents). The model learns grammar, word relationships, and general knowledge.

```bash
pip install sentencepiece datasets

# Tokenize 500M tokens of Common Corpus
python scripts/tokenize_pretrain.py \
  --tokenizer data/tokenizer.model \
  --output data/pretrain.bin \
  --max_tokens 500000000

# Train (~12 hours on RTX 3060)
mkdir -p checkpoints
stdbuf -oL ./build/train data/pretrain.bin --steps 5000 --save 500 --fp16 2>&1 | tee pretrain.log
```

#### Stage 2: Code Fine-Tune — Learn Python
Continues training on Python source code from [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata). The model learns syntax, function patterns, and coding conventions.

```bash
# Tokenize 200M tokens of Python code
python scripts/tokenize_code.py \
  --tokenizer data/tokenizer.model \
  --output data/code.bin \
  --max_tokens 200000000

# Continue from pretrained checkpoint (~2-3 hours)
stdbuf -oL ./build/train data/code.bin \
  --checkpoint checkpoints/step_5000.bin \
  --steps 3000 --lr 1e-4 --save 500 --fp16 2>&1 | tee code_train.log
```

#### Stage 3: Instruction SFT — Learn to Answer Questions
Fine-tunes on question-answer pairs from [Alpaca](https://huggingface.co/datasets/yahma/alpaca-cleaned) (52K instructions). The model learns to follow instructions and respond to prompts.

```bash
# Tokenize instruction pairs
python scripts/tokenize_instruct.py \
  --tokenizer data/tokenizer.model \
  --output data/instruct.bin \
  --dataset yahma/alpaca-cleaned

# Fine-tune from code checkpoint (~1 hour)
stdbuf -oL ./build/train data/instruct.bin \
  --checkpoint checkpoints/step_3000.bin \
  --steps 1000 --lr 2e-5 --save 500 --fp16 2>&1 | tee sft_train.log
```

#### Stage 4: Action SFT — Learn to Call Functions (the LAM in JrLAM)
Fine-tunes on function-calling conversations from [Glaive](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) (113K examples). The model learns to output structured tool calls — making it a Large **Action** Model, not just a language model.

```bash
# Tokenize function-calling data
python scripts/tokenize_sft.py \
  --tokenizer data/tokenizer.model \
  --output data/actions.bin \
  --dataset glaiveai/glaive-function-calling-v2

# Fine-tune from instruction checkpoint (~1-2 hours)
stdbuf -oL ./build/train data/actions.bin \
  --checkpoint checkpoints/step_1000.bin \
  --steps 2000 --lr 2e-5 --save 500 --fp16 2>&1 | tee action_train.log
```

After this stage, JrLAM can output structured function calls:
```
User: What's the weather in Las Vegas?
JrLAM: <function_call>get_weather(location="Las Vegas, NV")</function_call>
```

#### Summary — The Evolution of JrLAM

Each stage produces a distinct model. Save checkpoints separately to keep all four.

| Stage | Data | Source | Checkpoint | Model Name |
|---|---|---|---|---|
| Pretrain | 500M tokens | Common Corpus (public domain text) | `checkpoints/jrlm.bin` | **JrLM** — knows English |
| Code | 200M tokens | StarCoderData (Python source) | `checkpoints/jrcoder.bin` | **JrCoder** — writes Python |
| Instruct SFT | 52K examples | Alpaca (instruction-answer pairs) | `checkpoints/jrchat.bin` | **JrChat** — answers questions |
| Action SFT | 113K examples | Glaive function-calling v2 | `checkpoints/jrlam.bin` | **JrLAM** — calls functions |

```
JrLM  →  JrCoder  →  JrChat  →  JrLAM
 raw text    Python      Q&A     function calls
```

After each stage completes, copy the final checkpoint to preserve it:
```bash
cp checkpoints/step_5000.bin checkpoints/jrlm.bin      # after pretrain
cp checkpoints/step_3000.bin checkpoints/jrcoder.bin    # after code
cp checkpoints/step_1000.bin checkpoints/jrchat.bin     # after instruct
cp checkpoints/step_2000.bin checkpoints/jrlam.bin      # after actions
```

### Command-Line Options
| Flag | Default | Description |
|---|---|---|
| `--fp16` | off | Enable fp16 mixed precision (tensor cores) |
| `--steps N` | 30000 | Total optimizer steps |
| `--batch N` | 4 | Micro-batch size |
| `--accum N` | 16 | Gradient accumulation steps |
| `--lr F` | 3e-4 | Peak learning rate |
| `--warmup N` | 2000 | LR warmup steps |
| `--save N` | 2000 | Checkpoint interval |
| `--checkpoint PATH` | none | Load weights from file |

## Technical Highlights

Things that required non-trivial engineering:

- **CAS-based atomicAddHalf** — CUDA doesn't provide `atomicAdd` for `__half` on all architectures. Implemented a portable compare-and-swap loop with 4-byte alignment handling.

- **fp32 embedding gradients** — Discovered that scatter-add for high-frequency tokens overflows fp16 (53 inf values out of 124M). Solved by keeping `dwte` in fp32 with a mixed fp32-input/fp16-output backward kernel.

- **GPU-side gradient norm with sanitization** — Parallel reduction kernel that computes L2 norm across fp32 + fp16 gradient buffers in one pass, replacing inf/nan with zero in-place. Eliminates 280MB/step of GPU→CPU transfers that killed throughput.

- **fp32 gradient clipping inside optimizer** — Clipping in fp16 space zeros out small gradients (clip_scale ≈ 1e-7 underflows). Moving clip application into the AdamW kernel where it operates in fp32 preserves all gradient information.

- **Numerically stable mixed-precision backward pass** — All fp16 kernels read `__half`, accumulate in `float`, write `__half`. Attention softmax backward, RMSNorm backward, and cross-entropy backward all maintain fp32 internal precision.

## What's Next

1. **KV Cache + Inference** — autoregressive text generation from trained checkpoints
2. **Flash Attention** — fused tiled attention kernel, O(N) memory, 2-4x speedup
3. **SFT** — supervised fine-tuning on instruction data for conversation ability
4. **Scale to 320M** — same architecture, just bigger (fits on 12GB at ~6 GB total)

## Also By Me

**[RFX-Fuse](https://github.com/chriskuchar/RFX-Fuse)** — GPU-accelerated Random Forests engine in CUDA/C++ (46% CUDA, 50% C++). Published on [PyPI](https://pypi.org/project/rfx-fuse/) and [arXiv](https://arxiv.org/html/2603.13234v1). Scales to 25M+ samples with GPU acceleration, QLoRA compression, and explainable similarity.

## License

MIT
