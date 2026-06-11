# Training Guide: JrLAM (125M Large Action Model)

Step-by-step instructions to pretrain and fine-tune JrLAM, a 125M parameter
Large Action Model built from scratch on a single RTX 3060 12GB.

---

## Prerequisites

- NVIDIA GPU with CUDA support (tested on RTX 3060 12GB)
- CUDA Toolkit installed (`nvcc --version` should work)
- CMake 3.18+
- Python 3.8+
- ~10GB free disk space (data + checkpoints)

---

## Step 0: Build the C++ training binary

```bash
cd ~/llm_for_fun/llm_from_scratch
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..
```

This produces `build/train`. Verify it built:
```bash
ls -la build/train
```

**Clean rebuild** (if you've edited source files and want to start fresh):
```bash
cd ~/llm_for_fun/llm_from_scratch
rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..
```

### fp16 mixed precision build + quick test

Complete rebuild from scratch, then verify fp16 works on 100 steps:
```bash
cd ~/llm_for_fun/llm_from_scratch
rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..
```

Quick test (fp32, baseline — make sure nothing is broken):
```bash
./build/train data/pretrain_test.bin --steps 100 --accum 1
```

Quick test (fp16 mixed precision — should see higher tok/s):
```bash
./build/train data/pretrain_test.bin --steps 100 --fp16
```

Both should complete 100 steps with decreasing loss. Compare the `tok/s` between fp32 and fp16 to confirm tensor cores are active (expect ~2-4x speedup on RTX 3060).

---

## Step 1: Set up Python environment

```bash
cd ~/llm_for_fun/llm_from_scratch
python3 -m venv lam_env
source lam_env/bin/activate
pip install --upgrade pip
pip install datasets sentencepiece huggingface_hub
```

To reactivate later:
```bash
source ~/llm_for_fun/llm_from_scratch/lam_env/bin/activate
```

---

## Step 2: Download tokenizer

### What is a tokenizer?

A tokenizer converts human-readable text into numbers (token IDs) that the model
can process. The model never sees raw text — only sequences of integers.

For example: `"Hello world"` → `[15043, 3186]`

Each integer maps to a "token" — usually a word, subword, or character fragment.
The tokenizer has a fixed vocabulary (ours has 32,000 tokens) that covers all of
English plus common code patterns.

The tokenizer is used twice:
1. **Before training:** convert text datasets into `.bin` files of int32 token IDs
2. **At inference:** convert user input to token IDs, feed to model, convert output IDs back to text

### Why Open LLaMA's tokenizer?

We use Open LLaMA 3B's SentencePiece tokenizer because:
- **Vocab size = 32,000** — matches our `config.vocab_size` exactly. If the tokenizer
  produces token ID 31,999 but the model only has 20,000 embeddings, it crashes.
  The sizes must match.
- **SentencePiece format** — a single `.model` file, no dependencies beyond the
  `sentencepiece` Python package. Simple to use.
- **Apache 2.0 license** — fully open for any use, commercial included.
- **Good coverage** — trained on a large English corpus, handles code/JSON well.

We are NOT using Open LLaMA's model weights — just borrowing its tokenizer.
The tokenizer is completely independent from the model architecture. Any
SentencePiece tokenizer with 32k vocab would work. We pick this one because
it's high quality, freely available, and the right size.

```bash
mkdir -p data
python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download('openlm-research/open_llama_3b', 'tokenizer.model')
shutil.copy(path, 'data/tokenizer.model')
print('Saved tokenizer to data/tokenizer.model')
"
```

Verify:
```bash
python3 -c "
from sentencepiece import SentencePieceProcessor
sp = SentencePieceProcessor(model_file='data/tokenizer.model')
print(f'Vocab size: {sp.get_piece_size()}')
print(f'Test encode: {sp.encode(\"Hello world\")}')
"
```

Should print `Vocab size: 32000`.

---

## Phase 1: Pretraining

### 1a. Sanity Check Run (~20 minutes)

Before committing to the full pretrain, verify everything works with a
small 10M token dataset. This does NOT produce a useful model — it just
confirms the code compiles, loss decreases, and fp16 tensor cores work.

**Tokenize test data** (5 minutes, 10M tokens):
```bash
python scripts/tokenize_pretrain.py \
    --tokenizer data/tokenizer.model \
    --output data/pretrain_test.bin \
    --max_tokens 10000000
```

**Run sanity check** (fp32 baseline):
```bash
mkdir -p checkpoints
./build/train data/pretrain_test.bin --steps 100 --accum 1
```

**Run sanity check** (fp16 — should show ~3x higher tok/s):
```bash
./build/train data/pretrain_test.bin --steps 100 --accum 1 --fp16
```

**Run sanity check** (fp16 + B=16 — max throughput on 3060):
```bash
./build/train data/pretrain_test.bin --steps 100 --accum 1 --fp16 --batch 16
```

**Expected output:**
```
Building model (fp16 mixed precision)...
Model (fp16): 12 layers, 12 heads (4 kv), 768 embd, 2048 ffn
Parameters: 124.7M
DataLoader: 10120546 tokens from data/pretrain_test.bin
  4941 batches per epoch (B=4, T=512)
Training for 100 steps (accum=1, effective batch=2048 tokens)...
step     1 | loss 10.4857 | lr 1.50e-07 | ~7000 tok/s
step    50 | loss 10.4872 | lr 7.50e-06 | ~7000 tok/s
step   100 | loss 10.3878 | lr 1.50e-05 | ~7000 tok/s
Training complete.
```

**What to check:**
- Loss starts ~10.4 (random) and slowly decreases — confirms training works
- fp16 tok/s should be ~3x faster than fp32 (~7000 vs ~2275)
- No crashes, no NaN — confirms fp16 numerics are stable
- Loss barely moves because 10M tokens is 250x too little data for 124.7M params

| What | Value | Meaning |
|---|---|---|
| Starting loss ~10.4 | `-log(1/32000)` | Random guessing over 32k vocab |
| Ending loss ~10.3 | Barely moved | Only saw 0.1x tokens-per-param (need 20x) |
| 0.1x tokens/param | 10M / 125M | Way underfed — just a code test |

---

### 1b. Tokenize Full Pretraining Data (~1-2 hours download)

The real pretrain uses **TinyStories** — a 500M token dataset of simple
short stories from Microsoft Research. Clean, coherent English that teaches
JrLAM grammar, vocabulary, and basic narrative structure.

```bash
python scripts/tokenize_pretrain.py \
    --tokenizer data/tokenizer.model \
    --output data/pretrain.bin \
    --max_tokens 500000000
```

This downloads and tokenizes ~500M tokens. Takes 1-2 hours depending on
your internet speed. The output file will be ~2 GB.

| Dataset | Tokens | File Size | Tokens/Param | Quality |
|---|---|---|---|---|
| pretrain_test.bin | 10M | 40 MB | 0.1x | Code test only |
| **pretrain.bin** | **500M** | **2 GB** | **4x** | **Grammatical sentences, simple stories** |
| Ideal (Chinchilla) | 2.5B | 10 GB | 20x | Full language understanding |

4x tokens-per-param won't match Chinchilla optimal, but it's enough for
JrLAM to produce coherent text and serve as a solid base for SFT.

---

### 1c. Run Full Pretraining (~8-10 hours, fp16, B=16)

```bash
mkdir -p checkpoints
./build/train data/pretrain.bin \
    --fp16 \
    --batch 16 \
    --steps 15300 \
    --accum 2 \
    --lr 3e-4 \
    --warmup 500
```

**What these flags mean:**
| Flag | Value | Why |
|---|---|---|
| `--fp16` | mixed precision | 3x faster via tensor cores |
| `--batch 16` | sequences per micro-batch | 4x bigger batch, still only ~4.2 GB VRAM |
| `--steps 15300` | weight updates | `500M / (16 × 512 × 2)` ≈ 15,259 steps to see all tokens |
| `--accum 2` | gradient accumulation | 2 micro-batches per update = 16,384 effective tokens |
| `--lr 3e-4` | peak learning rate | Standard for 124M models |
| `--warmup 500` | LR warmup steps | Ramp up slowly to avoid early instability |

**Expected performance:**
| Metric | Value |
|---|---|
| Speed | ~12,000-15,000 tok/s |
| Time | ~8-10 hours |
| VRAM | ~4.2 GB of 12 GB |
| Checkpoints | Every 2000 steps in `checkpoints/` |

**Expected output over the run:**
```
Building model (fp16 mixed precision)...
Model (fp16): 12 layers, 12 heads (4 kv), 768 embd, 2048 ffn
Parameters: 124.7M
DataLoader: 500000000 tokens from data/pretrain.bin
Training for 30500 steps (accum=4, effective batch=16384 tokens)...
step     1 | loss 10.4741 | lr 4.50e-07 | ~8000 tok/s    <- random
step  1000 | loss  7.2000 | lr 3.00e-04 | ~8000 tok/s    <- learning common words
step  5000 | loss  5.5000 | lr 2.85e-04 | ~8000 tok/s    <- short phrases
step 10000 | loss  4.8000 | lr 2.55e-04 | ~8000 tok/s    <- basic grammar
step 20000 | loss  4.3000 | lr 1.50e-04 | ~8000 tok/s    <- coherent sentences
step 30000 | loss  4.0000 | lr 3.00e-05 | ~8000 tok/s    <- simple stories
step 30500 | loss  3.9500 | lr 3.00e-05 | ~8000 tok/s    <- done
Training complete.
```

**What loss means at each stage:**
| Loss | JrLAM can... | Approx. step |
|---|---|---|
| ~10.4 | Nothing (random guessing) | 0 |
| ~8.0 | Predict common tokens ("the", "a", spaces) | ~500 |
| ~6.0 | Output real words, broken grammar | ~3,000 |
| ~5.0 | Short phrases that mostly make sense | ~8,000 |
| ~4.5 | Grammatically correct sentences | ~15,000 |
| ~4.0 | Coherent paragraphs, simple stories | ~30,000 |

**You can safely stop early** if you're happy with the loss. Checkpoints
are saved every 2000 steps, so you can always resume later:
```bash
./build/train data/pretrain.bin \
    --fp16 --checkpoint checkpoints/step_20000.bin \
    --steps 10500 --accum 4 --lr 1e-4
```

**Checkpoints** are saved as fp32 master weights for compatibility.
They work with both `--fp16` and regular fp32 mode.

---

## Phase 2: SFT Fine-Tuning (~2 hours, makes JrLAM a LAM)

### 2a. Tokenize function-calling data

Downloads the Glaive function-calling v2 dataset (Apache 2.0 license,
113k examples of tool-use conversations).

```bash
python scripts/tokenize_sft.py \
    --tokenizer data/tokenizer.model \
    --output data/sft_train.bin
```

### 2b. Fine-tune the pretrained model

```bash
./build/train data/sft_train.bin \
    --checkpoint checkpoints/step_30000.bin \
    --lr 2e-5 \
    --steps 5000 \
    --accum 4 \
    --warmup 200
```

This loads the pretrained weights, then trains on function-calling data
with a much lower learning rate (2e-5 vs 3e-4) so it learns the new
format without forgetting English.

**Expected output:**
```
Building model...
Loaded checkpoint: checkpoints/step_30000.bin (125000000 params)
Opening data: data/sft_train.bin
Training for 5000 steps (accum=4, effective batch=8192 tokens)...
step     1 | loss 4.2103 | lr 1.00e-07 | 14500 tok/s
...
step  5000 | loss 1.8234 | lr 2.00e-06 | 14800 tok/s
Training complete.
```

Your JrLAM checkpoint is now at `checkpoints/step_5000.bin`.

---

## Resuming / Continuing Training

To continue pretraining with more data:
```bash
./build/train data/pretrain_part2.bin --checkpoint checkpoints/step_30000.bin
```

To continue SFT with more function-calling data:
```bash
./build/train data/more_sft.bin --checkpoint checkpoints/step_5000.bin --lr 2e-5
```

---

## File Structure After Training

```
llm_from_scratch/
├── build/
│   └── train                   # compiled binary
├── data/
│   ├── tokenizer.model         # SentencePiece tokenizer (32k vocab)
│   ├── pretrain.bin            # tokenized pretraining data (~4GB)
│   └── sft_train.bin           # tokenized SFT data (~200MB)
├── checkpoints/
│   ├── step_2000.bin           # early checkpoint
│   ├── step_30000.bin          # pretrained base model
│   └── step_5000.bin           # finished LAM (after SFT)
├── scripts/
│   ├── tokenize_pretrain.py
│   └── tokenize_sft.py
├── lam_env/                    # Python virtual environment
├── ROADMAP.md                  # project plan
└── TRAINING.md                 # this file
```

---

## Troubleshooting

**CUDA out of memory:**
Reduce `accum_steps` or `batch_size` in `src/config.h` and rebuild.

**Loss stuck at ~10.4 (not decreasing):**
Learning rate might be too low during warmup. Check that data isn't empty
(`wc -c data/pretrain.bin` should show >100MB).

**cmake can't find CUDA:**
Make sure `nvcc` is on your PATH: `export PATH=/usr/local/cuda/bin:$PATH`

**"could not open data/pretrain.bin":**
Run the tokenize script first. The binary must exist before training.

---

## Licensing

All components are fully open-source:
- Code: Apache 2.0
- Pretrain data: Common Corpus (public domain / CC-BY)
- SFT data: Glaive v2 (Apache 2.0)
- Tokenizer: Open LLaMA (Apache 2.0)

You can release everything — weights, code, data — for any use.
