# JrLAM Pretraining Guide

Everything about the full pretraining run for JrLAM (124.7M params) on an RTX 3060 12GB.

---

## Hardware: RTX 3060 12GB

| Spec | Value |
|---|---|
| VRAM | 12 GB GDDR6 |
| fp32 TFLOPS | 12.74 |
| fp16 Tensor Core TFLOPS | ~51 |
| Memory Bandwidth | 360 GB/s |
| Max Safe Temp | 83C (throttles above this) |
| TDP | 170W |

---

## Model: JrLAM-124M

| Parameter | Value | Meaning |
|---|---|---|
| n_layer | 12 | Transformer blocks stacked |
| n_head | 12 | Query attention heads |
| n_kv_head | 4 | Key/value heads (GQA, 3:1 sharing) |
| n_embd | 768 | Hidden dimension |
| ffn_hidden | 2048 | MLP expansion size |
| vocab_size | 32,000 | SentencePiece tokenizer |
| Total params | 124.7M | All trainable weights |

---

## Dataset: TinyStories (500M tokens)

| Field | Value |
|---|---|
| Source | Microsoft Research TinyStories |
| Content | Simple short stories, clean English |
| Total tokens | ~500M |
| File size | ~2 GB (pretrain.bin) |
| Tokens per param | 4x (Chinchilla optimal = 20x) |
| Download + tokenize time | ~1-2 hours |

**Why TinyStories?** Clean, coherent text that teaches grammar, vocabulary,
and simple narrative structure. At 500M tokens it's enough to get JrLAM
producing grammatical sentences and basic stories. Not GPT-4, but a real
working language model.

---

## Training Time Estimates by Batch Size

All estimates: fp16 mixed precision, T=512, 500M tokens, accum adjusted to
keep effective batch at ~16K tokens.

| B | accum | Effective Batch | VRAM | tok/s (est.) | GPU Temp | **Time** |
|---|---|---|---|---|---|---|
| 4 | 8 | 16,384 | 2.5 GB | ~7,000 | ~65C | **~20 hours** |
| 8 | 4 | 16,384 | 3.1 GB | ~10,000 | ~70C | **~14 hours** |
| 12 | 2 | 12,288 | 3.6 GB | ~12,000 | ~73C | **~11.5 hours** |
| **16** | **2** | **16,384** | **4.2 GB** | **~13,000** | **~75C** | **~10 hours** |
| 20 | 2 | 20,480 | 4.8 GB | ~14,000 | ~77C | **~9.5 hours** |
| 24 | 2 | 24,576 | 5.4 GB | ~15,000 | ~78C | **~9 hours** |
| 32 | 1 | 16,384 | 6.8 GB | ~16,000 | ~80C | **~8.5 hours** |

**Recommended: B=16** — best balance of speed, temperature, and VRAM headroom.

### Why B=16 is the sweet spot

- **Fast enough:** ~13,000 tok/s is ~90% of the GPU's peak throughput
- **Cool enough:** ~75C is safe for 24/7 operation, fans stay quiet
- **VRAM headroom:** 4.2 / 12 GB = only 35% used, room for OS + browser + other apps
- **Future-proof:** If you increase model size or T later, you have 7.8 GB of breathing room
- **Diminishing returns:** B=24 is only ~15% faster but uses 30% more VRAM and runs hotter

### When to choose differently

| Situation | Use |
|---|---|
| Running overnight, PC idle | B=16 (default, best balance) |
| Gaming/working while training | B=8 (uses less VRAM, GPU stays cooler) |
| Want absolute fastest, don't care about heat | B=24 |
| Debugging / testing | B=4, accum=1 (minimal VRAM) |

---

## Training Parameters Explained

### B (batch size) — `--batch 16`

How many sequences processed in parallel per micro-batch.
- Each sequence = T tokens = 512 tokens
- One micro-batch = B × T = 16 × 512 = 8,192 tokens
- Bigger B = more GPU utilization = faster, but more VRAM

### T (sequence length) — hardcoded 512

Tokens per training sequence. The model learns to predict each token from
all previous tokens within the sequence.
- T=512 = model can see up to 512 tokens of context
- VRAM scales as T² for attention, so doubling T roughly quadruples attention memory
- 512 is ideal for 124M on a 3060

### accum (gradient accumulation) — `--accum 2`

How many micro-batches of gradients to sum before one weight update.
- accum=1: update weights after every micro-batch (fastest per step, noisiest)
- accum=2: process 2 micro-batches, sum gradients, then update (smoother)
- accum=4: even smoother gradients
- **Does NOT change total training time** — same number of tokens either way
- Only changes how many weight updates you get

**Effective batch = B × T × accum:**
| B | accum | Effective batch |
|---|---|---|
| 4 | 8 | 16,384 |
| 8 | 4 | 16,384 |
| 16 | 2 | 16,384 |
| 32 | 1 | 16,384 |

All give the same effective batch size — the difference is whether the GPU
processes 4, 8, 16, or 32 sequences at a time.

### --fp16 (mixed precision)

Enables fp16 mixed precision training:
- Weights, activations, gradients stored as 16-bit (half VRAM)
- Matrix multiplications use tensor cores (3-4x faster)
- Master weights + optimizer stay fp32 (numerical stability)
- **Always use this on RTX 3060.** No reason not to.

### --lr (learning rate) — `--lr 3e-4`

Peak learning rate. Controls how big each weight update step is.
- Too high = training explodes (loss goes to NaN)
- Too low = training is painfully slow
- 3e-4 is standard for 124M parameter models
- Uses warmup (ramps from 0 to peak) + cosine decay (slowly decreases to min)

### --warmup (warmup steps) — `--warmup 500`

Number of steps to linearly ramp LR from 0 to peak.
- Prevents instability at the start when gradients are wild
- 500-1000 steps is typical
- After warmup, LR follows cosine decay down to min_lr

### --steps (total steps) — `--steps 15300`

Total number of weight updates. Determines how many tokens you train on:
- `tokens = steps × effective_batch`
- `15300 × 16384 = ~250M tokens per epoch` → 2 epochs over 500M tokens
- For exactly 1 epoch: `500M / 16384 ≈ 30,500 steps`

### --checkpoint (resume from saved weights)

Load a previously saved checkpoint and continue training:
```bash
./build/train data/pretrain.bin --fp16 --batch 16 --checkpoint checkpoints/step_10000.bin
```

Checkpoints are saved as fp32 master weights, compatible with both fp16 and fp32 mode.

---

## The Commands

### Step 1: Tokenize (run once, ~1-2 hours)

```bash
cd ~/llm_for_fun/llm_from_scratch
source lam_env/bin/activate
python scripts/tokenize_pretrain.py \
    --tokenizer data/tokenizer.model \
    --output data/pretrain.bin \
    --max_tokens 500000000
```

### Step 2: Sanity check (2 minutes)

```bash
./build/train data/pretrain.bin --steps 50 --accum 1 --fp16 --batch 16
```

Check that:
- Loss starts ~10.4 and ticks down slightly
- tok/s is ~13,000 (confirms tensor cores working)
- No crashes or NaN

### Step 3: Full pretrain (~10 hours)

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

### Step 4: Monitor (optional, in another terminal)

```bash
watch -n 5 nvidia-smi
```

Check GPU temp stays under 80C. If it's too hot, kill the run and restart with `--batch 8`.

---

## What To Expect During Training

```
Building model (fp16 mixed precision)...
Model (fp16): 12 layers, 12 heads (4 kv), 768 embd, 2048 ffn
Parameters: 124.7M
DataLoader: 500000000 tokens from data/pretrain.bin
Training for 15300 steps (accum=2, effective batch=16384 tokens)...
step     1 | loss 10.4741 | lr 6.00e-07 | ~13000 tok/s
step   500 | loss  7.8000 | lr 3.00e-04 | ~13000 tok/s
step  2000 | loss  5.8000 | lr 2.92e-04 | ~13000 tok/s
step  5000 | loss  4.9000 | lr 2.60e-04 | ~13000 tok/s
step  8000 | loss  4.4000 | lr 2.10e-04 | ~13000 tok/s
step 10000 | loss  4.2000 | lr 1.65e-04 | ~13000 tok/s
step 13000 | loss  4.0000 | lr 7.50e-05 | ~13000 tok/s
step 15300 | loss  3.9000 | lr 3.00e-05 | ~13000 tok/s
Training complete.
```

### Loss progression — what JrLAM learns at each stage

| Loss | Step (approx.) | Time | JrLAM can... |
|---|---|---|---|
| ~10.4 | 0 | 0 min | Nothing. Random guessing over 32K vocab |
| ~8.0 | ~300 | ~6 min | Predict common tokens ("the", "a", spaces, punctuation) |
| ~6.0 | ~1,500 | ~30 min | Output real English words, broken grammar |
| ~5.0 | ~4,000 | ~1.3 hours | Short phrases that mostly make sense |
| ~4.5 | ~7,000 | ~2.3 hours | Grammatically correct sentences |
| ~4.0 | ~13,000 | ~4.3 hours | Coherent paragraphs, simple stories |
| ~3.9 | ~15,300 | ~10 hours | Simple narratives with characters and plot |

**You can stop early and resume later.** Checkpoints save every 2000 steps.

---

## After Pretraining

JrLAM now understands English but doesn't know how to follow instructions
or call functions. That's what Phase 2 (SFT) is for — see `TRAINING.md`.

The pretrained checkpoint at `checkpoints/step_15300.bin` is your base model.
Back it up before starting SFT:
```bash
cp checkpoints/step_15300.bin checkpoints/jrlam_pretrained.bin
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA out of memory | Lower B: `--batch 8` |
| GPU too hot (>80C) | Lower B: `--batch 8`, improve airflow |
| Loss stuck at ~10.4 | Check data file isn't empty: `ls -la data/pretrain.bin` |
| Loss goes to NaN | Lower LR: `--lr 1e-4` |
| tok/s much lower than expected | Make sure `--fp16` is set |
| Want to resume training | `--checkpoint checkpoints/step_XXXXX.bin` |
