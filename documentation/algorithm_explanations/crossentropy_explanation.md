# Cross-Entropy Loss — Measuring How Wrong the Model Is

## What It Does

After the model processes a sequence, it outputs a **logit** for every token
in the vocabulary (32,000 values) at every position. Cross-entropy loss
measures how far these predictions are from the correct answer.

```
Input:    "The cat sat on the"
Target:   "cat sat on the mat"   ← shifted by one (next-token prediction)

At position 4, model should predict "mat" (token ID 15234)
Model outputs 32,000 logits — one score per possible token
Loss = how much probability the model put on the correct token
```

---

## Core Formula

**In one sentence: cross-entropy loss is the average of the negative log
probabilities the model assigned to each correct token.**

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│             1                                                       │
│  ℒ  =  ─────────  ×   Σ    -log( p_bt[ y_bt ] )                   │
│          B × T       b,t                                            │
│                                                                     │
│    average, over all B×T positions, of the negative log-probability │
│    the model assigned to that position's correct token y_bt.        │
│                                                                     │
│  where the probability for position (b,t) comes from softmax:       │
│                                                                     │
│                       exp( z_bt[ y_bt ] )                           │
│  p_bt[ y_bt ]  =  ──────────────────────────                       │
│                     Σ_v  exp( z_bt[ v ] )                           │
│                                                                     │
│  Where:                                                             │
│    z_bt ∈ ℝ^V        logits at position (b,t)  (V = 32,000)        │
│    y_bt              correct token ID at position (b,t)             │
│    v                 sums over all V vocab entries                  │
│    B × T             batch × seq_len = positions to average over    │
│                                                                     │
│  Simplified single-position loss (the log-sum-exp form):            │
│                                                                     │
│  ℒ_bt = -log( p_bt[ y_bt ] ) = -z_bt[ y_bt ] + log Σ_v exp(z_bt[v])│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Breaking Down Each Part

```
exp(z_v)              What it does: converts logit to positive number
                      Bigger logit → exponentially bigger value
                      This amplifies differences between predictions

Σ_v exp(z_v)          What it does: sums all exponentiated logits
                      This is the normalizer — makes probabilities sum to 1.0

exp(z_y) / Σ exp(z)   What it does: probability of the correct token
                       = "what fraction of total probability went to the right answer?"

-log(p_y)             What it does: penalizes low probability
                      p = 1.0 → loss = 0     (perfect)
                      p = 0.5 → loss = 0.69  (uncertain)
                      p = 0.01 → loss = 4.6  (very wrong)
                      The log makes the penalty grow faster as p → 0
```

---

## Why Negative Log? (Three Reasons)

The main reason isn't scale — it's **how the penalty behaves**. But the nice
scale is a real bonus.

```
1. Turns tiny probabilities into big, usable penalties

   Raw probability is awkward to optimize (p=0.01 is "almost zero").
   -log stretches it into a readable loss:

     p = 0.9    →  loss = 0.11   (good)
     p = 0.5    →  loss = 0.69   (meh)
     p = 0.01   →  loss = 4.6    (bad)
     p = 0.001  →  loss = 6.9    (very bad)

   This is why loss values land on a nice ~0–10 scale instead of
   0.00001 — making a loss of 7.4 readable at a glance.

2. Punishes confident wrong answers harder

   The curve is steep near 0, so being loudly wrong costs more:

     wrong but unsure     (p=0.1)   →  loss ≈ 2.3
     wrong but confident  (p=0.001) →  loss ≈ 6.9

3. Math convenience (the real "why" in theory)

   - Probabilities MULTIPLY across positions → logs turn that into
     SUMS, which give clean, stable gradients.
   - We MINIMIZE loss, so we flip the sign: negative log-likelihood
     = cross-entropy.
```


| Reason                   | Role                      |
| ------------------------ | ------------------------- |
| Nice scale (~0–10)       | Bonus — readable training |
| Harsh penalty when p → 0 | Main training signal      |
| Sums instead of products | Makes backprop clean      |


**So: scale is a bonus; the real point is a loss that punishes bad
predictions strongly and differentiably.** A loss of ~7.4 means "on average,
the model is pretty unsure about the right next token."

---

## The Two Steps: Softmax → Negative Log

### Step 1: Softmax — Turn Logits into Probabilities

```
                  exp(z_i)
  p_i  =  ──────────────────
            Σ_j exp(z_j)

  Logits → all positive, sum to 1.0
```

Example with a tiny 5-word vocab:

```
logits = [2.0,  5.0,  1.0,  0.5,  3.0]
                 ↑
            correct token (ID=1)

exp    = [7.39, 148.4, 2.72, 1.65, 20.09]     exponentiate each
sum    = 180.25                                 sum of all exp values
probs  = [0.041, 0.823, 0.015, 0.009, 0.111]   divide each by sum
                  ↑
          model puts 82.3% on correct answer — good!
```

### Step 2: Negative Log — Penalize Low Probability

```
  ℒ = -log(p_correct)

  Probability → Loss:
    p = 1.0    → ℒ = -log(1.0)   = 0.0     perfect prediction
    p = 0.5    → ℒ = -log(0.5)   = 0.693   uncertain
    p = 0.1    → ℒ = -log(0.1)   = 2.303   pretty wrong
    p = 0.01   → ℒ = -log(0.01)  = 4.605   very wrong
    p = 0.001  → ℒ = -log(0.001) = 6.908   terrible
```

```
                    loss
               7 │  ╲
               6 │   ╲
               5 │    ╲
               4 │     ╲
               3 │      ╲
               2 │        ╲
               1 │          ╲
               0 │─────────────╲───
                 0   0.2  0.4  0.6  0.8  1.0
                          probability
```

---

## Numerical Stability: The Max Trick

```
  Problem:  exp(1000) = ∞   (overflow!)

  Solution: subtract max before exponentiating

  Stable softmax:

                   exp(z_i - max(z))
    p_i  =  ───────────────────────────
              Σ_j exp(z_j - max(z))

  Why identical:

    exp(z_i - m)       exp(z_i) × exp(-m)       exp(z_i)
    ────────────── = ─────────────────────── = ──────────────
    Σ exp(z_j - m)   Σ exp(z_j) × exp(-m)     Σ exp(z_j)

    The exp(-m) cancels in numerator and denominator.
```

---

## Full Worked Example

**Setup:** position 4, model should predict "mat" (token ID 3), tiny vocab of 5

### Step 1: Raw Logits from Model

```
logits = [2.0, 5.0, 1.0, 0.5, 3.0]
                          ↑
                    target = token 3
```

### Step 2: Find Max (for stability)

```
max_val = 5.0
```

### Step 3: Exponentiate (shifted)

```
exp(2.0 - 5.0) = exp(-3.0) = 0.0498
exp(5.0 - 5.0) = exp(0.0)  = 1.0000
exp(1.0 - 5.0) = exp(-4.0) = 0.0183
exp(0.5 - 5.0) = exp(-4.5) = 0.0111    ← correct token
exp(3.0 - 5.0) = exp(-2.0) = 0.1353
```

### Step 4: Normalize (divide by sum)

```
sum = 0.0498 + 1.0000 + 0.0183 + 0.0111 + 0.1353 = 1.2145

probs = [0.041, 0.823, 0.015, 0.009, 0.111]
                                ↑
                    prob of correct token = 0.009
```

### Step 5: Compute Loss

```
ℒ = -log(0.009) = 4.71
```

That's bad — the model put only 0.9% probability on the correct answer.
The loss of 4.71 will create a strong gradient to fix this.

### Step 6: Average Across All Positions

```
                 1
total_loss = ─────── × Σ ℒ_bt
              B × T     bt

With B=4, T=512:
total_loss = (1/2048) × (sum of 2048 per-position losses)
```

This is the single scalar number printed in the training log.

---

## The Backward Pass — Why It's So Clean

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Gradient of cross-entropy + softmax w.r.t. logits:                │
│                                                                     │
│              p_v - 𝟙(v = y)                                        │
│  dz_v  =  ─────────────────                                       │
│                B × T                                                │
│                                                                     │
│  Where:                                                             │
│    p_v = softmax probability of token v                            │
│    𝟙(v = y) = 1 if v is the correct token, 0 otherwise            │
│                                                                     │
│  For CORRECT token:   dz = (p_y - 1) / (B×T)     ← NEGATIVE      │
│  For WRONG tokens:    dz = p_v / (B×T)            ← POSITIVE      │
│                                                                     │
│  Negative gradient → push logit UP (increase probability)          │
│  Positive gradient → push logit DOWN (decrease probability)        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Example:

```
probs    = [0.041, 0.823, 0.015, 0.009, 0.111]
one_hot  = [0,     0,     0,     1,     0    ]    (target = token 3)

dlogits  = [0.041, 0.823, 0.015, -0.991, 0.111]  / (B×T)
                                  ↑
                    negative! pushes this logit UP
                    all others are positive, pushing them DOWN
```

---

## Connection to Perplexity

```
  Perplexity = exp(ℒ)

  ℒ = 4.71  →  PPL = exp(4.71)  = 111    choosing among ~111 tokens
  ℒ = 2.0   →  PPL = exp(2.0)   = 7.4    choosing among ~7 tokens
  ℒ = 1.0   →  PPL = exp(1.0)   = 2.7    choosing among ~3 tokens
  ℒ = 0.5   →  PPL = exp(0.5)   = 1.6    almost certain

  Interpretation: perplexity = "how many tokens is the model
  choosing between?" Lower = more confident = better.
```

Your training log shows loss going from ~10.5 (perplexity 36,000 — random guessing
among 32k tokens) down to ~7.5 (perplexity 1,800) after 2000 steps.

---

## Compute Cost (FLOPs)

```
  Our config: B=4, T=512, V=32000, N=B×T=2048

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Operation                Shape / Count              FLOPs         │
  │  ─────────────────────    ───────────────────────    ──────────     │
  │  Final projection         [2048,768]@[768,32000]    2×2048×768×32k │
  │  (logits = hidden@W_out)                            = 100.7 GFLOPs │
  │                                                                    │
  │  Softmax (forward)        2048 rows × 32000 cols    ~3 ops/elem   │
  │  (max, exp, sum, div)                               = 197 MFLOPs   │
  │                                                                    │
  │  Log + loss               2048 positions             trivial       │
  │                                                                    │
  │  Backward dlogits         2048 × 32000              = 66 MFLOPs   │
  │  (p - one_hot, very cheap)                                         │
  │                                                                    │
  │  Backward projections     2 matmuls (dX, dW)        = 201 GFLOPs  │
  │  ─────────────────────────────────────────────────────────────────  │
  │  TOTAL forward:                                     ~101 GFLOPs   │
  │  TOTAL forward + backward:                          ~302 GFLOPs   │
  │                                                                    │
  │  The final projection is the SINGLE MOST EXPENSIVE operation      │
  │  in the entire model because V=32000 is huge.                     │
  │  It's 100 GFLOPs vs ~27 GFLOPs for an entire transformer layer.  │
  │                                                                    │
  │  At 101 TFLOPS: ~3.0 ms per step (theoretical)                   │
  └──────────────────────────────────────────────────────────────────────┘
```

