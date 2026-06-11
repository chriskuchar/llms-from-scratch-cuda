# Softmax — Turning Numbers into Probabilities

## What It Does

Softmax takes a vector of arbitrary real numbers (logits) and converts
them into a **probability distribution** — all values between 0 and 1,
summing to exactly 1.0.

```
logits = [2.0,  5.0,  1.0,  0.5,  3.0]     arbitrary numbers, any range
                 │
            softmax
                 │
probs  = [0.041, 0.823, 0.015, 0.009, 0.111]   all positive, sum = 1.0
```

---

## Core Formula

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                   exp(x_i)                                      │
│  softmax(x)_i = ──────────────                                 │
│                  Σ_j exp(x_j)                                   │
│                                                                 │
│  Numerically stable version (subtract max first):              │
│                                                                 │
│                   exp(x_i - max(x))                             │
│  softmax(x)_i = ────────────────────                           │
│                  Σ_j exp(x_j - max(x))                          │
│                                                                 │
│  Properties:                                                    │
│    • 0 < softmax(x)_i < 1     always positive                  │
│    • Σ_i softmax(x)_i = 1     sums to 1.0                     │
│    • monotonic: bigger input → bigger output                    │
│    • differentiable everywhere                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Breaking Down Each Part

```
exp(x_i)             What it does: makes each value positive
                     exp(-5) = 0.007   exp(0) = 1.0   exp(5) = 148.4
                     Amplifies differences between inputs exponentially

Σ_j exp(x_j)        What it does: sums all the positive values
                     This becomes the denominator — the normalizer

exp(x_i) / Σ exp     What it does: divide each by the total
                     Now each value is a fraction of the whole → probability
                     Guaranteed to sum to 1.0

x_i - max(x)         What it does: shifts values so largest = 0
                     Prevents exp from overflowing (exp(1000) = infinity)
                     Mathematically identical — the max cancels out
```

---

## Where It's Used

```
1. Attention Scores → Attention Weights

   scores = Q @ K^T / √d_k          [T × T] raw scores
   Each row:  [2.1, -0.3, 1.5, -∞, -∞]    (with causal mask)
                        │
                    softmax (per row)
                        │
   Weights:    [0.56, 0.05, 0.39, 0.00, 0.00]   probabilities

2. Final Logits → Token Probabilities (in cross-entropy)

   logits = model output [32000]       one score per vocab token
                        │
                    softmax
                        │
   probs  = [0.0001, ..., 0.042, ..., 0.0003]   probability of each token
```

---

## Why Exponentiation?

```
The exp function does two things:

1. Makes everything positive:
   exp(anything) > 0  always

2. Amplifies differences exponentially:
   input:     [1.0,   2.0,   3.0]       differences of 1.0 each
   exp:       [2.72,  7.39,  20.09]      differences GROW
   softmax:   [0.090, 0.245, 0.665]      biggest input dominates

   If inputs are close → spread-out distribution
   If one input is much larger → near one-hot distribution
```

---

## Numerical Stability: The Max Trick

```
Problem:

  logits = [1000.0, 1001.0, 999.0]
  exp(1000) = 1.97 × 10^434        ← INFINITY in float32!
  float32 overflows at exp(~88)

Solution:

  max = 1001.0
  shifted = [1000-1001, 1001-1001, 999-1001] = [-1.0, 0.0, -2.0]

  exp(-1.0) = 0.368
  exp(0.0)  = 1.000      ← max element always becomes exp(0) = 1.0
  exp(-2.0) = 0.135

  All exp values ≤ 1.0 — no overflow possible.

Why it's identical:

  exp(x_i - m)       exp(x_i) × exp(-m)       exp(x_i)
  ──────────────  =  ─────────────────────  =  ──────────
  Σ exp(x_j - m)    Σ exp(x_j) × exp(-m)     Σ exp(x_j)

  The exp(-m) cancels in numerator and denominator.
```

---

## Full Worked Example

**Input:** attention scores for one row (position 2 attending to positions 0-4)

### Step 1: Raw Scores (after causal mask)

```
scores = [0.16, 0.03, 0.35, -∞, -∞]
           ↑     ↑     ↑    ↑    ↑
         pos 0  pos 1  pos 2  future (masked)
```

### Step 2: Find Max

```
max = 0.35    (ignoring -∞)
```

### Step 3: Subtract Max and Exponentiate

```
scores - max = [-0.19, -0.32, 0.00, -∞, -∞]

exp(-0.19) = 0.827
exp(-0.32) = 0.726
exp(0.00)  = 1.000
exp(-∞)    = 0.000
exp(-∞)    = 0.000
```

### Step 4: Sum

```
sum = 0.827 + 0.726 + 1.000 + 0.000 + 0.000 = 2.553
```

### Step 5: Normalize

```
probs = [0.827/2.553, 0.726/2.553, 1.000/2.553, 0.000, 0.000]
      = [0.324,       0.284,       0.392,       0.000, 0.000]

sum check: 0.324 + 0.284 + 0.392 = 1.000 ✓
```

---

## The CUDA Kernel: Three Parallel Reductions

The kernel launches **one block per row** (`<<<rows, 256>>>`):

```
256 threads cooperate on one row of "cols" elements.

Phase 1: FIND MAX (parallel reduction)
──────────────────────────────────────
  Each thread finds max of its assigned columns:
    thread 0:   max(col[0], col[256], col[512], ...)
    thread 1:   max(col[1], col[257], col[513], ...)
    ...
    thread 255: max(col[255], col[511], col[767], ...)

  Tree reduction in shared memory:
    256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
    Result: sdata[0] = row_max

Phase 2: EXP AND SUM (parallel reduction)
─────────────────────────────────────────
  Each thread: for its columns j:
    e = exp(input[j] - row_max)
    output[j] = e                    store exp (normalize later)
    local_sum += e

  Tree reduction → row_sum in sdata[0]

Phase 3: NORMALIZE (parallel)
────────────────────────────
  Each thread: for its columns j:
    output[j] /= row_sum            divide by sum → probability
```

---

## Softmax Temperature

```
  softmax(x_i / τ)     where τ = temperature

  ┌────────────────────────────────────────────────────────────┐
  │  Temperature    Effect                                     │
  │  ──────────     ──────                                     │
  │  τ → 0          one-hot (argmax)      "very confident"    │
  │  τ = 1.0        normal softmax        "standard"          │
  │  τ → ∞          uniform (1/N each)    "maximum entropy"   │
  │                                                            │
  │  This is what "temperature" means in ChatGPT —            │
  │  it controls how peaked or flat the distribution is.      │
  └────────────────────────────────────────────────────────────┘
```

---

## Backward Pass Formula

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  dx_i = p_i × (dy_i - Σ_j dy_j × p_j)                        │
│                                                                 │
│  Where:                                                         │
│    p = softmax output (probabilities)                          │
│    dy = upstream gradient (dout)                                │
│    dx = gradient of raw inputs                                 │
│                                                                 │
│  Why the subtraction?                                           │
│    Increasing x[2] makes exp(x[2]) bigger                      │
│    → prob[2] increases                                          │
│    → BUT sum also increases → ALL probs decrease slightly       │
│    → net effect captured by the "Σ dy_j × p_j" term            │
│                                                                 │
│  Softmax couples ALL outputs — can't change one without         │
│  affecting all others. The backward must account for this.      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Compute Cost (FLOPs)

```
  Softmax is used in two places with very different costs:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  1. Attention softmax (per head, per layer):                       │
  │     Shape: [B×n_head, T, T] = [48, 512, 512]                     │
  │     Per element: ~5 ops (max, subtract, exp, sum, divide)         │
  │     Total: 48 × 512 × 512 × 5 = 62.9 MFLOPs per layer           │
  │     All 12 layers: 755 MFLOPs                                     │
  │                                                                    │
  │  2. Cross-entropy softmax (final logits):                          │
  │     Shape: [B×T, V] = [2048, 32000]                               │
  │     Per element: ~5 ops                                            │
  │     Total: 2048 × 32000 × 5 = 328 MFLOPs                        │
  │                                                                    │
  │  TOTAL softmax compute: ~1.1 GFLOPs                               │
  │                                                                    │
  │  Compare to matmul compute: ~1,300 GFLOPs                        │
  │  Softmax is only 0.08% of total compute!                          │
  │                                                                    │
  │  But softmax is MEMORY-BOUND:                                     │
  │    Three passes over the data (max, exp+sum, normalize)           │
  │    For attention: 3 × 48 × 512² × 2 bytes = 75 MB of traffic     │
  │    This is why fusing softmax into Flash Attention helps —        │
  │    it eliminates the HBM round-trips between passes.              │
  └──────────────────────────────────────────────────────────────────────┘
```
