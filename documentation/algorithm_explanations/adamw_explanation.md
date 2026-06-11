# AdamW — The Optimizer

## What It Does

After the backward pass computes gradients (how much to change each parameter),
AdamW decides **how much to actually move** each of the 124.7M parameters.
It's smarter than plain gradient descent — it adapts the learning rate
per-parameter based on the history of gradients.

---

## Core Formula

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  For each parameter w, at each step t:                             │
│                                                                     │
│  1. First moment (momentum):                                       │
│     m_t = β₁ · m_{t-1} + (1 - β₁) · g_t                          │
│                                                                     │
│  2. Second moment (variance):                                      │
│     v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²                         │
│                                                                     │
│  3. Bias correction:                                               │
│            m_t                       v_t                            │
│     m̂ = ─────────           v̂ = ─────────                         │
│          1 - β₁^t                1 - β₂^t                          │
│                                                                     │
│  4. Update:                                                        │
│                                         m̂                          │
│     w_t = w_{t-1} × (1 - η·λ)  -  η · ─────────                  │
│                                        √v̂ + ε                     │
│           ╰────────────────╯       ╰───────────╯                   │
│             weight decay            gradient step                  │
│                                                                     │
│  Where:                                                             │
│    g_t = gradient at step t       η = learning rate (3e-4)         │
│    β₁ = 0.9   (momentum decay)   β₂ = 0.95  (variance decay)      │
│    λ = 0.1    (weight decay)      ε = 1e-8   (prevent div by 0)   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Breaking Down Each Part

```
m = β₁·m + (1-β₁)·g        FIRST MOMENT — gradient smoothing
                            Exponential moving average of gradient direction
                            β₁ = 0.9 means: 90% old momentum + 10% new gradient
                            Effect: smooths out noise, builds up in consistent direction
                            Like a heavy ball rolling — doesn't change from every bump

v = β₂·v + (1-β₂)·g²       SECOND MOMENT — gradient magnitude tracking
                            Exponential moving average of squared gradients
                            Tracks HOW BIG gradients typically are (ignoring sign)
                            Used to normalize: big gradients → small steps

m̂ = m / (1 - β₁^t)         BIAS CORRECTION — warmup fix
                            At step 1, m ≈ 0.1×g (too small, since m starts at 0)
                            Dividing by (1-0.9^1) = 0.1 corrects back to ≈ g
                            Correction shrinks over time: (1-0.9^100) ≈ 1.0

m̂ / (√v̂ + ε)               ADAPTIVE UPDATE — per-parameter learning rate
                            Direction from m̂, magnitude normalized by √v̂
                            Big gradients (large v̂) → small step (÷ large √v̂)
                            Small gradients (tiny v̂) → big step (÷ small √v̂)

w × (1 - η·λ)              WEIGHT DECAY — regularization
                            Shrinks weights toward zero each step
                            Prevents overfitting by keeping weights small
                            "Decoupled" from gradient (applied separately)
```

---

## Why Not Just SGD?

Stochastic Gradient Descent (SGD) updates every parameter the same way:

```
  SGD:    w = w - η × g

  Problems:
  ┌──────────────────────────────────────────────────────────────┐
  │  1. Noisy gradients:  batch 1 gives g=+0.5, batch 2 gives  │
  │     g=-0.3. SGD jerks back and forth.                        │
  │     Adam fixes: momentum (m) smooths the noise.              │
  │                                                              │
  │  2. Different scales:  some params have g=0.001, others      │
  │     g=10.0. One learning rate can't fit both.                │
  │     Adam fixes: dividing by √v normalizes per-parameter.     │
  │                                                              │
  │  3. Saddle points:  gradient ≈ 0 doesn't mean minimum.      │
  │     SGD stops. Adam's momentum carries through.              │
  └──────────────────────────────────────────────────────────────┘
```

---

## Bias Correction: Why Divide by (1 - β^t)?

```
At step 1, m and v start at 0. After one gradient:

  m = 0.9 × 0 + 0.1 × g = 0.1 × g      ← 10× too small!

Bias correction:

  Step  1:  m̂ = m / (1 - 0.9^1)  = m / 0.1     = g         ← corrected!
  Step  2:  m̂ = m / (1 - 0.9^2)  = m / 0.19    ← less correction needed
  Step 10:  m̂ = m / (1 - 0.9^10) = m / 0.651   ← almost no correction
  Step 100: m̂ = m / (1 - 0.9^100) ≈ m / 1.0    ← correction vanishes
```

---

## Full Worked Example

**Setup:** One parameter, η=3e-4, β₁=0.9, β₂=0.95, λ=0.1, ε=1e-8

### Initial State

```
w = 0.5          (parameter value)
m = 0.0          (first moment)
v = 0.0          (second moment)
```

### Step 1: gradient = 0.3

```
m = 0.9 × 0.0 + 0.1 × 0.3      = 0.030
v = 0.95 × 0.0 + 0.05 × 0.09   = 0.0045

m̂ = 0.030 / (1 - 0.9^1) = 0.030 / 0.1    = 0.300
v̂ = 0.0045 / (1 - 0.95^1) = 0.0045 / 0.05 = 0.090

update = 0.300 / (√0.090 + 1e-8) = 0.300 / 0.300 = 1.000

w = 0.5 × (1 - 3e-4 × 0.1) - 3e-4 × 1.000
  = 0.5 × 0.99997 - 0.0003
  = 0.49999 - 0.0003
  = 0.49969
```

### Step 2: gradient = -0.1

```
m = 0.9 × 0.030 + 0.1 × (-0.1)  = 0.017
v = 0.95 × 0.0045 + 0.05 × 0.01 = 0.00478

m̂ = 0.017 / (1 - 0.9^2)  = 0.017 / 0.19  = 0.0895
v̂ = 0.00478 / (1 - 0.95^2) = 0.00478 / 0.0975 = 0.0490

update = 0.0895 / (√0.0490 + 1e-8) = 0.0895 / 0.2214 = 0.4043

w = 0.49969 × 0.99997 - 3e-4 × 0.4043
  = 0.49967 - 0.000121
  = 0.49955
```

Notice how:
- Step 1: gradient was 0.3, update magnitude was 1.0 (bias correction amplified it)
- Step 2: gradient was -0.1 (opposite direction!), but m still positive (momentum),
  so we still move in the original direction, just slower.

---

## Gradient Clipping (Before AdamW)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  global_norm = √( Σ_i  g_i² )    across all 124.7M parameters │
│                                                                 │
│  if global_norm > max_norm:                                     │
│                 max_norm                                        │
│      g = g × ──────────────                                    │
│              global_norm                                        │
│                                                                 │
│  Our config: max_norm = 1.0                                     │
│                                                                 │
│  Effect: preserves gradient direction,                          │
│          limits step size to prevent explosions                  │
│                                                                 │
│  Example:  global_norm = 5.0                                    │
│    Every gradient gets scaled by 1.0/5.0 = 0.2                 │
│    Direction unchanged, magnitude capped                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Learning Rate Schedule

```
  η(t) = ?

                                    ┌──────────── cosine decay ────────────┐
  η                                 │                                      │
  3e-4 │           ╱────────────────╲                                      │
       │          ╱                   ╲                                    │
       │         ╱                      ╲                                  │
       │        ╱                         ╲                                │
  3e-5 │───────╱                            ╲─────────                     │
       │ warmup                              cooldown                     │
       └──────────────────────────────────────────────                     │
       0      500              4500   5000                                │
                    step                                                   │
                                                                           │
  Warmup (0→500):                                                          │
    η = η_max × (t / t_warmup)      linear ramp from 0 to 3e-4           │
                                                                           │
  Cosine decay (500→4500):                                                │
    η = η_min + 0.5×(η_max - η_min)×(1 + cos(π × progress))             │
    where progress = (t - t_warmup) / (t_total - t_warmup)                │
                                                                           │
  Cooldown (4500+):                                                        │
    η = η_min = 3e-5                                                      │
```

---

## Mixed Precision: FP32 Master Weights

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Memory Layout                             │
│                                                                  │
│  BF16 parameters  [124.7M × 2 bytes = 249 MB]                  │
│    → used in forward/backward (fast, Tensor Cores)               │
│                                                                  │
│  FP32 master weights [124.7M × 4 bytes = 499 MB]               │
│  FP32 first moment m [124.7M × 4 bytes = 499 MB]               │
│  FP32 second moment v [124.7M × 4 bytes = 499 MB]              │
│    → used in optimizer update (precise)                          │
│                                                                  │
│  Why FP32?                                                       │
│    η × update ≈ 0.0003                                          │
│    BF16:  0.5 - 0.0003 = 0.5      (rounded away!)              │
│    FP32:  0.5 - 0.0003 = 0.4997   (correct)                    │
│                                                                  │
│  Workflow:                                                       │
│    1. Forward:    use BF16 weights (fast)                        │
│    2. Backward:   compute BF16 gradients (fast)                  │
│    3. Optimizer:  update FP32 master weights (precise)           │
│    4. Cast back:  FP32 master → BF16 weights                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## CUDA Kernel Implementation

```
Thread i handles parameter[i]:

  1. Load:     g = bf16_grad[i],  w32 = fp32_master[i],  m = m1[i],  v = m2[i]
  2. Convert:  g32 = bf16_to_float(g)
  3. Moments:  m = β₁×m + (1-β₁)×g32        v = β₂×v + (1-β₂)×g32²
  4. Correct:  m̂ = m/(1-β₁^t)               v̂ = v/(1-β₂^t)
  5. Update:   w32 = w32 × (1-η×λ) - η × m̂/(√v̂ + ε)
  6. Store:    fp32_master[i] = w32,  m1[i] = m,  m2[i] = v
  7. Cast:     bf16_params[i] = float_to_bf16(w32)
```

124.7M parameters, each handled by one thread → trivially parallel.

---

## Memory Overhead Summary

```
Component             Size        Format
───────────────────────────────────────
BF16 parameters       249 MB      bf16     (used in forward/backward)
BF16 gradients        249 MB      bf16     (computed in backward)
FP32 master weights   499 MB      fp32     (precise copy for updates)
FP32 first moment m   499 MB      fp32     (gradient mean)
FP32 second moment v  499 MB      fp32     (gradient variance)
───────────────────────────────────────
Total optimizer state: 1,995 MB  ≈ 2 GB

16 bytes per parameter for the optimizer.
2 bytes per parameter for the model itself (bf16).
Optimizer state is 8× larger than the model weights.
```

---

## Compute Cost (FLOPs)

```
  Our config: 124.7M parameters

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Operation                        Per-param FLOPs    Total FLOPs   │
  │  ──────────────────────────────   ──────────────     ──────────     │
  │  Grad norm (Σ g²)                 2 (square+add)    249 MFLOPs     │
  │  Grad clip (g × scale)            1 (multiply)      125 MFLOPs     │
  │  m = β₁m + (1-β₁)g               3 (mul,mul,add)   374 MFLOPs     │
  │  v = β₂v + (1-β₂)g²              4 (sq,mul,mul,add) 499 MFLOPs    │
  │  Bias correction (m̂, v̂)           2 (div, div)      249 MFLOPs     │
  │  Update (m̂/(√v̂+ε))               3 (sqrt,add,div)  374 MFLOPs     │
  │  Weight decay (w×(1-ηλ))          2 (mul, sub)      249 MFLOPs     │
  │  BF16 cast                        1 (convert)       125 MFLOPs     │
  │  ──────────────────────────────────────────────────────────────     │
  │  TOTAL:                           ~18 FLOPs/param   ~2.2 GFLOPs   │
  │                                                                    │
  │  Compare to forward+backward matmuls: ~1,300 GFLOPs               │
  │  Optimizer is only 0.2% of total compute!                         │
  │                                                                    │
  │  The optimizer is MEMORY-BOUND, not compute-bound:                │
  │    4 loads (g, w32, m, v) + 4 stores = 8 × 124.7M × 4 bytes     │
  │    = 4.0 GB of memory traffic per step                            │
  │    At 360 GB/s (RTX 3060): 4.0/360 = 11 ms (memory-limited)     │
  │    At 12.7 TFLOPS: 2.2G/12.7T = 0.17 ms (compute would be fast) │
  │    → bottleneck is memory bandwidth, not arithmetic              │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## Why 3e-4 as Peak Learning Rate?

```
  3e-4 is the standard peak LR for AdamW at the ~100M-300M param scale.
  It comes from empirical scaling laws (GPT, LLaMA, Chinchilla papers).

  ┌──────────────────────────────────────────────────────────────────┐
  │  Model Size        Peak LR        Source                        │
  │  ──────────────    ─────────      ──────────────────            │
  │  ~125M (ours)      3e-4           GPT-2, LLaMA                 │
  │  ~350M             3e-4           Same range                    │
  │  ~1.3B             1.5e-4         Halved — model is bigger      │
  │  ~7B               1e-4           LLaMA paper                   │
  │  ~70B              5e-5           Need very gentle steps        │
  │                                                                  │
  │  Rule: bigger model → smaller LR                                │
  │  (more parameters means each step affects more interactions)    │
  └──────────────────────────────────────────────────────────────────┘

  Why this number works with AdamW specifically:

  Adam's update ≈ lr × m̂ / (√v̂ + ε) ≈ lr × sign(gradient)
  So the actual weight change per step ≈ ±lr per parameter.

  With lr = 3e-4:
    Each weight moves ~0.0003 per step
    After 30,000 steps: max displacement ≈ 9.0
    Weights initialized at std ≈ 0.02
    → enough room to learn, not enough to overshoot

  What happens at different LRs:

  ┌──────────────────────────────────────────────────────────────────┐
  │  LR          Effect                                             │
  │  ──────      ─────────────────────────────────────              │
  │  1e-3        Too aggressive for 125M. Loss unstable, may       │
  │              diverge. OK for tiny models (<10M).                │
  │                                                                  │
  │  3e-4        Sweet spot. Fast convergence + stable.             │
  │              Used by GPT-2, LLaMA, most papers at this scale.  │
  │                                                                  │
  │  1e-4        Too conservative. Converges but ~2× slower.       │
  │              Need ~2× more steps to reach same loss.            │
  │                                                                  │
  │  3e-5        Way too slow. Model barely moves.                  │
  │              Need ~10× more steps. Wastes GPU hours.            │
  └──────────────────────────────────────────────────────────────────┘

  Our min_lr = 3e-5 (10× lower than peak) is also standard.
  The cosine decay drops LR by 10× over training for a smooth landing.

  There is no closed-form formula that derives 3e-4. It comes from
  thousands of LR sweep experiments across model scales by OpenAI,
  Meta, and DeepMind, formalized in the Chinchilla scaling laws.
```
