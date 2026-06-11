# RMSNorm — Root Mean Square Normalization

## Core Formula

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Forward:                                                           │
│                      x                                              │
│    RMSNorm(x) = ─────────── × γ                                   │
│                  RMS(x)                                             │
│                                                                     │
│  Where:                                                             │
│                   ┌─────────────────┐                               │
│    RMS(x) =      │  1              │                               │
│              √   │ ─── Σ x_i²  + ε │                              │
│                   │  C   i          │                               │
│                   └─────────────────┘                               │
│                                                                     │
│    x ∈ ℝ^C           input vector (C = 768)                       │
│    γ ∈ ℝ^C           learned per-channel scale (initialized to 1) │
│    ε = 1e-5          prevents division by zero                     │
│    rrms = 1/RMS(x)   cached for backward pass                     │
│                                                                     │
│  Backward:                                                          │
│    s = Σ_c (dy_c × γ_c × x_c)                                     │
│    dx_c = rrms × (dy_c × γ_c - x_c × rrms² × s / C)              │
│    dγ_c += dy_c × x_c × rrms          (atomicAdd across rows)     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Breaking Down Each Part

```
Σ x_i²              What it does: sum of squares of all 768 values
                     Measures total "energy" or scale of the vector

(1/C) × Σ x_i²      What it does: MEAN of squared values
                     Normalizes for the number of dimensions

√(mean + ε)          What it does: root mean square (RMS)
                     ε = 1e-5 prevents division by zero when x ≈ 0

x / RMS              What it does: normalize to unit RMS
                     After this, the RMS of the output ≈ 1.0
                     Regardless of the original scale of x

× γ                  What it does: learned per-channel rescaling
                     Each channel gets its own volume knob
                     γ[i] > 1 → amplify channel i
                     γ[i] < 1 → suppress channel i
```

---

## What It Does

RMSNorm normalizes each token's 768-dim vector so its values have a
consistent scale, then applies a learned per-channel scaling factor.

```
input  = [3.2, -1.5, 0.8, 2.1, ...]      (768 values, arbitrary scale)
                    │
                RMSNorm
                    │
output = [0.89, -0.42, 0.22, 0.58, ...]   (768 values, controlled scale)
```

---

## Where It Appears

```
Per layer (×12):
  x ──→ RMSNorm ──→ Attention ──→ + (residual) ──→
  x ──→ RMSNorm ──→ SwiGLU    ──→ + (residual) ──→

Plus one final RMSNorm before the output projection.

Total: 12 × 2 + 1 = 25 RMSNorm operations
```

---

## Why Normalize?

```
The Scale Problem:

  After layer 1:    x = [0.3, -0.1, 0.5, ...]        scale ≈ 0.3
  After layer 5:    x = [2.1,  1.3, -1.8, ...]        scale ≈ 2.0
  After layer 12:   x = [15.2, -8.7, 12.4, ...]       scale ≈ 12.0

  Each layer adds to the residual stream → values grow.
  Attention and SwiGLU can't handle wildly different scales.

  RMSNorm resets the scale:
    Before norm:  [15.2, -8.7, 12.4, ...]      scale ≈ 12.0
    After norm:   [1.05, -0.60, 0.86, ...]      scale ≈ 1.0     ← consistent!
```

---

## RMSNorm vs LayerNorm

```
┌────────────────────────────────────────────────────────────────┐
│  LayerNorm (GPT-2):                                           │
│    LN(x) = (x - mean(x)) / std(x) × γ + β                   │
│                                                                │
│    • Subtracts mean (centers values)                          │
│    • Divides by std (normalizes spread)                        │
│    • Has bias parameter β                                     │
│    • Needs 2 reductions + 2 parameter vectors                 │
│                                                                │
│  RMSNorm (LLaMA):                                             │
│    RN(x) = x / RMS(x) × γ                                    │
│                                                                │
│    • No mean subtraction                                       │
│    • No bias parameter                                         │
│    • Needs 1 reduction + 1 parameter vector                   │
│    • Empirically works just as well                           │
│    • Faster and fewer parameters                              │
│                                                                │
│  For C=768:                                                    │
│    LayerNorm: 1536 params (γ + β)                             │
│    RMSNorm:   768 params (γ only)                             │
└────────────────────────────────────────────────────────────────┘
```

---

## Full Worked Example

**Setup:** one token position, C=8 (simplified from 768)

### Input Vector

```
x = [2.0, -1.0, 0.5, 3.0, -0.5, 1.0, -2.0, 1.5]
```

### Step 1: Sum of Squares

```
x² = [4.0, 1.0, 0.25, 9.0, 0.25, 1.0, 4.0, 2.25]

Σx² = 4.0 + 1.0 + 0.25 + 9.0 + 0.25 + 1.0 + 4.0 + 2.25 = 21.75
```

### Step 2: RMS

```
mean(x²) = 21.75 / 8 = 2.71875

RMS = √(2.71875 + 1e-5) = √2.71876 = 1.6489
```

### Step 3: Normalize

```
rrms = 1/RMS = 1/1.6489 = 0.6065     ← saved for backward

x_norm = x × rrms

x_norm = [1.213, -0.607, 0.303, 1.820, -0.303, 0.607, -1.213, 0.910]

Check: RMS of x_norm ≈ 1.0  ✓
```

### Step 4: Apply Learned Weight

```
γ = [1.1, 0.9, 1.0, 1.2, 0.8, 1.0, 1.1, 0.95]

output = x_norm ⊙ γ

output = [1.334, -0.546, 0.303, 2.184, -0.243, 0.607, -1.334, 0.864]
```

---

## CUDA Kernel: One Block Per Row

```
  Launch: <<<B_T, 256>>>    one block of 256 threads per token position

  Phase 1: Sum of squares (parallel reduction)
    256 threads on 768 elements (3 each)
    thread 0: x[0]² + x[256]² + x[512]² → sdata[0]
    thread 1: x[1]² + x[257]² + x[513]² → sdata[1]
    ...
    Tree reduction: 256→128→64→32→16→8→4→2→1
    Result: sdata[0] = Σx²

  Phase 2: Compute 1/RMS
    ssq = sdata[0] / C
    rrms = rsqrtf(ssq + ε)           hardware fast inverse sqrt
    if (tid == 0) save rrms[row]     cache for backward

  Phase 3: Normalize + scale
    for each column j:
      out[j] = inp[j] × rrms × γ[j]
```

---

## Backward Pass

```
  Why it's complicated:
    Changing x[0] affects RMS, which affects ALL 768 channels.
    The gradient must account for this global coupling.

  Step 1: s = Σ_c (dy_c × γ_c × x_c)     (parallel reduction)

  Step 2: Per channel:
    dx_c = rrms × (dy_c × γ_c - x_c × rrms² × s / C)
           ╰──────────────────╯  ╰───────────────────╯
            direct contribution   coupling correction
                                  (x_c affects RMS of all channels)

  Step 3: Weight gradient (atomicAdd across all 2048 rows):
    dγ_c += dy_c × x_c × rrms
```

---

## Parameter Count

```
Per RMSNorm: γ vector = 768 parameters
Total: 25 norms × 768 = 19,200 params  (0.015% of model)

Tiny, but essential — without it, training diverges in ~100 steps.
```

---

## Compute Cost (FLOPs)

```
  Our config: B=4, T=512, C=768, N=B×T=2048
  25 RMSNorm operations total (2 per layer + 1 final)

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Per RMSNorm forward (one call, N=2048 rows):                      │
  │                                                                    │
  │    Sum of squares:    N × C multiply-adds  = 2048 × 768 × 2       │
  │                                            = 3.15 MFLOPs          │
  │    Mean + rsqrt:      N × 3 ops            = 6.1 KFLOPs           │
  │    Normalize × weight: N × C × 2 ops       = 3.15 MFLOPs          │
  │    ──────────────────────────────────────────────────               │
  │    Per call forward:                        ~6.3 MFLOPs            │
  │                                                                    │
  │  Per RMSNorm backward:                                             │
  │    Sum (dy × w × x):   N × C × 3 ops      = 4.7 MFLOPs           │
  │    dinp computation:   N × C × 6 ops       = 9.4 MFLOPs           │
  │    dweight (atomicAdd): N × C × 3 ops      = 4.7 MFLOPs           │
  │    ──────────────────────────────────────────────────               │
  │    Per call backward:                       ~18.8 MFLOPs           │
  │                                                                    │
  │  ALL 25 norms (forward + backward):                               │
  │    25 × (6.3 + 18.8) = 628 MFLOPs                                │
  │                                                                    │
  │  Compare to total model: ~1,300 GFLOPs                           │
  │  RMSNorm is 0.05% of total compute — negligible.                 │
  │                                                                    │
  │  Like softmax, RMSNorm is MEMORY-BOUND:                           │
  │    Parallel reduction requires __syncthreads() barriers           │
  │    Multiple passes over shared memory                              │
  │    Limited by latency, not arithmetic throughput                  │
  └──────────────────────────────────────────────────────────────────────┘
```
