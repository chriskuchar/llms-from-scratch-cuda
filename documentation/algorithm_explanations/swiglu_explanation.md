# SwiGLU — The Feed-Forward Block

## What It Does

Every transformer layer has two main blocks: attention (which mixes information
across token positions) and the **feed-forward / MLP block** (which processes
each token independently). SwiGLU is the MLP block used in LLaMA-style models.

It takes a token's hidden state [768] and runs it through a "think deeply about
this token" transformation, then projects it back to [768].

---

## Core Formula

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  SwiGLU(x) = (SiLU(x · W_gate) ⊙ (x · W_up)) · W_down           │
│                                                                     │
│  Expanded:                                                          │
│    gate   = x · W_gate        [768] → [2048]   "should it pass?"  │
│    up     = x · W_up          [768] → [2048]   "what to pass"     │
│    hidden = SiLU(gate) ⊙ up   [2048]           element-wise gate  │
│    output = hidden · W_down   [2048] → [768]   project back       │
│                                                                     │
│  Where:                                                             │
│    ⊙ = element-wise (Hadamard) product                             │
│    SiLU(x) = x × σ(x) = x × 1/(1 + e^(-x))                      │
│    W_gate ∈ ℝ^{768×2048}                                          │
│    W_up   ∈ ℝ^{768×2048}                                          │
│    W_down ∈ ℝ^{2048×768}                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Breaking Down Each Part

```
x · W_gate            What it does: produces a "confidence score" per feature
                      Large positive → gate OPEN → pass information through
                      Large negative → gate CLOSED → block information
                      Near zero → gate half-open → partial pass

SiLU(gate)            What it does: smooth activation on the gate values
                      SiLU(x) = x × sigmoid(x)
                      Positive → ≈ x (passes through)
                      Negative → ≈ 0 (blocked)
                      Smooth everywhere (no dead neurons like ReLU)

x · W_up              What it does: creates the actual information payload
                      These are the feature values that MIGHT pass through
                      Independent from the gate — different learned projection

SiLU(gate) ⊙ up       What it does: the gate controls what information survives
                      gate[i] decides: "should feature i pass?"
                      up[i] provides:  "what is feature i's value?"
                      Product = only gated-on features have nonzero output

hidden · W_down       What it does: compresses 2048 features back to 768
                      Recombines the surviving features into the output
```

---

## Standard MLP (GPT-2 Style) — The Baseline

Before SwiGLU, let's understand the standard MLP it replaced.

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Standard MLP:                                                      │
│    hidden = ReLU(x · W₁)          [768] → [3072] → ReLU           │
│    output = hidden · W₂           [3072] → [768]                   │
│                                                                     │
│  Where:                                                             │
│    ReLU(x) = max(0, x)            hard switch: on or off           │
│    W₁ ∈ ℝ^{768×3072}                                              │
│    W₂ ∈ ℝ^{3072×768}                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```
Architecture:

  input [768]
      │
      W₁ [768 → 3072]        ← "up project" to wider space
      │
    ReLU                      ← kill negatives, keep positives
      │
      W₂ [3072 → 768]        ← "down project" back to original size
      │
  output [768]
```

**Problem with Standard MLP:**
ReLU is a hard switch — either fully on or fully off. No "50% on."
W₁ does double duty: detect features AND decide activation strength.
SwiGLU splits these two jobs apart.

---

## SwiGLU Architecture: Gate + Up + Down

```
                    input x [768]
                    ════════╤════════
                            │
               ┌────────────┼────────────┐
               │            │            │
               ▼            │            ▼
          ┌─────────┐       │       ┌─────────┐
          │ W_gate  │       │       │  W_up   │
          │768→2048 │       │       │768→2048 │
          └────┬────┘       │       └────┬────┘
               │            │            │
               ▼            │            ▼
        gate [2048]         │       up [2048]
      "should it pass?"    │     "what to pass"
               │            │            │
               ▼            │            │
          ┌─────────┐       │            │
          │  SiLU   │       │            │
          │ smooth  │       │            │
          │  gate   │       │            │
          └────┬────┘       │            │
               │            │            │
               └──────┬─────┘────────────┘
                      │
                      ▼
                   ⊙ (element-wise multiply)
                      │
                      ▼
               hidden [2048]
            "gated information"
                      │
                      ▼
                ┌──────────┐
                │  W_down  │
                │2048→768  │
                └─────┬────┘
                      │
                      ▼
               output [768]
                      │
                      ▼
              x = x + output     (residual connection)
```

---

## What Each Weight Matrix Does

### W_gate — "Should this feature pass through?"

```
  gate = x · W_gate        [768] @ [768 × 2048] = [2048]

  Each column of W_gate asks a yes/no question about the input:

  input ──→ W_gate ──→  [ 5.0,  -3.0,   0.1,   8.0,  ... ]
  [768]     [768×2048]     │      │       │       │
                    ┌──────┴──────┴───────┴───────┴──────┐
                    │ "strongly  "no,    "hmm,  "YES,    │
                    │  yes"     block"  unsure" pass it!"│
                    └────────────────────────────────────┘

  Large positive → gate OPEN (pass info)
  Large negative → gate CLOSED (block info)
  Near zero      → gate half-open
```

### W_up — "What information should flow?"

```
  up = x · W_up            [768] @ [768 × 2048] = [2048]

  input ──→ W_up  ──→  [ 1.2,   0.9,   3.5,   0.4,  ... ]
  [768]     [768×2048]

  These are the "raw information" values.
  They could be big or small — doesn't matter
  until the gate decides which ones survive.
```

### Gate Values vs Up Values — The Key Difference

```
                     GATE values               UP values
                     (from W_gate)             (from W_up)
                     ─────────────             ───────────
  Purpose:           Control / switch           Content / information
  Question:          "Should this pass?"        "What to pass?"
  Analogy:           Volume knob per channel    Music signal per channel

   gate[i]    SiLU    up[i]                 hidden[i]
   ───────    ────    ─────                 ─────────

    5.0    →  4.97  ×  1.2    =  5.96       ← OPEN gate, info passes
   -3.0    →  0.14  ×  0.9    =  0.13       ← CLOSED gate, blocked
    0.1    →  0.05  ×  3.5    =  0.18       ← HALF gate, mostly lost
    8.0    →  7.99  ×  0.4    =  3.20       ← WIDE OPEN, amplified

  Notice: up[2]=3.5 is the BIGGEST up value, but its
  gate (0.1) is nearly closed, so almost nothing passes.
  The gate, not the content, controls the flow.
```

### W_down — "Compress back to model dimension"

```
  output = hidden · W_down     [2048] @ [2048 × 768] = [768]

  hidden [2048]                                 output [768]
  ┌──────────────────────────────┐              ┌──────────┐
  │ 5.96  0.13  0.18  3.20  ... │   W_down     │  0.12    │
  │                              │ ────────→    │ -0.34    │
  │  (only features that        │ [2048×768]   │  0.67    │
  │   survived the gate)        │              │  ...     │
  └──────────────────────────────┘              └──────────┘
```

---

## The SiLU Activation

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  SiLU(x) = x × σ(x) = x × 1/(1 + e^(-x))         │
│                                                     │
│         output                                      │
│    2.0  │              ╱                            │
│         │            ╱                              │
│    1.0  │          ╱                                │
│         │        ╱                                  │
│    0.0  │──────╱─────────── input                   │
│         │    ╱                                      │
│   -0.5  │  ╱                                        │
│         │╱                                          │
│   -1.0  │                                           │
│        -4  -2   0   2   4                           │
│                                                     │
│  x → +∞:  SiLU(x) ≈ x    (identity)               │
│  x → -∞:  SiLU(x) ≈ 0    (killed)                 │
│  x = 0:   SiLU(0) = 0    (zero)                   │
│                                                     │
│  Like ReLU but smooth — no dead neurons.           │
│                                                     │
│  SiLU vs ReLU:                                     │
│  input    ReLU    SiLU                             │
│  ─────    ────    ────                             │
│  -3.0      0.0     0.14    ← SiLU passes a bit    │
│  -1.0      0.0     0.27                            │
│   0.0      0.0     0.00    ← both zero             │
│   1.0      1.0     0.73                            │
│   3.0      3.0     2.86    ← nearly identical      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Why SwiGLU Beats Standard MLP

```
  Standard MLP:                         SwiGLU:
  ─────────────                         ──────
  hidden = ReLU(x·W₁)                  hidden = SiLU(x·W_gate) ⊙ (x·W_up)
  output = hidden·W₂                   output = hidden·W_down

  One projection does everything:       Two projections split the job:
    - detect features AND                 - W_gate: "should it pass?"
    - carry information                   - W_up:   "what to pass"

  ReLU: hard on/off switch              SiLU: smooth dimmer switch

  2 weight matrices:                    3 weight matrices:
    W₁: [768 × 3072]                     W_gate: [768 × 2048]
    W₂: [3072 × 768]                     W_up:   [768 × 2048]
    Total: 4,718,592                      W_down: [2048 × 768]
                                          Total:  4,718,592  (same!)

  Key insight: SwiGLU uses smaller hidden dim (2048 vs 3072)
  but gets better loss because gating is more expressive.
```

---

## Backward Pass Formulas

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Stage 1: Down projection backward                                │
│    dhidden = dout · W_down^T        gradient of hidden [2048]      │
│    dW_down = hidden^T · dout        gradient of down weights       │
│                                                                     │
│  Stage 2: SiLU gating backward (element-wise)                     │
│                                                                     │
│    SiLU'(x) = σ(x) × (1 + x × (1 - σ(x)))                       │
│                                                                     │
│    dup   = dhidden ⊙ SiLU(gate)          gradient to up branch     │
│    dgate = dhidden ⊙ up ⊙ SiLU'(gate)   gradient to gate branch   │
│                                                                     │
│  Stage 3: Input projection backward                               │
│    dinp  = dgate · W_gate^T              from gate path            │
│    dinp += dup · W_up^T                  from up path (accumulated)│
│    dW_gate = inp^T · dgate               weight gradient for gate  │
│    dW_up   = inp^T · dup                 weight gradient for up    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Parameter Count

```
Per layer:
  W_gate:  768 × 2048 = 1,572,864
  W_up:    768 × 2048 = 1,572,864
  W_down:  2048 × 768 = 1,572,864
                         ─────────
  Total:                 4,718,592 per layer
                       × 12 layers
                       ──────────
                       56,623,104  (45% of the model's 124.7M params!)
```

The MLP is the biggest part of the model — more parameters than attention.

---

## Analogy: Recording Studio Mixing Board

```
  Standard MLP = one microphone, one volume knob
    mic (W₁) picks up sound AND sets volume
    ReLU clips anything below zero
    speaker (W₂) outputs the result

  SwiGLU = two microphones, one is the volume control for the other
    mic 1 (W_gate): decides which channels to boost/cut (the mixer)
    mic 2 (W_up):   captures the actual content (the signal)
    mixer × signal = only the good parts survive
    speaker (W_down): outputs the mixed result
```

---

## Compute Cost (FLOPs)

```
  Our config: B=4, T=512, C=768, ffn=2048, N=B×T=2048

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Operation                Shape                   FLOPs            │
  │  ─────────────────────    ─────────────────────    ──────────       │
  │  Gate projection          [2048,768]@[768,2048]    2×2048×768×2048 │
  │                                                    = 6.44 GFLOPs   │
  │                                                                    │
  │  Up projection            [2048,768]@[768,2048]    2×2048×768×2048 │
  │                                                    = 6.44 GFLOPs   │
  │                                                                    │
  │  SiLU activation          2048 × 2048 elements     ~5 ops/elem    │
  │  (sigmoid + multiply)                              = 21 MFLOPs     │
  │                                                                    │
  │  Element-wise gate × up   2048 × 2048 elements     1 op/elem      │
  │                                                    = 4.2 MFLOPs    │
  │                                                                    │
  │  Down projection          [2048,2048]@[2048,768]   2×2048×2048×768 │
  │                                                    = 6.44 GFLOPs   │
  │  ─────────────────────────────────────────────────────────────────  │
  │  TOTAL per layer (forward):                        ~19.3 GFLOPs   │
  │  TOTAL per layer (forward + backward):             ~58.0 GFLOPs   │
  │  TOTAL all 12 layers:                              ~696 GFLOPs    │
  │                                                                    │
  │  SwiGLU is 2.4× more FLOPs than attention per layer!             │
  │  This is because it has 3 large matmuls vs attention's smaller    │
  │  score/value matmuls. The projections (gate+up+down) dominate.    │
  │                                                                    │
  │  At 101 TFLOPS: ~6.9 ms per step (theoretical)                   │
  │                                                                    │
  │  SiLU and element-wise multiply are negligible (~0.1% of FLOPs). │
  │  SwiGLU is essentially "three big matmuls per layer."            │
  └──────────────────────────────────────────────────────────────────────┘
```
