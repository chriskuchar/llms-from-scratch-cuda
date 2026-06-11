# Residual Connections — The Highway for Gradients

## Core Formula

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Forward:                                                           │
│    y = x + F(x)                                                    │
│                                                                     │
│  Where:                                                             │
│    x = input (residual stream)                                     │
│    F(x) = sub-layer output (attention or SwiGLU)                   │
│    y = output (updated residual stream)                            │
│                                                                     │
│  Backward:                                                          │
│    ∂y/∂x = 1 + ∂F/∂x                                              │
│                                                                     │
│    The "1" means gradient ALWAYS passes through at full strength.  │
│    Even if ∂F/∂x ≈ 0, the gradient never vanishes.                │
│                                                                     │
│  CUDA kernel (per element):                                        │
│    out[i] = a[i] + b[i]           forward                          │
│    da[i] += dout[i]               backward (copies to both)        │
│    db[i] += dout[i]                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## What It Does

A residual (skip) connection simply **adds** a block's output to its input:

```
x = x + block(x)
```

That's it. Element-wise addition. The simplest operation in the entire model.
But without it, deep transformers can't train at all.

---

## Where They Appear — Exactly

Every transformer layer has **two** residual connections.
Here's exactly where they sit, with concrete shapes:

```
  One full layer in our model (repeated 12 times):

  INPUT x [2048, 768]
    │
    ├────────────────────────────────────────────┐ (save x for later)
    │                                            │
    ▼                                            │
  RMSNorm(x) [2048, 768]                        │
    │                                            │
    ▼                                            │
  Attention [2048, 768]                          │
    │  (Q,K,V projections → RoPE → scores       │
    │   → softmax → weighted V → output proj)    │
    │                                            │
    ▼                                            │
  x = x + attention_output ◄────────────────────┘  RESIDUAL #1
    │
    │  x is now the ORIGINAL input plus a small
    │  nudge from attention. The original signal
    │  is preserved — attention just added to it.
    │
    ├────────────────────────────────────────────┐ (save x again)
    │                                            │
    ▼                                            │
  RMSNorm(x) [2048, 768]                        │
    │                                            │
    ▼                                            │
  SwiGLU [2048, 768]                             │
    │  (gate proj → SiLU → up proj              │
    │   → element-wise multiply → down proj)     │
    │                                            │
    ▼                                            │
  x = x + swiglu_output ◄──────────────────────┘  RESIDUAL #2
    │
    │  x now has nudges from BOTH attention AND SwiGLU,
    │  plus the original input is still in there.
    │
    ▼
  OUTPUT x [2048, 768] → feeds into next layer
```

### The Full 12-Layer Picture

```
  Token embedding [2048, 768]
    │
    ▼
  Layer 0:   x = x + Attention(RMSNorm(x))    ← residual #1
             x = x + SwiGLU(RMSNorm(x))       ← residual #2
    │
    ▼
  Layer 1:   x = x + Attention(RMSNorm(x))    ← residual #3
             x = x + SwiGLU(RMSNorm(x))       ← residual #4
    │
    ▼
  ...
    │
    ▼
  Layer 11:  x = x + Attention(RMSNorm(x))    ← residual #23
             x = x + SwiGLU(RMSNorm(x))       ← residual #24
    │
    ▼
  Final RMSNorm → Output projection → Softmax → Loss

  24 total residual additions. Each one is just:
    output[i] = input[i] + sublayer_output[i]
  for all 2048 × 768 = 1,572,864 elements.

  The original token embedding from layer 0 is STILL present
  in the final output — buried under 24 layers of small nudges,
  but never erased. That's the power of residual connections.
```

Across 12 layers, there are **24 residual additions** total.

---

## Why They're Essential

```
WITHOUT RESIDUALS (vanishing gradients):

  loss ← layer 12 ← layer 11 ← ... ← layer 2 ← layer 1

  Each layer multiplies gradient by its Jacobian.
  12 multiplications → gradient shrinks exponentially.
  Layer 1 gets gradient ~0.0001 → barely learns.

WITH RESIDUALS (gradient highway):

  ∂ℒ/∂x = ∂ℒ/∂y × (1 + ∂F/∂x)
                      ↑
                   always ≥ 1

  loss
    │
    ├──→ direct to layer 12    gradient ≈ 1.0
    ├──→ direct to layer 11    gradient ≈ 1.0
    ...
    ├──→ direct to layer 2     gradient ≈ 1.0
    └──→ direct to layer 1     gradient ≈ 1.0

  Every layer gets a strong gradient signal.
```

---

## The Residual Stream

```
Token embedding ──→ [768-dim vector]
                         │
                    ┌────┴────┐
                    │ STREAM  │  ← this vector accumulates information
                    └────┬────┘
                         │
Layer 1:  attn adds   ──┼──→  stream = stream + attn_output
          mlp adds    ──┼──→  stream = stream + mlp_output
                         │
Layer 2:  attn adds   ──┼──→  stream = stream + attn_output
          mlp adds    ──┼──→  stream = stream + mlp_output
                         │
          ...           │
                         │
Layer 12: attn adds   ──┼──→  stream = stream + attn_output
          mlp adds    ──┼──→  stream = stream + mlp_output
                         │
                    ┌────┴────┐
                    │ STREAM  │  ← now contains info from all 24 sub-layers
                    └────┬────┘
                         │
                    Final RMSNorm → Logit projection → Output
```

---

## Backward: Why += Not =

```
  da[idx] += dout[idx];     ← note: +=, not =

  Multiple operations contribute gradients to the same tensor.
  The residual backward is called AFTER attention backward has
  already written some gradients into da. We accumulate, not overwrite.

        dout
         │
    ┌────┴────┐
    │         │
    ▼         ▼
   da        db
  (+= dout) (+= dout)

  Gradient COPIES to both branches. Both get 100%.
```

---

## Compute Cost (FLOPs)

```
  Our config: B=4, T=512, C=768, N=B×T=2048
  24 residual additions total (2 per layer: post-attn + post-MLP)

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Forward (one residual add):                                       │
  │    out[i] = x[i] + sublayer[i]    for i in 0..N×C                │
  │    Elements: 2048 × 768 = 1,572,864                               │
  │    FLOPs: 1,572,864 (one add per element)                        │
  │    = 1.57 MFLOPs per call                                         │
  │                                                                    │
  │  Backward (one residual):                                          │
  │    dx = dout       (copy, no arithmetic)                          │
  │    dsublayer = dout (copy, no arithmetic)                          │
  │    FLOPs: 0 (just pointer/copy operations)                        │
  │                                                                    │
  │  ALL 24 residual connections:                                      │
  │    Forward:  24 × 1.57 MFLOPs = 37.7 MFLOPs                      │
  │    Backward: 0 FLOPs                                               │
  │    Total: 37.7 MFLOPs                                              │
  │                                                                    │
  │  Compare to total model: ~1,300 GFLOPs                           │
  │  Residual is 0.003% of compute — the cheapest operation.          │
  │                                                                    │
  │  But it is the MOST IMPORTANT for training:                       │
  │    Without residuals, gradients vanish exponentially with depth.  │
  │    Cost is nothing. Benefit is everything.                        │
  └──────────────────────────────────────────────────────────────────────┘
```
