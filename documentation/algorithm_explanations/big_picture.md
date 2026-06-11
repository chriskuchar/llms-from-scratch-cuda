# LLaMA Architecture — Forward & Backward Pass Visual Guide

## The Big Picture

The model is a pipeline. Data flows down during forward, gradients flow up during backward.

```
FORWARD (top → bottom)              BACKWARD (bottom → top)
═══════════════════                  ═══════════════════════

  token IDs [B, T]                   dwte (embedding gradients)
       │                                  ▲
       ▼                                  │
  ┌──────────┐                       ┌──────────┐
  │ EMBEDDING │                      │ EMBEDDING │
  │           │  embedding.cu        │ BACKWARD  │  embedding.cu
  └─────┬─────┘                      └─────┬─────┘
        │ x [B,T,C]                        │ dx [B,T,C]
        ▼                                  ▲
  ╔═══════════════╗                  ╔═══════════════╗
  ║ TRANSFORMER   ║                  ║ TRANSFORMER   ║
  ║ BLOCK × 12    ║  (see below)    ║ BLOCK × 12    ║  (in reverse)
  ╚══════╤════════╝                  ╚══════╤════════╝
        │ x [B,T,C]                        │ dx [B,T,C]
        ▼                                  ▲
  ┌──────────┐                       ┌──────────┐
  │ RMSNORM  │  rmsnorm.cu          │ RMSNORM  │  rmsnorm.cu
  │ (final)  │                       │ BACKWARD │
  └─────┬────┘                       └─────┬────┘
        │                                  │
        ▼                                  ▲
  ┌──────────┐                       ┌──────────┐
  │ MATMUL   │  matmul.cu           │ MATMUL   │  matmul.cu
  │ → logits │  x @ W_out           │ BACKWARD │  dlogits → dx, dw_out
  └─────┬────┘                       └─────┬────┘
        │ logits [B,T,V]                   │ dlogits [B,T,V]
        ▼                                  ▲
  ┌──────────┐                       ┌──────────┐
  │  CROSS   │  cross_entropy.cu    │ CE+SOFT  │  cross_entropy.cu
  │ ENTROPY  │                       │ BACKWARD │  dlogits = probs - one_hot
  └─────┬────┘                       └──────────┘
        │                                  ▲
        ▼                                  │
      loss ─────── this single number ─────┘
      (scalar)     starts the entire
                   backward chain
```

## Inside One Transformer Block

Each block has two branches: **attention** and **MLP**. Both use residual connections (skip connections that add the input back to the output).

### Forward Pass (one block)

```
x ─────────────────────────────────────────────────────┐
│                                                       │
│  ┌──────────────────────────────────────────────┐    │
│  │                                                │    │
│  │  1. RMSNorm(x)                    rmsnorm.cu  │    │
│  │     │                                          │    │
│  │     │  Normalize each token's 768 floats       │    │
│  │     │  by their root-mean-square               │    │
│  │     ▼                                          │    │
│  │  2. Q = norm @ W_q  [B*T, 768]   matmul.cu   │    │
│  │     K = norm @ W_k  [B*T, 256]   matmul.cu   │    │
│  │     V = norm @ W_v  [B*T, 256]   matmul.cu   │    │
│  │     │                                          │    │
│  │     │  Project into query, key, value spaces   │    │
│  │     │  K,V smaller than Q because of GQA       │    │
│  │     ▼                                          │    │
│  │  3. RoPE(Q, K)                    rope.cu     │    │
│  │     │                                          │    │
│  │     │  Rotate Q,K pairs by position-dependent  │    │
│  │     │  angles. Encodes "where" each token is.  │    │
│  │     ▼                                          │    │
│  │  4. Attention                     attention.cu │    │
│  │     │                                          │    │
│  │     │  ┌─────────────────────────────────┐    │    │
│  │     │  │ Reshape Q → [B, 12, T, 64]     │    │    │
│  │     │  │ Reshape K → [B, 4, T, 64]      │    │    │
│  │     │  │ Reshape V → [B, 4, T, 64]      │    │    │
│  │     │  │                                  │    │    │
│  │     │  │ For each query head group:       │    │    │
│  │     │  │   scores = Q @ K^T / sqrt(64)   │    │    │
│  │     │  │                                  │    │    │
│  │     │  │   ┌─────────────────────┐       │    │    │
│  │     │  │   │ scores before mask: │       │    │    │
│  │     │  │   │  0.5  0.3  0.8  0.1│       │    │    │
│  │     │  │   │  0.2  0.7  0.4  0.6│       │    │    │
│  │     │  │   │  0.9  0.1  0.5  0.3│       │    │    │
│  │     │  │   │  0.4  0.6  0.2  0.8│       │    │    │
│  │     │  │   └─────────────────────┘       │    │    │
│  │     │  │              │                   │    │    │
│  │     │  │              ▼                   │    │    │
│  │     │  │   Causal mask: future = -inf     │    │    │
│  │     │  │   ┌─────────────────────┐       │    │    │
│  │     │  │   │  0.5 -inf -inf -inf│       │    │    │
│  │     │  │   │  0.2  0.7 -inf -inf│       │    │    │
│  │     │  │   │  0.9  0.1  0.5 -inf│       │    │    │
│  │     │  │   │  0.4  0.6  0.2  0.8│       │    │    │
│  │     │  │   └─────────────────────┘       │    │    │
│  │     │  │              │                   │    │    │
│  │     │  │              ▼                   │    │    │
│  │     │  │   probs = softmax(scores)        │    │    │
│  │     │  │   (each row sums to 1.0)         │    │    │
│  │     │  │              │                   │    │    │
│  │     │  │              ▼                   │    │    │
│  │     │  │   context = probs @ V            │    │    │
│  │     │  │   (weighted average of V rows)   │    │    │
│  │     │  └─────────────────────────────────┘    │    │
│  │     │                                          │    │
│  │     ▼                                          │    │
│  │  5. out = context @ W_o            matmul.cu  │    │
│  │     │                                          │    │
│  │     │  Project attention output back to C=768  │    │
│  │                                                │    │
│  └──────────────────────────────────────────────┘    │
│        │ attn_out [B,T,C]                             │
│        ▼                                              │
├── x = x + attn_out ◄─────────────────────────────────┘
│                                          residual.cu
│
│   (x now has attention information mixed in)
│
x ─────────────────────────────────────────────────────┐
│                                                       │
│  ┌──────────────────────────────────────────────┐    │
│  │                                                │    │
│  │  6. RMSNorm(x)                    rmsnorm.cu  │    │
│  │     │                                          │    │
│  │     ▼                                          │    │
│  │  7. SwiGLU MLP                    swiglu.cu   │    │
│  │     │                                          │    │
│  │     │  ┌─────────────────────────────────┐    │    │
│  │     │  │                                  │    │    │
│  │     │  │ gate = norm @ W_gate  [B*T,2048]│    │    │
│  │     │  │ up   = norm @ W_up    [B*T,2048]│    │    │
│  │     │  │                                  │    │    │
│  │     │  │        gate          up          │    │    │
│  │     │  │         │             │          │    │    │
│  │     │  │         ▼             │          │    │    │
│  │     │  │      SiLU(gate)       │          │    │    │
│  │     │  │         │             │          │    │    │
│  │     │  │         └─── * ───────┘          │    │    │
│  │     │  │              │                   │    │    │
│  │     │  │              ▼                   │    │    │
│  │     │  │           hidden  [B*T, 2048]    │    │    │
│  │     │  │              │                   │    │    │
│  │     │  │              ▼                   │    │    │
│  │     │  │   out = hidden @ W_down [B*T,C]  │    │    │
│  │     │  │                                  │    │    │
│  │     │  └─────────────────────────────────┘    │    │
│  │     │                                          │    │
│  │     │  SiLU(x) = x * sigmoid(x)               │    │
│  │     │  The gate controls how much info flows   │    │
│  │                                                │    │
│  └──────────────────────────────────────────────┘    │
│        │ mlp_out [B,T,C]                              │
│        ▼                                              │
└── x = x + mlp_out ◄──────────────────────────────────┘
                                           residual.cu

    x [B, T, C] → output of this block, input to next block
```

### Backward Pass (one block, reversed)

Gradients flow upward. Each step receives `dout` from below, produces `dinp` going up.

```
    dx [B, T, C] ← coming from the block above (or final norm)
        │
        ▼
┌── residual_backward ──────────────────────────────── residual.cu
│       │
│       │  dx splits into two: dx stays on the main path,
│       │  dmlp_out goes to the MLP branch.
│       │  (addition in forward = copy gradient in backward)
│       │
│       ▼
│   swiglu_backward                                     swiglu.cu
│       │
│       │  Reverse the MLP:
│       │  dhidden = dmlp_out @ W_down^T               (matmul backward)
│       │  dgate = dhidden * up * SiLU'(gate)          (SiLU derivative)
│       │  dup = dhidden * SiLU(gate)
│       │  dln2 += dgate @ W_gate^T                    (accumulate both)
│       │  dln2 += dup @ W_up^T
│       │  dW_gate, dW_up, dW_down updated             (weight gradients)
│       │
│       ▼
│   rmsnorm_backward                                    rmsnorm.cu
│       │
│       │  dx += rmsnorm gradient
│       │  drms2_w updated
│       │
│       ▼
├── residual_backward ──────────────────────────────── residual.cu
│       │
│       │  dx splits again: dx stays, dattn_out goes to attention
│       │
│       ▼
│   matmul_backward (W_o)                               matmul.cu
│       │  dcontext = dattn_out @ W_o^T
│       │  dW_o = context^T @ dattn_out
│       │
│       ▼
│   attention_backward                                  attention.cu
│       │
│       │  Reverse attention:
│       │  dprobs = dcontext_heads @ V^T
│       │  dV = probs^T @ dcontext_heads
│       │  dscores = softmax_backward(dprobs, probs)
│       │  dscores zeroed where mask was -inf
│       │  dscores /= sqrt(head_dim)
│       │  dQ = dscores @ K
│       │  dK = dscores^T @ Q
│       │
│       ▼
│   rope_backward                                       rope.cu
│       │
│       │  Inverse rotation: negate the angle
│       │  dQ_unrotated, dK_unrotated
│       │
│       ▼
│   matmul_backward (W_q, W_k, W_v)  × 3               matmul.cu
│       │  dln1 += dQ @ W_q^T     (accumulate all three)
│       │  dln1 += dK @ W_k^T
│       │  dln1 += dV @ W_v^T
│       │  dW_q, dW_k, dW_v updated
│       │
│       ▼
│   rmsnorm_backward                                    rmsnorm.cu
│       │  dx += rmsnorm gradient
│       │  drms1_w updated
│       │
│       ▼
└──► dx [B, T, C] ← pass to the block below (or embedding backward)
```

## Grouped-Query Attention (GQA) Detail

Standard MHA: every head has its own Q, K, V.
GQA: Q has 12 heads, but K and V only have 4. Groups of 3 query heads share 1 KV head.

```
Query heads:     Q0  Q1  Q2 │ Q3  Q4  Q5 │ Q6  Q7  Q8 │ Q9  Q10 Q11
                  │   │   │  │  │   │   │  │  │   │   │  │  │   │   │
                  └───┴───┘  │  └───┴───┘  │  └───┴───┘  │  └───┴───┘
                      │      │      │      │      │      │      │
KV heads:            K0,V0  │    K1,V1    │    K2,V2    │    K3,V3
                             │             │             │
                         group 0       group 1       group 2       group 3
```

Each KV head is [T, 64]. Each Q head is [T, 64].
Within a group, the 3 query heads all attend against the same K and V.

Memory savings: KV cache is 4 heads instead of 12 = 1/3 the size at inference.

## RoPE Rotation Visualization

Each pair of adjacent dimensions gets rotated by a position-dependent angle:

```
Token at position t=0:     no rotation (angle = 0 for all dims)
Token at position t=1:     small rotation
Token at position t=100:   large rotation

Dimension pair 0,1:        rotates FAST  (high frequency)
Dimension pair 2,3:        rotates slightly slower
Dimension pair 4,5:        even slower
...
Dimension pair 62,63:      rotates SLOW  (low frequency)

Visually (angle vs position for different dim pairs):

angle
  ▲
  │  dim 0,1 ╱╱╱╱╱╱╱╱╱╱    (steep = high frequency)
  │         ╱
  │  dim 2,3 ╱╱╱╱╱╱╱
  │           ╱
  │  dim 4,5   ╱╱╱╱╱
  │              ╱
  │  dim 62,63     ╱       (shallow = low frequency)
  │
  └──────────────────────► position t
```

The dot product Q·K after rotation depends on the DISTANCE between two positions,
not their absolute position. That's why RoPE encodes relative position naturally.

## SwiGLU Gating Visualization

```
input ──┬──────────────┐
        │              │
        ▼              ▼
    x @ W_gate     x @ W_up
        │              │
        ▼              │
     SiLU(·)           │
        │              │
        │   SiLU output acts as a "gate":
        │   values near 0 → block the information
        │   values near 1 → let it through
        │              │
        └──── × ───────┘
              │
              ▼
           hidden
              │
              ▼
        hidden @ W_down
              │
              ▼
           output

SiLU(x) = x · σ(x) where σ = sigmoid

     SiLU(x)
      ▲
    2 │           ╱
      │         ╱
    1 │       ╱
      │     ╱
    0 ├───╱──────────►  x
      │ ╱
   -1 │╱
      │
```

## Residual Connections

Why they matter: without them, gradients would have to flow through every operation
in sequence. With 12 layers of matmuls, norms, and activations, gradients shrink
exponentially (vanishing gradient problem).

The residual connection creates a "gradient highway" — a direct path from the loss
back to early layers:

```
WITHOUT residual:                    WITH residual:
gradient must pass through           gradient has a shortcut
every operation                      straight through

dx ← op12 ← op11 ← ... ← op1      dx ← op12 + skip ← op11 + skip ← ... ← op1 + skip
     (gradients shrink at                (shortcut preserves gradient magnitude)
      each step)

In code:
  x = x + sublayer(x)

Forward: output = input + processed_input
Backward: dinput = doutput + dprocessed    (gradient flows to BOTH paths)
```

## Training Loop Overview

```
┌─────────────────────────────────────────────────────────┐
│ for step = 0 to 20000:                                   │
│                                                           │
│   1. LOAD BATCH          dataloader.cpp                  │
│      read B*T+1 tokens from disk                         │
│      input_ids = tokens[0..B*T-1]                        │
│      targets   = tokens[1..B*T]    (shifted by 1)        │
│      cudaMemcpy to GPU                                   │
│                                                           │
│   2. FORWARD PASS        model.cpp → all .cu files       │
│      embedding → 12×(norm→attn→residual→norm→mlp→res)    │
│      → final norm → logits → loss                        │
│                                                           │
│   3. ZERO GRADIENTS      model.cpp                       │
│      cudaMemset entire grads buffer to 0                 │
│                                                           │
│   4. BACKWARD PASS       model.cpp → all .cu files       │
│      loss → dlogits → reverse everything                  │
│      every weight now has a gradient                      │
│                                                           │
│   5. CLIP GRADIENTS      optimizer.cpp                   │
│      norm = sqrt(sum of all grad^2)                      │
│      if norm > 1.0: scale all grads by (1.0 / norm)     │
│                                                           │
│   6. ADAMW UPDATE        adamw.cu                        │
│      for each parameter:                                  │
│        m = 0.9*m + 0.1*grad           (momentum)         │
│        v = 0.95*v + 0.05*grad^2       (velocity)         │
│        param -= lr * (m̂/√v̂ + 0.1*param) (update)        │
│                                                           │
│   7. LOG                                                  │
│      print step, loss, learning_rate                     │
│                                                           │
│   8. SAVE CHECKPOINT     (every 2000 steps)              │
│      cudaMemcpy params to CPU, fwrite to disk            │
│                                                           │
└─────────────────────────────────────────────────────────┘

Learning rate over training:

  LR
   ▲
   │       3e-4
   │      ╱‾‾‾‾╲
   │     ╱       ╲
   │    ╱          ╲
   │   ╱             ╲
   │  ╱                ‾‾‾‾‾‾  3e-5
   │ ╱ warmup
   └──────────────────────────► steps
     0  200               20000
```
