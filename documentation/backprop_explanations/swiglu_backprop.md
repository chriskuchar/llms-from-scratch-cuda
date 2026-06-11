# Backprop: SwiGLU (MLP)

The most compute-heavy backward pass — 6 matmuls vs 3 in the forward.
The gating mechanism creates two parallel paths that both need gradients.

Config: B=4, T=512, C=768, ffn=2048, N=B×T=2048.

---

## Forward

```
  gate = X @ W_gate       [N,C] @ [C,ffn]     = [N,ffn]    ffn=2048
  up   = X @ W_up         [N,C] @ [C,ffn]     = [N,ffn]
  act  = silu(gate) ⊙ up                       = [N,ffn]
  out  = act @ W_down      [N,ffn] @ [ffn,C]   = [N,C]

  Where silu(x) = x × sigmoid(x) = x × (1/(1+e^{-x}))
```

---

## Backward Derivation

```
  Given dout (shape [N, C]):

  ─── Step 1: Down projection backward ───
  dact   = dout @ W_down^T           [N,C] @ [C,ffn]       = [N,ffn]
  dW_down = act^T @ dout             [ffn,N] @ [N,C]       = [ffn,C]

  ─── Step 2: Element-wise multiply backward ───
  (act = silu(gate) ⊙ up)

  dsilu_gate = dact ⊙ up            element-wise
  dup        = dact ⊙ silu(gate)    element-wise

  ─── Step 3: SiLU backward ───
  Need derivative of silu(x) = x × σ(x)  where σ = sigmoid

  Using product rule:
    d/dx [x × σ(x)] = σ(x) + x × σ(x) × (1 - σ(x))
                     = σ(x) × (1 + x × (1 - σ(x)))
                     = σ(x) + x × σ(x) - x × σ(x)²

  Alternatively, factor:
    silu'(x) = σ(x) × (1 + x - x × σ(x))
             = σ(x) × (1 + x × (1 - σ(x)))

  So:
    dgate = dsilu_gate × silu'(gate)     element-wise

  ─── Step 4: Up and gate projection backward ───
  dX_gate = dgate @ W_gate^T         [N,ffn] @ [ffn,C]     = [N,C]
  dW_gate = X^T @ dgate              [C,N] @ [N,ffn]       = [C,ffn]

  dX_up   = dup @ W_up^T             [N,ffn] @ [ffn,C]     = [N,C]
  dW_up   = X^T @ dup                [C,N] @ [N,ffn]       = [C,ffn]

  dX = dX_gate + dX_up               (two paths merge at input)

  ┌─────────────────────────────────────────────────────────────────┐
  │  SwiGLU backward has 6 matmuls (2 per projection × 3):        │
  │    2 for W_down backward (dact, dW_down)                       │
  │    2 for W_gate backward (dX_gate, dW_gate)                    │
  │    2 for W_up backward   (dX_up, dW_up)                        │
  │                                                                 │
  │  Plus the SiLU derivative, which is element-wise (cheap).      │
  │  Forward was 3 matmuls → backward is 6 → ratio is 2:1.       │
  │                                                                 │
  │  Key insight: the gating means the backward flows through     │
  │  TWO separate paths (gate and up), each requiring its own     │
  │  projection backward. Standard MLP has only ONE path.         │
  └─────────────────────────────────────────────────────────────────┘
```

---

## SiLU Derivative Detail

```
  Let σ(x) = 1/(1+e^{-x})       (sigmoid)
  silu(x) = x × σ(x)

  d(silu)/dx = σ(x) + x × dσ/dx
             = σ(x) + x × σ(x) × (1 - σ(x))

  At different values:
    x = 0:    silu'(0) = 0.5 + 0 = 0.5
    x = 2:    silu'(2) = 0.88 + 2(0.88)(0.12) = 1.09
    x = -2:   silu'(-2) = 0.12 + (-2)(0.12)(0.88) = -0.09

  The SiLU derivative can be negative for x < -1, which is what
  allows the gate to actively suppress (push to zero) certain
  features. ReLU derivative is always 0 or 1 — less expressive.
```

---

## Gradient Flow Diagram

```
                   dout
                     │
              ┌──────┴──────┐
              ▼              ▼
          dact = dout @ W_down^T      dW_down = act^T @ dout
              │
       ┌──────┴──────┐
       │              │
       ▼              ▼
  dsilu_gate      dup = dact ⊙ silu(gate)
  = dact ⊙ up         │
       │               │
       ▼               │
  dgate = dsilu_gate   │
  × silu'(gate)        │
       │               │
       ▼               ▼
  dX_gate = dgate    dX_up = dup
  @ W_gate^T         @ W_up^T
       │               │
       └───────┬───────┘
               ▼
          dX = dX_gate + dX_up
```

---

## Compare: SwiGLU vs Standard MLP Backward

```
  Standard MLP:  out = ReLU(X @ W1) @ W2
  ─────────────────────────────────────────
  Backward: 4 matmuls (2 for W2, 2 for W1)
  Element-wise: ReLU' (just mask by sign, trivial)

  SwiGLU:  out = (silu(X @ W_gate) ⊙ (X @ W_up)) @ W_down
  ─────────────────────────────────────────
  Backward: 6 matmuls (2 for each of 3 weight matrices)
  Element-wise: silu' (more expensive but still cheap)

  SwiGLU uses 50% more matmuls but produces better features
  because the gate can smoothly scale (0 to 1+) rather than
  hard-threshold (0 or 1 for ReLU).
```
