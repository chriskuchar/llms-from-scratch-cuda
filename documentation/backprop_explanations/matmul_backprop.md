# Backprop: Linear Projection (Matmul)

Every projection in the model (Q, K, V, O, gate, up, down, output) uses this.
This is the most reused backward pass in the entire model.

Config: B=4, T=512, C=768, N=B×T=2048.

---

## Forward

```
  Y = X @ W

  X: [N, C_in]     input activations
  W: [C_in, C_out]  weight matrix
  Y: [N, C_out]     output activations
```

---

## Backward Derivation

```
  Given dY (shape [N, C_out]) from upstream:

  ─── Gradient w.r.t. input X ───

  Y_ij = Σ_k X_ik × W_kj

  dL/dX_ik = Σ_j (dL/dY_ij × dY_ij/dX_ik)
           = Σ_j (dL/dY_ij × W_kj)
           = (dY @ W^T)_ik

  Therefore:
    dX = dY @ W^T      shape: [N, C_out] @ [C_out, C_in] = [N, C_in]

  ─── Gradient w.r.t. weight W ───

  dL/dW_kj = Σ_i (dL/dY_ij × dY_ij/dW_kj)
           = Σ_i (dL/dY_ij × X_ik)
           = (X^T @ dY)_kj

  Therefore:
    dW = X^T @ dY      shape: [C_in, N] @ [N, C_out] = [C_in, C_out]


  ┌─────────────────────────────────────────────────────────────────┐
  │  Forward:   Y  = X   @ W                                       │
  │  Backward:  dX = dY  @ W^T    (to pass gradient downstream)   │
  │             dW = X^T @ dY     (to update weights)              │
  │                                                                 │
  │  The backward has 2 matmuls per forward matmul.                │
  │  This is why backward ≈ 2× forward compute.                   │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Compute Cost

```
  Forward:   2 × N × C_in × C_out   FLOPs
  Backward:  2 × N × C_out × C_in   (dX)
           + 2 × C_in × N × C_out   (dW)
           = 4 × N × C_in × C_out   FLOPs

  Backward = 2× forward.  Total = 3× forward.
```

---

## Where This Appears in Our Model

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  Projection         Shape               Forward GFLOPs         │
  │  ──────────────     ──────────────────   ──────────────         │
  │  W_q  (×12 layers)  [768, 768]           2.42 each             │
  │  W_k  (×12 layers)  [768, 256]           0.81 each             │
  │  W_v  (×12 layers)  [768, 256]           0.81 each             │
  │  W_o  (×12 layers)  [768, 768]           2.42 each             │
  │  W_gate (×12)        [768, 2048]          6.44 each             │
  │  W_up   (×12)        [768, 2048]          6.44 each             │
  │  W_down (×12)        [2048, 768]          6.44 each             │
  │  W_out               [768, 32000]         100.7                 │
  │                                                                  │
  │  Each backward = 2× forward FLOPs for that projection.         │
  │  Matmuls account for ~99% of total model FLOPs.                │
  └──────────────────────────────────────────────────────────────────┘
```
