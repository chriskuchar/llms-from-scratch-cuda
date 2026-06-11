# Backprop: Attention (Grouped Query)

The most complex backward pass — 4 stages, GQA gradient accumulation,
and a softmax Jacobian embedded inside.

Config: B=4, T=512, C=768, n_head=12, n_kv_head=4, head_dim=64.

---

## Forward

```
  Q = X @ W_q    [N, C] @ [C, C]      → [N, C]    → reshape [B,T,12,64]
  K = X @ W_k    [N, C] @ [C, kv_dim] → [N, 256]  → reshape [B,T,4,64]
  V = X @ W_v    [N, C] @ [C, kv_dim] → [N, 256]  → reshape [B,T,4,64]

  Apply RoPE to Q, K

  For each query head h (0..11):
    kv_head = h / 3    (GQA: 3 query heads share 1 KV head)

    scores = Q_h @ K_{kv_head}^T / sqrt(64)     [T, 64] @ [64, T] → [T, T]
    scores += causal_mask                         (-inf for future positions)
    probs = softmax(scores)                       [T, T]
    head_out = probs @ V_{kv_head}                [T, T] @ [T, 64] → [T, 64]

  Concat all heads → [N, C]
  output = concat @ W_o                          [N, C] @ [C, C] → [N, C]
```

---

## Backward Derivation

```
  Given dout (shape [N, C]):

  ─── Step 1: Output projection backward ───
  dconcat = dout @ W_o^T                [N,C] @ [C,C]      = [N,C]
  dW_o    = concat^T @ dout             [C,N] @ [N,C]      = [C,C]

  Reshape dconcat → [B, T, 12, 64] → dhead_out per head

  ─── Step 2: Per-head backward (for each query head h) ───

  kv_head = h / 3

  # Backward through probs @ V
  dprobs = dhead_out @ V_{kv_head}^T    [T,64] @ [64,T]    = [T,T]
  dV_{kv_head} += probs^T @ dhead_out   [T,T]  @ [T,64]    = [T,64]

  Note: dV accumulates from all 3 query heads that share this KV head!

  # Backward through softmax
  # For each row i of the attention matrix:
  dot_i = Σ_j probs_ij × dprobs_ij
  dscores_ij = probs_ij × (dprobs_ij - dot_i)

  # Backward through scaling
  dscores_unscaled = dscores / sqrt(64)

  # Backward through Q @ K^T
  dQ_h = dscores_unscaled @ K_{kv_head}        [T,T] @ [T,64]   = [T,64]
  dK_{kv_head} += dscores_unscaled^T @ Q_h      [T,T] @ [T,64]   = [T,64]

  Note: dK also accumulates from all 3 query heads sharing this KV head!

  ─── Step 3: RoPE backward ───
  (see rope_backprop.md — just flip the sign of sin)
  dQ_pre_rope, dK_pre_rope = rope_backward(dQ, dK)

  ─── Step 4: Projection backward ───
  dX_q = dQ_pre_rope @ W_q^T            [N,C] @ [C,C]      = [N,C]
  dW_q = X^T @ dQ_pre_rope              [C,N] @ [N,C]      = [C,C]

  dX_k = dK_pre_rope @ W_k^T            [N,256] @ [256,C]  = [N,C]
  dW_k = X^T @ dK_pre_rope              [C,N] @ [N,256]    = [C,256]

  dX_v = dV @ W_v^T                     [N,256] @ [256,C]  = [N,C]
  dW_v = X^T @ dV                       [C,N] @ [N,256]    = [C,256]

  dX = dX_q + dX_k + dX_v               (three paths merge at input)

  ┌─────────────────────────────────────────────────────────────────┐
  │  Key insight for GQA backward:                                 │
  │                                                                 │
  │  Forward: 3 Q heads READ from the same K,V head (broadcast)   │
  │  Backward: 3 Q heads WRITE gradients to the same dK, dV       │
  │            (accumulate via +=)                                  │
  │                                                                 │
  │  This is why our CUDA code uses atomicAddBf16 for dK, dV —    │
  │  multiple query heads update the same gradient buffer.         │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Attention Softmax Backward Detail

```
  The attention softmax backward deserves special attention.
  Each row of the [T, T] attention matrix is an independent softmax.

  For row i:
    p = softmax(s)    where s = scores[i, :]    (length T)

    Given dp (upstream gradient from probs @ V backward):

    Step 1: Compute the dot product
      dot = Σ_j p_j × dp_j

    Step 2: Compute dscores
      ds_j = p_j × (dp_j - dot)

  Why this formula?
    The Jacobian of softmax is:  ∂p_i/∂s_j = p_i(δ_ij - p_j)
    Where δ_ij = 1 if i=j, else 0.

    Multiplying the upstream gradient through this Jacobian gives:
      ds_j = Σ_i dp_i × p_i × (δ_ij - p_j)
           = dp_j × p_j - p_j × Σ_i dp_i × p_i
           = p_j × (dp_j - dot)
```

---

## GQA Gradient Flow Diagram

```
  Query heads:      Q_0   Q_1   Q_2      Q_3   Q_4   Q_5      ...
                     │     │     │         │     │     │
                     ▼     ▼     ▼         ▼     ▼     ▼
  KV head 0:       K_0,V_0 ◄───────     K_1,V_1 ◄───────      ...

  Forward:  3 Q heads each READ from the same K,V (broadcast)
  Backward: 3 Q heads each WRITE dK, dV (accumulate += )

  dK_0 = dscores_0^T @ Q_0  +  dscores_1^T @ Q_1  +  dscores_2^T @ Q_2
         ──────────────────     ──────────────────     ──────────────────
         from query head 0      from query head 1      from query head 2
```
