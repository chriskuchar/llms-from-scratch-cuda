# Backprop: Softmax

The softmax Jacobian is a dense matrix (every output depends on every input),
but we can compute the backward in O(n) instead of O(n²) with a clever trick.

---

## Forward

```
  Given scores s (shape [N]):

  p_i = exp(s_i - max(s)) / Σ_j exp(s_j - max(s))

  The subtraction of max(s) is for numerical stability only —
  it doesn't change the output or the gradient.
```

---

## Backward Derivation

```
  The Jacobian of softmax is:

    ∂p_i/∂s_j = p_i × (δ_ij - p_j)

  Where δ_ij = 1 if i=j, else 0.

  ─── Why this Jacobian? ───

  Case i = j (diagonal):
    p_i = exp(s_i) / Σ
    ∂p_i/∂s_i = (exp(s_i) × Σ - exp(s_i) × exp(s_i)) / Σ²
              = exp(s_i)/Σ × (1 - exp(s_i)/Σ)
              = p_i × (1 - p_i)
              = p_i × (δ_ii - p_i)  ✓

  Case i ≠ j (off-diagonal):
    ∂p_i/∂s_j = (0 × Σ - exp(s_i) × exp(s_j)) / Σ²
              = -exp(s_i)/Σ × exp(s_j)/Σ
              = -p_i × p_j
              = p_i × (0 - p_j)
              = p_i × (δ_ij - p_j)  ✓

  ─── Computing ds from dp ───

  ds_j = Σ_i dp_i × ∂p_i/∂s_j
       = Σ_i dp_i × p_i × (δ_ij - p_j)
       = dp_j × p_j - p_j × Σ_i (dp_i × p_i)

  Let dot = Σ_i (dp_i × p_i)

  Then:
    ds_j = p_j × (dp_j - dot)

  ┌─────────────────────────────────────────────────────────────────┐
  │  The trick: we avoid the O(n²) Jacobian multiplication!        │
  │                                                                 │
  │  Instead of ds = J^T × dp    (matrix-vector, O(n²))           │
  │  We compute:                                                    │
  │    dot = Σ(p ⊙ dp)           (one dot product, O(n))           │
  │    ds  = p ⊙ (dp - dot)      (element-wise, O(n))             │
  │                                                                 │
  │  Total: O(n) instead of O(n²). Critical when n = 32000 (vocab) │
  │  or n = 512 (sequence length in attention).                    │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Numerical Example

```
  scores = [2.0, 1.0, 0.5]
  probs  = [0.506, 0.234, 0.113]

  dp (upstream gradient) = [0.3, -0.1, 0.2]

  Step 1: dot = 0.506×0.3 + 0.234×(-0.1) + 0.113×0.2
              = 0.152 - 0.023 + 0.023
              = 0.152

  Step 2: ds_0 = 0.506 × (0.3 - 0.152) = 0.506 × 0.148 = 0.075
          ds_1 = 0.234 × (-0.1 - 0.152) = 0.234 × (-0.252) = -0.059
          ds_2 = 0.113 × (0.2 - 0.152) = 0.113 × 0.048 = 0.005

  Verify: sum(ds) = 0.075 - 0.059 + 0.005 = 0.021 ≈ 0
  (softmax gradients always approximately sum to zero because
   increasing one probability must decrease others — they sum to 1)
```

---

## Where Softmax Backward Appears

```
  1. Attention softmax (per head, per layer):
     Shape: [T, T] = [512, 512]
     Called 12 heads × 4 batch × 12 layers = 576 times per step
     Using the O(n) trick: 576 × 512 = 295K ops (trivial)
     Without it (O(n²)): 576 × 512² = 151M ops (still small, but worse)

  2. Cross-entropy softmax (final logits):
     Shape: [V] = [32000]
     Called 2048 times (once per token)
     But we NEVER compute this softmax Jacobian explicitly!
     Cross-entropy + softmax fuse into: dlogits = probs - one_hot
```
