# Backprop: RMSNorm

The most mathematically tricky backward pass in the model, because
normalizing one element changes the denominator for ALL elements.

Config: B=4, T=512, C=768, 25 RMSNorm ops (2 per layer + 1 final).

---

## Forward

```
  Given input x (shape [N, C]) and learnable weight γ (shape [C]):

  RMS(x) = sqrt( (1/C) × Σ_j x_j² + ε )

  out_i = (x_i / RMS(x)) × γ_i
```

---

## Backward Derivation

```
  Let:
    ss = (1/C) × Σ_j x_j²         (mean of squares)
    rsqrt = 1 / sqrt(ss + ε)       (reciprocal square root)
    norm_i = x_i × rsqrt           (normalized value)
    out_i = norm_i × γ_i           (scaled output)

  Given dout (shape [N, C]):

  ─── Gradient w.r.t. γ (weight) ───

  dγ_i = Σ_n dout_ni × norm_ni

  (sum over the batch dimension because γ is shared)

  ─── Gradient w.r.t. x ───

  This requires careful derivation because rsqrt depends on ALL x_j.

  Step 1: Direct path (x_i → norm_i → out_i)
    dout_i × γ_i × rsqrt

  Step 2: Indirect path (x_i → ss → rsqrt → all norm_j → all out_j)
    Each x_i contributes to ss:  dss/dx_i = 2x_i/C
    rsqrt depends on ss:         drsqrt/dss = -0.5 × (ss + ε)^{-3/2}
    Each norm_j depends on rsqrt: dnorm_j/drsqrt = x_j

    Combining via chain rule:
    indirect = Σ_j (dout_j × γ_j) × x_j × (-0.5) × (ss+ε)^{-3/2} × (2x_i/C)

  Let dot_product = Σ_j (dout_j × γ_j × norm_j)

  Step 3: Combine
    dx_i = rsqrt × (dout_i × γ_i  -  norm_i × dot_product / C)

  ┌─────────────────────────────────────────────────────────────────┐
  │  In compact form:                                               │
  │                                                                 │
  │  dx = rsqrt × (dout ⊙ γ  -  norm ⊙ (1/C) × Σ(dout ⊙ γ ⊙ norm))│
  │                                                                 │
  │  The second term is the correction for the normalization —      │
  │  changing one x_i affects the RMS denominator for ALL outputs. │
  │  This coupling is what makes the derivative non-trivial.       │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Derivation (Full Detail)

```
  out_i = γ_i × x_i × (ss + ε)^{-1/2}

  where ss = (1/C) × Σ_j x_j²

  Apply the product rule — x_i appears in TWO places:
    1. In the numerator: x_i
    2. In the denominator: ss contains x_i²

  ∂out_i/∂x_k = γ_i × [ δ_{ik} × (ss+ε)^{-1/2}
                        + x_i × (-1/2)(ss+ε)^{-3/2} × (2x_k/C) ]

  Where δ_{ik} = 1 if i=k, else 0.

  Multiply by dout_i and sum over i:

  dx_k = Σ_i dout_i × γ_i × [ δ_{ik} × rsqrt
                              - x_i × x_k × rsqrt³ / C ]

       = dout_k × γ_k × rsqrt
         - (x_k × rsqrt³ / C) × Σ_i (dout_i × γ_i × x_i)

       = rsqrt × [ dout_k × γ_k
                    - (x_k × rsqrt² / C) × Σ_i (dout_i × γ_i × x_i) ]

  Since norm_i = x_i × rsqrt:
       = rsqrt × [ dout_k × γ_k
                    - norm_k × (1/C) × Σ_i (dout_i × γ_i × norm_i) ]

  Which matches our compact form above.
```

---

## CUDA Implementation Note

```
  The backward kernel uses shared memory reduction:
    1. Each thread computes dout_i × γ_i × norm_i for its elements
    2. Parallel reduction sums these across the row → dot_product
    3. Each thread then computes its dx_i using the shared dot_product

  The atomicAddBf16 is used for dγ because multiple rows in the
  batch contribute to the same weight gradient.
```
