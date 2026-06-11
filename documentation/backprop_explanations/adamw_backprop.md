# Backprop: AdamW Optimizer

AdamW is not part of backprop (it runs AFTER), but it consumes the gradients
computed by backprop to update all 124.7M parameters.

Config: β₁=0.9, β₂=0.95, ε=1e-8, weight_decay=0.1, peak_lr=3e-4.

---

## Update Rules

```
  Given gradient g_t for parameter w at step t:

  ─── Step 1: Moment updates ───
  m_t = β₁ × m_{t-1} + (1 - β₁) × g_t           (first moment / momentum)
  v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²           (second moment / variance)

  ─── Step 2: Bias correction ───
  m̂_t = m_t / (1 - β₁^t)
  v̂_t = v_t / (1 - β₂^t)

  ─── Step 3: Parameter update ───
  w_t = w_{t-1} - η × (m̂_t / (√v̂_t + ε) + λ × w_{t-1})
                       ──────────────────   ───────────────
                       adaptive update       weight decay
```

---

## Derivative of the Update (why it converges)

```
  The effective step size per parameter:

  step_size = η × m̂ / (√v̂ + ε)

  Think of it as automatic learning rate per parameter:

  If gradient is CONSISTENT (same sign):
    m̂ ≈ recent gradient  (momentum carries it)
    v̂ ≈ gradient²
    step ≈ η × g / (|g| + ε) ≈ η × sign(g)
    → takes consistent steps of size ~η regardless of gradient magnitude

  If gradient is NOISY (flipping sign):
    m̂ ≈ 0  (positive and negative cancel)
    v̂ ≈ gradient²  (still large)
    step ≈ η × 0 / (|g| + ε) ≈ 0
    → automatically reduces step for noisy parameters

  ┌─────────────────────────────────────────────────────────────────┐
  │  AdamW separates weight decay from the adaptive update.        │
  │                                                                 │
  │  L2 regularization:  g' = g + λw, then update w -= η × g'     │
  │    Problem: the adaptive denominator √v̂ also scales the        │
  │    regularization, which means large-gradient params get       │
  │    less regularization — the opposite of what you want.         │
  │                                                                 │
  │  AdamW (decoupled): update w -= η × (m̂/√v̂) + η × λ × w      │
  │    Weight decay is applied DIRECTLY to weights, unscaled.      │
  │    Every parameter decays at the same rate regardless of       │
  │    gradient magnitude.                                          │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Bias Correction Derivation

```
  Why do we need bias correction?

  At step 1, m and v are initialized to 0:
    m_1 = 0.9 × 0 + 0.1 × g_1 = 0.1 × g_1
    v_1 = 0.95 × 0 + 0.05 × g_1² = 0.05 × g_1²

  These are biased LOW — m should estimate E[g], not 0.1 × E[g].

  The expected value of m_t (assuming constant gradient g):
    E[m_t] = (1 - β₁^t) × g

  So dividing by (1 - β₁^t) removes the bias:
    m̂_t = m_t / (1 - β₁^t) → unbiased estimate of E[g]

  At step 1:   1 - 0.9^1  = 0.1   → m̂ = m/0.1 = g     (fully corrected)
  At step 10:  1 - 0.9^10 = 0.65  → m̂ = m/0.65         (partial correction)
  At step 100: 1 - 0.9^100 ≈ 1.0  → m̂ ≈ m              (correction vanishes)

  Same logic for v with β₂ = 0.95:
  At step 1:   1 - 0.95^1  = 0.05  → v̂ = v/0.05 = g²
  At step 100: 1 - 0.95^100 ≈ 0.994 → v̂ ≈ v

  ┌─────────────────────────────────────────────────────────────────┐
  │  This is why --start-step matters for checkpoint resume:       │
  │                                                                 │
  │  At step 1:   bias correction divides by 0.1 and 0.05         │
  │               → amplifies everything 10-20×                    │
  │                                                                 │
  │  At step 2001: bias correction divides by ≈1.0                │
  │               → no amplification, stable updates               │
  │                                                                 │
  │  Using --start-step 2000 tells Adam "you're at step 2001"     │
  │  so it doesn't over-amplify the cold m and v values.           │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Gradient Clipping (Pre-Optimizer)

```
  Before AdamW runs, we clip the global gradient norm:

  global_norm = sqrt( Σ_i g_i² )     over all 124.7M parameters

  if global_norm > max_norm (1.0):
      scale = max_norm / global_norm
      g_i *= scale                     for all parameters

  This prevents any single batch from causing an enormous update.
  It rescales ALL gradients proportionally to keep the direction
  but reduce the magnitude.

  Clipping happens BEFORE Adam sees the gradients, so Adam's
  moment estimates are based on the clipped values.
```
