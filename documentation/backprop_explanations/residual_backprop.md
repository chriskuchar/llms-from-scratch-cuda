# Backprop: Residual Connection

The simplest derivative in the model — but the most important for training.

Config: B=4, T=512, C=768, 24 residual connections (2 per layer).

---

## Forward

```
  out = x + sublayer(x)
```

---

## Backward Derivation

```
  This is the simplest derivative in the model.

  dL/dx = dL/dout × dout/dx

  out = x + f(x)

  dout/dx = 1 + df/dx

  But we don't compute it as one combined derivative.
  Instead, the residual splits the gradient:

    dx_residual = dout           (gradient through the skip path)
    dx_sublayer = dout           (gradient through the sublayer path)

  The sublayer's backward pass computes df/dx internally.
  The residual just ADDS the sublayer gradient to the skip gradient.

  In code:
    // dx already has gradient from residual path (= dout)
    // sublayer backward adds its own gradient to dx
    dx += sublayer_backward(dout)
```

---

## Why Residuals Prevent Vanishing Gradients

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  The residual connection is a GRADIENT HIGHWAY.                │
  │                                                                 │
  │  Without residuals (y = f(x)):                                 │
  │    dy/dx = f'(x)    ← if f'(x) < 1, gradient vanishes        │
  │    After 12 layers: gradient ≈ f'₁ × f'₂ × ... × f'₁₂ → 0   │
  │                                                                 │
  │  With residuals (y = x + f(x)):                                │
  │    dy/dx = 1 + f'(x)  ← the "1" ensures gradient ≥ 1         │
  │    After 12 layers: gradient = 1 + all the sublayer terms     │
  │    → gradient always has a direct path from loss to any layer  │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Gradient Flow Visualization

```
  Layer 12 ──→ Layer 11 ──→ Layer 10 ──→ ... ──→ Layer 1 ──→ Embedding

  Without residuals (gradient must pass through every sublayer):
  ────────────────────────────────────────────────────────────────
  dout → f'₁₂ × dout → f'₁₁ × f'₁₂ × dout → ... → Π f'ᵢ × dout
                                                      ─────────────
                                                      shrinks to 0!

  With residuals (gradient has a direct highway):
  ────────────────────────────────────────────────────────────────
  dout ─────────────────────────────────────────────────→ dout (full!)
    └→ + f'₁₂×dout                                         (bonus)
              └→ + f'₁₁×dout                                (bonus)
                         └→ ...                              (bonus)

  Every layer gets the FULL upstream gradient plus bonuses from
  all the sublayers above it.
```
