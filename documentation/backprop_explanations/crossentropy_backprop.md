# Backprop: Cross-Entropy + Softmax

This is where backprop STARTS — the loss function produces the first gradient.

Config: B=4, T=512, V=32000, N=B×T=2048.

---

## Forward

```
  Given logits z (shape [N, V]) and target labels y (shape [N]):

  Step 1: Softmax
    p_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))

  Step 2: Cross-entropy loss
    L = -(1/N) × Σ_n log(p_{y_n})

  Where p_{y_n} is the softmax probability of the correct token.
```

---

## Backward Derivation

```
  We want dL/dz_i for each logit z_i.

  Case 1: i = y (the correct class)
  ──────────────────────────────────
    L = -log(p_y)

    dL/dp_y = -1/p_y

    dp_y/dz_y = p_y(1 - p_y)        (softmax derivative, same index)

    dL/dz_y = dL/dp_y × dp_y/dz_y
            = (-1/p_y) × p_y(1 - p_y)
            = -(1 - p_y)
            = p_y - 1

  Case 2: i ≠ y (wrong classes)
  ──────────────────────────────────
    dp_y/dz_i = -p_y × p_i          (softmax derivative, different index)

    dL/dz_i = dL/dp_y × dp_y/dz_i
            = (-1/p_y) × (-p_y × p_i)
            = p_i

  Combined (both cases in one formula):
  ──────────────────────────────────
    dL/dz_i = p_i - 1_{i=y}

  Where 1_{i=y} is 1 if i is the correct class, 0 otherwise.

  In code (per token):
    dlogits[i] = probs[i]              for all i
    dlogits[y] = probs[y] - 1.0        for the correct class
    dlogits[i] /= N                    average over batch

  ┌─────────────────────────────────────────────────────────────────┐
  │  This is the MOST ELEGANT gradient in all of deep learning.    │
  │                                                                 │
  │  dlogits = softmax(logits) - one_hot(target)                   │
  │                                                                 │
  │  Softmax + cross-entropy cancel out, leaving a simple          │
  │  subtraction. The gradient for the correct class is (p - 1),   │
  │  pushing it toward 1. All wrong classes get gradient p,        │
  │  pushing them toward 0.                                        │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Numerical Example

```
  logits  = [2.0, 1.0, 0.5]    target = 0
  softmax = [0.506, 0.234, 0.113]

  dlogits = [0.506 - 1, 0.234, 0.113]
          = [-0.494,    0.234, 0.113]

  Correct class (idx 0) gets negative gradient → push logit UP
  Wrong classes get positive gradient → push logits DOWN
```

---

## Gradient Flow Diagram

```
  ┌──────────┐      ┌──────────┐      ┌──────────┐
  │  logits   │ ───→ │ softmax  │ ───→ │   -log   │ ───→ loss
  │ [2048,32k]│      │  probs   │      │ (correct)│
  └──────────┘      └──────────┘      └──────────┘
       ↑                                    │
       │          dlogits = p - one_hot     │
       └────────────────────────────────────┘

  The beauty: we never actually compute the softmax Jacobian (32000×32000).
  The combined cross-entropy+softmax gradient simplifies to a subtraction.
```
