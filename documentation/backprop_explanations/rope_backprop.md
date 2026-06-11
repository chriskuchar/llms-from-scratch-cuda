# Backprop: RoPE (Rotary Position Embeddings)

The easiest backward pass to derive — rotation matrices are orthogonal,
so the backward is just the inverse rotation (flip the sign of sin).

Config: B=4, T=512, n_head=12, n_kv_head=4, head_dim=64, θ=10000.

---

## Forward

```
  For position pos, dimension pair (2i, 2i+1) in a head:

  freq = 1 / (θ^(2i/d))          where θ=10000, d=64 (head_dim)
  angle = pos × freq

  q'[2i]   = q[2i] × cos(angle) - q[2i+1] × sin(angle)
  q'[2i+1] = q[2i] × sin(angle) + q[2i+1] × cos(angle)

  Same rotation applied to K.
```

---

## Backward Derivation

```
  RoPE is a 2D rotation matrix applied per pair:

  Forward:  [q'_0]   [cos  -sin] [q_0]
            [q'_1] = [sin   cos] [q_1]

  The Jacobian of a rotation matrix:
    dq'_0/dq_0 = cos      dq'_0/dq_1 = -sin
    dq'_1/dq_0 = sin      dq'_1/dq_1 = cos

  Backward (transpose of rotation = inverse rotation):
    [dq_0]   [cos   sin] [dq'_0]
    [dq_1] = [-sin  cos] [dq'_1]

  Expanding:
    dq[2i]   = dq'[2i] × cos(angle) + dq'[2i+1] × sin(angle)
    dq[2i+1] = -dq'[2i] × sin(angle) + dq'[2i+1] × cos(angle)

  ┌─────────────────────────────────────────────────────────────────┐
  │  The backward is just a rotation by NEGATIVE angle.            │
  │                                                                 │
  │  Rotation matrices are orthogonal: R^T = R^{-1}               │
  │  So the "undo" of rotation(+θ) is rotation(-θ).               │
  │                                                                 │
  │  In code: literally flip the sign of sin in the forward kernel │
  │  Same kernel, same cost, same everything.                      │
  │                                                                 │
  │  No learned parameters → no weight gradients to compute.       │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Why the Transpose Equals the Inverse

```
  A rotation matrix R has the property R^T × R = I (identity).

  Proof for 2D:
    R   = [cos  -sin]     R^T = [cos   sin]
          [sin   cos]           [-sin  cos]

    R^T × R = [cos   sin] [cos  -sin]
              [-sin  cos] [sin   cos]

            = [cos²+sin²     -cos×sin+sin×cos]
              [-sin×cos+cos×sin   sin²+cos²   ]

            = [1  0]
              [0  1]  ✓

  So for backprop:
    Forward:   q' = R × q
    Backward:  dq = R^T × dq'  =  R^{-1} × dq'

  The gradient "unrotates" the upstream gradient.
```

---

## Numerical Example

```
  pos = 3, pair i = 0, head_dim = 64, θ = 10000

  freq = 1 / 10000^(0/64) = 1.0
  angle = 3 × 1.0 = 3.0
  cos(3) = -0.99,  sin(3) = 0.14

  Forward:
    q_in = [1.5, 0.8]
    q'_0 = 1.5 × (-0.99) - 0.8 × 0.14 = -1.597
    q'_1 = 1.5 × 0.14    + 0.8 × (-0.99) = -0.582

  Backward (given dq' = [0.3, -0.2]):
    dq_0 = 0.3 × (-0.99) + (-0.2) × 0.14  = -0.325
    dq_1 = -0.3 × 0.14   + (-0.2) × (-0.99) = 0.156

  Verification: if we rotate dq forward, we should get dq' back:
    dq'_0 = -0.325 × (-0.99) - 0.156 × 0.14 = 0.322 - 0.022 = 0.300 ✓
    dq'_1 = -0.325 × 0.14    + 0.156 × (-0.99) = -0.046 - 0.154 = -0.200 ✓
```
