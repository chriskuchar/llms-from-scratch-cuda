# Backprop: Embedding

The end of the gradient chain — where backprop stops.
No gradient flows to the input tokens (they're integers, not differentiable).

Config: B=4, T=512, C=768, V=32000, N=B×T=2048.

---

## Forward

```
  out[n] = W_e[token[n]]       lookup row from [V, C] table

  For each of the N=2048 positions, copy one row (768 floats)
  from the embedding table based on the token ID at that position.
```

---

## Backward Derivation

```
  The embedding forward is a table lookup — no multiplication.
  The backward is a scatter-add:

  For each position n with token id t = token[n]:
    dW_e[t] += dout[n]

  That's it. The gradient for each token's embedding row is the
  sum of all dout vectors at positions where that token appeared.

  ┌─────────────────────────────────────────────────────────────────┐
  │  Why += (accumulation)?                                        │
  │                                                                 │
  │  If token "the" appears at positions 3, 15, and 42:           │
  │    dW_e["the"] = dout[3] + dout[15] + dout[42]               │
  │                                                                 │
  │  Each occurrence contributes its own gradient.                 │
  │  Rare tokens get fewer gradient updates → learn slower.        │
  │  Common tokens get many updates → learn faster.                │
  │                                                                 │
  │  No gradient flows to the input tokens (they're integers,     │
  │  not differentiable). The embedding table is the end of the    │
  │  gradient chain.                                                │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Formal Derivation

```
  Forward:  out_nj = W_e[token[n], j]     for j = 0..C-1

  dL/dW_e[v, j] = Σ_n dL/dout_nj × ∂out_nj/∂W_e[v,j]

  ∂out_nj/∂W_e[v,j] = 1 if token[n] = v, else 0

  Therefore:
    dL/dW_e[v, j] = Σ_{n: token[n]=v} dout[n, j]

  In words: sum up all the upstream gradients at positions
  where token v appeared.
```

---

## Gradient Flow Diagram

```
  Input tokens:     [42,  7, 42, 15,  7, ...]
                      │   │   │   │   │
                      ▼   ▼   ▼   ▼   ▼
  Embedding table:  ┌──────────────────────┐
                    │ row 0:  [............] │
                    │ ...                    │
                    │ row 7:  [............] │ ← gets dout[1] + dout[4]
                    │ ...                    │
                    │ row 15: [............] │ ← gets dout[3]
                    │ ...                    │
                    │ row 42: [............] │ ← gets dout[0] + dout[2]
                    │ ...                    │
                    │ row 31999: [........] │
                    └──────────────────────┘

  Most rows (31,995 out of 32,000) get ZERO gradient in any given batch.
  Only the ~2048 tokens that appeared get updates.
```

---

## CUDA Note

```
  Multiple tokens with the same ID create a race condition:
    Thread for position 3: dW_e["the"] += dout[3]
    Thread for position 15: dW_e["the"] += dout[15]   ← concurrent!

  Solution: atomicAddBf16 — serializes the adds at the hardware level.
  This is slow when one token repeats many times (contention), but
  tokens are usually diverse enough that collisions are rare.
```
