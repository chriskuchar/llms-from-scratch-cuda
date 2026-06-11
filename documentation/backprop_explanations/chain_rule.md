# Backprop: Chain Rule Refresher

All derivations for our 124.7M parameter LLaMA-style transformer.
Config: B=4, T=512, C=768, n_head=12, n_kv_head=4, head_dim=64, ffn=2048, V=32000.

Convention: `dL/dX` is shortened to `dX`. Upstream gradient always called `dout`.

---

```
  The chain rule is the ONLY tool you need for backprop.

  If   y = f(g(x))     then   dy/dx = df/dg × dg/dx

  For neural nets, we always compute:

    dL/dx = dL/dy × dy/dx
            ─────   ─────
            "dout"   "local gradient"
            (comes    (we derive
             from      this for
             above)    each op)

  Matrix version:
    If  Y = X @ W        (forward)
    Then dX = dY @ W^T   (backward w.r.t. input)
         dW = X^T @ dY   (backward w.r.t. weight)
```

---

## Why the Chain Rule Works for Deep Networks

```
  A transformer is just a chain of functions:

    loss = CrossEntropy(Softmax(Linear(RMSNorm(... Embedding(tokens)))))

  The chain rule telescopes through ALL of them:

    dL/dW_embed = dL/dlogits × dlogits/dhidden × dhidden/d... × .../dW_embed
                  ───────────   ────────────────   ────────────   ────────────
                  cross-entropy   output proj       all layers     embedding
                  gradient        gradient          gradients      gradient

  Each layer only needs to compute its LOCAL gradient and multiply
  by the upstream gradient (dout) that was passed down from above.

  ┌─────────────────────────────────────────────────────────────────┐
  │  This is why backprop is O(n) not O(n²):                       │
  │                                                                 │
  │  Each layer computes ONE local gradient and passes it down.    │
  │  We never need to differentiate through the entire network     │
  │  at once — just one layer at a time, in reverse order.         │
  └─────────────────────────────────────────────────────────────────┘
```
