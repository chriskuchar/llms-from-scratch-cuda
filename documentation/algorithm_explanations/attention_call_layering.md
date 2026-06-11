# Call Layering — Attention (Standard vs Flash)

How a single attention block flows through the codebase, from the model
loop down to the fused CUDA kernels. Covers **forward** and **backward**,
the **`#ifdef USE_FLASH_ATTN` toggle**, **fp32 vs bf16** overload resolution,
and the **internal steps** of the fused kernels.

There are three layers:

```
┌────────────────────────────────────────────────────────────────────┐
│  Layer 1: model.cpp        orchestration (per-layer loop)          │
│           Model::forward / Model::backward                         │
│           ModelBF16::forward / ModelBF16::backward                 │
│                              │                                      │
│                              ▼                                      │
│  Layer 2: attention.cu      block wrapper                          │
│           attention_forward / attention_backward                  │
│           (Q/K/V projections, RoPE, W_o, weight grads)            │
│                              │                                      │
│                              ▼                                      │
│  Layer 3: flash_attn_2.cu   fused core                            │
│           flash_attention_forward / flash_attention_backward      │
│           (tiled scores → softmax → @V, online softmax)           │
└────────────────────────────────────────────────────────────────────┘
```

`model.cpp` NEVER calls Layer 3 directly. It only ever calls
`attention_forward` / `attention_backward`. The flash-vs-standard choice
is made inside Layer 2.

---

## Step 0: The Toggle

`USE_FLASH_ATTN` is a compile-time switch. It shows up in two places:
the **call site** (which saved buffer to pass) and **inside Layer 2**
(which inner kernels to run).

```
                    USE_FLASH_ATTN defined?
                          │
            ┌─────────────┴─────────────┐
           YES                          NO
            │                            │
   pass  lse  [B,n_head,T]      pass  att  [B,n_head,T,T]
   fused tiled kernels          standard score/softmax/value kernels
   O(T) memory                  O(T²) memory
```

---

## Step 1: Layer 1 — model.cpp orchestration (FORWARD)

The per-layer loop calls `attention_forward`. Only the saved-stat buffer
differs between branches (`lse` vs `att`).

```
Model::forward  (per layer l)
┌──────────────────────────────────────────────────────────────────┐
│  rmsnorm_forward(ln1_out, ...)                                    │
│           │                                                       │
│           ▼                                                       │
│  #ifdef USE_FLASH_ATTN                                            │
│     attention_forward(attn_out, q,k,v, lse, ln1_out, wq,wk,wv,wo)│
│  #else                                                            │
│     attention_forward(attn_out, q,k,v, att, ln1_out, wq,wk,wv,wo)│
│  #endif                                                           │
│           │                                                       │
│           ▼                                                       │
│  residual_forward(x, x, attn_out)                                │
└──────────────────────────────────────────────────────────────────┘

  inputs : ln1_out [B*T, C]  (normalized residual stream)
           wq,wk,wv,wo       (this layer's weights)
  output : attn_out [B*T, C]
  saved  : q,k,v  + (lse OR att)   ← consumed by backward
```

---

## Step 2: Layer 2 — attention_forward internals

The wrapper is identical for both paths EXCEPT the fused middle. It always
does projections + RoPE + output projection.

```
attention_forward(out, q,k,v, [lse|att], inp, wq,wk,wv,wo, ...)
┌──────────────────────────────────────────────────────────────────┐
│  1. q = inp @ wq          matmul_forward   [B*T, C]              │
│  2. k = inp @ wk          matmul_forward   [B*T, kv_dim]         │
│  3. v = inp @ wv          matmul_forward   [B*T, kv_dim]         │
│                                                                  │
│  4. rope_forward(q, k)    rotate Q,K in place                   │
│                                                                  │
│  5. ┌────────────────── FUSED MIDDLE ──────────────────┐        │
│     │ #ifdef USE_FLASH_ATTN                            │        │
│     │   flash_attention_forward(context, lse, q,k,v)   │        │
│     │ #else                                            │        │
│     │   attention_score_kernel   → att                 │        │
│     │   softmax_forward          → att                 │        │
│     │   attention_value_kernel   → context             │        │
│     │ #endif                                           │        │
│     └──────────────────────────────────────────────────┘        │
│                                                                  │
│  6. out = context @ wo    matmul_forward   [B*T, C]             │
└──────────────────────────────────────────────────────────────────┘

  The projections (1-3), RoPE (4), and output proj (6) are SHARED.
  Only step 5 changes. That's why model.cpp keeps one call site.
```

---

## Step 3: Layer 3 — flash_attention_forward (the fused core)

This is where tiling + online softmax live. One thread block per
(batch, query head, Q-tile). Loops over K/V tiles, keeps only a tile in SRAM.

```
flash_attention_forward(context, lse, q, k, v, B,T,nh,nkv,hd)

  For each Q tile (rows of queries):
  ┌────────────────────────────────────────────────────────────┐
  │  init   m = -inf   (running max per row)                   │
  │         l = 0      (running sum of exp)                    │
  │         O = 0      (running output accumulator)            │
  │                                                            │
  │  for each K/V tile (causal: only t2 <= t1):               │
  │  ┌──────────────────────────────────────────────────┐    │
  │  │ load K_tile, V_tile  → SRAM                       │    │
  │  │ S = Q_tile @ K_tile^T * scale     (local scores) │    │
  │  │ m_new = max(m, rowmax(S))                         │    │
  │  │ p     = exp(S - m_new)                            │    │
  │  │ l     = l*exp(m - m_new) + rowsum(p)             │    │
  │  │ O     = O*exp(m - m_new) + p @ V_tile           │    │
  │  │ m     = m_new                                    │    │
  │  └──────────────────────────────────────────────────┘    │
  │                                                            │
  │  context_tile = O / l          (final normalize)          │
  │  lse_row      = m + log(l)     (SAVE for backward)         │
  └────────────────────────────────────────────────────────────┘

  Writes: context [B*T, C]   and   lse [B, n_head, T]
  Never materializes the T×T score matrix in HBM.
```

The rescaling `exp(m - m_new)` is the correction applied when a later tile
reveals a larger max than earlier tiles saw. See `flash_attention_explanation.md`.

---

## Step 4: Layer 1 — model.cpp orchestration (BACKWARD)

Reverse loop. Same single call site, only `lse` vs `att` swapped.

```
Model::backward  (per layer l, reversed)
┌──────────────────────────────────────────────────────────────────┐
│  residual_backward(dx, dattn_out, dx_tmp)                        │
│           │                                                       │
│           ▼                                                       │
│  #ifdef USE_FLASH_ATTN                                            │
│     attention_backward(dln1_out, dwq,dwk,dwv,dwo,                │
│                        dattn_out, ln1_out, q,k,v, lse, wq..wo)   │
│  #else                                                            │
│     attention_backward(dln1_out, dwq,dwk,dwv,dwo,                │
│                        dattn_out, ln1_out, q,k,v, att, wq..wo)   │
│  #endif                                                           │
│           │                                                       │
│           ▼                                                       │
│  rmsnorm_backward(dx_tmp, ...)                                   │
└──────────────────────────────────────────────────────────────────┘

  inputs : dattn_out [B*T, C]  (grad of attn output)
           q,k,v + (lse OR att) saved from forward
  output : dln1_out [B*T, C]
           dwq, dwk, dwv, dwo  (weight grads, accumulated)
```

---

## Step 5: Layer 2 — attention_backward internals

Mirror of forward. Projections' weight grads are shared; the fused middle
differs.

```
attention_backward(dinp, dwq,dwk,dwv,dwo, dout, inp, q,k,v, [lse|att], wq..wo)
┌──────────────────────────────────────────────────────────────────┐
│  6b. context @ wo = out  →  dcontext, dwo   matmul_backward      │
│                                                                  │
│  5b. ┌────────────────── FUSED MIDDLE ──────────────────┐       │
│      │ #ifdef USE_FLASH_ATTN                            │       │
│      │   flash_attention_backward(dq,dk,dv,             │       │
│      │       dcontext, q,k,v, lse)                      │       │
│      │ #else                                            │       │
│      │   attention_value_backward_datt → datt          │       │
│      │   attention_value_backward_dv   → dv            │       │
│      │   attention_softmax_backward    → dscores       │       │
│      │   attention_score_backward_dq   → dq            │       │
│      │   attention_score_backward_dk   → dk            │       │
│      │ #endif                                           │       │
│      └──────────────────────────────────────────────────┘       │
│                                                                  │
│  4b. rope_backward(dq, dk)            inverse rotation           │
│                                                                  │
│  1b. dq → dinp,  dwq    matmul_backward                          │
│  2b. dk → dinp_k, dwk   matmul_backward                          │
│  3b. dv → dinp_v, dwv   matmul_backward                          │
│  --. dinp = dinp + dinp_k + dinp_v    residual_forward accumulate│
└──────────────────────────────────────────────────────────────────┘

  dk, dv must be zeroed before the fused middle (GQA → atomicAdd).
```

---

## Step 6: Layer 3 — flash_attention_backward (recompute, don't store)

Key idea: the T×T probs were never saved. Recompute scores from q,k,
renormalize with the saved `lse` (no second softmax pass needed).

```
flash_attention_backward(dq, dk, dv, dcontext, q, k, v, lse, ...)

  For each Q tile:
  ┌────────────────────────────────────────────────────────────┐
  │  load lse_row (saved per query row)                        │
  │                                                            │
  │  for each K/V tile (causal):                              │
  │  ┌──────────────────────────────────────────────────┐    │
  │  │ S = Q_tile @ K_tile^T * scale     (RECOMPUTE)     │    │
  │  │ P = exp(S - lse_row)   ← exact probs, one pass    │    │
  │  │                                                   │    │
  │  │ dV += P^T @ dcontext_tile        (atomicAdd)      │    │
  │  │ dP  = dcontext_tile @ V_tile^T                    │    │
  │  │ dS  = P * (dP - rowsum(dP * P))   softmax bwd     │    │
  │  │ dQ += dS @ K_tile  * scale                        │    │
  │  │ dK += dS^T @ Q_tile * scale       (atomicAdd)     │    │
  │  └──────────────────────────────────────────────────┘    │
  └────────────────────────────────────────────────────────────┘

  Writes: dq [B*T, C], dk/dv [B*T, kv_dim]
  Recompute costs ~33% extra FLOPs, saves O(T²) HBM. Net win.
```

Because `lse = m + log(l)`, the term `exp(S - lse)` reproduces the exact
softmax probability in a single pass — no running max/sum needed in backward.

---

## Step 7: Overload resolution (fp32 vs bf16)

The compiler picks the overload by pointer type. This is why `lse` typing matters.

```
  fp32 Model:
    att  : float*          lse : float*
    → both fit the SAME float* slot in attention_backward.
    → swapping att↔lse just works.

  bf16 ModelBF16:
    att  : __nv_bfloat16*   lse : float*    ← DIFFERENT types
    → cannot share one parameter slot.
    → REQUIRES a bf16 overload whose stat arg is `const float* lse`:

      void attention_forward (__nv_bfloat16* out, ..., float* lse, ...);
      void attention_backward(__nv_bfloat16* dinp, ..., const float* lse, ...);

  Rule of thumb (matches rrms/losses): per-row scalars stay float*
  even in the bf16 model, for numerical stability.
```

---

## Full Stack at a Glance

```
FORWARD                                BACKWARD
───────                                ────────
model.cpp                              model.cpp
  attention_forward                      attention_backward
    matmul_forward  (Q,K,V)               matmul_backward (W_o)
    rope_forward                          ┌ flash_attention_backward
    ┌ flash_attention_forward             │   recompute S from q,k
    │   tiled online softmax              │   exp(S - lse)
    │   write context + lse               │   dV, dP, dS, dQ, dK
    └ (or standard score/softmax/value)   └ (or standard 5 kernels)
    matmul_forward  (W_o)                 rope_backward
                                          matmul_backward (W_q,W_k,W_v)
                                          residual accumulate → dinp

saved across the boundary:  q, k, v, and (lse if flash | att if standard)
```

---

## Memory Footprint (per layer, T=512, B=4, n_head=12)

```
  att  [B, n_head, T, T] = 4·12·512·512 · 2B  ≈  25 MB   (standard)
  lse  [B, n_head, T]    = 4·12·512     · 4B  ≈  98 KB   (flash)

  ~256× smaller saved state. The gap grows as O(T) vs O(T²):
  at T=2048 the att buffer alone is ~400 MB/layer.
```
