# Grouped-Query Attention (GQA)

## What Attention Does

Attention answers the question: **"For each token, which other tokens should
it pay attention to, and how much?"**

Given the sentence "The cat sat on the mat", when processing "sat", attention
might decide to focus heavily on "cat" (who sat?) and lightly on "the" (not useful).

---

## Core Formula

```
                    ┌                          ┐
                    │       Q · K^T             │
Attention(Q,K,V) = │ softmax(────────) · V     │
                    │         √d_k              │
                    └                          ┘

Where:
  Q ∈ ℝ^{T × d_k}   — query matrix     ("what am I looking for?")
  K ∈ ℝ^{T × d_k}   — key matrix       ("what do I contain?")
  V ∈ ℝ^{T × d_v}   — value matrix     ("what information do I offer?")
  d_k = 64           — head dimension   (controls scaling)
  T = 512            — sequence length
```

### Breaking Down Each Part

```
Q · K^T           What it does: computes a score between every pair of tokens
─────────         Q row i dotted with K row j = "how relevant is token j to token i?"
  √d_k            The √d_k scaling keeps scores moderate so softmax has good gradients

                  Score matrix shape: [T × T] — every token vs every other token

                        K[0]   K[1]   K[2]   K[3]
                  Q[0] [ 0.26   0.10   0.19   0.04 ]   ← token 0's relevance to all others
                  Q[1] [ 0.06   0.40  -0.09   0.14 ]
                  Q[2] [ 0.16   0.03   0.35   0.11 ]
                  Q[3] [ 0.09   0.18   0.08   0.31 ]
```

```
softmax(scores)   What it does: turns raw scores into probabilities (0 to 1, sum to 1)
                  Each ROW independently becomes a probability distribution

                  Before:  [0.26,  0.10,  0.19,  0.04]   raw scores
                  After:   [0.34,  0.26,  0.28,  0.12]   probabilities (sum=1.0)

                  High score → high probability → "pay attention here"
                  Low score  → low probability  → "ignore this token"
```

```
probs · V         What it does: weighted average of value vectors
                  Each token's output = blend of all values, weighted by attention probs

                  context[2] = 0.30 × V[0] + 0.26 × V[1] + 0.44 × V[2]
                               ↑              ↑              ↑
                          30% of V[0]    26% of V[1]    44% of V[2]

                  The output is a rich mixture of information from relevant tokens
```

---

## Multi-Head Attention Formula

```
MultiHead(X) = Concat(head_0, head_1, ..., head_11) · W_o

  where head_i = Attention(X · W_q_i,  X · W_k_g(i),  X · W_v_g(i))

  ┌────────────────────────────────────────────────────────────────┐
  │  W_q_i  ∈ ℝ^{768 × 64}   query projection for head i        │
  │  W_k_j  ∈ ℝ^{768 × 64}   key projection for KV-head j       │
  │  W_v_j  ∈ ℝ^{768 × 64}   value projection for KV-head j     │
  │  W_o    ∈ ℝ^{768 × 768}   output projection                  │
  │  g(i) = floor(i / 3)      GQA group mapping (3 Q per 1 KV)   │
  └────────────────────────────────────────────────────────────────┘

  Each head sees a different 64-dim "view" of the 768-dim input.
  12 heads × 64 dims = 768 dims total → concat back to original size.
```

---

## Causal Mask Formula

```
                 ┌  Q_i · K_j
                 │  ─────────    if j ≤ i    (can see past + self)
  S_{i,j}  =    │    √d_k
                 │
                 └   -∞          if j > i    (cannot see future)

  e^{-∞} = 0  in softmax  →  future tokens get zero attention weight

  Visual for T=4:
                K[0]   K[1]   K[2]   K[3]
  Q[0]       [  ✓      ✗      ✗      ✗  ]     sees only self
  Q[1]       [  ✓      ✓      ✗      ✗  ]     sees pos 0-1
  Q[2]       [  ✓      ✓      ✓      ✗  ]     sees pos 0-2
  Q[3]       [  ✓      ✓      ✓      ✓  ]     sees all
```

---

## The Three Players: Q, K, V

Every token gets projected into three vectors:

```
Q (Query):  "What am I looking for?"
K (Key):    "What do I contain?"
V (Value):  "What information do I offer?"
```

Attention score = how well a Query matches a Key.
High score = "this token is relevant to me" → pull in its Value.

```
Token: "sat"
  Q = "I need to know WHO did the action"

Token: "cat"
  K = "I am a noun, a subject"         ← matches Q well! high score
  V = [rich representation of "cat"]   ← this gets pulled in

Token: "the"
  K = "I am a determiner"              ← doesn't match Q, low score
  V = [representation of "the"]        ← mostly ignored
```

---

## The Full Forward Pass

```
Step 1:  Q = input @ W_q          [B×T, 768] @ [768, 768]   = [B×T, 768]
Step 2:  K = input @ W_k          [B×T, 768] @ [768, 256]   = [B×T, 256]
Step 3:  V = input @ W_v          [B×T, 768] @ [768, 256]   = [B×T, 256]

Step 4:  RoPE(Q, K)               rotate Q and K to encode position

Step 5:  scores = Q @ K^T / √64   [B, 12, T, T]  — every Q vs every K
Step 6:  causal mask               scores[t2 > t1] = -∞  (can't see future)
Step 7:  probs = softmax(scores)   normalize to probabilities [0,1]
Step 8:  context = probs @ V       weighted sum of values [B×T, 768]

Step 9:  output = context @ W_o    [B×T, 768] @ [768, 768]  = [B×T, 768]
```

---

## Grouped-Query Attention (GQA)

Standard multi-head attention has 12 Q heads AND 12 K,V heads.
GQA reduces K,V heads to 4, with every 3 Q heads sharing one K,V head:

```
GQA group mapping:  g(i) = floor(i / r),   r = n_head / n_kv_head = 12/4 = 3

Standard MHA:                    GQA (our model):

Q heads:  0  1  2  3  4  5      Q heads:  0  1  2  3  4  5  6  7  8  9  10 11
K heads:  0  1  2  3  4  5      K heads:  0  0  0  1  1  1  2  2  2  3  3  3
V heads:  0  1  2  3  4  5      V heads:  0  0  0  1  1  1  2  2  2  3  3  3
          ↕  ↕  ↕  ↕  ↕  ↕                ╰──┬──╯  ╰──┬──╯  ╰──┬──╯  ╰──┬──╯
          1:1 pairing                      group 0  group 1  group 2  group 3
```

Why? K,V heads are often redundant — different heads learn similar keys/values.
Sharing saves 2/3 of K,V parameters with minimal quality loss.

```
K,V projection size:
  MHA:  [768, 768] for K and V each    = 1,179,648 params
  GQA:  [768, 256] for K and V each    =   393,216 params  ← 3× smaller
```

---

## The 1/√d_k Scaling — Why It Matters

```
Without scaling:
  Q and K entries have variance σ² each
  dot product = sum of d_k products
  Var(Q · K) = d_k × σ⁴         ← grows with dimension!

  d_k = 64  →  dot products are ~8× too large
  softmax on large values → one-hot output → vanishing gradients

With scaling:
  Var(Q · K / √d_k) = σ⁴        ← constant regardless of d_k
  softmax on moderate values → smooth probabilities → healthy gradients
```

---

## Multi-Head: Why 12 Heads?

Each head attends to different things:

```
Head 0:  might learn to attend to the subject of the sentence
Head 1:  might learn to attend to the previous verb
Head 2:  might learn to attend to the most recent noun
Head 3:  might learn syntactic structure (brackets, commas)
...
Head 11: might learn long-range topic coherence
```

Each head operates on its own 64-dim slice (768 / 12 = 64).
After all heads compute their context vectors, the results are
concatenated back to 768 dims and projected through W_o.

```
head 0 context [64]  ─┐
head 1 context [64]  ─┤
head 2 context [64]  ─┤
...                    ├──→ concatenate [768] ──→ W_o [768,768] ──→ output [768]
head 10 context [64] ─┤
head 11 context [64] ─┘
```

---

## Memory: The T×T Problem

The attention score matrix has shape [B, n_head, T, T]:

```
T=512:    4 × 12 × 512 × 512   = 12.6M values  = 25 MB (bf16)
T=2048:   4 × 12 × 2048 × 2048 = 201M values   = 402 MB (bf16)
T=8192:   4 × 12 × 8192 × 8192 = 3.2B values   = 6.4 GB (bf16)  ← doesn't fit!
```

This is why Flash Attention exists — it avoids materializing this matrix entirely.
At T=512 it's manageable, but it becomes the bottleneck as you scale up.

---

## Backward Pass Formulas

```
Step 9b:  dcontext = dout @ W_o^T               dW_o = context^T @ dout
Step 8b:  dprobs   = dcontext @ V^T              dV   = probs^T @ dcontext
Step 7b:  dscores  = probs × (dprobs - Σ_j(dprobs_j × probs_j))     ← softmax backward
Step 6b:  dscores masked (zero where causal mask was -∞)
Step 5b:  dscores /= √d_k
          dQ = dscores @ K                       dK = dscores^T @ Q
Step 4b:  RoPE backward (inverse rotation on dQ, dK)
Step 3b:  dinp += dV @ W_v^T                     dW_v = inp^T @ dV
Step 2b:  dinp += dK @ W_k^T                     dW_k = inp^T @ dK
Step 1b:  dinp += dQ @ W_q^T                     dW_q = inp^T @ dQ
```

The softmax backward (step 7b) is the trickiest — because softmax couples
all elements in a row, changing one score affects all probabilities.

```
Softmax backward formula:

  dS_i = P_i × (dP_i - Σ_j dP_j × P_j)

  Where P = softmax output (probabilities)
        dP = upstream gradient into probabilities
        dS = gradient into raw scores

  Why the subtraction?  Increasing one score pushes ALL probabilities down
  (because the denominator grows). The Σ term accounts for this coupling.
```

---

## Compute Cost (FLOPs)

```
  Our config: B=4, T=512, C=768, d_k=64, n_head=12, n_kv_head=4, kv_dim=256
  N = B×T = 2048

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Operation                Shape                   FLOPs            │
  │  ─────────────────────    ─────────────────────    ──────────       │
  │  Q projection             [2048,768]@[768,768]     2×2048×768×768  │
  │                                                    = 2.42 GFLOPs   │
  │                                                                    │
  │  K projection (GQA)       [2048,768]@[768,256]     2×2048×768×256  │
  │                                                    = 0.81 GFLOPs   │
  │                                                                    │
  │  V projection (GQA)       [2048,768]@[768,256]     2×2048×768×256  │
  │                                                    = 0.81 GFLOPs   │
  │                                                                    │
  │  RoPE                     2048×16×32 threads       trivial         │
  │                           6 FLOPs each             = 6.3 MFLOPs    │
  │                                                                    │
  │  Q @ K^T (per head)       [512,64]@[64,512]        2×512×64×512   │
  │    × 12 heads × 4 batch                            = 0.81 GFLOPs  │
  │                                                                    │
  │  Softmax                  4×12×512×512              ~100 MFLOPs    │
  │                           (exp, sum, div per elem)                 │
  │                                                                    │
  │  probs @ V (per head)     [512,512]@[512,64]       2×512×512×64   │
  │    × 12 heads × 4 batch                            = 0.81 GFLOPs  │
  │                                                                    │
  │  Output projection        [2048,768]@[768,768]     2×2048×768×768 │
  │                                                    = 2.42 GFLOPs  │
  │  ─────────────────────────────────────────────────────────────────  │
  │  TOTAL per layer (forward):                        ~8.1 GFLOPs    │
  │  TOTAL per layer (forward + backward):             ~24.3 GFLOPs   │
  │  TOTAL all 12 layers:                              ~291 GFLOPs    │
  │                                                                    │
  │  At 101 TFLOPS (BF16 Tensor Cores):                               │
  │    Attention per step ≈ 291G / 101T = 2.9 ms (theoretical)       │
  │                                                                    │
  │  Note: matmuls dominate (~99%). Softmax and RoPE are negligible.  │
  │  The Q,K,V,O projections are 4× more compute than the score/     │
  │  value matmuls because they operate on the full 768-dim space.    │
  └──────────────────────────────────────────────────────────────────────┘

  FLOPs rule for matmul: A[M,K] @ B[K,N] = 2×M×K×N FLOPs
  (one multiply + one add per output element, K times)
```
