# KV Cache — Fast Autoregressive Inference

## What Problem Does It Solve?

During **inference** (text generation), the model generates one token at a time.
Each new token needs to attend to ALL previous tokens. Without a cache, we'd
recompute K and V for every previous token at every step — wasteful.

```
  Generating "The cat sat on the mat":

  Step 1: process "The"                          → predict "cat"
  Step 2: process "The cat"                      → predict "sat"
  Step 3: process "The cat sat"                  → predict "on"
  Step 4: process "The cat sat on"               → predict "the"
  Step 5: process "The cat sat on the"           → predict "mat"

  WITHOUT cache: step 5 recomputes K,V for ALL 5 tokens from scratch
  WITH cache:    step 5 only computes K,V for "the" (new token),
                 reuses cached K,V for "The cat sat on" from steps 1-4
```

---

## Core Idea

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  KV Cache stores K and V from all previous tokens.                 │
│  At each new step, we only compute K,V for the NEW token           │
│  and APPEND it to the cache.                                       │
│                                                                     │
│  Step t:                                                            │
│    Q_new = x_t × W_q           [1, d_k]     only new token        │
│    K_new = x_t × W_k           [1, d_k]     only new token        │
│    V_new = x_t × W_v           [1, d_v]     only new token        │
│                                                                     │
│    K_cache = concat(K_cache, K_new)   [t, d_k]  append            │
│    V_cache = concat(V_cache, V_new)   [t, d_v]  append            │
│                                                                     │
│    scores = Q_new × K_cache^T / √d_k    [1, t]  new Q vs ALL K   │
│    probs = softmax(scores)               [1, t]                    │
│    output = probs × V_cache              [1, d_v]                  │
│                                                                     │
│  Complexity per step:                                               │
│    Without cache: O(t × d × n_head)   for ALL tokens              │
│    With cache:    O(d × n_head)        for ONE token (+ cache ops) │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Visual: Cache Growing Over Time

```
  Step 1: Generate token 1 ("The")
  ┌──────────────────────────────┐
  │  Q = query for "The"         │
  │  K_cache = [K_"The"]         │  ← cache has 1 entry
  │  V_cache = [V_"The"]         │
  │  scores = Q × K_cache^T      │  = [1×1] — only self-attention
  │  output = softmax × V_cache  │
  └──────────────────────────────┘

  Step 2: Generate token 2 ("cat")
  ┌──────────────────────────────┐
  │  Q = query for "cat"          │
  │  K_cache = [K_"The", K_"cat"]│  ← cache grows to 2
  │  V_cache = [V_"The", V_"cat"]│
  │  scores = Q × K_cache^T      │  = [1×2] — attend to "The" and "cat"
  │  output = softmax × V_cache  │
  └──────────────────────────────┘

  Step 5: Generate token 5 ("the")
  ┌─────────────────────────────────────────────────────────────┐
  │  Q = query for "the"                                        │
  │  K_cache = [K_"The", K_"cat", K_"sat", K_"on", K_"the"]   │  ← 5 entries
  │  V_cache = [V_"The", V_"cat", V_"sat", V_"on", V_"the"]   │
  │  scores = Q × K_cache^T                                     │  = [1×5]
  │  output = softmax × V_cache                                 │
  └─────────────────────────────────────────────────────────────┘
```

---

## Compute Savings

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Generating a sequence of length T:                           │
│                                                                │
│  WITHOUT KV Cache:                                             │
│    Step 1: compute K,V for 1 token     (1 projection)         │
│    Step 2: compute K,V for 2 tokens    (2 projections)        │
│    Step 3: compute K,V for 3 tokens    (3 projections)        │
│    ...                                                         │
│    Step T: compute K,V for T tokens    (T projections)        │
│                                                                │
│    Total K,V projections: 1+2+3+...+T = T×(T+1)/2 = O(T²)   │
│                                                                │
│  WITH KV Cache:                                                │
│    Step 1: compute K,V for 1 token, cache it                  │
│    Step 2: compute K,V for 1 token, append to cache           │
│    Step 3: compute K,V for 1 token, append to cache           │
│    ...                                                         │
│    Step T: compute K,V for 1 token, append to cache           │
│                                                                │
│    Total K,V projections: T = O(T)                            │
│                                                                │
│    Speedup: T/2 ×  (for T=512, that's 256× fewer projections)│
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Memory Cost

```
  KV Cache size per layer:

    K: [T, n_kv_head, d_k] = [T, 4, 64] = 256×T values
    V: [T, n_kv_head, d_v] = [T, 4, 64] = 256×T values
    Total per layer: 512×T values

  For all 12 layers, BF16:

    T=512:   12 × 512 × 512 × 2 bytes = 6.3 MB
    T=2048:  12 × 512 × 2048 × 2 bytes = 25.2 MB
    T=8192:  12 × 512 × 8192 × 2 bytes = 100.7 MB

  GQA helps: with 4 KV heads instead of 12, the cache is 3× smaller
  than standard multi-head attention.

  ┌────────────────────────────────────────────────────────────────┐
  │  Cache size comparison (T=2048, BF16):                        │
  │                                                                │
  │  Standard MHA (12 KV heads):  12 × 2 × 2048 × 768 = 75.5 MB │
  │  GQA (4 KV heads):            12 × 2 × 2048 × 256 = 25.2 MB │
  │                                                   3× smaller  │
  │                                                                │
  │  This is WHY GQA was invented — to shrink the KV cache.      │
  │  During training, GQA saves some parameters.                  │
  │  During inference, GQA saves massive KV cache memory.         │
  └────────────────────────────────────────────────────────────────┘
```

---

## When KV Cache is Used vs Not Used

```
  TRAINING (no KV cache):
    Process entire sequences in parallel
    All T tokens computed at once
    Causal mask handles autoregressive property
    No need to cache — everything is computed together

  INFERENCE (KV cache):
    Generate one token at a time
    Each new token needs to attend to all previous
    Cache avoids recomputing K,V for past tokens
    Massive speedup for long generations

  PREFILL PHASE (inference startup):
    Process the entire prompt at once (like training)
    All K,V from the prompt go into the cache
    Then switch to cached generation for new tokens
```

---

## Implementation: Pre-allocated Buffer

```
  Naive approach: grow the cache with each token (realloc/concat)
  Better approach: pre-allocate for max sequence length

  ┌────────────────────────────────────────────────────────────────┐
  │                                                                │
  │  Pre-allocate:                                                 │
  │    K_cache = zeros(max_T, n_kv_head, d_k)    [2048, 4, 64]   │
  │    V_cache = zeros(max_T, n_kv_head, d_v)    [2048, 4, 64]   │
  │    pos = 0                                    current length  │
  │                                                                │
  │  At each step:                                                 │
  │    K_cache[pos] = K_new     write new K to slot "pos"         │
  │    V_cache[pos] = V_new     write new V to slot "pos"         │
  │    pos += 1                                                    │
  │                                                                │
  │    Q × K_cache[0:pos]^T     attend to filled slots only      │
  │                                                                │
  │  No memory allocation during generation.                       │
  │  One cudaMalloc at startup, then just pointer arithmetic.     │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

---

## RoPE + KV Cache

```
  RoPE rotates Q and K based on position.
  With KV cache, K at position t was rotated by angle t×freq
  during the prefill/generation step when it was first computed.

  Two approaches:

  1. Apply RoPE when inserting into cache (our approach):
     K_new = RoPE(x_t × W_k, pos=t)
     K_cache[t] = K_new                already rotated
     scores = Q_new × K_cache^T        rotations are baked in

  2. Store un-rotated K, apply RoPE on-the-fly:
     K_cache[t] = x_t × W_k            raw K
     At query time: rotate each cached K by its position
     More flexible (e.g., for position interpolation) but slower

  Approach 1 is simpler and faster — rotate once, use many times.
```

---

## Inference vs Training: Full Comparison

```
                        Training            Inference (with KV cache)
                        ────────            ─────────────────────────
Tokens processed        B×T at once         1 at a time
Q,K,V computed for      all T positions     1 new position
Attention matrix        [T × T]             [1 × t] (t = current length)
K,V storage             activation memory   dedicated KV cache
Backward pass           yes                 no (inference only)
GPU utilization         high (big batches)  often low (1 token at a time)
Bottleneck              compute (matmuls)   memory bandwidth (cache reads)
```

---

## Batched Inference: Multiple Users

```
  Serving many users at once:

  User A at position 100:  Q_A × K_cache_A[0:100]^T
  User B at position 50:   Q_B × K_cache_B[0:50]^T
  User C at position 200:  Q_C × K_cache_C[0:200]^T

  Each user has their OWN KV cache (different conversations).
  Total memory = Σ (per-user cache size)

  For 100 concurrent users at T=2048:
    100 × 25.2 MB = 2.52 GB just for KV caches

  This is why KV cache size matters for serving —
  it limits how many users you can handle simultaneously.
  GQA's 3× smaller cache = 3× more concurrent users.
```

---

## Compute Cost (FLOPs)

```
  KV Cache is about INFERENCE, not training. It trades memory for compute.

  Our config: B=1, C=768, n_head=12, n_kv_head=4, head_dim=64

  ┌──────────────────────────────────────────────────────────────────────┐
  │  WITHOUT KV Cache (recompute all tokens every step):               │
  │                                                                    │
  │  At position t, recompute K,V for ALL t positions:                │
  │    K projection: 2 × t × 768 × 256 = 393K × t FLOPs             │
  │    V projection: 2 × t × 768 × 256 = 393K × t FLOPs             │
  │    Q @ K^T:      2 × t × 64 × 12 heads = 1.5K × t FLOPs         │
  │    probs @ V:    2 × t × 64 × 12 heads = 1.5K × t FLOPs         │
  │                                                                    │
  │  Total for all T tokens:  Σ_{t=1}^{T} O(t) = O(T²)              │
  │  At T=2048: ~800 GFLOPs per layer × 12 layers = 9.6 TFLOPs      │
  │                                                                    │
  │  WITH KV Cache (compute only new token):                          │
  │                                                                    │
  │  At position t, only compute K,V for the NEW token (1 row):      │
  │    K projection: 2 × 1 × 768 × 256 = 393K FLOPs                 │
  │    V projection: 2 × 1 × 768 × 256 = 393K FLOPs                 │
  │    Q @ K^T:      2 × t × 64 × 12 heads = 1.5K × t FLOPs         │
  │    probs @ V:    2 × t × 64 × 12 heads = 1.5K × t FLOPs         │
  │                                                                    │
  │  Total for all T tokens:  T × O(1) + Σ O(t) = O(T) + O(T²)     │
  │  But the O(T²) part is only the small score/value matmuls,       │
  │  not the expensive projections.                                    │
  │                                                                    │
  │  Savings:                                                          │
  │    K,V projections: T× cheaper (biggest win)                      │
  │    At T=2048: saves ~1.6 GFLOPs per layer × 12 = 19.2 GFLOPs    │
  │                                                                    │
  │  Memory cost:                                                      │
  │    Cache size: T × n_kv_head × head_dim × 2 bytes × 2 (K+V)     │
  │    = 2048 × 4 × 64 × 2 × 2 = 2.1 MB per layer                   │
  │    × 12 layers = 25.2 MB total                                    │
  │                                                                    │
  │  The tradeoff:                                                     │
  │    25.2 MB memory → saves recomputing projections every step     │
  │    Without cache: 9.6 TFLOPs for 2048 tokens                     │
  │    With cache:    ~0.8 TFLOPs for 2048 tokens = 12× faster       │
  └──────────────────────────────────────────────────────────────────────┘
```
