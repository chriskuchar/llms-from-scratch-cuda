# Embedding — From Token IDs to Vectors

## Core Formula

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Forward (table lookup):                                           │
│    out[t, c] = W_e[token_id[t], c]                                │
│                                                                     │
│  Where:                                                             │
│    W_e ∈ ℝ^{V × C}       embedding table (V=32000, C=768)         │
│    token_id[t] ∈ {0..V-1}  integer token ID at position t         │
│    out ∈ ℝ^{B×T × C}      output embeddings                       │
│                                                                     │
│  Backward (scatter-add):                                           │
│    dW_e[token_id[t], c] += dout[t, c]     (atomicAdd for safety)  │
│                                                                     │
│  No matrix multiply — just index into a lookup table.              │
│  The "learning" happens when gradients update the table rows.      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## What It Does

The embedding layer is the **very first operation** in the model.
It converts integer token IDs into dense vectors that the transformer
can actually process.

```
Input:   [1542,   284,  3290,    13]     ← token IDs (integers)
                    │
              embedding lookup
                    │
Output:  [[0.12, -0.34, 0.56, ...],     ← 768-dim vector for token 1542
          [0.78,  0.23, -0.11, ...],     ← 768-dim vector for token 284
          [-0.45, 0.67,  0.89, ...],     ← 768-dim vector for token 3290
          [0.33, -0.12,  0.05, ...]]     ← 768-dim vector for token 13
```

---

## The Embedding Table

```
                         768 dimensions →
                    ┌──────────────────────────────────┐
  Token 0     ("") │  0.012  -0.034   0.056  ...      │
  Token 1    ("a") │  0.078   0.023  -0.011  ...      │
  Token 2    ("b") │ -0.045   0.067   0.089  ...      │
  ...              │   ...     ...     ...   ...      │
  Token 1542       │  0.120  -0.340   0.560  ...      │  ← row 1542
  ...              │   ...     ...     ...   ...      │
  Token 31999      │  0.033  -0.012   0.050  ...      │
                    └──────────────────────────────────┘

  32,000 rows × 768 columns = 24,576,000 parameters
  That's ~20% of the entire model!
```

Each row is a **learned** 768-dimensional representation of one token.
Tokens with similar meanings end up with similar vectors after training.

---

## Forward Pass: Table Lookup

```
  out[t, c] = W_e[token_id[t], c]       just copy one row per token

  input_ids = [1542, 284, 3290, 13]

  out[0] = W_e[1542]     ← copy row 1542
  out[1] = W_e[284]      ← copy row 284
  out[2] = W_e[3290]     ← copy row 3290
  out[3] = W_e[13]       ← copy row 13

  CUDA thread idx:
    bt = idx / C            which token position
    c  = idx % C            which dimension
    token_id = input_ids[bt]
    out[idx] = wte[token_id × C + c]
```

No math at all — just memory reads.

---

## Backward Pass: Scatter-Add

```
  dW_e[token_id[t], c] += dout[t, c]

  Problem: multiple positions might use the SAME token ID
  ("the" appears many times). All gradients must accumulate.

  input_ids = ["the", "cat", "sat", "on", "the", "mat"]
                  ↑                              ↑
             position 0                     position 4
             both are token ID 1997

  dW_e[1997] = dout[pos 0] + dout[pos 4]    BOTH accumulate!

  This requires atomicAdd to prevent race conditions:

  WITHOUT atomicAdd (BROKEN):
    Thread A reads  dW_e[1997][0] = 0.0
    Thread B reads  dW_e[1997][0] = 0.0      ← stale value
    Thread A writes 0.0 + 0.01 = 0.01
    Thread B writes 0.0 + 0.04 = 0.04        ← OVERWRITES A!
    Result: 0.04  (should be 0.05)

  WITH atomicAdd (CORRECT):
    Thread A: atomicAdd(&dW_e[1997][0], 0.01)
    Thread B: atomicAdd(&dW_e[1997][0], 0.04)
    Result: 0.05 ✓
```

---

## Mixed Precision: Why FP32 Gradients?

```
  BF16 activations (dout) → FP32 gradient output (dW_e)

  Why?  High-frequency tokens appear hundreds of times per batch.
  Each does an atomicAdd to the same row:

  Token "the" appears 200 times:
    dW_e["the"] = Σ of 200 gradient vectors

    FP32: 200 small additions → precise result
    BF16: 200 small additions → accumulated rounding errors
          BF16 has only ~3 decimal digits of precision
          0.5 + 0.001 in BF16 = 0.5  (rounded away!)
```

---

## What the Vectors Mean

```
  After training, embedding vectors capture semantic meaning:

  Nearby in vector space:              Far apart:
    "cat" ≈ "dog" ≈ "kitten"           "cat" ≠ "algorithm"
    "run" ≈ "jog" ≈ "sprint"           "run" ≠ "photosynthesis"
    "Paris" ≈ "London" ≈ "Tokyo"       "Paris" ≠ "subtract"

  Famous example:
    vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

---

## Weight Tying (Not in Our Model, but Common)

```
  Embedding:     token_id → W_e[token_id] → 768-dim vector       (input)
  Output layer:  768-dim vector @ W_e^T → 32,000 logits           (output)

  Same matrix for both!
  Output logit for token v = dot_product(hidden_state, W_e[v])
  Saves 24.6M parameters.
  Our model uses separate weights.
```

---

## Compute Cost (FLOPs)

```
  Our config: B=4, T=512, C=768, V=32000, N=B×T=2048

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Forward (table lookup):                                           │
  │    2048 lookups into [32000, 768] table                            │
  │    Each lookup copies 768 values (no arithmetic, just memcpy)      │
  │    FLOPs: 0  (pure memory operation)                               │
  │    Memory: 2048 × 768 × 2 bytes = 3.1 MB read                    │
  │                                                                    │
  │  Backward (scatter-add):                                           │
  │    2048 atomic adds, each adding 768 values to the gradient table │
  │    FLOPs: 2048 × 768 = 1.57 MFLOPs (one add per element)         │
  │    Memory: same 3.1 MB but with atomics (much slower)              │
  │                                                                    │
  │  The embedding is a ZERO-FLOP forward pass!                       │
  │  It's purely memory-bound — one random access per token.          │
  │                                                                    │
  │  Memory traffic:                                                   │
  │    The embedding table itself: 32000 × 768 × 2 bytes = 47 MB     │
  │    But each forward only touches 2048 rows = 3.1 MB              │
  │    (likely all different tokens in a batch)                        │
  │                                                                    │
  │  Backward bottleneck:                                              │
  │    atomicAdd collisions when the same token appears multiple       │
  │    times in the batch. Worst case: all same token → serialized.   │
  │    Our implementation processes per-token with atomicAddBf16.     │
  └──────────────────────────────────────────────────────────────────────┘
```
