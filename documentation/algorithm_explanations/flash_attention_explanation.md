# Flash Attention — Memory-Efficient Attention

## What Problem Does It Solve?

Standard attention materializes the full T×T score matrix in GPU memory (HBM).
This is the bottleneck for long sequences:

```
  Standard attention memory for score matrix [B, n_head, T, T]:

  T=512:     25 MB      ← fits easily
  T=2,048:   402 MB     ← tight
  T=8,192:   6.4 GB     ← doesn't fit on most GPUs
  T=32,768:  103 GB     ← impossible

  Flash Attention: O(T) memory instead of O(T²)
  Keeps only a small tile in fast SRAM, never writes full matrix to HBM.
```

---

## Core Idea

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Standard Attention (3 passes over HBM):                           │
│    1. S = Q × K^T / √d_k              write T×T scores to HBM    │
│    2. P = softmax(S)                   read/write T×T probs       │
│    3. O = P × V                        read T×T probs from HBM    │
│                                                                     │
│    Total HBM reads/writes: O(B × H × T² × d)                      │
│    Memory: O(B × H × T²) for the score matrix                     │
│                                                                     │
│  Flash Attention (1 fused pass, tiled):                            │
│    for each block of Q rows (tile):                                │
│      for each block of K,V cols (tile):                            │
│        - load Q_tile, K_tile, V_tile into SRAM                     │
│        - compute local scores S_tile = Q_tile × K_tile^T / √d_k   │
│        - update running softmax (online softmax)                   │
│        - accumulate O_tile += softmax_tile × V_tile               │
│      end                                                           │
│      write O_tile back to HBM                                     │
│    end                                                              │
│                                                                     │
│    Total HBM reads/writes: O(B × H × T × d² / M)                 │
│    Memory: O(B × H × T × d) — no T² term!                        │
│                                                                     │
│    M = SRAM size. The SRAM is ~100× faster than HBM.              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## GPU Memory Hierarchy

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Register File   ~256 KB total    ~0 cycles     fastest     │
│       ↕                                                      │
│  SRAM / Shared   ~128-228 KB      ~5 cycles     ← tiles    │
│  Memory          per SM            (on-chip)      live here │
│       ↕                                                      │
│  L2 Cache        ~3-6 MB          ~30 cycles                │
│       ↕                                                      │
│  HBM (DRAM)      ~12 GB           ~300 cycles    slowest    │
│                  (RTX 3060)                                  │
│                                                              │
│  HBM bandwidth:  360 GB/s  (RTX 3060)                       │
│  SRAM bandwidth: ~19 TB/s  (on-chip, per SM)                │
│                                                              │
│  SRAM is ~50× faster than HBM.                              │
│  Flash Attention's key insight: keep the score matrix        │
│  in SRAM, never write it to HBM.                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## The Tiling Strategy

Standard attention computes the FULL T×T matrix at once.
Flash Attention processes it in small tiles that fit in SRAM:

```
  Q [T × d]     K [T × d]     V [T × d]
  ┌─────────┐   ┌─────────┐   ┌─────────┐
  │ block 0 │   │ block 0 │   │ block 0 │
  │─────────│   │─────────│   │─────────│
  │ block 1 │   │ block 1 │   │ block 1 │
  │─────────│   │─────────│   │─────────│
  │ block 2 │   │ block 2 │   │ block 2 │
  │─────────│   │─────────│   │─────────│
  │ block 3 │   │ block 3 │   │ block 3 │
  └─────────┘   └─────────┘   └─────────┘

  Block size B_r × B_c ≈ 64×64 (tuned to fit in SRAM)

  For each Q block (outer loop):
    Initialize running output O = 0, running max m = -∞, running sum l = 0
    For each K,V block (inner loop):
      Load Q_block, K_block, V_block into SRAM
      Compute local scores: S = Q_block × K_block^T / √d
      Update online softmax statistics (m, l)
      Accumulate: O += rescaled_softmax × V_block
    Write final O back to HBM
```

---

## Online Softmax — The Key Innovation

Standard softmax needs TWO passes over the data:
1. Find max (for numerical stability)
2. Compute exp and sum

Flash Attention uses **online softmax** — processes data in ONE pass
by maintaining running statistics:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Online Softmax Algorithm:                                         │
│                                                                     │
│  Initialize:  m = -∞  (running max)                                │
│               l = 0   (running sum of exp)                         │
│               O = 0   (running output)                             │
│                                                                     │
│  For each new block of scores S_new:                               │
│                                                                     │
│    1. m_new = max(m_old, max(S_new))        update running max     │
│                                                                     │
│    2. l_new = l_old × exp(m_old - m_new)    rescale old sum        │
│             + Σ exp(S_new - m_new)           add new contributions │
│                                                                     │
│    3. O_new = O_old × exp(m_old - m_new) × (l_old / l_new)        │
│             + exp(S_new - m_new) / l_new × V_block                 │
│                    ╰──────────────────────────────╯                 │
│                    rescale: if the max changes, all previous        │
│                    softmax values need to be adjusted               │
│                                                                     │
│  After all blocks:                                                  │
│    O contains the correct attention output                         │
│    m contains the row max                                           │
│    l contains the row sum                                           │
│                                                                     │
│  Key property: mathematically IDENTICAL to standard softmax.       │
│  Just computed incrementally instead of all at once.                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Rescaling is Needed

```
  Problem: when processing block 2, the max might be larger
  than what we saw in blocks 0-1. This means our previous
  softmax values were computed with the WRONG max.

  Example:
    Block 0 max = 3.0  →  used exp(x - 3.0) for softmax
    Block 1 max = 5.0  →  true max is 5.0, not 3.0!

    All block 0's exp values are too large by a factor of exp(3.0 - 5.0) = exp(-2)

    Fix: multiply all old values by exp(m_old - m_new) = exp(3-5) = 0.135
    This corrects for the updated max.

  This is why flash attention tracks m and l — they enable
  retroactive correction as new blocks reveal larger values.
```

---

## Worked Example: 2 Blocks

**Setup:** T=4, d=2, block_size=2 (2 Q rows per block, 2 K rows per block)

### Block (0,0): Q[0:2] × K[0:2]

```
Q_block = [[1, 0],    K_block = [[1, 1],    V_block = [[1, 0],
            [0, 1]]                [0, 1]]                [0, 1]]

S = Q × K^T / √2 = [[1.0, 0.0],     (raw scores)
                      [1.0, 1.0]] / √2

S = [[0.71, 0.0],
     [0.71, 0.71]]

m = [0.71, 0.71]                      row maxes
exp(S - m) = [[1.0, 0.49],
               [1.0, 1.0]]
l = [1.49, 2.0]                       row sums

P_local = [[0.67, 0.33],
            [0.50, 0.50]]

O = P × V = [[0.67, 0.33],            partial output
              [0.50, 0.50]]
```

### Block (0,1): Q[0:2] × K[2:4]

```
New scores S_new from K[2:4]
m_new = max(m_old, max(S_new))         might update running max
l_new = l_old × exp(m_old - m_new) + Σ exp(S_new - m_new)

Rescale old output:
  O = O_old × exp(m_old - m_new) × (l_old / l_new) + new_contribution

After all blocks: O is the correct final attention output.
```

---

## Memory Comparison

```
                    Standard Attention      Flash Attention
                    ──────────────────      ───────────────
Score matrix        O(B×H×T²)              O(B_r × B_c) ← tile size
                    stored in HBM           only in SRAM

Extra storage       T×T per head            2×T per head (m, l vectors)
                    = 262,144 (T=512)       = 1,024 (T=512)

HBM reads/writes    O(T² × d)              O(T × d² / M)
                    ~3 passes over T²       ~1 pass, tiled

For T=512, d=64, one head:
  Standard:  512×512×4 = 1 MB in HBM (score matrix alone)
  Flash:     ~8 KB in SRAM (one tile) + 4 KB (m, l vectors)
```

---

## Speed Improvement

```
  Flash Attention is IO-bound, not compute-bound.
  The speedup comes from fewer HBM reads/writes:

  ┌──────────────────────────────────────────────────────┐
  │  Sequence Length    Standard    Flash    Speedup     │
  │  ───────────────    ────────    ─────    ───────     │
  │  T = 512            1.0×        1.3×     ~1.3×      │
  │  T = 1024           1.0×        1.7×     ~1.7×      │
  │  T = 2048           1.0×        2.4×     ~2.4×      │
  │  T = 4096           1.0×        3.5×     ~3.5×      │
  │  T = 8192           OOM         4.0×     ∞          │
  │                                                      │
  │  Speedup grows with T because the O(T²) HBM         │
  │  traffic of standard attention becomes dominant.     │
  │                                                      │
  │  At T=512 (our model), speedup is modest (~1.3×).   │
  │  The bigger win is training stability and the        │
  │  ability to scale to longer sequences later.         │
  └──────────────────────────────────────────────────────┘
```

---

## Backward Pass

Flash Attention backward is also tiled. The key insight: we DON'T save the
T×T attention matrix from the forward pass. Instead, we **recompute** it
from Q, K during the backward (recomputation is cheaper than the HBM
read/write of storing T×T).

```
  Standard backward: read saved P [T×T] from HBM → compute gradients
  Flash backward:    recompute P from Q,K on-the-fly in SRAM tiles

  Recomputation costs extra FLOPs but saves massive HBM bandwidth.
  On modern GPUs, compute is cheap but memory bandwidth is expensive.
  Net result: flash backward is faster despite doing more math.
```

---

## Implementation Approaches

```
  1. Use NVIDIA's cuDNN Flash Attention (easiest)
     - Drop-in replacement, highly optimized
     - Available in cuDNN 8.9+

  2. Use Tri Dao's FlashAttention library
     - pip install flash-attn
     - Works with PyTorch, can extract kernels

  3. Write your own (hardest, most educational)
     - Implement tiled attention with online softmax
     - Manage SRAM allocation manually
     - Handle causal masking within tiles
     - Implement backward with recomputation

  For our C/CUDA project, option 3 is the most aligned
  with the learning-from-scratch philosophy.
```

---

## Compute Cost (FLOPs)

```
  Flash Attention does the SAME arithmetic as standard attention.
  The FLOPs are identical — the speedup comes from memory, not math.

  Our config: B=4, T=512, n_head=12, head_dim=64

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Operation              Standard Attn         Flash Attn           │
  │  ──────────────────     ──────────────        ──────────────       │
  │  Q @ K^T                2×T²×d = 33.6M       same                 │
  │  Softmax                ~3×T²  = 786K         same (online)       │
  │  probs @ V              2×T²×d = 33.6M       same                 │
  │  ─────────────────────────────────────────────────────────────────  │
  │  FLOPs per head:        ~67.2 MFLOPs          ~67.2 MFLOPs        │
  │  × 12 heads × 4 batch:  3.2 GFLOPs           3.2 GFLOPs          │
  │                                                                    │
  │  FLOPs are IDENTICAL.                                             │
  │                                                                    │
  │  The difference is HBM (memory) traffic:                          │
  │                                                                    │
  │  Standard:                                                         │
  │    Write scores [T,T]:     T² × 2 bytes = 512K per head           │
  │    Write probs [T,T]:      T² × 2 bytes = 512K per head           │
  │    Total HBM traffic:      48 heads × 1MB = 48 MB                 │
  │                                                                    │
  │  Flash:                                                            │
  │    Scores stay in SRAM:    0 bytes written to HBM                 │
  │    Probs stay in SRAM:     0 bytes written to HBM                 │
  │    Only read Q,K,V and write O:                                    │
  │    Total HBM traffic:      4 × B×T×C × 2 bytes = 12.6 MB         │
  │                                                                    │
  │  Memory reduction:  48 MB → 12.6 MB = 3.8× less HBM traffic     │
  │                                                                    │
  │  At T=2048: 768 MB → 12.6 MB = 61× less!                        │
  │  This is where Flash Attention really shines — the savings       │
  │  scale as O(T²) for standard vs O(T) for Flash.                  │
  │                                                                    │
  │  Backward:                                                         │
  │    Standard: store probs [T,T] for backward = O(T²) memory       │
  │    Flash: recompute probs from Q,K on the fly = O(T) memory      │
  │    Recomputation costs ~33% more FLOPs but saves O(T²) memory.   │
  │                                                                    │
  │  At 101 TFLOPS + 360 GB/s (RTX 3060):                           │
  │    Standard attention is MEMORY-BOUND (low arithmetic intensity) │
  │    Flash Attention moves the bottleneck toward COMPUTE-BOUND     │
  │    → actually utilizes the Tensor Cores instead of waiting on HBM│
  └──────────────────────────────────────────────────────────────────────┘
```
