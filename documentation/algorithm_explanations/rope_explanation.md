# RoPE — Rotary Position Embedding

## What Problem Does RoPE Solve?

A transformer processes tokens in parallel — it has no built-in sense of order.
Without position information, "the cat sat on the mat" and "mat the on sat cat the"
would look identical. RoPE encodes position by **rotating** the Q and K vectors
before the attention dot product, so the model knows where each token is.

---

## Core Formula

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  2D Rotation (one pair of dimensions):                             │
│                                                                     │
│  ┌          ┐   ┌              ┐   ┌     ┐                        │
│  │ x'       │   │ cos θ  -sin θ│   │  x  │                        │
│  │          │ = │              │ × │     │                        │
│  │ y'       │   │ sin θ   cos θ│   │  y  │                        │
│  └          ┘   └              ┘   └     ┘                        │
│                                                                     │
│  Element-wise:                                                      │
│    x' = x·cos(θ) - y·sin(θ)                                       │
│    y' = x·sin(θ) + y·cos(θ)                                       │
│                                                                     │
│  Where the angle θ depends on position t and dimension pair i:     │
│                                                                     │
│    θ(t, i) = t × freq(i)                                          │
│                                                                     │
│                   1                                                 │
│    freq(i) = ──────────────                                        │
│               θ_base^(2i/d)                                        │
│                                                                     │
│    θ_base = 10000    d = head_dim = 64    i = pair index (0..31)   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Breaking Down Each Part

```
cos(θ), sin(θ)        What they do: define the rotation angle
                      θ = position × frequency
                      Position 0 → θ=0 (no rotation)
                      Position 100 → θ=100×freq (big rotation)

x·cos - y·sin         What it does: rotates point (x,y) by angle θ
y = x·sin + y·cos     Standard 2D rotation matrix applied to a pair
                      Preserves the LENGTH of the vector (orthogonal)

freq(i)               What it does: different speed for each pair
                      pair 0: freq=1.0    → fast rotation (seconds hand)
                      pair 15: freq=0.01  → slow rotation (hours hand)
                      pair 31: freq=0.0001 → very slow (year hand)

                      Fast pairs → sensitive to nearby token positions
                      Slow pairs → sensitive to distant token positions
```

---

## The Core Idea: 2D Rotation

Take a pair of dimensions from a Q vector and treat them as a point on a 2D plane.
Rotate that point by an angle that depends on the token's position in the sequence.

```
        y (dim 1)
        │
        │    • (0.5, 0.8)  ← original point
        │   ╱
        │  ╱  angle = 3.0 rad
        │ ╱
        │╱
   ─────┼──────── x (dim 0)
        │╲
        │ ╲
        │  ╲
        │   • (-0.61, -0.72)  ← rotated point
        │
```

---

## Why Rotation Encodes Relative Position

```
  Key insight:

    Q at position t₁, rotated by angle t₁·freq
    K at position t₂, rotated by angle t₂·freq

    Q·K depends ONLY on the angle DIFFERENCE = (t₁ - t₂)·freq

  Example:

    Q at pos 5,   K at pos 8     →  angle diff = 3·freq
    Q at pos 100, K at pos 103   →  angle diff = 3·freq   ← SAME!

    Tokens the same distance apart produce the SAME attention pattern
    regardless of absolute position. That's relative position encoding.
```

---

## What Does "Frequency" Mean Here?

```
  Frequency = how much rotation angle you gain per position step.

  angle = position × frequency

  ┌──────────────────────────────────────────────────────────────────┐
  │  freq = 1.0:     each position adds 1.0 radians of rotation    │
  │                   full circle (2π ≈ 6.28 rad) in ~6 positions  │
  │                                                                  │
  │  freq = 0.01:    each position adds 0.01 radians               │
  │                   full circle in ~628 positions                  │
  │                                                                  │
  │  freq = 0.0001:  each position adds 0.0001 radians             │
  │                   full circle in ~62,800 positions               │
  └──────────────────────────────────────────────────────────────────┘

  Same meaning as everyday frequency:

  Sound:  frequency = wave cycles per second
          high freq → fast vibration → high pitch
          low freq  → slow vibration → low pitch

  RoPE:   frequency = radians of rotation per position
          high freq → fast rotation → wraps around quickly
          low freq  → slow rotation → takes many positions to go around

  A full circle = 2π ≈ 6.28 radians.
  So the "wavelength" (positions per full cycle) = 2π / freq.
```

---

## Multiple Frequencies: The Clock Analogy

```
  freq(i) = 1 / (10000^(2i / 64))

  Each head has 64 dimensions → 32 pairs (i = 0 to 31).
  Pair i uses dimensions [2i, 2i+1] from the head's 64-dim vector.

  ┌──────────────────────────────────────────────────────────────────┐
  │  head vector: [d0, d1, d2, d3, d4, d5, ... d60, d61, d62, d63] │
  │                ─────  ─────  ─────          ──────  ──────      │
  │                pair 0  pair 1  pair 2        pair 30  pair 31    │
  │                i=0     i=1     i=2           i=30     i=31      │
  └──────────────────────────────────────────────────────────────────┘

  Each pair gets its own frequency based on i:

  Pair  Dims     2i/64   10000^(2i/64)    freq = 1/that    Rotation
  ────  ──────   ─────   ─────────────    ─────────────    ────────────
  i=0   [0, 1]   0/64    10000^0   = 1         1.0        FASTEST
  i=1   [2, 3]   2/64    10000^.031 = 1.35     0.74       very fast
  i=2   [4, 5]   4/64    10000^.063 = 1.82     0.55       fast
  i=3   [6, 7]   6/64    10000^.094 = 2.45     0.41
  i=4   [8, 9]   8/64    10000^.125 = 3.31     0.30
  i=5   [10,11]  10/64   10000^.156 = 4.47     0.22
  i=8   [16,17]  16/64   10000^.25  = 10.0     0.10       medium
  i=12  [24,25]  24/64   10000^.375 = 47.3     0.021
  i=16  [32,33]  32/64   10000^.5   = 100      0.01       slow
  i=20  [40,41]  40/64   10000^.625 = 473      0.0021
  i=24  [48,49]  48/64   10000^.75  = 1000     0.001      very slow
  i=28  [56,57]  56/64   10000^.875 = 4735     0.00021
  i=31  [62,63]  62/64   10000^.969 = 9120     0.00011    SLOWEST

  Example at position t=5:
    pair i=0:  angle = 5 × 1.0     = 5.0 rad    (almost full circle!)
    pair i=1:  angle = 5 × 0.74    = 3.7 rad
    pair i=4:  angle = 5 × 0.30    = 1.5 rad    (quarter turn)
    pair i=8:  angle = 5 × 0.10    = 0.5 rad    (small turn)
    pair i=16: angle = 5 × 0.01    = 0.05 rad   (barely moved)
    pair i=31: angle = 5 × 0.00011 = 0.00055 rad (didn't move)

  pair  i=0:   freq = 1.0        — seconds hand (spins fast)
  pair  i=8:   freq ≈ 0.1        — minutes hand
  pair  i=16:  freq = 0.01       — hours hand
  pair  i=24:  freq ≈ 0.001      — day hand
  pair  i=31:  freq ≈ 0.00011    — year hand (barely moves)

  FAST PAIRS (low i)              SLOW PAIRS (high i)
  ┌───────────┐                   ┌───────────┐
  │     •     │                   │           │
  │    ╱      │  moves a lot      │     •     │  barely moves
  │   ╱       │  between          │     │     │  between
  │  ╱        │  adjacent         │     │     │  adjacent
  │ •         │  positions        │     •     │  positions
  └───────────┘                   └───────────┘
  Sensitive to nearby tokens      Sensitive to distant tokens

  Together: multi-scale position encoding
    Fast pairs → local syntax (word order within a phrase)
    Slow pairs → long-range structure (paragraph-level context)
```

---

## Why Low i = Nearby Tokens, High i = Distant Tokens

```
  The dot product Q·K measures "how much should token t₁ attend to t₂?"
  RoPE makes this depend on the DISTANCE (t₁ - t₂) through rotation.

  The key: after rotation, Q·K for one pair contains cos(Δt × freq(i))
  where Δt = t₁ - t₂ = distance between the two tokens.

  ─── What cos(Δt × freq) looks like for different i ───

  pair i=0, freq=1.0 (LOW i, FAST rotation):

  cos(Δt × 1.0)
   1 │x                                x
     │ x                              x
     │  x                            x
   0 │───x──────────────────────────x────
     │    x                        x
     │     x                      x
  -1 │      xxxxxxxxxxxxxxxxxxxxxxx
     └──────────────────────────────────
     Δt=0  1  2  3  4  5  6  7  8  9

  The signal completes a full cycle in ~6 positions.
  Tokens 1 apart look VERY different (cos drops fast).
  → This pair can distinguish: "position 5 vs position 6"
  → Sensitive to LOCAL word order (syntax, grammar)


  pair i=16, freq=0.01 (MID i):

  cos(Δt × 0.01)
   1 │xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
     │
     │
   0 │
     │
     │
  -1 │
     └──────────────────────────────────
     Δt=0  1  2  3  4  5  6  7  8  9

  Signal barely changes over 10 positions. Needs ~300 positions
  for a half cycle. Tokens 1 apart look identical.
  → This pair can distinguish: "position 50 vs position 350"
  → Sensitive to PARAGRAPH-level structure


  pair i=31, freq=0.00011 (HIGH i, SLOW rotation):

  cos(Δt × 0.00011)
   1 │xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
     │  (literally flat — needs ~28,000 positions to change)
     └──────────────────────────────────
     Δt=0  1  2  3  4  5  6  7  8  9

  → This pair can distinguish: "beginning vs end of a document"
  → Sensitive to DOCUMENT-level position
```

### Why This Multi-Scale Design?

```
  Language has structure at multiple scales:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Scale          Distance    Which pairs    Example             │
  │  ─────────      ────────    ───────────    ─────────────────── │
  │  Word order     1-5 tokens  i=0,1,2        "the cat" vs       │
  │                                             "cat the"          │
  │                                                                 │
  │  Phrase         5-30 tokens i=5-10         subject-verb        │
  │                                             agreement          │
  │                                                                 │
  │  Sentence       30-100      i=10-20        pronoun reference   │
  │                                             ("he" → who?)      │
  │                                                                 │
  │  Paragraph      100-500     i=20-28        topic tracking      │
  │                                                                 │
  │  Document       500+        i=28-31        overall context     │
  └─────────────────────────────────────────────────────────────────┘

  The 32 pairs together give the model a MULTI-RESOLUTION view of
  position. It's like having 32 clocks running at different speeds —
  from a stopwatch (pair 0) to a calendar (pair 31).

  The attention score sums all 32 pair dot products:

  score(t₁, t₂) = Σ_i  q_pair_i · k_pair_i

  Each pair votes on "should these tokens attend to each other?"
  based on its own frequency scale. The model LEARNS which
  frequencies matter for which task during training.
```

### Numerical: How Distance Affects the Dot Product

```
  Two tokens with identical Q,K vectors (before RoPE):
    q = k = [1.0, 0.0] for pair i=0 (freq=1.0)

  After RoPE at positions t₁ and t₂:
    Dot product of rotated pair = cos((t₁ - t₂) × freq)

  Pair i=0 (freq=1.0):
    distance 0:  cos(0) = 1.00    ← identical, max attention
    distance 1:  cos(1) = 0.54    ← nearby, moderate attention
    distance 3:  cos(3) = -0.99   ← close but out of phase
    distance 6:  cos(6) = 0.96    ← wraps around!

  Pair i=16 (freq=0.01):
    distance 0:  cos(0) = 1.00    ← identical
    distance 1:  cos(0.01) = 1.00 ← basically the same!
    distance 3:  cos(0.03) = 1.00 ← still the same
    distance 100: cos(1.0) = 0.54 ← NOW it notices the gap

  Low i pairs: sharp distance discrimination at close range
  High i pairs: smooth distance discrimination at long range
```

---

## Full Worked Example

```
  Starting scenario:

  Input sentence:  "The  cat  sat  on  the  mat"
                    tok0 tok1 tok2 tok3 tok4 tok5

  We'll trace token "on" (tok3):
    Batch:       b = 0  (first sequence in the batch)
    Position:    t = 3  (4th token, 0-indexed)
    Head:        h = 0  (first query head out of 12)
    Pair:        i = 0  (first pair: dimensions [0, 1] of head 0)

  After the Q projection (X @ W_q), head 0 has 64 values.
  Pair i=0 uses the first two:
    q[0] = 0.5   (this came from W_q projecting the "on" embedding)
    q[1] = 0.8

  These are just regular numbers — no position info yet.
  RoPE adds position by rotating this (0.5, 0.8) point.
```

### Step 1: Compute Frequency

```
  Which pair?  i = 0 (dimensions [0, 1])

  freq(i) = 1 / (10000^(2i / 64))
  freq(0) = 1 / (10000^(0 / 64))
           = 1 / (10000^0)
           = 1 / 1
           = 1.0

  This is the FASTEST frequency — pair 0 rotates the most per position.
```

### Step 2: Compute Angle

```
  angle = position × frequency
        = 3 × 1.0
        = 3.0 radians

  Token "on" at position 3 gets rotated by 3.0 radians (~172 degrees).
  For comparison:
    "The" (pos 0):  angle = 0 × 1.0 = 0.0 rad (no rotation)
    "cat" (pos 1):  angle = 1 × 1.0 = 1.0 rad (~57 degrees)
    "sat" (pos 2):  angle = 2 × 1.0 = 2.0 rad (~115 degrees)
    "on"  (pos 3):  angle = 3 × 1.0 = 3.0 rad (~172 degrees)  ← this one
```

### Step 3: Compute cos and sin

```
  cos(3.0) = -0.9900
  sin(3.0) =  0.1411

  The angle 3.0 rad is just past π (180°), so:
    cos is negative (pointing left on x-axis)
    sin is small positive (slightly above x-axis)
```

### Step 4: Load Original Q Values

```
  From the Q projection for token "on", head 0, pair 0:

  q[0] = 0.5     (x coordinate — dimension 0 of head 0)
  q[1] = 0.8     (y coordinate — dimension 1 of head 0)

  These values encode what W_q learned about the word "on" —
  its meaning, syntactic role, etc. But NO position info yet.
```

### Why Pair 0? All 32 Fire in Parallel

```
  The example uses pair i=0 for simplicity, but ALL 32 pairs
  rotate simultaneously for every token. Pair i is not "for" a
  specific word — every word goes through all 32 pairs.

  Where do the 64 dimensions come from?

  The embedding for each token is 768 numbers (C=768).
  The Q projection (X @ W_q) splits those 768 numbers evenly
  across 12 attention heads:

    768 ÷ 12 heads = 64 dimensions per head

  Each dimension is just a float — one number that W_q learned
  to extract from the token's embedding. A head's 64 dimensions
  together form a vector that represents what that head "cares
  about" for this token.

  ┌──────────────────────────────────────────────────────────────┐
  │  Token "on" embedding: 768 floats                           │
  │  ──────────────────────────────────────────                  │
  │  Q = embedding @ W_q → 768 floats reshaped to 12 heads:    │
  │                                                              │
  │  Head 0:  q[0] q[1] q[2] q[3] ... q[62] q[63]   (64 dims) │
  │  Head 1:  q[0] q[1] q[2] q[3] ... q[62] q[63]   (64 dims) │
  │  Head 2:  q[0] q[1] q[2] q[3] ... q[62] q[63]   (64 dims) │
  │  ...                                                         │
  │  Head 11: q[0] q[1] q[2] q[3] ... q[62] q[63]   (64 dims) │
  │                                                              │
  │  12 heads × 64 dims = 768 total (matches embedding size)    │
  └──────────────────────────────────────────────────────────────┘

  RoPE groups each head's 64 dimensions into pairs of 2:

  64 dimensions ÷ 2 per pair = 32 pairs

  Each pair is an (x, y) point on a 2D plane. You need 2 dimensions
  to do a rotation — you can't rotate a single number.

  All 32 pairs for head 0's Q vector of token "on" at position 3:

  Pair  Dims       x=q[2i]  y=q[2i+1]  freq       angle(pos=3)
  ────  ─────────  ───────  ─────────  ─────────  ────────────
  i=0   q[0],q[1]   0.50     0.80     1.0        3.000
  i=1   q[2],q[3]   0.12    -0.34     0.74       2.220
  i=2   q[4],q[5]  -0.91     0.15     0.55       1.650
  i=3   q[6],q[7]   0.44     0.67     0.41       1.230
  i=4   q[8],q[9]  -0.23     0.88     0.30       0.900
  i=5   q[10],q[11]  0.71   -0.52     0.22       0.660
  i=6   q[12],q[13] -0.18    0.39     0.17       0.510
  i=7   q[14],q[15]  0.55   -0.11     0.12       0.360
  i=8   q[16],q[17]  0.33    0.46     0.10       0.300
  i=9   q[18],q[19] -0.62    0.28     0.07       0.210
  i=10  q[20],q[21]  0.19   -0.73     0.055      0.165
  i=11  q[22],q[23]  0.84    0.02     0.041      0.123
  i=12  q[24],q[25] -0.37    0.59     0.030      0.090
  i=13  q[26],q[27]  0.48   -0.41     0.022      0.066
  i=14  q[28],q[29] -0.15    0.76     0.017      0.051
  i=15  q[30],q[31]  0.63   -0.29     0.012      0.036
  i=16  q[32],q[33]  0.27    0.51     0.010      0.030
  i=17  q[34],q[35] -0.44    0.38     0.0074     0.022
  i=18  q[36],q[37]  0.81   -0.16     0.0055     0.017
  i=19  q[38],q[39] -0.09    0.64     0.0041     0.012
  i=20  q[40],q[41]  0.35   -0.55     0.0030     0.009
  i=21  q[42],q[43]  0.72    0.23     0.0022     0.007
  i=22  q[44],q[45] -0.58    0.41     0.0017     0.005
  i=23  q[46],q[47]  0.14   -0.82     0.0012     0.004
  i=24  q[48],q[49]  0.46    0.37     0.0010     0.003
  i=25  q[50],q[51] -0.33    0.69     0.00074    0.002
  i=26  q[52],q[53]  0.57   -0.25     0.00055    0.002
  i=27  q[54],q[55] -0.21    0.43     0.00041    0.001
  i=28  q[56],q[57]  0.68   -0.14     0.00030    0.001
  i=29  q[58],q[59]  0.39    0.56     0.00022    0.001
  i=30  q[60],q[61] -0.47    0.31     0.00017    0.001
  i=31  q[62],q[63]  0.25   -0.61     0.00011    0.0003

  (q values are illustrative — actual values come from W_q projection)

  Notice:
    pair 0:  angle = 3.0 rad (~172°) — nearly half a circle!
    pair 8:  angle = 0.3 rad (~17°)  — small turn
    pair 16: angle = 0.03 rad (~1.7°) — barely moved
    pair 31: angle = 0.0003 rad (~0.02°) — essentially didn't rotate

  Each pair rotates the SAME token by a different amount.
  All 32 rotations happen simultaneously on the GPU (one thread each).
```

### Step 5: Apply Rotation

```
  q[0]_new = x·cos(θ) - y·sin(θ)
           = 0.5 × (-0.9900) - 0.8 × (0.1411)
           = -0.4950 - 0.1129
           = -0.6079

  q[1]_new = x·sin(θ) + y·cos(θ)
           = 0.5 × (0.1411) + 0.8 × (-0.9900)
           = 0.0706 - 0.7920
           = -0.7214

  Now q = (-0.61, -0.72) encodes BOTH:
    - what the token "on" means (from W_q)
    - that it's at position 3 (from the rotation)
```

```
  Before RoPE         After RoPE
       y                    y
       │                    │
       │  • (0.5, 0.8)     │
       │                    │
       │                    │
  ─────┼─────── x     ─────┼─────── x
       │                    │
       │                    │  • (-0.61, -0.72)
       │                    │
```

### What Happens to the Same Token at a Different Position?

```
  If "on" appeared at position 10 instead of 3:
    angle = 10 × 1.0 = 10.0 rad
    cos(10) = -0.839,  sin(10) = -0.544

    q[0]_new = 0.5×(-0.839) - 0.8×(-0.544) = -0.420 + 0.435 = 0.016
    q[1]_new = 0.5×(-0.544) + 0.8×(-0.839) = -0.272 - 0.671 = -0.943

  Same word, same Q values, but DIFFERENT rotation → different result.
  The attention mechanism will see these as different because
  they've been rotated by different amounts.

  Position 3:  (-0.61, -0.72)     ← "on" early in sentence
  Position 10: (0.02, -0.94)      ← "on" later in sentence
```

---

## All 32 Pairs in Parallel

```
  Total attention score for one Q-K pair:

                  1      31
  score(t₁,t₂) = ─── × Σ  (q'_pair · k'_pair)
                  √64   i=0

  pair 0:  freq=1.0       → fast rotation  → fine position detail
  pair 1:  freq=0.74      → ...
  pair 2:  freq=0.55      → ...
  ...
  pair 15: freq=0.01      → ...
  ...
  pair 31: freq=0.00011   → slow rotation  → coarse position detail

  32 pair dot products summed → rich multi-scale position information
```

---

## CUDA Kernel Mapping

```
  Each GPU thread handles ONE pair in ONE head at ONE position:

  thread idx → decomposed into:
    i  = idx % 32                          → which pair (0-31)
    h  = (idx / 32) % 16                   → which head (0-11=Q, 12-15=K)
    bt = idx / (32 × 16)                   → which (batch, position)
    t  = bt % T                            → sequence position → angle

  Total threads: B × T × 16 × 32 = B × T × 512

  For B=4, T=512: 4 × 512 × 512 = 1,048,576 threads
  Each does: 1 cos, 1 sin, 2 multiplies, 1 add, 1 subtract
```

---

## Backward Pass

```
  Rotation is ORTHOGONAL (length-preserving)
  → inverse = rotate by NEGATIVE angle

  Forward:   rotate by +θ  →  x' = x·cos(θ) - y·sin(θ)
                               y' = x·sin(θ) + y·cos(θ)

  Backward:  rotate by -θ  →  dx = dx'·cos(θ) + dy'·sin(θ)     ← sign flip on sin
                               dy = -dx'·sin(θ) + dy'·cos(θ)

  No learned parameters → no weight gradients
  Gradients just flow through the inverse rotation back to dQ, dK.
```

---

## Compute Cost (FLOPs)

```
  Our config: B=4, T=512, n_head=12, n_kv_head=4, head_dim=64
  total_heads = 12 + 4 = 16 (Q heads + K heads; V is not rotated)
  half_hd = 32 (pairs per head)

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Per thread (one pair, one head, one position):                    │
  │    1 freq computation:    2 ops (power, divide)                    │
  │    1 angle:               1 op  (multiply)                         │
  │    cos + sin:             2 ops (transcendental, expensive on GPU) │
  │    2 rotations (x',y'):   4 ops (2 mul + 1 add + 1 sub each = 6) │
  │    Total per thread:      ~11 FLOPs                                │
  │                                                                    │
  │  Total threads:                                                    │
  │    B × T × total_heads × half_hd = 4 × 512 × 16 × 32            │
  │    = 1,048,576 threads                                             │
  │                                                                    │
  │  Forward FLOPs:   1,048,576 × 11 = 11.5 MFLOPs                   │
  │  Backward FLOPs:  same (just flip sin sign) = 11.5 MFLOPs         │
  │  Total:           23 MFLOPs per layer × 12 layers = 276 MFLOPs   │
  │                                                                    │
  │  Compare to total model: ~1,300 GFLOPs                           │
  │  RoPE is 0.00002% of compute — completely negligible.             │
  │  Each thread does ~11 simple ops. Trivially parallel.             │
  │                                                                    │
  │  RoPE is LATENCY-bound: the kernel launch overhead is likely      │
  │  more expensive than the actual computation.                      │
  └──────────────────────────────────────────────────────────────────────┘
```
