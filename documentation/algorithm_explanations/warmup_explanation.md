# Learning Rate Warmup — Why, How, and When

---

## The Problem: Why Can't We Just Start at Full Learning Rate?

```
  At step 0, the model is in a BAD state:

  ┌─────────────────────────────────────────────────────────────────┐
  │  1. Weights are RANDOM (or near-random from initialization)    │
  │  2. Adam moments m=0, v=0 (no gradient history yet)            │
  │  3. Gradients are HUGE and NOISY (random weights → wild loss)  │
  └─────────────────────────────────────────────────────────────────┘

  If you apply a large LR to huge noisy gradients:

    weight_update = lr × gradient / sqrt(variance)
                  = 3e-4 × (huge) / sqrt(tiny)     ← v≈0 early on!
                  = MASSIVE update

  The weights overshoot wildly → loss explodes → training dies.
```

---

## What Warmup Does

```
  Instead of jumping to peak LR, ramp up gradually:

  Learning Rate vs Step
  ─────────────────────

  lr
  3e-4 |                    ╭────────────────────╮
       |                   ╱                      ╲
       |                  ╱                        ╲
       |                 ╱         cosine            ╲
       |                ╱          decay               ╲
       |               ╱                                ╲
       |              ╱                                  ╲
       |             ╱                                    ╲
       |            ╱                                      ╲
  3e-5 |           ╱                                        ╲______
       |          ╱
       |         ╱
       |        ╱
       |       ╱  ← linear warmup
       |      ╱
       |     ╱
       |    ╱
       |   ╱
       |  ╱
   0   |_╱_____|_____________________________________________|______
       0     2000                                          7000
            warmup                    cosine decay          max_steps
            ends                                           (min_lr)
```

---

## The Three Phases of Training

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │  Phase 1: WARMUP (steps 0 → 2000)                                  │
  │  ─────────────────────────────────                                   │
  │  lr = peak_lr × (step / warmup_steps)                               │
  │                                                                      │
  │  Step    0:  lr = 3e-4 × (0/2000)    = 0                           │
  │  Step  100:  lr = 3e-4 × (100/2000)  = 1.5e-5                     │
  │  Step  500:  lr = 3e-4 × (500/2000)  = 7.5e-5                     │
  │  Step 1000:  lr = 3e-4 × (1000/2000) = 1.5e-4                     │
  │  Step 2000:  lr = 3e-4 × (2000/2000) = 3.0e-4  ← peak!           │
  │                                                                      │
  │  Purpose: Let Adam's moments (m, v) accumulate gradient stats      │
  │  before making big updates. Small LR = small steps = safe.         │
  │                                                                      │
  ├──────────────────────────────────────────────────────────────────────┤
  │                                                                      │
  │  Phase 2: COSINE DECAY (steps 2000 → max_steps)                    │
  │  ───────────────────────────────────────────────                     │
  │  progress = (step - warmup) / (max_steps - warmup)                  │
  │  lr = min_lr + 0.5 × (peak_lr - min_lr) × (1 + cos(π × progress)) │
  │                                                                      │
  │  Starts at peak_lr, smoothly decays to min_lr.                      │
  │  The cosine curve spends more time near peak (slow start of decay) │
  │  then accelerates the drop toward the end.                          │
  │                                                                      │
  ├──────────────────────────────────────────────────────────────────────┤
  │                                                                      │
  │  Phase 3: MIN LR FLOOR (after max_steps)                            │
  │  ─────────────────────────────────                                   │
  │  lr = min_lr = 3e-5                                                 │
  │                                                                      │
  │  Never goes to zero — the model can always make tiny adjustments.  │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## Why Adam Specifically Needs Warmup

```
  Adam's update rule:

    m = β₁ × m + (1 - β₁) × g          momentum (running average of gradients)
    v = β₂ × v + (1 - β₂) × g²         variance (running average of squared gradients)
    update = m / (√v + ε)

  At step 1:
    m = 0.1 × g          (β₁ = 0.9, so (1-0.9) × g)
    v = 0.05 × g²        (β₂ = 0.95, so (1-0.95) × g²)

  The bias correction makes this worse:
    m̂ = m / (1 - 0.9¹)  = m / 0.1  = g          (fully amplified!)
    v̂ = v / (1 - 0.95¹) = v / 0.05 = g²         (fully amplified!)

  So the update ≈ g / |g| = ±1 regardless of gradient magnitude!
  Every parameter takes a step of size ≈ lr.

  With lr = 3e-4 and 124.7M parameters all stepping by ±3e-4:
  → model weights get scrambled in ONE step.

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  After ~500 steps, v has seen enough gradients to estimate     │
  │  the true variance per parameter. Now Adam can properly        │
  │  scale: parameters with large gradients get small steps,       │
  │  parameters with small gradients get large steps.               │
  │                                                                 │
  │  The warmup gives Adam time to learn WHICH parameters need     │
  │  big steps and which need small steps, before the LR gets      │
  │  large enough to cause damage.                                  │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Visual: What Happens Without Warmup

```
  Loss vs Step — WITH warmup (safe)
  ─────────────────────────────────

  loss
   11 |×
      |  ×
   10 |    ×
      |      ×
    9 |        ×
      |          ××
    8 |            ××
      |              ×××
    7 |                 ×××××
      |                      ××××××
    6 |                            ××××××××
      |                                    ×××××××××××
    5 |                                               ×××××
      |____________________________________________________________
      0        500      1000     1500     2000     3000     4000


  Loss vs Step — WITHOUT warmup (divergence)
  ──────────────────────────────────────────

  loss
   11 |×
      |  ×
   10 |    ×
      |     ×
    9 |       ×
      |        ×
    8 |         ×
      |           ×
    7 |            ×
      |              ×        ×
    6 |                ×    ×   ×
      |                 × ×      ×                  ×
    5 |                  ×        ×  ×            × ×
      |                            ×   ×  ×     ×     ×
    4 |                                 ×   × ×         ×
      |                                      ×
  inf |                                        BOOM ← diverged
      |____________________________________________________________
      0        500      1000     1500     2000     3000     4000

  Without warmup, the model initially learns fast (loss drops to ~4)
  but becomes unstable as Adam's poor variance estimates cause
  oscillations that grow until the loss explodes.
```

---

## Why Linear Warmup (Not Exponential or Step)?

```
  ┌────────────────────────────────────────────────────────────────┐
  │  Warmup Shape        Formula              Why / Why Not       │
  │  ────────────────    ─────────────────    ──────────────────  │
  │                                                                │
  │  Linear (ours)       lr × step/warmup     Simple, predictable │
  │    ╱                                      Most commonly used  │
  │   ╱                                       Good enough for     │
  │  ╱                                        nearly all cases    │
  │                                                                │
  │  Exponential         lr × (1-e^{-step})   Too aggressive at   │
  │    ╱──               the end (jumps to    the end of warmup — │
  │   ╱                  peak too fast)       can still overshoot  │
  │  ╱                                                             │
  │                                                                │
  │  Step function       0, 0, 0, ... peak    Sudden jump causes  │
  │         ┌──          (no ramp at all)     instability at the  │
  │  ───────┘                                 transition point    │
  │                                                                │
  │  Cosine warmup       lr×(1-cos(π×s/w))/2  Slightly smoother   │
  │     ╱─               than linear but      Used in some newer  │
  │    ╱                 minimal benefit       papers. Overkill.   │
  │  ──                                                            │
  └────────────────────────────────────────────────────────────────┘
```

---

## How Long Should Warmup Be?

```
  Rule of thumb: warmup = 0.5% to 5% of total training steps

  ┌────────────────────────────────────────────────────────────────┐
  │  Total Steps    Warmup Steps    Warmup %    Notes             │
  │  ────────────   ────────────    ────────    ───────────────── │
  │  5,000          100-250         2-5%        Small run         │
  │  30,000         1,000-2,000     3-7%        Our config        │
  │  100,000        2,000-5,000     2-5%        Medium run        │
  │  600,000        2,000-5,000     0.3-0.8%    GPT-3 scale       │
  │                                                                │
  │  Our model: 2000 warmup / 30000 total = 6.7%                  │
  │  This is on the generous side. Could be 1000 and still work.  │
  └────────────────────────────────────────────────────────────────┘

  Too short (< 100 steps):
    Adam's v hasn't converged → unstable updates → may diverge

  Too long (> 10% of training):
    Wasted steps at low LR → model learns slowly for no benefit

  Sweet spot:
    Long enough for Adam's second moment (v) to roughly converge.
    v uses β₂ = 0.95, so its half-life is ~14 steps.
    After ~100 steps, v has a reasonable estimate.
    After ~500 steps, v is well-calibrated.
    We use 2000 to be extra safe with a 124M parameter model.
```

---

## When to Skip Warmup (Resuming from Checkpoint)

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Scenario                         Warmup?    Why               │
  │  ──────────────────────────────   ────────   ────────────────  │
  │                                                                 │
  │  Training from scratch            YES        Random weights,   │
  │                                              cold optimizer     │
  │                                                                 │
  │  Resume with saved optimizer      NO         Adam state is     │
  │  (m, v, step counter)                        already warm      │
  │                                                                 │
  │  Resume weights only (our case)   MAYBE      Weights are good  │
  │                                              but Adam is cold.  │
  │                                              Short warmup       │
  │                                              (100-200 steps)    │
  │                                              or --warmup 0      │
  │                                              if you're careful. │
  │                                                                 │
  │  Fine-tuning on new data          YES        New data has       │
  │                                              different gradient  │
  │                                              statistics          │
  └─────────────────────────────────────────────────────────────────┘

  Your situation:
    You loaded weights from step 2000 but Adam's m and v reset to 0.
    Using --start-step 2000 skips past warmup in the LR schedule,
    but Adam will still take ~100 steps to re-warm its moments.
    This is fine — the LR is at 3e-4 but the updates are naturally
    damped by Adam's cold variance estimates for those first steps.
```

---

## Our Implementation

```
  From main.cpp:

  float get_lr(int step, float peak_lr, float min_lr,
               int warmup_steps, int max_steps) {

      // Phase 1: Linear warmup
      if (step < warmup_steps) {
          return peak_lr * ((float)step / (float)warmup_steps);
      }

      // Phase 2: Cosine decay
      float progress = (float)(step - warmup_steps)
                      / (float)(max_steps - warmup_steps);
      if (progress > 1.0f) progress = 1.0f;

      return min_lr + 0.5f * (peak_lr - min_lr)
                            * (1.0f + cosf(M_PI * progress));
  }

  With --start-step 2000:
    global_step starts at 2001 (past warmup_steps=2000)
    → jumps straight to cosine decay at peak LR
```

---

## Compute Cost (FLOPs)

```
  Warmup has ZERO extra compute cost.

  The model does exactly the same forward + backward pass regardless
  of the learning rate. Warmup only changes the scalar multiplier
  on the weight update:

    w -= lr × adam_update

  The single multiply by lr costs 124.7M FLOPs = 0.1 GFLOPs.
  Compare to the ~1,300 GFLOPs of the actual training step.

  The "cost" of warmup is in WALL TIME:
    2000 warmup steps × ~0.4 sec/step = ~13 minutes
    of training where the model learns slowly.

  This is why skipping warmup on checkpoint resume saves time —
  not because the compute per step changes, but because every
  step now produces meaningful learning.
```

---

## Why 3e-4 as Peak Learning Rate?

```
  3e-4 is the standard peak LR for AdamW at the ~100M-300M param scale.
  It comes from empirical scaling laws (GPT, LLaMA, Chinchilla papers).

  ┌──────────────────────────────────────────────────────────────────┐
  │  Model Size        Peak LR        Source                        │
  │  ──────────────    ─────────      ──────────────────            │
  │  ~125M (ours)      3e-4           GPT-2, LLaMA                 │
  │  ~350M             3e-4           Same range                    │
  │  ~1.3B             1.5e-4         Halved — model is bigger      │
  │  ~7B               1e-4           LLaMA paper                   │
  │  ~70B              5e-5           Need very gentle steps        │
  │                                                                  │
  │  Rule: bigger model → smaller LR                                │
  │  (more parameters means each step affects more interactions)    │
  └──────────────────────────────────────────────────────────────────┘

  Why this number works:

  Adam's update ≈ lr × sign(gradient) ≈ ±lr per parameter.

  With lr = 3e-4:
    Each weight moves ~0.0003 per step
    After 30,000 steps: max displacement ≈ 9.0
    Weights initialized at std ≈ 0.02
    → enough room to learn, not enough to overshoot

  What happens at different LRs:

  ┌──────────────────────────────────────────────────────────────────┐
  │  LR          Effect                                             │
  │  ──────      ─────────────────────────────────────              │
  │  1e-3        Too aggressive for 125M. Loss unstable, may       │
  │              diverge. OK for tiny models (<10M).                │
  │                                                                  │
  │  3e-4        Sweet spot. Fast convergence + stable.             │
  │              Used by GPT-2, LLaMA, most papers at this scale.  │
  │                                                                  │
  │  1e-4        Too conservative. Converges but ~2× slower.       │
  │              Need ~2× more steps to reach same loss.            │
  │                                                                  │
  │  3e-5        Way too slow. Model barely moves.                  │
  │              Need ~10× more steps. Wastes GPU hours.            │
  └──────────────────────────────────────────────────────────────────┘

  Our min_lr = 3e-5 (10× lower than peak) is also standard.
  The cosine decay drops LR by 10× over training for a smooth landing.

  There is no closed-form formula that derives 3e-4. It comes from
  thousands of LR sweep experiments across model scales by OpenAI,
  Meta, and DeepMind, formalized in the Chinchilla scaling laws.
```
