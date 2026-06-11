# Adam Warmup vs. Learning Rate Warmup

Two completely different "warmups" that happen at the same time
but for different reasons. Understanding the distinction explains
why checkpoint resume behaves the way it does.

---

## The Two Warmups

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  LR Warmup                    Adam Warmup                      │
  │  ──────────                   ───────────                      │
  │  What:  Learning rate ramps   What:  Adam's m and v buffers    │
  │         from 0 to peak_lr            accumulate gradient       │
  │                                      statistics                │
  │                                                                 │
  │  Where: get_lr() function     Where: Inside adamw_update()    │
  │         in main.cpp                  kernel on GPU             │
  │                                                                 │
  │  How:   lr = peak × step/N    How:   m = β₁m + (1-β₁)g       │
  │         (we control this)            v = β₂v + (1-β₂)g²       │
  │                                      (happens automatically)   │
  │                                                                 │
  │  Time:  2000 steps            Time:  ~200 steps (for v)       │
  │         (our config)                 ~50 steps (for m)         │
  │                                                                 │
  │  Saved: NO (it's just a       Saved: YES (m and v buffers     │
  │         formula from step)           must be checkpointed)     │
  │                                                                 │
  │  Cost:  Zero FLOPs            Cost:  Zero FLOPs               │
  │         (just a scalar)              (happens during update)    │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

---

## LR Warmup — What You Control

```
  LR warmup is EXPLICIT. You set the schedule in code:

  lr
  3e-4 |              ╭──────────
       |             ╱
       |            ╱
       |           ╱
       |          ╱  ← you chose this ramp
       |         ╱
       |        ╱
       |       ╱
       |      ╱
       |     ╱
   0   |____╱
       0   2000
       warmup_steps

  Formula:  lr = peak_lr × (step / warmup_steps)

  Purpose: Prevent large weight updates when Adam's v is unreliable.
           The small LR acts as a safety net while Adam figures out
           which parameters need big vs small steps.

  On checkpoint resume: Skip it. The model already went through
  warmup. Use --start-step to jump past the warmup region.
  LR warmup is determined purely by step number — no state to save.
```

---

## Adam Warmup — What Happens Automatically

```
  Adam warmup is IMPLICIT. It happens inside the optimizer:

  At step 1 (cold start):
    m = 0 + 0.1 × g₁           = 0.1g    (way too small)
    v = 0 + 0.05 × g₁²         = 0.05g²  (way too small)

    Bias correction fixes the scale:
    m̂ = m / (1 - 0.9¹)  = m / 0.1 = g₁     (OK now)
    v̂ = v / (1 - 0.95¹) = v / 0.05 = g₁²   (OK but noisy — one sample!)

  At step 50:
    m ≈ exponential average of last ~10 gradients    (decent)
    v ≈ exponential average of last ~20 grad²        (getting there)

  At step 200:
    m ≈ smooth running average                       (good)
    v ≈ stable variance estimate per parameter       (good)
    Adam now knows which params need big/small steps

  v convergence:
  ─────────────
  accuracy
  100% |                              xxxxxxxxxxxxxxxxxx
       |                    xxxxxxxxxx
   90% |              xxxxxx
       |          xxxx
   80% |        xx
       |      xx
   70% |    xx
       |   x
   50% |  x
       | x
    0% |x
       └──────────────────────────────────────────────
       0    50    100    150    200    250    300
                      steps

  v uses β₂ = 0.95 → half-life ≈ 14 steps.
  After 200 steps, v has "seen" the equivalent of ~20 independent
  gradient samples. Reliable enough for stable updates.
```

---

## Why They're Coupled (Training from Scratch)

```
  When training from scratch, BOTH warmups happen simultaneously
  and serve the same goal: prevent early instability.

  Step 1:
    LR = tiny (0.00015)     ← LR warmup protecting you
    v = unreliable           ← Adam is cold
    Effective update = tiny LR × noisy direction = small & messy
    → no damage, model is safe

  Step 500:
    LR = moderate (0.075)    ← LR warmup halfway
    v = pretty good           ← Adam warming up
    Effective update = moderate LR × decent direction = productive
    → model starts learning

  Step 2000:
    LR = peak (0.0003)       ← LR warmup complete
    v = converged             ← Adam fully warm
    Effective update = full LR × accurate direction = maximum learning
    → peak training efficiency

  ┌─────────────────────────────────────────────────────────────────┐
  │  The LR warmup is REDUNDANT with Adam's bias correction.      │
  │                                                                 │
  │  In theory, bias correction alone prevents the huge early      │
  │  updates. But in practice, v is so noisy with 1-10 samples    │
  │  that bias correction isn't enough. The LR warmup provides    │
  │  a second layer of safety.                                     │
  │                                                                 │
  │  Belt AND suspenders.                                          │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Why They Decouple (Checkpoint Resume)

```
  When resuming from a checkpoint, the two warmups decouple:

  ┌──────────────────────────────────────────────────────────────────┐
  │  Scenario              LR Warmup    Adam Warmup    Why         │
  │  ──────────────────    ─────────    ───────────    ────────    │
  │                                                                  │
  │  Fresh training        NEEDED       NEEDED         Both cold   │
  │  (no checkpoint)                                                │
  │                                                                  │
  │  Resume: weights       SKIP         NEEDED         Weights OK  │
  │  only (our case)       (--start-    (~200 steps)   but Adam    │
  │                        step 2000)                  m,v = 0     │
  │                                                                  │
  │  Resume: weights +     SKIP         SKIP           Everything  │
  │  optimizer state       (auto from   (m,v loaded)   is warm     │
  │  (new checkpoint)      saved step)                              │
  │                                                                  │
  │  Fine-tune on          MAYBE        NEEDED         New data    │
  │  new data              (short,      (new gradient  has differ- │
  │                        100 steps)    statistics)   ent stats   │
  └──────────────────────────────────────────────────────────────────┘
```

---

## What We Experienced

```
  Timeline of our training:

  ═══════════════════════════════════════════════════════════════
  FP16 run (original):
    Steps 0-2000:     LR warmup + Adam warmup (both cold)
    Step 2000:        Saved checkpoint (weights only)
    Step 2220:        DIVERGED (FP16 overflow) ← killed

  BF16 run attempt 1 (no --start-step):
    Loaded step_2000 checkpoint
    Steps 1-810:      LR warmup AGAIN (wasted!)
                      Adam cold AGAIN (m,v = 0)
                      LR only reached 1.22e-4 at step 810
                      → barely any learning ← killed

  BF16 run attempt 2 (--start-step 2000):
    Loaded step_2000 checkpoint
    Steps 2001-2200:  LR at peak 3e-4 immediately ✓
                      Adam cold (m,v = 0) for ~200 steps
                      Loss noisy but no divergence
    Steps 2200+:      Adam warm, LR cosine decaying
                      Passed step 2220 without diverging! ✓
    Step 4000:        Will save with optimizer state (new format)

  Future resume from step_4000:
    LR:   auto-resumes cosine decay from step 4001
    Adam: m,v loaded from checkpoint → instant warm start
    → zero wasted steps
  ═══════════════════════════════════════════════════════════════
```

---

## Why Each Is Necessary

```
┌──────────────────────────────────────────────────────────────────────┐
│  WHY LR WARMUP IS NECESSARY                                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Problem it prevents: the first few updates wrecking the model.       │
│                                                                      │
│  At step 1, the weights are random and Adam's variance estimate (v)  │
│  is basically empty. If you apply the full peak LR immediately, the  │
│  update is a large step in an essentially random direction:          │
│                                                                      │
│     full LR × garbage direction = huge, destructive update           │
│                                                                      │
│  That can push weights into a bad region (huge activations, dead     │
│  ReLUs/saturated units, NaNs) that the model never recovers from.    │
│                                                                      │
│  The fix: start LR at ~0 and ramp up. Early updates are tiny, so     │
│  even if the direction is wrong, no permanent damage is done. By     │
│  the time LR reaches peak, the model is in a sane region and Adam    │
│  has reliable statistics.                                            │
│                                                                      │
│  Without it: loss spikes or diverges in the first few hundred steps  │
│  (especially with large LR, big batches, or low precision).          │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  WHY ADAM WARMUP IS NECESSARY                                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Problem it prevents: unreliable per-parameter step sizes.           │
│                                                                      │
│  Adam divides each update by sqrt(v), its estimate of that           │
│  parameter's gradient variance. With only 1-2 gradient samples, v    │
│  is a terrible estimate:                                             │
│                                                                      │
│     update = m / (sqrt(v) + eps)                                     │
│                                                                      │
│  If v is tiny by chance (one small early gradient), sqrt(v) is tiny, │
│  the division blows the update UP → instability. If v is too big,    │
│  the update is uselessly small.                                      │
│                                                                      │
│  Bias correction (m_hat, v_hat) helps but isn't enough when v is     │
│  built from just a handful of noisy samples.                         │
│                                                                      │
│  The fix: let m and v accumulate over ~200 steps until they reflect  │
│  the true gradient statistics. Then sqrt(v) is trustworthy and the   │
│  per-parameter scaling actually means something.                     │
│                                                                      │
│  Without it: early updates have wildly miscalibrated magnitudes —    │
│  some parameters lurch, others stall.                               │
└──────────────────────────────────────────────────────────────────────┘
```

Together they cover both failure modes: **LR warmup controls the overall
step size while the model is fragile; Adam warmup fixes the per-parameter
step size until the variance estimates are trustworthy.** One is a global
safety throttle, the other is local calibration — you need both early on.

---

## Summary

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  LR Warmup:                                                    │
  │    Controlled by YOU via get_lr() function                     │
  │    Ramps learning rate from 0 to peak over N steps             │
  │    Saved implicitly via step number (no state needed)          │
  │    Skip on resume with --start-step                            │
  │                                                                 │
  │  Adam Warmup:                                                   │
  │    Happens AUTOMATICALLY inside the optimizer                  │
  │    m and v buffers accumulate gradient statistics               │
  │    Must be SAVED to checkpoint for seamless resume             │
  │    Takes ~200 steps if not saved (v with β₂=0.95)             │
  │    Takes 0 steps if saved (instant warm start)                 │
  │                                                                 │
  │  They serve the same purpose (prevent early instability)       │
  │  but through different mechanisms. LR warmup is the blunt      │
  │  tool (just make all updates small). Adam warmup is the        │
  │  precise tool (learn which updates should be big vs small).    │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```
