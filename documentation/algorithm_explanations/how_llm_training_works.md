# How LLM Training Works

A plain-English guide to what JrLAM is actually doing when it trains,
and why pretraining and SFT are two separate phases.

---

## The Core Loop (Same for Everything)

Every single training step — whether pretraining, SFT, or any other kind
of training — does the exact same 5 things:

```
1. Feed in tokens:     [The, cat, sat, on, the]
2. Model predicts:     [cat, sat, on, the, mat]  (next token at each position)
3. Compare to actual:  [cat, sat, on, the, mat]  (ground truth from data)
4. Compute loss:       How wrong was it? (cross-entropy)
5. Update weights:     Nudge weights to be less wrong next time
```

That's it. The model never "reads" or "understands" — it just gets
slightly better at predicting the next token, over and over, billions
of times.

### What is "loss"?

Loss = how surprised the model is by the correct answer.

- `loss = -log(probability the model assigned to the correct token)`
- If the model was 100% sure of the right token: loss = 0 (perfect)
- If the model gave 1/32000 chance (random guess): loss = 10.4
- Lower loss = better predictions

### What is "next-token prediction"?

Given a sequence like:

```
Input:  [Once, upon, a, time, there, was, a, little]
Target: [upon, a, time, there, was, a, little, girl]
```

At each position, the model tries to predict what comes next.
It can only see tokens BEFORE the current position (causal masking).

Position 0: sees [Once] → should predict "upon"
Position 1: sees [Once, upon] → should predict "a"
Position 2: sees [Once, upon, a] → should predict "time"
...and so on for all 512 positions.

The model gets a gradient signal from EVERY position in EVERY sequence.
With B=16 and T=512, that's 16 × 512 = 8,192 prediction tasks per micro-batch.

---

## Phase 1: Pretraining — Learning Language

### What data looks like

Raw text. No structure, no instructions, just natural language:

```
Once upon a time, there was a little girl named Lily. She liked to play
in the garden. One day, she found a shiny red ball under a big tree.
"Look what I found!" she said to her mom.
```

### What the model learns

By predicting billions of next tokens from raw text, the model absorbs:

- **Vocabulary:** "shiny" is a word, "xqzpt" is not
- **Grammar:** "She liked to play" not "She liked to playing"
- **Common sense:** After "shiny red" → "ball", "car", "apple" (not "sadness")
- **Narrative:** Stories have characters, settings, events
- **World knowledge:** Gardens have flowers, balls are round, moms exist

The model doesn't "know" any of this explicitly. It just gets really good
at predicting what word comes next in English text. But that requires
implicitly learning all of the above.

### What it can do after pretraining

Complete text. Give it a prompt, it continues:

```
Input:  "The dog ran across the"
Output: "park and jumped into the lake. He was a very happy dog."
```

It generates fluent English but has no concept of:
- Following instructions ("Summarize this article")
- Having a conversation (user/assistant turns)
- Calling functions or tools
- Being helpful, honest, or safe

It's a text-completion engine, not an assistant.

### Training settings

| Setting | Value | Why |
|---|---|---|
| Data | TinyStories (~500M tokens) | Clean English text |
| Learning rate | 3e-4 | High — learning from scratch |
| Steps | ~15,300 | Process all 500M tokens |
| Starting loss | ~10.4 | Random (1/32000 chance) |
| Ending loss | ~3.9 | Decent language model |

---

## Phase 2: SFT (Supervised Fine-Tuning) — Learning a Skill

### What data looks like

Structured conversations with tool use:

```
<user>What's the weather in Tokyo?</user>
<assistant><function_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</function_call></assistant>
<function_response>{"temp": 72, "condition": "sunny"}</function_response>
<assistant>The weather in Tokyo is 72°F and sunny.</assistant>
```

### What the model learns

By predicting the next token in these structured examples, it learns:

- **Conversation format:** `<user>` → `<assistant>` → back and forth
- **When to call functions:** User asks about weather → call get_weather
- **JSON structure:** `{"name": "...", "arguments": {...}}`
- **How to summarize results:** Function returns data → write natural response
- **Which function to pick:** Weather question → get_weather, not search_web

### The key insight

The model is STILL just doing next-token prediction. When it sees:

```
<user>What's the weather in Tokyo?</user>
<assistant><function_call>{"name": "
```

It learns to predict `get_weather` as the next token. It's not "reasoning"
about which function to call — it's pattern-matching from thousands of
examples where weather questions led to get_weather calls.

But this pattern-matching is surprisingly effective. With enough examples,
the model generalizes to new questions it hasn't seen.

### Why a low learning rate?

SFT uses ~15x lower LR than pretraining (2e-5 vs 3e-4) because:

- **Don't destroy English.** High LR would overwrite the language knowledge
  from pretraining. The model would learn JSON format but forget grammar.
- **Small dataset.** Only ~50M tokens of SFT data. High LR + small data
  = overfitting (memorizes examples instead of generalizing).
- **Nudge, don't rebuild.** SFT is teaching a new format, not a new language.
  The weights just need small adjustments.

Analogy: Pretraining is teaching a child to speak English (takes years).
SFT is teaching them to write formal emails (takes a few lessons).
You don't re-teach English — you just show them the format.

### Training settings

| Setting | Value | Why |
|---|---|---|
| Data | Glaive function-calling v2 (~50M tokens) | Structured tool-use conversations |
| Learning rate | 2e-5 | Low — preserve English knowledge |
| Steps | ~5,000 | Enough to learn the format |
| Starting loss | ~4.2 | Starts from pretrained knowledge |
| Ending loss | ~1.8 | Good at predicting tool-use patterns |
| Checkpoint | `jrlam_pretrained.bin` | Must start from pretrained weights |

---

## Side-by-Side Comparison

| | Pretraining | SFT |
|---|---|---|
| **Goal** | Learn language | Learn to follow instructions + use tools |
| **Data** | Raw text (stories, articles) | Structured conversations (user/assistant/tools) |
| **Tokens** | 500M | ~50M |
| **Learning rate** | 3e-4 (high) | 2e-5 (low) |
| **Starting point** | Random weights | Pretrained weights |
| **Loss start → end** | 10.4 → 3.9 | 4.2 → 1.8 |
| **Time** | ~10 hours | ~2 hours |
| **Analogy** | Teaching a child to read | Teaching them to fill out forms |
| **Forward pass code** | Identical | Identical |
| **Backward pass code** | Identical | Identical |
| **Optimizer code** | Identical | Identical |

The ONLY differences are:
1. Which `.bin` file the DataLoader reads
2. The learning rate
3. Whether `--checkpoint` is set

The model architecture, the forward pass, the loss function, the backward
pass, the optimizer — all completely identical. The "intelligence" comes
from the data, not the code.

---

## Why Two Phases? Why Not Just Train on SFT Data?

**Can't we skip pretraining and just train on function-calling data?**

No. Here's what happens:

| Approach | Result |
|---|---|
| Pretrain only | Fluent English, no tool-use ability |
| SFT only (no pretrain) | Broken English, memorized JSON templates |
| **Pretrain → SFT** | **Fluent English + correct tool-use** |

With only 50M tokens of SFT data, the model can't learn English AND
JSON format AND function selection simultaneously. 50M tokens isn't enough
to learn a language from scratch (needs ~500M+).

But 50M tokens IS enough to teach a model that already knows English
how to format its responses as function calls. It just needs to learn
the pattern, not the language.

---

## The Full JrLAM Pipeline

```
Random weights (knows nothing)
        │
        ▼
[Phase 1: Pretrain on TinyStories, 500M tokens, ~10 hours]
        │
        ▼
Pretrained model (fluent English, no instructions)
        │
        ▼
[Phase 2: SFT on Glaive function-calling, 50M tokens, ~2 hours]
        │
        ▼
JrLAM (fluent English + follows instructions + calls functions)
```

Total: ~12 hours on a single RTX 3060. Not bad for a from-scratch LAM.

---

## Common Questions

**Q: Does the model "understand" anything?**
A: No. It predicts the next token. But next-token prediction at scale
requires implicitly modeling grammar, facts, reasoning patterns, etc.
It's "understanding" in the same way a weather model "understands" physics —
it doesn't, but it produces correct outputs because it learned the patterns.

**Q: Why is loss measured in nats, not percentage?**
A: Loss = `-log(probability)`. It's in nats (natural log units).
Loss 4.0 means the model assigns ~1.8% probability to the correct token
on average. Loss 2.0 means ~13.5%. Loss 1.0 means ~36.8%.

**Q: Can I do more pretraining later?**
A: Yes. Load the checkpoint and train on more data:
`./build/train data/more_data.bin --checkpoint checkpoints/jrlam_pretrained.bin`
The optimizer state (m, v) resets but the weights continue from where they were.

**Q: What if I want a smarter model?**
A: Three options, in order of effectiveness:
1. More pretraining data (2B+ tokens instead of 500M)
2. Bigger model (350M params — needs ~8 GB VRAM with fp16)
3. Higher quality SFT data (more diverse tool-use examples)
