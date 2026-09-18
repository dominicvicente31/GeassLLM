# Lelouch-LLM: Project Roadmap

A from-scratch GPT-style language model trained on Lelouch's dialogue,
built to actually learn transformer internals (not just ship a demo).

**End goal:** a conversational interface — you send a message, the model
responds in Lelouch's voice. Achieved via prompt-conditioned generation:
training data is formatted as `[USER]: ...\n[LELOUCH]: ...` turns, so the
model learns the Q&A pattern from the corpus itself (no architectural change
needed, no separate instruction-tuning step).

## Project Vision (tiered)

```
Tier 1 (core)     -> Lelouch-LLM: from-scratch char-level GPT trained
                      on his dialogue, with a conversational chat interface
Tier 2 (stretch)  -> Chess algorithm tuned to mimic his playstyle,
                      with the LLM generating in-character move commentary
Tier 3 (aspirational, later) -> TouchDesigner visual system that reacts
                      to the LLM (e.g. a "Geass activation" trigger)
```

Only Tier 1 is scoped week-by-week below. Tier 2 starts only after
Tier 1 is finished and written up -- not in parallel. Tier 3 is a
possible future layer once there's something (text output, attention
weights) worth piping into it.

---

## Week 1 — Language modeling fundamentals ✓

- ✓ Implement a bigram language model on toy text.
- ✓ Understand *why* a bigram model can't capture long-range structure.
- **Checkpoint:** explain in your own words why a bigram model can't
  produce coherent dialogue.

## Week 2 — Self-attention, from math to code ✓

- ✓ Implement single-head self-attention from the raw matrix math (Q,
  K, V, scaled dot-product, softmax) with the PyTorch shortcut shown
  as a commented alternative.
- ✓ Implement causal masking; understand why it's needed for
  autoregressive generation (vs. bidirectional attention like BERT).
- ✓ Extend to multi-head attention.
- **Checkpoint:** draw the attention matrix for a 4-token sequence and
  what masking removes from it.

## Week 3 — Transformer block + corpus pipeline ✓

Architecture:
- ✓ MLP/feedforward block (4× expansion, GELU activation).
- ✓ Residual connections and pre-norm LayerNorm on both sublayers.
- ✓ Full transformer block assembled; stacked `n_layer` deep with a
  final LayerNorm before the output head (GPT-2 style).
- ✓ Understand why residuals + LayerNorm matter for training stability
  at depth.

Corpus pipeline:
- ✓ Assembled 49 episodes of Lelouch-only dialogue (155,608 characters)
  sourced and cleaned manually into per-episode `.txt` files.
- ✓ Inspected corpus: 44,523 BPE tokens, ~130 unique characters.

## Week 4 — Full model + training loop ✓

- ✓ Stacked transformer blocks with configurable depth (`n_layer`).
- ✓ Token + positional embeddings, output head (`lm_head`).
- ✓ Training loop with cross-entropy loss and AdamW.
- ✓ LR schedule (linear warmup + cosine decay).
- ✓ Gradient clipping (`torch.nn.utils.clip_grad_norm_`).
- ✓ `chat()` method seeds generation from `"[USER]: ...\n[LELOUCH]: "` prefix.
- ✓ Full training run on Lelouch corpus via Google Colab (GPU).
- **Checkpoint:** intentionally break something (remove residuals,
  remove masking) and observe how training degrades.

## Week 5 — Real training, diagnosis, iteration ✓

- ✓ Trained from-scratch model on full corpus; tracked train/val loss.
- ✓ Diagnosed overfitting: original config (n_layer=6, dropout=0.2)
  produced train/val gap of 0.40 at 5k steps.
- ✓ Iterated: reduced n_layer 6→4, increased dropout 0.2→0.25,
  added weight_decay=0.05, extended to 10k steps.
  Final gap: 0.08 (train 1.40, val 1.48).
- ✓ Identified data ceiling (~155K chars) as the primary bottleneck.
- ✓ Implemented GPT-2 fine-tuning (`finetune_gpt2.py`) as the path
  past the data ceiling: pretrained weights provide English fluency,
  fine-tuning adapts to Lelouch's style. Output quality is
  significantly better from the same corpus.
- **Focus skill:** reading a loss curve and making informed changes --
  arguably the most interview-relevant skill in the project.

## Week 6 — Eval, visualization, polish

- Add attention visualization (which tokens the model attends to) and
  a clean loss-curve plot for the writeup.
- Qualitative conversation testing at different sampling temperatures:
  does the model stay in character? Does it maintain the `[LELOUCH]:` 
  response boundary, or bleed past it?
- Write the README/report: architecture decisions, what worked, what
  didn't, and why. This document carries as much portfolio weight as
  the code.

## Week 7 — Conversational chat interface

- Gradio or Streamlit chat UI: text input box, response displayed in
  character as Lelouch, conversation history fed back as context.
- This is the deliverable that makes the project feel real -- not a
  stretch goal. It's the difference between "I trained a model" and
  "I built something you can talk to."

## Weeks 8+ (separate phase, not parallel) — Chess algorithm

- Only start once Tier 1 is finished and written up.
- `python-chess` + minimax/alpha-beta, tuned toward Lelouch's
  playstyle; LLM generates in-character commentary on moves.

---

## Throughline

At every week, before moving to the next section: be able to explain
the *why* out loud, not just "it worked." That's what separates a
portfolio piece that survives an interviewer's follow-up questions
from one that doesn't.
