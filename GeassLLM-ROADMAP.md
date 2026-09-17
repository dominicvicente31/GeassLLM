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

- Implement a bigram language model on toy text (don't just follow
  along with Karpathy -- type it yourself).
- Understand *why* a bigram model can't capture long-range structure.
- **Checkpoint:** explain in your own words why a bigram model can't
  produce coherent dialogue.

## Week 2 — Self-attention, from math to code ✓

- Implement single-head self-attention from the raw matrix math (Q,
  K, V, scaled dot-product, softmax) before jumping to the PyTorch
  shortcut version.
- Implement causal masking; understand why it's needed for
  autoregressive generation (vs. bidirectional attention like BERT).
- Extend to multi-head attention.
- **Checkpoint:** draw the attention matrix for a 4-token sequence and
  what masking removes from it.

## Week 3 — Transformer block + corpus pipeline

- Add the MLP/feedforward block, residual connections, LayerNorm to
  complete the transformer block.
- Understand why residuals + LayerNorm matter for training stability
  at depth (common interview question -- be able to answer it, not
  just cite it).
- **Corpus work:** run `extract_dialogue.py` → `filter_and_clean.py` on
  Code Geass scripts. Output format must be `[USER]: ...\n[LELOUCH]: ...`
  turn pairs -- this is the pattern the model learns conversation from.
  Inspect corpus quality: token count, turn count, coverage of topics.

## Week 4 — Full model + training loop

- Stack blocks into the complete GPT: positional embeddings, output
  head, configurable depth.
- Write the training loop: cross-entropy loss, AdamW, LR schedule,
  gradient clipping.
- Update `generate()` to accept a prompt prefix string rather than
  empty context -- this is what enables conversation at inference time:
  `"[USER]: <input>\n[LELOUCH]: "` becomes the seed.
- Run `prepare.py` on the cleaned corpus, do a first small training
  run (expect bad output -- that's fine).
- **Checkpoint:** intentionally break something (remove residuals,
  remove masking) and observe how training degrades.

## Week 5 — Real training, diagnosis, iteration

- Train on the full Lelouch corpus, track train/val loss.
- Diagnose overfitting (val loss rising while train falls) vs.
  underfitting; adjust model size, dropout, context length.
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
