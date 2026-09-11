# Lelouch-LLM: Project Roadmap

A from-scratch GPT-style language model trained on Lelouch's dialogue,
built to actually learn transformer internals (not just ship a demo).

## Project Vision (tiered)

```
Tier 1 (core)     -> Lelouch-LLM: from-scratch char-level GPT trained
                      on his dialogue from the show/movie scripts
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

## Week 1 — Language modeling fundamentals

- Implement a bigram language model on toy text (don't just follow
  along with Karpathy -- type it yourself).
- Understand *why* a bigram model can't capture long-range structure.
- **Checkpoint:** explain in your own words why a bigram model can't
  produce coherent dialogue.

## Week 2 — Self-attention, from math to code

- Implement single-head self-attention from the raw matrix math (Q,
  K, V, scaled dot-product, softmax) before jumping to the PyTorch
  shortcut version.
- Implement causal masking; understand why it's needed for
  autoregressive generation (vs. bidirectional attention like BERT).
- **Checkpoint:** draw the attention matrix for a 4-token sequence and
  what masking removes from it.

## Week 3 — Multi-head attention + transformer block

- Extend to multi-head attention; add the MLP/feedforward block,
  residual connections, LayerNorm.
- Understand why residuals + LayerNorm matter for training stability
  at depth (common interview question -- be able to answer it, not
  just cite it).
- **Parallel data work:** run `extract_dialogue.py` -> `filter_and_clean.py`
  on the scripts, inspect corpus quality.

## Week 4 — Full model + training loop

- Stack blocks into the complete GPT: positional embeddings, output
  head.
- Write the training loop: cross-entropy loss, AdamW, LR schedule,
  gradient clipping.
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
- Qualitative generation testing at different sampling temperatures.
- Write the README/report: architecture decisions, what worked, what
  didn't, and why. This document carries as much portfolio weight as
  the code.

## Week 7 (stretch, cut without guilt if behind) — Interactive demo

- Gradio/Streamlit interface for live generation.
- First genuinely optional week in the plan.

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
