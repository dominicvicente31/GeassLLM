# GeassLLM

A GPT-style language model built from scratch and trained on Lelouch vi Britannia's dialogue from *Code Geass*. The end goal is a conversational interface — you send a message, the model responds in Lelouch's voice.

Built to deeply understand transformer internals, not just run a demo. Every architectural decision has a reason.

## What this is

A character-level autoregressive transformer, built incrementally from the ground up:

- Bigram language model (baseline)
- Single-head self-attention with causal masking (Q, K, V, scaled dot-product)
- Multi-head attention
- Full transformer block: MLP, residual connections, LayerNorm (pre-norm)
- Stacked transformer blocks with configurable depth
- Conversational generation via prompt-conditioned inference — no instruction tuning required

The model learns Lelouch's voice from his dialogue corpus. At inference, a `[USER]: ...\n[LELOUCH]: ` prefix seeds the generation, producing in-character responses without any separate fine-tuning step.

## Architecture

| Component | Detail |
|---|---|
| Tokenization | Character-level |
| Attention | Multi-head, causal masking |
| Block | Pre-norm transformer (MHA → FFN, residuals on both) |
| FFN | 4× expansion, GELU |
| Depth | Configurable via `n_layer` |
| Optimizer | AdamW |

## Project structure

```
train.py                        — model architecture and training loop
GeassLLM-ROADMAP.md             — week-by-week build plan
GeassLLM-dev-KnowledgeChecks/   — weekly knowledge checks (questions only)
data/scripts/                   — Lelouch dialogue scripts, one file per episode
```

## Current state

Full transformer block implemented (MHA + FFN + residuals + LayerNorm, stacked `n_layer` deep). Training on toy text. Lelouch corpus and real training run in progress.

## Running it

Requires Python 3.11+ and PyTorch:

```
pip install torch
py train.py
```

## Notes

- Trained on *Code Geass* scripts — not for distribution due to copyright
- Built for learning and portfolio purposes
