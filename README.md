# GeassLLM

A GPT-style language model built from scratch and trained on Lelouch vi Britannia's dialogue from *Code Geass*. The goal is to deeply understand transformer internals — not just run a demo, but be able to explain every architectural decision.

## What this is

This project builds a character-level autoregressive transformer, incrementally, from the ground up:

- Bigram language model (baseline)
- Single-head self-attention with causal masking
- Multi-head attention
- Full transformer block (MLP, residual connections, LayerNorm)
- Complete GPT training loop on the Lelouch corpus

The model learns to generate text in Lelouch's speaking style — declarative, strategic, theatrical.

## Project structure

```
train.py              — model architecture and training loop (evolves each week)
GeassLLM-ROADMAP.md   — week-by-week build plan
```

## Current state

Single-head self-attention with causal masking implemented. Multi-head attention added. Training on toy text. Full corpus and transformer block coming in subsequent weeks.

## Running it

Requires Python 3.11+ and PyTorch:

```
pip install torch
py train.py
```

## Notes

- Trained on publicly available *Code Geass* scripts
- Not for distribution due to copyright
- Built for learning and portfolio purposes
