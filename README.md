# GeassLLM

A GPT-style language model built from scratch and fine-tuned on Lelouch vi Britannia's dialogue from *Code Geass*. The end goal is a conversational interface — you send a message, the model responds in Lelouch's voice.

Built to deeply understand transformer internals, not just run a demo. Every architectural decision has a reason.

## What this is

Two parallel tracks:

**Track 1 — From scratch (`train.py`)**
A character-level autoregressive transformer built incrementally:
- Bigram language model (baseline)
- Single-head self-attention with causal masking (Q, K, V, scaled dot-product)
- Multi-head attention
- Full transformer block: MLP, residual connections, LayerNorm (pre-norm)
- Stacked transformer blocks with configurable depth
- LR schedule (linear warmup + cosine decay), gradient clipping, dropout
- Trained on 49 episodes / 155K characters of Lelouch dialogue on Google Colab

**Track 2 — GPT-2 fine-tuning (`finetune_gpt2.py`)**
Fine-tunes GPT-2 small on the same corpus via HuggingFace Transformers. Pretrained weights provide English fluency; fine-tuning adapts the style to Lelouch's voice. Produces significantly more coherent output from the same data.

## Architecture (from-scratch model)

| Component | Detail |
|---|---|
| Tokenization | Character-level |
| Attention | Multi-head, causal masking |
| Block | Pre-norm transformer (MHA → FFN, residuals on both) |
| FFN | 4× expansion, GELU |
| Depth | 4 layers (`n_layer`) |
| Embedding dim | 128 (`n_embd`) |
| Optimizer | AdamW + weight decay |
| LR schedule | Linear warmup + cosine decay |

## Training results

| Model | Train Loss | Val Loss | Train/Val Gap |
|---|---|---|---|
| From scratch (10k steps) | 1.40 | 1.48 | 0.08 |
| GPT-2 fine-tuned (5 epochs) | 2.51 | 2.99 | 0.49 (different scale) |

GPT-2 loss operates on a 50K BPE vocab vs 130-char vocab, so the numbers are not directly comparable. Output quality from GPT-2 fine-tuning is substantially better.

## Project structure

```
train.py                        — from-scratch character-level GPT
finetune_gpt2.py                — GPT-2 fine-tuning via HuggingFace
GeassLLM-ROADMAP.md             — week-by-week build plan
GeassLLM-dev-KnowledgeChecks/   — weekly knowledge checks (questions only)
data/scripts/                   — Lelouch dialogue scripts, one file per episode
```

## Running it

**From scratch:**
```
pip install torch
py train.py
```

**GPT-2 fine-tuning:**
```
pip install torch transformers
py finetune_gpt2.py
```

Both scripts are also designed to run on Google Colab with Google Drive mounted for data access.

## Notes

- Trained on *Code Geass* scripts — not for distribution due to copyright
- Built for learning and portfolio purposes
