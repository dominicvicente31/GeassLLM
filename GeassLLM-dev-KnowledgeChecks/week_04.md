# Week 4 Knowledge Check — Full Model + Training Loop

---

## Positional Embeddings

1. Why do we need positional embeddings at all?
   What specific property of attention makes position information invisible without them?

2. The model uses a learned positional embedding table rather than the fixed sinusoidal
   encoding from the original Transformer paper. What is the tradeoff between the two?

3. What would the model output if you shuffled the tokens in a sequence but kept the
   positional embeddings fixed to their original positions? What if you shuffled both?

## The Full Stack

4. Trace a batch of shape `(B=2, T=8)` through the full `LanguageModel.forward()`.
   Write out the tensor shape after each stage: token embed → pos embed → add → blocks → ln_f → lm_head.

5. Why is there a final `LayerNorm` (`ln_f`) applied after all the blocks but before `lm_head`?
   What would happen if you removed it?

## Training Loop

6. What is AdamW, and what does the "W" specifically add over standard Adam?
   Why does that matter for language models with large embedding tables?

7. What is gradient clipping? What symptom in training would tell you that you need it?
   What is a typical clip value and why?

8. What is a learning rate schedule? Describe one common shape (e.g., warmup + decay)
   and explain why each phase exists.

## Checkpoint Exercise

9. The roadmap says: "intentionally break something and observe how training degrades."
   For each modification below, predict what you expect to happen to the loss curve and why:
   - Remove all residual connections
   - Remove causal masking
   - Set learning rate to 1.0
   - Remove positional embeddings entirely

## Prompt-Conditioned Generation

10. Before this week's changes, `generate()` started from `torch.zeros((1,1))` — an empty context.
    Explain step by step how the new prompt-prefix approach enables conversation:
    what is passed in, what does the model see, and how does the response get extracted?

11. What limits how long the conversation history can be when feeding it back as context?
    What happens when that limit is exceeded?
