# Week 6 Knowledge Check — Eval, Visualization, Polish

---

## Attention Visualization

1. When you visualize an attention head's weight matrix, what exactly are you looking at?
   What does a bright cell at position (row i, col j) mean?

2. Different attention heads often specialize. Give two examples of patterns a head might learn
   in a dialogue corpus (e.g., attending to the previous turn boundary, attending to pronouns).

3. If every head in every layer produces nearly uniform attention weights, what does that suggest
   about the model?

## Temperature and Sampling

4. What is temperature in text generation? Write the modified softmax formula that incorporates it.

5. What happens to the output distribution as temperature → 0? As temperature → ∞?
   What is the name for temperature → 0 decoding?

6. You notice that at temperature 1.0 the model sounds generic, but at temperature 0.3 it starts
   repeating phrases. What tradeoff is temperature controlling, and where is the sweet spot likely to be?

7. What is top-k sampling? How does it differ from pure temperature sampling, and why might it
   produce better results for staying in character?

## Qualitative Evaluation

8. List three concrete things you would check to decide whether the model has learned
   Lelouch's style, not just general English.

9. The model occasionally generates `[USER]:` mid-response, breaking the conversation format.
   What is happening, and what could you do about it at inference time (no retraining)?

10. What does it mean for a language model to "hallucinate," and is that concept applicable here?
    Can this model hallucinate in the same way a large instruction-tuned model can?

## The README / Report

11. Your project README needs to justify the architectural choices you made.
    For each of the following, write one sentence explaining the *why*:
    - Causal masking
    - Residual connections
    - Pre-norm instead of post-norm
    - Char-level tokenization instead of BPE

12. What is perplexity, and how does it relate to cross-entropy loss?
    Write the formula and explain what a perplexity of 5 means in plain language.
