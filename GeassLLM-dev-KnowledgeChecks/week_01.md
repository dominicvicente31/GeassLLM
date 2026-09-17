# Week 1 Knowledge Check — Language Modeling Fundamentals

Answer these out loud or in writing before moving to Week 2.
The goal is explanation, not recall — if you can only restate a definition, dig deeper.

---

## The Bigram Model

1. What does a bigram model predict, and what is its only input at each step?

2. Why can't a bigram model produce coherent dialogue, even if trained on a large corpus?
   What structural property of language is it blind to?

3. If you doubled the training data for a bigram model, would coherence improve significantly?
   Why or why not?

## Loss and Optimization

4. What is cross-entropy loss? Why is it used for language modeling instead of, say, MSE?

5. At initialization (random weights, uniform logits), what should the loss be approximately equal to,
   given a vocab size of 65? Show the math.

6. What does it mean for loss to decrease during training? What is the model actually learning?

## Train / Val Split

7. Why do we hold out a validation set instead of training on all available data?

8. If your training loss is 1.1 and your validation loss is 1.1 after 1000 steps, is that good or bad?
   What if validation loss is 2.4?

## Thinking Ahead

9. A bigram model looks back one token. What would you need to look back 8 tokens? 100 tokens?
   Why does this get expensive fast with naive approaches?

10. In your own words: what is the single biggest limitation of the bigram model that
    self-attention is designed to solve?
