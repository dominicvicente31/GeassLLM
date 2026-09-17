# Week 5 Knowledge Check — Real Training, Diagnosis, Iteration

This week is about reading a loss curve and making principled decisions.
Memorizing the vocabulary is not enough — you should be able to diagnose
a curve you've never seen and justify a specific fix.

---

## Reading Loss Curves

1. Your train loss is 1.1 and val loss is 2.9, and the gap is widening.
   What is this called? Name two hyperparameter changes you would try first and why.

2. Your train loss and val loss are both stuck at 2.5 after 3000 steps.
   What is this called? Name two things you would try and why.

3. Your val loss decreases, then plateaus, then slowly rises while train loss keeps falling.
   Sketch this curve. What is the name for the point at which val loss starts rising?

4. Your loss curve has sudden large spikes upward at random steps.
   What is likely causing this, and what is the standard fix?

5. Train loss and val loss both drop steeply early, then the gap between them slowly widens
   over the rest of training. Is this a problem? At what point does it become one?

## Interventions

6. Explain what dropout does mechanically (not just "it prevents overfitting").
   Why does randomly zeroing activations during training help generalization?

7. If you increase `n_layer` from 3 to 6, what are you changing about the model's capacity?
   When would this help, and when would it hurt?

8. If you increase `block_size` from 8 to 64, what changes about what the model can learn?
   What is the computational cost of doing this?

9. A char-level model needs significantly more data than a word-level model to produce
   coherent text at the same quality level. Explain why.

## Lelouch-Specific

10. What would a loss curve from a model that has genuinely learned Lelouch's dialogue style
    look like, compared to one that has just memorized training sequences?
    How would you tell the difference from the numbers alone?

11. After training, you sample from the model and it produces generic-sounding text rather
    than anything that sounds like Lelouch. Name three possible causes and how you would
    investigate each.
