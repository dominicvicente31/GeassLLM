# Week 3 Knowledge Check — Transformer Block + Corpus Pipeline

---

## Feedforward / MLP Block

1. Attention handles communication between positions — each token can look at others.
   What does the feedforward block do that attention doesn't?

2. The FFN expands from `n_embd` to `4 * n_embd` before projecting back down.
   Where does the 4x factor come from, and what is the effect of that bottleneck shape?

3. The original Transformer used ReLU; GPT-2 and later models use GELU.
   What is the practical difference, and why might GELU be preferred?

## Residual Connections

4. What problem do residual connections (skip connections) solve in deep networks?
   Describe what happens to gradients in a 12-layer network without them.

5. In `Block.forward()`, the pattern is `x = x + sublayer(x)`.
   What would change if it were just `x = sublayer(x)`?

6. Residuals are sometimes described as creating an "ensemble" of shallow models.
   Explain what that means.

## LayerNorm

7. What does LayerNorm normalize, and over which dimensions?

8. How does LayerNorm differ from BatchNorm? Why is LayerNorm preferred for sequence models?

9. This implementation uses pre-norm: `x = x + sublayer(LayerNorm(x))`.
   The original Transformer paper used post-norm: `x = LayerNorm(x + sublayer(x))`.
   What is the practical difference, and why has pre-norm become standard?

## Full Block

10. Trace a single token's vector through one full `Block.forward()` pass.
    List every operation in order and the shape of the tensor after each step.

## Corpus Pipeline

11. The training data must be formatted as `[USER]: ...\n[LELOUCH]: ...` turn pairs.
    Why does the data format alone teach the model to respond conversationally,
    without any separate instruction-tuning step?

12. What would you check when inspecting corpus quality after running
    `extract_dialogue.py` → `filter_and_clean.py`?
    Name at least three things that could make the corpus unusable or poor quality.
