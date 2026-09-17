# Week 2 Knowledge Check — Self-Attention, From Math to Code

---

## Q, K, V

1. What do Query, Key, and Value each represent conceptually?
   Don't just name them — explain what role each plays in the computation.

2. Why are the Q, K, V projections linear layers with `bias=False`?
   What would adding bias change, if anything?

3. The attention scores are computed as `Q @ K.T * head_size**-0.5`.
   Why the scaling factor? What goes wrong without it, and why does it hurt training?

## The Attention Matrix

4. For a sequence of length T=4 with tokens ["I", " ", "a", "m"], sketch the attention weight
   matrix after causal masking. Which cells are zero, and why?

5. What does a single row of the attention weight matrix represent?
   What does the model "decide" by choosing those weights?

6. After softmax, what property do the rows of the weight matrix have?
   Why is that property important?

## Causal Masking

7. Why does autoregressive generation require causal masking during training?
   What specific problem arises without it?

8. BERT is trained without causal masking — it uses bidirectional attention.
   Why is that valid for BERT but not for a text-generation model?

## Multi-Head Attention

9. What does running multiple attention heads in parallel give you that a single head doesn't?
   Give a concrete example of what different heads might learn to attend to.

10. After concatenating the head outputs, why do we apply a final linear projection?
    What dimension does it map from and to, and why does that matter?

## Checkpoint

11. Implement single-head attention from memory (just the forward pass, no PyTorch shortcuts).
    If you have to look anything up, note which step — that's exactly what to review.
