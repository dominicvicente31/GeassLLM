# Week 7 Knowledge Check — Conversational Chat Interface

---

## Prompt-Conditioned Generation

1. This model was never explicitly trained to follow instructions — there is no RLHF,
   no system prompt, no instruction tuning. Explain exactly how it still produces
   conversational responses in Lelouch's voice.

2. What is the difference between this approach and fine-tuning an existing instruction-tuned
   model (e.g., GPT-3.5) on Lelouch's dialogue? What does each approach give up?

3. The `chat()` method feeds the full `[USER]: ...\n[LELOUCH]: ` prefix as the generation seed.
   What would happen if you passed only `[LELOUCH]: ` with no user message?

## Context and Memory

4. A user has a 10-turn conversation. How does the interface feed conversation history back
   to the model? What is the hard limit on how much history can be included, and what happens
   when you exceed it?

5. Describe two strategies for handling context overflow (when the conversation history exceeds
   `block_size`). What does each one lose?

6. Does this model have any "memory" of conversations between sessions?
   If a user comes back the next day, what does the model know about them?

## The Interface

7. You're building a Gradio chat UI. What state needs to persist across turns,
   and what gets reconstructed fresh on every generation call?

8. A user types a very long message — longer than `block_size`. What happens?
   How would you handle this gracefully in the UI?

9. What is streaming output in a chat interface, and why does it make the experience
   feel more responsive? Can this model support it as currently implemented?
   What would you need to change?

## End-to-End

10. Walk through the full lifecycle of a single user message from keypress to displayed response:
    UI input → encoding → context construction → generate() → decoding → response extraction → display.
    Identify every place where something could go wrong.

11. What makes this project a portfolio piece worth talking about in a technical interview,
    beyond "I trained a language model"? Write the 3-sentence pitch you would give.
