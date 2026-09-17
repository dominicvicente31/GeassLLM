import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Toy text ---
# A small self-contained corpus, enough to see bigram patterns emerge.
# Replace this with the Lelouch corpus once we get there (Week 4).
text = (
    "All men are not created equal. Some are born swifter afoot, some with greater beauty, "
    "some are born into poverty, and others are born sick and feeble. "
    "Both in birth and in upbringing, in sheer scope of ability, every human is inherently different. "
    "Yes, that is why people discriminate against one another, which is why there is struggle, "
    "competition, and the unfaltering march of progress. "
    "Inequality is not wrong -- it is the very foundation of civilization. "
    "What I propose is the elimination of inequality itself. "
    "No more. I will stand in the way of that inequality. "
    "I hereby proclaim: all people are created equal, and I will enforce that principle "
    "by any means necessary. This is my declaration. This is my absolute command. "
    "The sword that strikes down falsehood shall be mine. I am Zero, the man who will remake the world."
)

# --- Character-level tokenizer ---
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)

# --- Hyperparameters (bigram-appropriate, much smaller than the full GPT) ---
batch_size = 32
block_size = 8       # context length; bigram only uses the last token anyway
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2  # bigram is shallow, so a higher LR is fine
eval_iters = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

# --- Train / val split ---
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size]     for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# --- Model ---
class BigramLanguageModel(nn.Module):
    """
    Simplest possible language model: each token predicts the next token
    purely from itself via a learned embedding table.  No attention, no
    memory of tokens further back than 1.
    """
    def __init__(self):
        super().__init__()
        # Row i of this table = logits over the next character given character i
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            # Bigram: only the last token's logits matter
            probs    = F.softmax(logits[:, -1, :], dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# --- Training ---
model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step:4d}: train loss {losses['train']:.4f}  val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- Sample ---
print("\n--- generated text ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))
