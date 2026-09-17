import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Toy text ---
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

# --- Hyperparameters ---
batch_size    = 32
block_size    = 8
max_iters     = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters    = 100
n_embd        = 32   # embedding dimension
head_size     = n_embd  # single head gets the full embedding dimension
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
    x = torch.stack([d[i : i + block_size]         for i in ix])
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


# ---------------------------------------------------------------------------
# Single-head self-attention
# ---------------------------------------------------------------------------
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        # W_Q, W_K, W_V are learned linear projections — no bias, standard practice
        self.W_Q = nn.Linear(n_embd, head_size, bias=False)
        self.W_K = nn.Linear(n_embd, head_size, bias=False)
        self.W_V = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular mask stored as a buffer (not a trainable parameter)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _, T, _ = x.shape

        # Step 1 — project into query / key / value spaces
        Q = self.W_Q(x)   # (B, T, head_size)  "what am I looking for?"
        K = self.W_K(x)   # (B, T, head_size)  "what do I have to offer?"
        V = self.W_V(x)   # (B, T, head_size)  "what do I actually send?"

        # Step 2 — scaled dot-product scores: how much does each Q match each K?
        # Scaling by 1/sqrt(head_size) keeps the variance of the dot products ≈ 1,
        # which prevents softmax from saturating into near-zero gradients.
        scores = Q @ K.transpose(-2, -1) * head_size**-0.5  # (B, T, T)

        # Step 3 — causal mask
        # Set future positions to -inf so softmax gives them zero weight.
        # Without this, position t can see position t+1 during training, but that
        # token doesn't exist yet during generation — it would be data leakage.
        # BERT-style bidirectional models skip this mask intentionally; we need it
        # because we generate left-to-right.
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Step 4 — softmax: turn scores into a probability distribution per row
        weights = F.softmax(scores, dim=-1)  # (B, T, T)

        # Step 5 — weighted sum of values
        out = weights @ V  # (B, T, head_size)
        return out

    # --- PyTorch shortcut (steps 2–5 fused into one call) ---
    # def forward(self, x):
    #     B, T, C = x.shape
    #     Q = self.W_Q(x)
    #     K = self.W_K(x)
    #     V = self.W_V(x)
    #     return F.scaled_dot_product_attention(Q, K, V, is_causal=True)


# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------
class LanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        # Positional embedding is necessary because attention is permutation-invariant:
        # without it, "abc" and "bca" produce identical attention outputs.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(head_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                                # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x       = tok_emb + pos_emb   # broadcast: (B, T, n_embd)
        x       = self.sa_head(x)     # (B, T, head_size)
        logits  = self.lm_head(x)     # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]   # attention window can't exceed block_size
            logits, _ = self(idx_cond)
            probs    = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# --- Training ---
model = LanguageModel().to(device)
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
