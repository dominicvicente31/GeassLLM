import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

# --- Corpus ---
scripts_dir = Path(__file__).parent / "data" / "scripts"
script_files = [f for f in sorted(scripts_dir.glob("*.txt")) if f.stat().st_size > 0]

if not script_files:
    raise FileNotFoundError(f"No non-empty .txt files found in {scripts_dir}. Add episode scripts before training.")

text = "\n".join(f.read_text(encoding="utf-8") for f in script_files)
print(f"Loaded {len(script_files)} episode(s) — {len(text):,} characters")

# --- Character-level tokenizer ---
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)

# --- Hyperparameters ---
batch_size    = 32
block_size    = 128  # ~2-4 lines of dialogue per context window
max_iters     = 10000
eval_interval = 500
learning_rate = 1e-3
eval_iters    = 100
n_embd        = 128  # embedding dimension
n_head        = 4    # number of attention heads; each gets n_embd // n_head = 32 dims
n_layer       = 4    # transformer depth
dropout       = 0.25 # fraction of activations zeroed during training
warmup_iters  = 200  # steps over which LR ramps from 0 → learning_rate
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


def get_lr(step):
    # Linear warmup: avoids a large gradient spike on step 0 when weights are random.
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    # Cosine decay from learning_rate → 0 over the remaining steps.
    progress = (step - warmup_iters) / (max_iters - warmup_iters)
    return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


# Single-head self-attention
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        # W_Q, W_K, W_V are learned linear projections — no bias, standard practice
        self.W_Q = nn.Linear(n_embd, head_size, bias=False)
        self.W_K = nn.Linear(n_embd, head_size, bias=False)
        self.W_V = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular mask stored as a buffer (not a trainable parameter)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape

        # Step 1 — project into query / key / value spaces
        Q = self.W_Q(x)   # (B, T, head_size)  "what am I looking for?"
        K = self.W_K(x)   # (B, T, head_size)  "what do I have to offer?"
        V = self.W_V(x)   # (B, T, head_size)  "what do I actually send?"

        # Step 2 — scaled dot-product scores: how much does each Q match each K?
        # Scaling by 1/sqrt(head_size) keeps the variance of the dot products ≈ 1,
        # which prevents softmax from saturating into near-zero gradients.
        scores = Q @ K.transpose(-2, -1) * self.head_size**-0.5  # (B, T, T)

        # Step 3 — causal mask
        # Set future positions to -inf so softmax gives them zero weight.
        # Without this, position t can see position t+1 during training, but that
        # token doesn't exist yet during generation — it would be data leakage.
        # BERT-style bidirectional models skip this mask intentionally; we need it
        # because we generate left-to-right.
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Step 4 — softmax: turn scores into a probability distribution per row
        weights = F.softmax(scores, dim=-1)  # (B, T, T)
        weights = self.attn_dropout(weights)

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


# Multi-head attention
class MultiHeadAttention(nn.Module):
    """Run n_head attention heads in parallel, then concatenate and project."""

    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        # Project the concatenated output back to n_embd so the residual
        # stream dimension stays constant throughout the model.
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Each head produces (B, T, head_size); cat along the last dim → (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj_dropout(self.proj(out))


# Feedforward / MLP block
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        # 4x expansion follows the original Transformer paper ("Attention Is All You Need").
        # Attention is about communication between positions; this MLP is about
        # each position independently processing what it gathered from attention.
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Transformer block: attention + feedforward with residuals and LayerNorm
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa  = MultiHeadAttention(n_head, head_size)
        self.ff  = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-norm: normalize *before* the sublayer rather than after.
        # Original Transformer used post-norm, but pre-norm is empirically more
        # stable at depth because the residual path stays unscaled.
        # Residuals let gradients flow directly back through the network,
        # making deep stacks trainable where they'd otherwise vanish.
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# Language model
class LanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        # Positional embedding is necessary because attention is permutation-invariant:
        # without it, "abc" and "bca" produce identical attention outputs.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)  # final norm before lm_head (GPT-2 style)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                                # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x       = tok_emb + pos_emb   # broadcast: (B, T, n_embd)
        x       = self.blocks(x)      # n_layer transformer blocks
        x       = self.ln_f(x)
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

    def chat(self, user_message, max_new_tokens=200):
        # Requires a corpus trained with [USER]:/[LELOUCH]: turn format.
        # Will raise ValueError on the toy corpus — [, ], : are not in vocab.
        prompt = f"[USER]: {user_message}\n[LELOUCH]: "
        try:
            idx = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        except KeyError as e:
            raise ValueError(
                f"Character {e} not in vocab. chat() requires a corpus "
                "formatted with [USER]:/[LELOUCH]: turns."
            ) from e

        out  = self.generate(idx, max_new_tokens)
        full = decode(out[0].tolist())

        # Extract only the final [LELOUCH]: response
        marker = "[LELOUCH]: "
        start  = full.rfind(marker)
        if start == -1:
            return full.strip()
        response = full[start + len(marker):]
        # Truncate if the model bleeds into the next [USER]: turn
        cutoff = response.find("[USER]:")
        if cutoff != -1:
            response = response[:cutoff]
        return response.strip()


# --- Training ---
model = LanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step:4d}: train loss {losses['train']:.4f}  val loss {losses['val']:.4f}")

    for param_group in optimizer.param_groups:
        param_group['lr'] = get_lr(step)

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# --- Sample (toy corpus) ---
print("\n--- generated text ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))

# --- Conversation (uncomment once trained on real corpus) ---
# print(model.chat("What is your true goal, Zero?"))
