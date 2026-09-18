import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

# --- Corpus ---
scripts_dir = Path(__file__).parent / "data" / "scripts"
# In Colab, replace the line above with:
# scripts_dir = Path("/content/drive/MyDrive/GeassLLM/data/scripts")

script_files = [f for f in sorted(scripts_dir.glob("*.txt")) if f.stat().st_size > 0]
if not script_files:
    raise FileNotFoundError(f"No non-empty .txt files found in {scripts_dir}")

text = "\n".join(f.read_text(encoding="utf-8") for f in script_files)
print(f"Loaded {len(script_files)} episode(s) — {len(text):,} characters")

# --- Tokenizer & Model ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# --- Dataset ---
class TextDataset(Dataset):
    def __init__(self, tokens, block_size=256):
        self.examples = [
            torch.tensor(tokens[i : i + block_size], dtype=torch.long)
            for i in range(0, len(tokens) - block_size, block_size)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


# --- Hyperparameters ---
block_size    = 256
batch_size    = 8
num_epochs    = 5
learning_rate = 5e-5
warmup_steps  = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Train / val split (90/10 on token level) ---
tokens = tokenizer.encode(text)
print(f"Tokenized: {len(tokens):,} tokens  (vocab size: {tokenizer.vocab_size:,})")

split = int(0.9 * len(tokens))
train_dataset = TextDataset(tokens[:split], block_size)
val_dataset   = TextDataset(tokens[split:], block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)


@torch.no_grad()
def eval_loss():
    model.eval()
    total = 0
    for batch in val_loader:
        inputs = batch.to(device)
        total += model(inputs, labels=inputs).loss.item()
    model.train()
    return total / len(val_loader)


# --- Training ---
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        inputs = batch.to(device)
        loss = model(inputs, labels=inputs).loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    val_loss   = eval_loss()
    print(f"Epoch {epoch+1:2d}/{num_epochs}: train loss {train_loss:.4f}  val loss {val_loss:.4f}")

# --- Generate sample ---
model.eval()
prompt = "I am Zero, and my power of Geass"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=200,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
print("\n--- Generated ---")
print(tokenizer.decode(output[0], skip_special_tokens=True))

# --- Save to Drive (uncomment in Colab) ---
# model.save_pretrained("/content/drive/MyDrive/GeassLLM/gpt2_lelouch")
# tokenizer.save_pretrained("/content/drive/MyDrive/GeassLLM/gpt2_lelouch")
