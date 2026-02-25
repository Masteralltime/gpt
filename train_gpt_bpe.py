"""
train_gpt_bpe.py

Trains a GPT-style Transformer using a custom Byte-Pair Encoding (BPE) tokenizer.
Includes mixed-precision training (AMP) for optimized performance on CUDA devices.
"""

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tqdm import tqdm
from dotenv import load_dotenv

# Load configuration from environment variables
load_dotenv()
DATASET_PATH = os.getenv("BPE_DATASET_PATH", "conversation.txt")
TOKENIZER_SAVE_PATH = os.getenv("TOKENIZER_PATH", "tokenizer.json")

# Hyperparameters
N_EMBD = 1024
N_HEAD = 16
N_LAYER = 24
BATCH_SIZE = 16
BLOCK_SIZE = 256
DROPOUT = 0.2
LEARNING_RATE = 3e-4
MAX_ITERS = 10000
EVAL_INTERVAL = 500
EVAL_ITERS = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'
scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

def train_tokenizer(dataset_path, save_path):
    """Trains a BPE tokenizer from scratch using the provided text dataset."""
    print(f"Training tokenizer on {dataset_path}...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = BpeTrainer(
        vocab_size=10000,
        special_tokens=["<pad>", "<unk>"],
        initial_alphabet=[" "],
    )
    tokenizer.train([dataset_path], trainer)
    tokenizer.save(save_path)
    return tokenizer

# Initialize or load tokenizer
if not os.path.exists(TOKENIZER_SAVE_PATH):
    tokenizer = train_tokenizer(DATASET_PATH, TOKENIZER_SAVE_PATH)
else:
    tokenizer = Tokenizer.from_file(TOKENIZER_SAVE_PATH)

vocab_size = tokenizer.get_vocab_size()

def encode(text): return tokenizer.encode(text).ids
def decode(ids): return tokenizer.decode(ids)

# Test tokenizer sanity
test_str = "Hello, how are you?"
encoded = encode(test_str)
print(f"Tokenizer Test -> Original: '{test_str}' | Encoded: {encoded} | Decoded: '{decode(encoded)}'")

# Load Training Data
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    text = f.read()

data = torch.tensor(encode(text), dtype=torch.long)
n_split = int(0.9 * len(data))
train_data = data[:n_split]
val_data = data[n_split:]

def get_batch(split):
    """Generates a small batch of data of inputs x and targets y."""
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    """Estimates the loss across both training and validation sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- Architecture ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

# --- Main Training Execution ---
if __name__ == "__main__":
    model = GPTLanguageModel().to(device)
    num_parameters = sum(p.numel() for p in model.parameters())
    bytes_per_parameter = model.parameters().__next__().element_size()

    print(f"Total parameter memory: {(num_parameters * bytes_per_parameter) / (1024 ** 2):.2f} MB")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    param_str = f"model_{int(num_parameters/1e6)}M_n_embd-{N_EMBD}_n_head-{N_HEAD}_n_layer-{N_LAYER}"
    os.makedirs(param_str, exist_ok=True)

    for iter in tqdm(range(MAX_ITERS), desc="Training Model"):
        if iter % EVAL_INTERVAL == 0 and iter > 0:
            losses = estimate_loss(model)
            print(f"Step {iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")
            torch.save(model.state_dict(), os.path.join(param_str, f"checkpoint_iter{iter}.pth"))

        xb, yb = get_batch('train')
        optimizer.zero_grad(set_to_none=True)

        # Automatic Mixed Precision
        with torch.amp.autocast('cuda'):
            logits, loss = model(xb, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Save final artifact
    torch.save(model.state_dict(), f"{param_str}.pth")
    print("Training complete and model saved.")