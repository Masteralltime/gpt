"""
train_bigram_basic.py

A rudimentary character-level bigram language model. This script serves as a simple
baseline and educational reference, relying only on single token lookup tables without
attention mechanisms or neural network layers.
"""

import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dotenv import load_dotenv

# Log Hardware Configuration
if torch.cuda.is_available():
    current_device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device_index)
    print(f"PyTorch is currently using GPU: {device_name} (Index: {current_device_index})")
else:
    print("No GPU in use as CUDA is not available.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1337)

# Load configuration
load_dotenv()
DATASET_PATH = os.getenv("CHAR_DATASET_PATH", "input.txt")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Missing text dataset at {DATASET_PATH}. Please provide a valid text file.")

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

str_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_str = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [str_to_int[c] for c in s]
def decode(l): return ''.join([int_to_str[i] for i in l])

# Prepare Data Splits
data = torch.tensor(encode(text), dtype=torch.long)
n_split = int(0.8 * len(data))
train_data = data[:n_split]
val_data = data[n_split:]

BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 10000

def get_batch(split):
    """Retrieve random chunks of input x and target y sequences."""
    dataset = train_data if split == 'train' else val_data
    ix = torch.randint(len(dataset) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([dataset[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([dataset[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y

class BigramLanguageModel(nn.Module):
    """
    A foundational Bigram Model.
    Each token directly predicts the next token via a simple lookup table,
    independent of wider context.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # Shape: (B, T, C)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generates characters sequentially based on the highest probability token."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    # Initialize model
    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("Pre-training sample output:")
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))

    print("\nTraining...")
    for epoch in range(MAX_ITERS):
        xb, yb = get_batch('train')
        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Final Training Loss: {loss.item():.4f}")

    print("\nPost-training sample output:")
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(idx, max_new_tokens=250)[0].tolist()))