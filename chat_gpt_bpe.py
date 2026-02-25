"""
chat_gpt_bpe.py

Interactive chat interface for the BPE-trained GPT model.
Loads model weights and tokenizers using environment variables and supports
hardware acceleration via CUDA or OpenVINO NPU.
"""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MODEL_PATH = os.getenv("MODEL_CHECKPOINT_PATH", "checkpoint.pth")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "tokenizer.json")

# Hyperparameters
BLOCK_SIZE = 256
DROPOUT = 0.2

# Hardware configuration
if hasattr(torch, "npu") and torch.npu.is_available():
    device = "npu"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using compute device: {device.upper()}")

def extract_params(fname):
    """Parses model architecture parameters from the checkpoint filename."""
    match = re.search(r'n_embd-(\d+)_n_head-(\d+)_n_layer-(\d+)', fname)
    if not match:
        raise ValueError("Filename must include 'n_embd', 'n_head', and 'n_layer' to parse architecture.")
    return int(match[1]), int(match[2]), int(match[3])

N_EMBD, N_HEAD, N_LAYER = extract_params(MODEL_PATH)

# Tokenizer setup
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()

def encode(text):
    return tokenizer.encode(text).ids

def decode(token_ids):
    return tokenizer.decode(token_ids)

class Head(nn.Module):
    """One head of self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
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
    """Multiple heads of self-attention in parallel."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """Main GPT Language Model architecture."""
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
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1)) if targets is not None else None
        return logits, loss

    def generate(self, idx, max_new_tokens=100, temperature=0.9, top_k=50):
        """Generates new tokens autoregressively."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    # Load model
    model = GPTLanguageModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Interactive loop
    print("\n[ChatBot GPT Ready — type 'exit' to quit]\n")
    while True:
        prompt = input("You: ")
        if prompt.strip().lower() == "exit":
            break

        input_ids = torch.tensor([encode("User: " + prompt)], dtype=torch.long, device=device)

        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=500)

        generated = output[0].tolist()
        response = decode(generated)

        # Format output for terminal readability
        response_with_lines = re.sub(r"\s*(User|Bot)\s*:", r"\n\1:", response)
        print("Bot:" + response_with_lines)