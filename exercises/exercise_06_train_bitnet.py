"""
Exercise 6: Train a Tiny BitNet
================================

Train the full MiniBitNet model on character-level text generation
and observe the quantization behavior during training.

Goals:
  - Train a complete BitNet language model
  - Monitor weight distributions during training
  - Generate text from a 1.58-bit model
  - Compare with a standard (non-quantized) model

Run:
    python exercises/exercise_06_train_bitnet.py
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import from the main implementation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'module-06-implementation'))
from bitnet_from_scratch import (
    MiniBitNet, CharDataset, weight_quant, activation_quant,
    analyze_quantization, count_parameters, RMSNorm
)


# ============================================================================
# Standard (non-quantized) model for comparison
# ============================================================================

class StandardBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

    def forward(self, x, mask=None):
        B, T, C = x.shape
        # Attention
        h = self.norm1(x)
        qkv = self.attn_qkv(h).chunk(3, dim=-1)
        Q, K, V = [t.view(B, T, self.n_heads, self.d_k).transpose(1, 2) for t in qkv]
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1) @ V
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.attn_out(attn)
        # FFN
        h = self.norm2(x)
        x = x + self.ff2(F.relu(self.ff1(h)) ** 2)
        return x


class StandardModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            StandardBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, tokens):
        B, T = tokens.shape
        device = tokens.device
        x = self.tok_emb(tokens) + self.pos_emb(torch.arange(T, device=device))
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))


# ============================================================================
# Training with monitoring
# ============================================================================

def train_with_monitoring(model, dataset, epochs=30, batch_size=16, lr=1e-3,
                          device='cpu', name="Model"):
    """Train model and return loss history."""
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        losses.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            ppl = math.exp(min(avg_loss, 20))
            print(f"  [{name}] Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | PPL: {ppl:.1f}")

    return losses


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 6: Train a Tiny BitNet (with Comparison)")
    print("=" * 60)

    # Prepare data
    TEXT = """
    The quick brown fox jumps over the lazy dog. Pack my box with five dozen
    liquor jugs. How vexingly quick daft zebras jump. The five boxing wizards
    jump quickly. Amazingly few discotheques provide jukeboxes. Heavy boxes
    perform quick waltzes and jigs. Jackdaws love my big sphinx of quartz.
    The job requires extra pluck and zeal from every young wage earner.
    A wizard's job is to vex chumps quickly in fog. We promptly judged
    antique ivory buckles for the next prize. Crazy Frederick bought many
    very exquisite opal jewels. Sixty zippers were quickly picked from the
    woven jute bag. Grumpy wizards make toxic brew for the evil queen and jack.
    """ * 100

    dataset = CharDataset(TEXT, seq_len=64)
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Dataset: {len(dataset)} samples")

    config = dict(
        vocab_size=dataset.vocab_size,
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_layers=4,
        max_seq_len=128,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train BitNet model
    print("\n--- Training BitNet (1.58-bit weights) ---")
    bitnet_model = MiniBitNet(**config)
    count_parameters(bitnet_model)
    bitnet_losses = train_with_monitoring(
        bitnet_model, dataset, epochs=30, device=device, name="BitNet"
    )

    # Train standard model
    print("\n--- Training Standard (FP32 weights) ---")
    standard_model = StandardModel(**config)
    standard_losses = train_with_monitoring(
        standard_model, dataset, epochs=30, device=device, name="Standard"
    )

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"\n{'Epoch':>6} {'BitNet Loss':>12} {'Standard Loss':>14} {'Gap':>8}")
    print("-" * 44)
    for i, (bl, sl) in enumerate(zip(bitnet_losses, standard_losses)):
        if (i + 1) % 5 == 0:
            print(f"{i+1:>6} {bl:>12.4f} {sl:>14.4f} {bl-sl:>8.4f}")

    print(f"\nFinal BitNet loss:    {bitnet_losses[-1]:.4f} "
          f"(PPL: {math.exp(min(bitnet_losses[-1], 20)):.1f})")
    print(f"Final Standard loss:  {standard_losses[-1]:.4f} "
          f"(PPL: {math.exp(min(standard_losses[-1], 20)):.1f})")

    # Analyze BitNet quantization
    analyze_quantization(bitnet_model)

    # Generate text from BitNet
    print("\n" + "=" * 60)
    print("TEXT GENERATION (BitNet)")
    print("=" * 60)
    bitnet_model.eval()
    start = torch.zeros(1, dtype=torch.long, device=device)
    generated = bitnet_model.generate(start, max_new_tokens=200, temperature=0.7)
    print(dataset.decode(generated[0]))
