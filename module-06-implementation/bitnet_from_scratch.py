"""
BitNet from Scratch — Complete Implementation
=============================================

This file contains a full, working implementation of a 1-bit (1.58-bit) LLM
following the BitNet b1.58 architecture.

You can run this file directly to train a tiny BitNet on synthetic data.

Usage:
    python bitnet_from_scratch.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Core Quantization Functions
# ============================================================================

def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """
    Quantize weights to ternary {-1, 0, 1} using absmean scaling.

    Formula:
        scale = 1 / mean(|W|)
        W_q = clamp(round(W * scale), -1, 1) / scale

    The division by scale at the end dequantizes back to the original scale,
    so the output has the same magnitude as the input but only 3 distinct values.
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """
    Per-token quantization to 8 bits using absmax scaling.

    Formula:
        scale = 127 / max(|x|)     (per token, along hidden dim)
        x_q = clamp(round(x * scale), -128, 127) / scale

    Per-token means each token gets its own scale factor, ensuring
    the full INT8 range [-128, 127] is used for each token.
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


# ============================================================================
# RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in BitNet instead of LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================================
# BitLinear: The Core Layer
# ============================================================================

class BitLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear with ternary weight quantization
    and 8-bit activation quantization.

    During training:
    - self.weight holds full-precision shadow weights (updated by optimizer)
    - Forward pass quantizes weights to {-1, 0, 1} via STE
    - Forward pass quantizes activations to INT8 via STE
    - Backward pass: gradients flow through STE to shadow weights

    The STE (Straight-Through Estimator) is implemented via the detach trick:
        x_quant = x + (quant(x) - x).detach()
    This gives quantized values in the forward pass but identity gradients in backward.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight

        # Normalize input
        x_norm = self.norm(x)

        # STE: quantize activations (forward=quantized, backward=identity)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()

        # STE: quantize weights (forward=ternary, backward=identity)
        w_quant = w + (weight_quant(w) - w).detach()

        # Linear operation
        return F.linear(x_quant, w_quant)


# ============================================================================
# BitNet Transformer Components
# ============================================================================

class BitNetAttention(nn.Module):
    """Multi-head self-attention with BitLinear projections."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = BitLinear(d_model, d_model)
        self.W_k = BitLinear(d_model, d_model)
        self.W_v = BitLinear(d_model, d_model)
        self.W_o = BitLinear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V and reshape for multi-head
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ V

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class BitNetFFN(nn.Module):
    """Feed-forward network with BitLinear and Squared ReLU activation."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = BitLinear(d_model, d_ff)
        self.w2 = BitLinear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)) ** 2)


class BitNetBlock(nn.Module):
    """Single transformer block: attention + FFN with pre-norm and residuals."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = BitNetAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = BitNetFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# Full BitNet Language Model
# ============================================================================

class MiniBitNet(nn.Module):
    """
    A minimal BitNet language model.

    - Token embedding: full precision (not quantized)
    - Transformer blocks: all Linear layers are BitLinear (ternary)
    - Output head: full precision (weight-tied with embedding)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings (NOT quantized)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks (BitLinear layers)
        self.blocks = nn.ModuleList([
            BitNetBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        # Output (NOT quantized, weight-tied with embedding)
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight tying

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        device = tokens.device

        # Embeddings
        x = self.tok_emb(tokens) + self.pos_emb(torch.arange(T, device=device))

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Output logits
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=50, temperature=0.8):
        """Autoregressive text generation."""
        self.eval()
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        for _ in range(max_new_tokens):
            logits = self(tokens[:, -self.pos_emb.num_embeddings:])
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_tok], dim=1)

        return tokens


# ============================================================================
# Analysis Utilities
# ============================================================================

def analyze_quantization(model: nn.Module):
    """Print quantization statistics for all BitLinear layers."""
    print("\n{'='*60}")
    print("QUANTIZATION ANALYSIS")
    print("=" * 60)

    total_params = 0
    total_zeros = 0

    for name, param in model.named_parameters():
        if 'blocks' in name and 'weight' in name and 'norm' not in name:
            w = param.data
            scale = w.abs().mean()
            w_scaled = w / scale.clamp(min=1e-5)
            ternary = w_scaled.round().clamp(-1, 1)

            n_neg = (ternary == -1).sum().item()
            n_zero = (ternary == 0).sum().item()
            n_pos = (ternary == 1).sum().item()
            total = ternary.numel()
            error = (w - weight_quant(w)).abs().mean().item()

            total_params += total
            total_zeros += n_zero

            print(f"\n{name} [{param.shape[0]}x{param.shape[1]}]:")
            print(f"  -1: {n_neg/total:6.1%} | 0: {n_zero/total:6.1%} | "
                  f"+1: {n_pos/total:6.1%}")
            print(f"  Scale: {scale:.4f} | Quant error: {error:.6f}")

    print(f"\n{'─'*60}")
    print(f"Overall sparsity (zero weights): {total_zeros/total_params:.1%}")
    print(f"Total quantized params: {total_params:,}")


def count_parameters(model: nn.Module):
    """Count total, quantized, and full-precision parameters."""
    total = sum(p.numel() for p in model.parameters())
    quantized = sum(
        p.numel() for n, p in model.named_parameters()
        if 'blocks' in n and 'weight' in n and 'norm' not in n
    )
    fp = total - quantized

    print(f"\nParameter count:")
    print(f"  Total:     {total:>12,}")
    print(f"  Quantized: {quantized:>12,} (1.58 bits each)")
    print(f"  Full prec: {fp:>12,} (16 bits each)")
    print(f"\nMemory estimate:")
    print(f"  FP16 model:    {total * 2 / 1e6:>8.1f} MB")
    print(f"  BitNet model:  {(quantized * 0.2 + fp * 2) / 1e6:>8.1f} MB")
    print(f"  Savings:       {total * 2 / (quantized * 0.2 + fp * 2):>8.1f}x")


# ============================================================================
# Simple Dataset for Testing
# ============================================================================

class CharDataset(Dataset):
    """Character-level text dataset."""

    def __init__(self, text: str, seq_len: int = 64):
        self.seq_len = seq_len
        # Build vocabulary from printable ASCII
        chars = sorted(set(text))
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(chars)

        # Encode text
        self.data = torch.tensor([self.char2idx[c] for c in text], dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        return self.data[idx : idx + self.seq_len + 1]

    def decode(self, indices):
        return ''.join(self.idx2char.get(i.item(), '?') for i in indices)


# ============================================================================
# Training
# ============================================================================

def train(
    model: nn.Module,
    dataset: CharDataset,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = 'cpu',
):
    """Train the BitNet model on character-level data."""
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\nTraining on {len(dataset)} samples, {len(loader)} batches/epoch")
    print(f"Device: {device}\n")

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
        ppl = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | PPL: {ppl:.1f}")

            # Generate a sample
            model.eval()
            start = torch.zeros(1, dtype=torch.long, device=device)
            generated = model.generate(start, max_new_tokens=60, temperature=0.8)
            text = dataset.decode(generated[0])
            print(f"  Sample: {text[:80]}...")
            print()

    return model


# ============================================================================
# Main: Demo Training
# ============================================================================

if __name__ == "__main__":
    # Sample text for training (repeat for more data)
    SAMPLE_TEXT = """
    The key advantage of 1-bit large language models is their extreme efficiency.
    By constraining every weight to just three values: minus one, zero, and plus one,
    BitNet eliminates the need for floating-point multiplication entirely.
    Matrix multiplication becomes a series of additions and subtractions.
    This reduces memory usage by roughly ten times and energy consumption by over
    seventy times compared to standard sixteen-bit models.
    The breakthrough came when researchers discovered that ternary weights,
    with their natural sparsity from the zero value, could match the performance
    of full-precision models at three billion parameters and above.
    Training uses the straight-through estimator to handle the non-differentiable
    rounding operation. Full-precision shadow weights are maintained by the optimizer,
    while quantized ternary weights are used in the forward pass.
    This elegant trick allows standard gradient descent to train discrete models.
    """ * 50  # Repeat for more training data

    # Create dataset
    dataset = CharDataset(SAMPLE_TEXT, seq_len=64)
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Dataset size: {len(dataset)} samples")

    # Create model
    model = MiniBitNet(
        vocab_size=dataset.vocab_size,
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_layers=4,
        max_seq_len=128,
    )

    count_parameters(model)

    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train(model, dataset, epochs=20, batch_size=16, lr=1e-3, device=device)

    # Analyze quantization
    analyze_quantization(model)

    # Final generation
    print("\n" + "=" * 60)
    print("FINAL GENERATION")
    print("=" * 60)
    model.eval()
    start = torch.zeros(1, dtype=torch.long, device=device)
    generated = model.generate(start, max_new_tokens=200, temperature=0.7)
    print(dataset.decode(generated[0]))
