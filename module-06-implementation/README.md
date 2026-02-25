# Module 6: Hands-on Implementation

Time to write code. We'll build a complete BitNet model from scratch in PyTorch.

All code is in `bitnet_from_scratch.py` — this document explains every piece.

## 6.1 Implementing `weight_quant()` from Scratch

### The Function

```python
def weight_quant(w):
    """
    Quantize weights to ternary {-1, 0, 1} using absmean scaling.

    Args:
        w: Weight tensor of any shape (typically [out_features, in_features])

    Returns:
        Quantized and dequantized weight tensor (same shape, but values
        are constrained to {-scale, 0, +scale} where scale = mean(|w|))
    """
    # Compute the average absolute value (absmean)
    # This serves as our scaling factor
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)

    # Scale weights, round to nearest integer, clamp to [-1, 1]
    # Then rescale back (divide by scale = multiply by 1/scale)
    u = (w * scale).round().clamp_(-1, 1) / scale

    return u
```

### What Each Step Does

```python
# Example weight matrix:
w = tensor([[ 0.50, -0.20,  0.80],
            [-0.10,  0.60, -0.40]])

# Step 1: Compute scale
mean_abs = mean(|w|) = (0.50+0.20+0.80+0.10+0.60+0.40) / 6 = 0.433
scale = 1 / 0.433 = 2.308

# Step 2: Scale and round
w * scale = [[ 1.15, -0.46,  1.85],
             [-0.23,  1.38, -0.92]]

rounded   = [[ 1,     0,     2],
             [ 0,     1,    -1]]

# Step 3: Clamp to [-1, 1]
clamped   = [[ 1,     0,     1],
             [ 0,     1,    -1]]

# Step 4: Rescale (dequantize)
result    = [[ 0.433,  0,     0.433],
             [ 0,      0.433,-0.433]]
```

---

## 6.2 Implementing `activation_quant()` from Scratch

### The Function

```python
def activation_quant(x):
    """
    Per-token quantization to 8 bits using absmax scaling.

    Args:
        x: Activation tensor, shape (batch, seq_len, hidden_dim)

    Returns:
        Quantized and dequantized activation tensor (same shape,
        values are approximately preserved but discretized)
    """
    # Per-token scaling: find max absolute value along hidden_dim
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)

    # Quantize to INT8 range and dequantize
    y = (x * scale).round().clamp_(-128, 127) / scale

    return y
```

### What Each Step Does

```python
# Example: single token activation
x = tensor([0.5, -1.2, 0.3, 0.8])

# Step 1: Find max absolute value
max_abs = 1.2

# Step 2: Compute scale
scale = 127.0 / 1.2 = 105.83

# Step 3: Scale, round, clamp
x * scale = [52.92, -127.0, 31.75, 84.67]
rounded   = [53,    -127,   32,    85]
clamped   = [53,    -127,   32,    85]  # All within [-128, 127]

# Step 4: Dequantize
result    = [0.501, -1.200, 0.302, 0.803]
# Very close to original! (quantization error is small at 8-bit)
```

---

## 6.3 Building the Full `BitLinear` Layer

### The Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class BitLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear with ternary weight quantization
    and 8-bit activation quantization.

    During training:
    - Maintains full-precision shadow weights (self.weight)
    - Quantizes weights to {-1, 0, 1} in forward pass via STE
    - Quantizes activations to INT8 in forward pass via STE
    - Gradients flow to shadow weights through STE

    During inference:
    - Weights can be pre-quantized to ternary
    - Matrix multiply becomes add/subtract (no multiply needed)
    """

    def __init__(self, in_features, out_features, bias=False):
        # BitNet doesn't use bias
        super().__init__(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features)

    def forward(self, x):
        w = self.weight

        # Step 1: Normalize input
        x_norm = self.norm(x)

        # Step 2: Quantize activations with STE
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()

        # Step 3: Quantize weights with STE
        w_quant = w + (weight_quant(w) - w).detach()

        # Step 4: Linear operation
        # In training: this is float matmul with quantized values
        # In optimized inference: this would be INT8 add/subtract
        y = F.linear(x_quant, w_quant)

        return y
```

### Testing It

```python
# Create a BitLinear layer
layer = BitLinear(768, 768)

# Forward pass
x = torch.randn(2, 10, 768)  # batch=2, seq_len=10, hidden=768
y = layer(x)
print(y.shape)  # torch.Size([2, 10, 768])

# Check that weights are quantized in forward pass
with torch.no_grad():
    w = layer.weight
    w_q = weight_quant(w)
    unique_scaled = torch.unique((w_q / w_q[w_q != 0].abs().min()).round())
    print(f"Unique quantized levels: {unique_scaled}")  # tensor([-1., 0., 1.])
```

---

## 6.4 Assembling a Mini-BitNet Transformer

### Full Model

```python
class BitNetAttention(nn.Module):
    """Multi-head attention with BitLinear layers."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # All projections are BitLinear (ternary weights)
        self.W_q = BitLinear(d_model, d_model)
        self.W_k = BitLinear(d_model, d_model)
        self.W_v = BitLinear(d_model, d_model)
        self.W_o = BitLinear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class BitNetFFN(nn.Module):
    """Feed-forward network with BitLinear and Squared ReLU."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = BitLinear(d_model, d_ff)
        self.w2 = BitLinear(d_ff, d_model)

    def forward(self, x):
        # Squared ReLU activation (used in BitNet)
        return self.w2(F.relu(self.w1(x)) ** 2)


class BitNetBlock(nn.Module):
    """Single transformer block with BitLinear layers."""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = BitNetAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = BitNetFFN(d_model, d_ff)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class MiniBitNet(nn.Module):
    """
    A minimal BitNet language model.

    Architecture:
    - Token embedding (full precision)
    - N BitNet transformer blocks (ternary weights)
    - RMSNorm
    - Output head (full precision)
    """

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len=512):
        super().__init__()

        # Embedding (NOT quantized — remains full precision)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (simple learned positions for this mini model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks (ALL linear layers are BitLinear)
        self.blocks = nn.ModuleList([
            BitNetBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        # Final norm and output head (NOT quantized)
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (optional but common)
        self.head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights — standard practice for Transformers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens):
        B, T = tokens.shape
        device = tokens.device

        # Token + positional embeddings
        x = self.embedding(tokens) + self.pos_embedding(
            torch.arange(T, device=device)
        )

        # Causal mask for autoregressive generation
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        return logits

    def count_parameters(self):
        """Count total and quantized parameters."""
        total = sum(p.numel() for p in self.parameters())
        quantized = sum(
            p.numel() for name, p in self.named_parameters()
            if 'blocks' in name and 'weight' in name and 'norm' not in name
        )
        return total, quantized
```

### Model Configurations

```python
# Tiny model for learning (runs on any machine)
tiny_config = dict(
    vocab_size=256,   # Character-level
    d_model=128,
    n_heads=4,
    d_ff=512,
    n_layers=4,
)
# ~2M params, ~1.5M quantized

# Small model for experiments
small_config = dict(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    d_ff=2048,
    n_layers=8,
)
# ~50M params, ~40M quantized

# Reference: BitNet b1.58 2B4T
bitnet_2b_config = dict(
    vocab_size=128256,
    d_model=2560,
    n_heads=20,
    d_ff=6912,
    n_layers=30,
)
# ~2.4B params
```

---

## 6.5 Training Loop: STE + Adam Optimizer

```python
def train_bitnet(model, train_data, epochs=10, lr=1e-3, device='cpu'):
    """
    Training loop for BitNet.

    Key insight: The optimizer updates FULL PRECISION shadow weights.
    Quantization happens only in the forward pass via STE.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_data:
            tokens = batch.to(device)

            # Input: all tokens except last
            # Target: all tokens except first (next-token prediction)
            x = tokens[:, :-1]
            y = tokens[:, 1:]

            # Forward pass (quantization happens here via STE)
            logits = model(x)

            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            # Backward pass (gradients flow through STE to shadow weights)
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (important for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update shadow weights (full precision)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
              f"PPL: {math.exp(avg_loss):.2f}")


def generate(model, start_tokens, max_new_tokens=100, temperature=0.8):
    """Autoregressive text generation."""
    model.eval()
    tokens = start_tokens.unsqueeze(0) if start_tokens.dim() == 1 else start_tokens

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

    return tokens
```

---

## 6.6 Monitoring Quantization Quality During Training

```python
def analyze_quantization(model):
    """Analyze the quantization state of a BitNet model."""
    print("\n=== Quantization Analysis ===\n")

    for name, param in model.named_parameters():
        if 'blocks' in name and 'weight' in name and 'norm' not in name:
            w = param.data
            w_q = weight_quant(w)

            # Get the ternary values
            scale = w.abs().mean()
            ternary = (w_q / scale).round()

            # Count distribution
            n_neg = (ternary == -1).sum().item()
            n_zero = (ternary == 0).sum().item()
            n_pos = (ternary == 1).sum().item()
            total = ternary.numel()

            # Quantization error
            error = (w - w_q).abs().mean().item()

            print(f"{name}:")
            print(f"  Distribution: -1={n_neg/total:.1%}, "
                  f"0={n_zero/total:.1%}, +1={n_pos/total:.1%}")
            print(f"  Scale: {scale:.4f}, Quant error: {error:.4f}")
            print()
```

---

## 6.7 Comparing BitLinear vs. nn.Linear on a Toy Task

```python
def comparison_experiment():
    """
    Train identical models — one with nn.Linear, one with BitLinear —
    and compare performance.
    """
    import copy

    # Create two identical models
    standard_model = MiniLLM(vocab_size=256, d_model=128, n_heads=4,
                             d_ff=512, n_layers=4)
    bitnet_model = MiniBitNet(vocab_size=256, d_model=128, n_heads=4,
                              d_ff=512, n_layers=4)

    # Count parameters
    std_total = sum(p.numel() for p in standard_model.parameters())
    bit_total, bit_quantized = bitnet_model.count_parameters()

    print(f"Standard model: {std_total:,} params (all FP16 = "
          f"{std_total * 2 / 1e6:.1f} MB)")
    print(f"BitNet model:   {bit_total:,} params ({bit_quantized:,} quantized)")
    print(f"  FP16 size: {bit_total * 2 / 1e6:.1f} MB")
    print(f"  1.58-bit size: {(bit_quantized * 0.2 + (bit_total - bit_quantized) * 2) / 1e6:.1f} MB")

    # Train both on same data and compare loss curves
    # ... (see exercise file for complete implementation)
```

---

## Exercise 6: Train a Tiny BitNet

See `../exercises/exercise_06_train_bitnet.py`.

Tasks:
1. Build the MiniBitNet model with the tiny config
2. Create a character-level dataset from a text file
3. Train for 20 epochs, monitoring loss and perplexity
4. Run `analyze_quantization()` to see weight distributions
5. Generate text samples at various checkpoints
6. Compare loss curves with a standard nn.Linear model

---

## Key Takeaways

1. `weight_quant()` uses absmean scaling → round → clamp to get ternary values
2. `activation_quant()` uses absmax per-token scaling → round → clamp for INT8
3. `BitLinear` is a drop-in replacement for `nn.Linear` with STE
4. The full model: embeddings (FP16) + BitLinear blocks + output head (FP16)
5. Training uses Adam on full-precision shadow weights; quantization is in the forward pass only
6. Monitor weight distributions: expect ~30-50% zeros in trained models
