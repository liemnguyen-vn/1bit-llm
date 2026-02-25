# Module 2: Transformer Architecture Deep Dive

## 2.1 Attention Mechanism: Intuition and Math

The Transformer's key innovation is **attention**: allowing every token to look at every other token and decide what's relevant.

### Intuition

Consider the sentence: "The cat sat on the mat because **it** was tired."

What does "it" refer to? To answer this, the model needs "it" to **attend to** "cat". Attention learns these relationships.

### The Attention Formula

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Where:
- `Q` (Query): "What am I looking for?" — shape (seq_len, d_k)
- `K` (Key): "What do I contain?" — shape (seq_len, d_k)
- `V` (Value): "What information do I provide?" — shape (seq_len, d_v)
- `d_k`: dimension of keys (used for scaling)

### Step by Step

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V):
    d_k = Q.shape[-1]

    # Step 1: Compute attention scores
    scores = Q @ K.transpose(-2, -1)  # (seq_len, seq_len)

    # Step 2: Scale to prevent vanishing gradients in softmax
    scores = scores / (d_k ** 0.5)

    # Step 3: Softmax to get attention weights (probabilities)
    weights = F.softmax(scores, dim=-1)  # Each row sums to 1

    # Step 4: Weighted sum of values
    output = weights @ V  # (seq_len, d_v)

    return output
```

---

## 2.2 Self-Attention: Q, K, V Projections

In **self-attention**, Q, K, and V all come from the same input, projected through learned weight matrices:

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=False)  # These are the weight
        self.W_k = nn.Linear(d_model, d_k, bias=False)  # matrices that BitNet
        self.W_v = nn.Linear(d_model, d_k, bias=False)  # will quantize to {-1,0,1}

    def forward(self, x):
        Q = self.W_q(x)   # x @ W_q
        K = self.W_k(x)   # x @ W_k
        V = self.W_v(x)   # x @ W_v
        return attention(Q, K, V)
```

**BitNet connection:** The `W_q`, `W_k`, `W_v` matrices are exactly where BitNet replaces `nn.Linear` with `BitLinear`. Instead of FP16 weights, they become ternary {-1, 0, 1}.

---

## 2.3 Multi-Head Attention

Instead of one attention function, use multiple "heads" that attend to different aspects:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        # Project and split into heads
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Attention per head
        out = attention(Q, K, V)

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)
```

**BitNet b1.58 2B4T uses:**
- 20 attention heads
- 5 key-value heads (Grouped Query Attention / GQA)
- Hidden size 2560 → d_k = 128 per head

---

## 2.4 Feed-Forward Networks in Transformers

After attention, each position passes through a feed-forward network (FFN):

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)    # Up-projection
        self.w2 = nn.Linear(d_ff, d_model, bias=False)    # Down-projection

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))
```

**BitNet uses Squared ReLU** instead of standard ReLU:

```python
def squared_relu(x):
    return F.relu(x) ** 2
```

Squared ReLU produces sparser activations, which works well with ternary weights.

The FFN typically has `d_ff = 4 * d_model`. For BitNet 2B4T: d_model=2560, d_ff=6912.

---

## 2.5 Positional Encoding

Transformers have no inherent notion of position. We must add it.

### Rotary Position Embedding (RoPE)

Modern LLMs (including BitNet) use RoPE, which encodes position by rotating the Q and K vectors:

```python
def apply_rope(x, freqs):
    """Apply rotary position embedding."""
    # Split into pairs and rotate
    x_r, x_i = x[..., ::2], x[..., 1::2]
    cos, sin = freqs.cos(), freqs.sin()
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos
    return torch.stack([out_r, out_i], dim=-1).flatten(-2)
```

RoPE encodes **relative** position, meaning the model learns distance between tokens rather than absolute positions.

---

## 2.6 Layer Normalization

Normalization stabilizes training by keeping activations at a reasonable scale.

### RMSNorm (used in BitNet)

Simpler and faster than LayerNorm — no mean subtraction:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

**BitNet uses SubLayerNorm:** RMSNorm is applied **before** each sub-layer (attention and FFN), and the normalization parameters are NOT quantized — they remain in full precision.

---

## 2.7 The Full Transformer Block

Putting it all together:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        # Pre-norm architecture with residual connections
        x = x + self.attn(self.norm1(x))    # Attention + residual
        x = x + self.ffn(self.norm2(x))      # FFN + residual
        return x
```

**Residual connections** (`x + ...`) are critical — they allow gradients to flow directly through the network, making deep models trainable.

---

## 2.8 From Transformer to LLM: Decoder-Only Architecture

Modern LLMs use **decoder-only** architecture (no encoder):

```python
class MiniLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        x = self.embedding(tokens)          # Token IDs → embeddings
        for block in self.blocks:
            x = block(x)                     # N transformer blocks
        x = self.norm(x)                     # Final normalization
        logits = self.head(x)                # Project to vocabulary
        return logits
```

**What BitNet changes:** Every `nn.Linear` inside the transformer blocks becomes `BitLinear`. The embedding and output head typically remain in full precision.

### Where the Parameters Live

For a typical LLM, the vast majority of parameters are in `nn.Linear` layers:

```
Component               Params (7B model)    BitNet savings
─────────────────────────────────────────────────────────────
Embedding               ~130M (FP16)         Not quantized
Attention W_q,W_k,W_v   ~2.1B → ternary     ~10x smaller
Attention W_o            ~700M → ternary      ~10x smaller
FFN w1, w2, w3          ~3.8B → ternary      ~10x smaller
Output head             ~130M (FP16)          Not quantized
Norm layers             ~0.5M (FP32)          Not quantized
```

---

## Exercise 2: Build a Minimal GPT

See `../exercises/exercise_02_mini_gpt.py`.

Build a character-level language model using the components above:
1. Implement self-attention from scratch
2. Build a transformer block
3. Stack blocks into a decoder-only model
4. Train on a small text corpus
5. Generate text

---

## Key Takeaways

1. The Transformer is built from **attention** + **FFN** blocks with residual connections
2. Nearly all parameters live in `nn.Linear` layers (Q, K, V, O projections + FFN)
3. BitNet replaces these `nn.Linear` layers with `BitLinear` (ternary weights)
4. Normalization layers, embeddings, and the output head remain in full precision
5. Modern features like RoPE, GQA, Squared ReLU, and RMSNorm are used in BitNet
