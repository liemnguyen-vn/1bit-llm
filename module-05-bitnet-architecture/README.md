# Module 5: BitNet Architecture Deep Dive

This is the heart of the course. We'll dissect every component of BitNet.

## 5.1 BitNet b1 (2023): The 1-Bit LLM Breakthrough

**Paper:** "BitNet: Scaling 1-bit Transformers for Large Language Models" (Ma et al., 2023)

BitNet b1 was the first paper to show that a Transformer LLM could be trained with binary {-1, +1} weights while remaining competitive at scale.

### Architecture

BitNet b1 replaces every `nn.Linear` in the Transformer with `BitLinear`:

```
Standard Transformer:          BitNet Transformer:
  x → LayerNorm → Linear        x → LayerNorm → BitLinear
                                      ↓
                               [quantize weights to {-1,+1}]
                               [quantize activations to INT8]
                               [compute, dequantize]
```

### BitLinear v1 (Binary)

```python
class BitLinear_v1(nn.Linear):
    """BitNet b1: Binary weights {-1, +1}, INT8 activations."""

    def forward(self, x):
        # 1. Normalize input
        x_norm = LayerNorm(x)

        # 2. Quantize weights to {-1, +1}
        alpha = self.weight.abs().mean()             # Scaling factor
        w_binary = self.weight.sign()                # Binarize
        # STE: w_q = w + (sign(w) - w).detach()

        # 3. Quantize activations to INT8
        gamma = x_norm.abs().max()                   # Max absolute value
        x_q = round(x_norm * 127 / gamma)            # Scale to [-127, 127]
        x_q = clamp(x_q, -128, 127)

        # 4. Integer matrix multiply (conceptually)
        y = x_q @ w_binary.T

        # 5. Dequantize output
        y = y * alpha * gamma / 127

        return y
```

### Results

BitNet b1 showed promising scaling — the gap with full-precision models **narrowed** as model size increased. But it didn't close completely.

---

## 5.2 BitNet b1.58 (2024): "All LLMs Are in 1.58 Bits"

**Paper:** "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (Ma et al., 2024)

The breakthrough: change from binary {-1, +1} to **ternary {-1, 0, +1}**.

### Why This Small Change Matters So Much

```
Binary {-1, +1}:
  - Every weight MUST participate (no zero means no pruning)
  - Matrix multiply: all additions (no skipping)
  - 1.00 bit per parameter

Ternary {-1, 0, +1}:
  - Zero weights are FREE — no computation needed
  - Natural sparsity: 30-50% of weights become zero
  - Effectively learned pruning built into training
  - 1.58 bits per parameter (only 0.58 bits more than binary)
  - MATCHES full-precision models at 3B+ parameters
```

### The Key Results

Comparison at 3B model size, trained on same data:

```
Model            Perplexity   Memory    Latency   Energy
─────────────────────────────────────────────────────────
LLaMA 3B (FP16)   baseline     1x        1x        1x
BitNet b1.58 3B    matches     3.55x     2.71x     71.4x
                   LLaMA!      less      faster    less
```

At 3.9B parameters:
- BitNet outperforms LLaMA on ARC-Challenge, HellaSwag, Winogrande
- Matches or exceeds on most other benchmarks

---

## 5.3 The BitLinear Layer — Complete Math

The BitLinear layer is the core building block. Here's every detail.

### Overview

```
Input x (FP16) → RMSNorm → Activation Quantization (INT8)
                                      ↓
Weight W (FP16 shadow) → Weight Quantization (ternary {-1,0,1})
                                      ↓
                              INT8 @ Ternary MatMul
                                      ↓
                           Dequantize → Output (FP16)
```

### The Full Forward Pass

```python
class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight  # Shadow weight, FP16, shape (out_features, in_features)

        # Step 1: Normalize input with RMSNorm
        x_norm = rms_norm(x)

        # Step 2: Quantize activations to INT8 (per-token absmax)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()

        # Step 3: Quantize weights to ternary (per-tensor absmean)
        w_quant = w + (weight_quant(w) - w).detach()

        # Step 4: Linear operation (in practice, INT8 x ternary)
        y = F.linear(x_quant, w_quant)

        return y
```

---

## 5.4 Weight Quantization: The AbsMean Function

### Formula

```
Given weight matrix W with shape (out_features, in_features):

1. Compute scale:
   γ = mean(|W|) = (1 / (n*m)) * Σ|W_ij|

2. Scale and round:
   W̃ = round(W / γ)

3. Clamp to ternary range:
   W̃ = clamp(W̃, -1, 1)

4. Rescale (for dequantization):
   W_q = W̃ * γ
```

### In PyTorch

```python
def weight_quant(w):
    """Quantize weights to ternary {-1, 0, 1} using absmean scaling."""
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u
```

### Step-by-Step Example

```python
W = tensor([[ 0.42, -0.13,  0.87, -0.55],
            [ 0.03, -0.71,  0.29,  0.64]])

# Step 1: scale = 1 / mean(|W|)
mean_abs = (0.42+0.13+0.87+0.55+0.03+0.71+0.29+0.64) / 8 = 0.455
scale = 1 / 0.455 = 2.198

# Step 2: Multiply by scale and round
W * scale = [[ 0.92, -0.29,  1.91, -1.21],
             [ 0.07, -1.56,  0.64,  1.41]]

round() =   [[ 1,     0,     2,    -1],
             [ 0,    -2,     1,     1]]

# Step 3: Clamp to [-1, 1]
clamp() =   [[ 1,     0,     1,    -1],
             [ 0,    -1,     1,     1]]

# Step 4: Rescale for dequantization
W_q = clamp_result / scale
     = [[ 0.455,  0,     0.455, -0.455],
        [ 0,     -0.455, 0.455,  0.455]]
```

### Why AbsMean?

AbsMean (mean of absolute values) is more robust than AbsMax (maximum absolute value):
- AbsMax is sensitive to outliers — one extreme weight skews everything
- AbsMean gives a representative scale for the entire matrix
- Works well with per-tensor granularity (simpler than per-channel)

---

## 5.5 Activation Quantization: The AbsMax Function (8-bit per-token)

### Formula

```
Given activation tensor x with shape (batch, seq_len, hidden_dim):

For each token t (quantize along hidden_dim):

1. Compute scale:
   η_t = 127 / max(|x_t|)

2. Quantize:
   x̃_t = round(x_t * η_t)

3. Clamp:
   x̃_t = clamp(x̃_t, -128, 127)

4. Dequantize:
   x_q_t = x̃_t / η_t
```

### In PyTorch

```python
def activation_quant(x):
    """Per-token quantization to 8 bits using absmax scaling."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y
```

### Why Per-Token?

Different tokens have different activation magnitudes. Per-token scaling ensures each token uses the full INT8 range:

```
Token "the":  activations range [-0.5, 0.5]  → scale = 254
Token "cat":  activations range [-2.0, 2.0]  → scale = 63.5
Token ".":    activations range [-0.1, 0.1]  → scale = 1270
```

Without per-token scaling, tokens with small activations would be quantized to mostly zeros.

### Why 8-bit (not 1-bit)?

Keeping activations at 8-bit precision is the key insight that separates BitNet from earlier binary networks:

1. **Information preservation:** Activations carry the "signal" through the network
2. **Hardware support:** INT8 operations are native on all modern GPUs and CPUs
3. **Training stability:** Binary activations caused severe training difficulties
4. **Minimal overhead:** Activation storage is temporary (not stored in the model)

---

## 5.6 STE in BitLinear: The Detach Trick

### The Core Pattern

```python
# STE for weights
w_quant = w + (weight_quant(w) - w).detach()

# STE for activations
x_quant = x + (activation_quant(x) - x).detach()
```

### Why This Works

Let's trace through the math:

```
Forward pass:
  w_quant = w + (weight_quant(w) - w).detach()
          = w + constant           # detach() makes (wq - w) a constant
          = w + wq - w             # But the VALUE is wq - w
          = wq                     # So we get the quantized weight

Backward pass:
  dL/dw_quant = dL/dw             # Gradient flows through the 'w' term
  The detached part has zero gradient
  So: dL/dw = dL/dw_quant * 1     # STE: treat quantization as identity
```

### Visualizing the Gradient Flow

```
Forward:  w ──→ weight_quant(w) ──→ w_quant ──→ F.linear ──→ loss
                                                     ↓
Backward: w ←── gradient = 1 ←──── dL/dw_quant ←── dL/dy
          (STE skips quantization)
```

The full-precision shadow weight `w` receives gradients as if it were used directly in the linear operation. But the actual computation used the quantized value.

---

## 5.7 Why 1.58 Bits? Information Theory of Ternary {-1, 0, 1}

### The Math

```
A ternary value can be one of 3 states: {-1, 0, 1}
Bits needed = log₂(3) = 1.5849... ≈ 1.58 bits

For comparison:
  Binary {0, 1}:     log₂(2) = 1.00 bits
  Ternary {-1,0,1}:  log₂(3) = 1.58 bits
  Quaternary:         log₂(4) = 2.00 bits
  INT4:               log₂(16) = 4.00 bits
  INT8:               log₂(256) = 8.00 bits
  FP16:               16.00 bits
```

### The Title Explained

"All Large Language Models are in 1.58 Bits" means:
1. You can train an LLM where every weight is {-1, 0, 1}
2. Each weight needs only 1.58 bits of information
3. This ternary LLM **matches** full 16-bit LLMs in quality
4. Therefore, all the "information" in an LLM can be compressed to 1.58 bits per parameter

---

## 5.8 The Role of Zero: Sparsity as a Feature

The addition of `0` to the value set was the key innovation of b1.58 over b1.

### What Zero Does

When `W[i][j] = 0`:
- The connection from input j to output i is **turned off**
- `x[j] * 0 = 0` — no computation needed
- The model has **learned to prune** this connection

### Sparsity Statistics

In trained BitNet models:
```
Typical weight distribution:
  -1:  ~25-35% of weights
   0:  ~30-50% of weights
  +1:  ~25-35% of weights
```

This means:
- ~30-50% of matrix multiply operations can be **skipped**
- Natural **structured sparsity** without separate pruning
- Different layers learn different sparsity patterns

### Why This Beats Binary

```
Binary {-1, +1}:
  y[i] = sum(x[j] * W[i][j]) for all j
  Every weight contributes → full dense computation

Ternary {-1, 0, +1}:
  y[i] = sum(x[j] * W[i][j]) for j where W[i][j] ≠ 0
  Zero weights are skipped → naturally sparse

  Equivalent to:
  y[i] = sum(x[j]) for j where W=+1  MINUS  sum(x[j]) for j where W=-1
```

---

## 5.9 Scaling Laws for 1-bit LLMs

BitNet b1.58 follows its own scaling law, distinct from full-precision models.

### Key Finding

The performance gap between BitNet and full-precision models **decreases** as model size increases:

```
Model Size    Perplexity Gap (BitNet vs FP16)
─────────────────────────────────────────────
125M          Significant gap (~20%)
350M          Large gap (~12%)
1.3B          Moderate gap (~5%)
3B            Gap closes (< 2%)
3.9B          BitNet matches or exceeds FP16
7B+           BitNet matches FP16
```

### Interpretation

At small sizes, every parameter carries critical information — quantizing to 3 levels loses too much.

At large sizes, there's enough redundancy that ternary representation suffices. The model learns to distribute information across many more (but simpler) parameters.

### Implication

The scaling law suggests that **larger 1-bit models will continue to match full-precision counterparts**. A 70B ternary model would likely match a 70B FP16 model, while using ~10x less memory.

---

## 5.10 Performance Benchmarks: BitNet vs. LLaMA

### BitNet b1.58 3B vs. LLaMA 3B

| Benchmark | LLaMA 3B (FP16) | BitNet b1.58 3B |
|-----------|-----------------|-----------------|
| Perplexity | Baseline | Matches |
| ARC-Challenge | 34.8 | 37.7 (+2.9) |
| ARC-Easy | 68.1 | 68.8 (+0.7) |
| HellaSwag | 71.1 | 71.2 (+0.1) |
| Winogrande | 65.0 | 66.2 (+1.2) |
| MMLU | - | Competitive |

### Efficiency Gains

| Metric | LLaMA 3B | BitNet b1.58 3B | Improvement |
|--------|----------|-----------------|-------------|
| Memory (GB) | 6.0 | 1.7 | 3.55x less |
| Latency | 1x | 0.37x | 2.71x faster |
| Energy (MatMul) | 1x | 0.014x | 71.4x less |

### BitNet b1.58 2B4T (2025 Release)

The first open-source 1-bit LLM:
```
Parameters: 2.4B (2,412,820,480)
Training tokens: 4 Trillion
Hidden size: 2560
Layers: 30
Attention heads: 20 (5 KV heads with GQA)
FFN hidden: 6912
Context length: 4096
Activation: Squared ReLU
```

---

## Exercise 5: BitLinear Forward Pass by Hand

Work through the BitLinear forward pass manually on a 4x4 weight matrix and a 4-dimensional input vector.

Tasks:
1. Start with a random 4x4 weight matrix W
2. Compute weight_quant(W) step by step (absmean, round, clamp)
3. Start with a random 4-dim input x
4. Compute activation_quant(x) step by step (absmax, scale, round, clamp)
5. Compute y = x_quant @ W_quant.T
6. Verify: the matmul only involves additions and subtractions

---

## Key Takeaways

1. BitNet b1 used binary {-1, +1} weights — good but not enough
2. BitNet b1.58 uses ternary {-1, 0, +1} — matches full-precision at scale
3. **Weight quantization** uses absmean: `scale = mean(|W|)`, round, clamp to [-1, 1]
4. **Activation quantization** uses absmax: `scale = 127/max(|x|)`, round, clamp to [-128, 127]
5. **STE via detach trick**: `w + (quant(w) - w).detach()` — forward uses quantized, backward ignores it
6. The **zero** in ternary provides natural sparsity (30-50% of weights)
7. Performance matches FP16 at 3B+ parameters with 3.5x less memory and 71x less energy
