# Module 3: Quantization Theory

## 3.1 What is Quantization?

Quantization maps values from a large set (e.g., all floating-point numbers) to a smaller set (e.g., 256 integers).

```
Full precision:  0.3721, -1.2055, 0.0087, 2.5412, -0.9931, ...
                 (infinite possible values)

Quantized INT8:  47, -153, 1, 127, -126, ...
                 (only 256 possible values: -128 to 127)

Quantized 1-bit: 1, -1, 1, 1, -1, ...
                 (only 2 possible values)

Quantized 1.58b: 0, -1, 0, 1, -1, ...
                 (only 3 possible values)
```

### Why Quantize?

| Metric | FP16 | INT8 | INT4 | 1.58-bit |
|--------|------|------|------|----------|
| Memory per param | 2 bytes | 1 byte | 0.5 bytes | 0.2 bytes |
| 7B model size | 14 GB | 7 GB | 3.5 GB | 1.4 GB |
| MatMul energy | 1x | 0.2x | 0.05x | 0.014x |
| Compute type | FP multiply | INT multiply | INT multiply | INT add only |

---

## 3.2 Symmetric vs. Asymmetric Quantization

### Symmetric Quantization

Zero maps to zero. The range is symmetric around zero.

```
quantize(x) = round(x / scale)
dequantize(q) = q * scale

where scale = max(|x|) / (2^(bits-1) - 1)
```

Example (INT8, symmetric):
```
x = [0.5, -1.0, 0.25, 0.75]
scale = 1.0 / 127 = 0.00787
q = [63, -127, 32, 95]
```

### Asymmetric Quantization

Uses a zero-point offset. Better for distributions not centered on zero.

```
quantize(x) = round(x / scale) + zero_point
dequantize(q) = (q - zero_point) * scale
```

**BitNet uses symmetric quantization** — weights are centered around zero by design, and the ternary set {-1, 0, 1} is inherently symmetric.

---

## 3.3 Per-Tensor vs. Per-Channel vs. Per-Token Quantization

### Granularity

```
Per-Tensor:  One scale for entire weight matrix
             + Simple, fast
             - Less accurate (outliers affect all values)

Per-Channel: One scale per output channel (row of weight matrix)
             + More accurate
             - More storage for scales

Per-Token:   One scale per token (for activations)
             + Handles varying activation magnitudes
             - Computed at runtime
```

**BitNet uses:**
- **Per-tensor** quantization for weights (absmean)
- **Per-token** quantization for activations (absmax)

This is simpler than per-channel methods but works well because the ternary constraint already heavily regularizes the weights.

---

## 3.4 Post-Training Quantization (PTQ)

Quantize a model **after** it's been trained in full precision.

```python
def post_training_quantize(weight, bits=8):
    """Quantize a pre-trained weight matrix."""
    scale = weight.abs().max() / (2**(bits-1) - 1)
    q_weight = torch.round(weight / scale).clamp(-2**(bits-1), 2**(bits-1) - 1)
    return q_weight, scale
```

**Pros:** Quick, no retraining needed.
**Cons:** Accuracy drops, especially at very low bit widths.

Popular PTQ methods: GPTQ, AWQ, SqueezeLLM

**This is NOT what BitNet does.** PTQ at 1.58 bits would destroy the model. BitNet trains with quantization from the start.

---

## 3.5 Quantization-Aware Training (QAT)

Include quantization in the **training loop** so the model learns to work with quantized weights.

```python
def qat_forward(x, weight):
    """Forward pass with quantization-aware training."""
    # Quantize
    q_weight = quantize(weight)
    # Dequantize (simulate quantization error)
    dq_weight = dequantize(q_weight)
    # Use dequantized weights for forward pass
    return x @ dq_weight.T
```

The model "sees" quantization errors during training and learns to be robust to them.

**BitNet IS a form of QAT** — specifically, it's the most extreme form, training with ternary weights from the very beginning.

---

## 3.6 The Quantization Spectrum

```
FP32 ──→ FP16/BF16 ──→ INT8 ──→ INT4 ──→ Ternary (1.58b) ──→ Binary (1b)
 32 bits    16 bits     8 bits   4 bits     1.58 bits           1 bit

 ←── Higher accuracy                          Lower memory ──→
 ←── More compute                             Less compute ──→
```

### The BitNet Sweet Spot

Why ternary {-1, 0, 1} instead of binary {-1, 1}?

The zero value adds **sparsity**. When a weight is 0:
- That connection is effectively **pruned**
- No computation needed (skip the addition)
- The model can learn which connections don't matter

In practice, ~30-50% of BitNet weights are zero, giving a natural form of structured sparsity.

---

## 3.7 Information Theory: Bits per Parameter

Claude Shannon showed that to encode N equally likely values, you need:

```
bits = log2(N)
```

| Values | N | Bits needed | Name |
|--------|---|-------------|------|
| {-1, 1} | 2 | log2(2) = 1.00 | Binary |
| {-1, 0, 1} | 3 | log2(3) = 1.58 | Ternary |
| {-2, -1, 0, 1} | 4 | log2(4) = 2.00 | 2-bit |
| {-8..7} | 16 | log2(16) = 4.00 | INT4 |
| {-128..127} | 256 | log2(256) = 8.00 | INT8 |

### Practical Storage

You can't use 1.58 bits per value in actual hardware (memory is byte-addressed). So BitNet **packs** 4 ternary values into 1 byte (INT8):

```
4 ternary values: each has 3 states
3^4 = 81 possible combinations
log2(81) = 6.34 bits needed
Stored in 8 bits (1 byte) → 2 bits per value effective

Alternative: 2-bit encoding per value
  -1 → 00
   0 → 01
   1 → 10
  4 values → 1 byte
```

---

## 3.8 The Straight-Through Estimator (STE)

**The most important concept for training quantized models.**

### The Problem

The `round()` function is not differentiable:

```
round(2.3) = 2
round(2.7) = 3

d/dx round(x) = 0 almost everywhere (flat)
                 undefined at x.5 (discontinuity)
```

If gradients are zero, backpropagation can't update the weights. Training stops.

### The Solution: STE

Proposed by Bengio et al. (2013). The idea: **pretend round() is the identity function during backpropagation**.

```
Forward pass:  w_q = round(w)         ← Use quantized weights
Backward pass: dL/dw ≈ dL/dw_q * 1   ← Treat round() as identity
```

### STE in PyTorch: The Detach Trick

```python
def ste_round(x):
    """Round with straight-through gradient."""
    return x + (x.round() - x).detach()
    #      ^                    ^^^^^^^
    #      |                    This stops gradient flow through (round - x)
    #      Gradient flows through x directly
```

**How it works:**
1. Forward: `x + (round(x) - x) = round(x)` (the `x` cancels out)
2. Backward: `detach()` makes `(round(x) - x)` a constant with zero gradient
3. So gradient flows through just `x`, as if round() wasn't there

### In BitLinear

```python
# Weight quantization with STE
w_quant = w + (weight_quant(w) - w).detach()

# Activation quantization with STE
x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
```

Forward pass uses quantized values. Backward pass computes gradients as if quantization didn't happen. The optimizer updates the full-precision shadow weights.

### The Shadow Weight Pattern

```
Training step:
1. shadow_weight (FP16): [0.342, -0.891, 0.015, 0.723]
2. Quantize (forward):    [1, -1, 0, 1]  (ternary)
3. Compute loss with quantized weights
4. Backprop gradients to shadow_weight (STE)
5. Optimizer updates shadow_weight:
   shadow_weight (FP16): [0.341, -0.893, 0.013, 0.725]
6. Next forward pass: requantize [1, -1, 0, 1]
```

The shadow weights slowly drift. Eventually, they cross quantization boundaries and the quantized weight flips (e.g., from 0 to 1).

---

## Exercise 3: INT8 Quantization by Hand

See `../exercises/exercise_03_quantization.py`.

Tasks:
1. Implement symmetric INT8 quantization and dequantization
2. Measure quantization error on random tensors
3. Implement STE and verify gradients flow correctly
4. Implement ternary quantization {-1, 0, 1}
5. Compare accuracy of different quantization levels on a simple model

---

## Key Takeaways

1. **Quantization** reduces precision to save memory and compute
2. **QAT** (training-aware) beats **PTQ** (post-training), especially at low bit widths
3. **Ternary (1.58-bit)** is the sweet spot: extreme compression with sparsity benefits
4. **STE** is the key trick: forward pass uses quantized values, backward pass pretends quantization didn't happen
5. The **detach trick** in PyTorch implements STE elegantly: `x + (quant(x) - x).detach()`
6. **Shadow weights** in full precision are maintained during training and updated by the optimizer
