# Module 4: Binary Neural Networks — A Brief History

## 4.1 BinaryConnect (2015): The First Binary Weights

**Paper:** Courbariaux et al., "BinaryConnect: Training Deep Neural Networks with binary weights during propagations"

The idea that started it all: what if we constrain neural network weights to just {-1, +1}?

### The Method

```python
def binaryconnect_forward(weight_fp, x):
    """BinaryConnect: binarize weights, keep full-precision shadow copy."""
    # Stochastic binarization (training)
    # Probability of +1 is proportional to how positive the weight is
    p = (weight_fp + 1) / 2  # Map [-1,1] → [0,1]
    weight_binary = torch.where(torch.rand_like(p) < p,
                                 torch.ones_like(p),
                                 -torch.ones_like(p))

    # OR deterministic binarization (inference)
    # weight_binary = weight_fp.sign()

    return x @ weight_binary.T
```

### Key Innovations
1. **Shadow weights:** Maintain full-precision weights for gradient updates
2. **Stochastic binarization:** Randomly round to +1 or -1 based on weight value
3. **STE for backprop:** Gradients pass through the binarization as if it were identity

### Limitations
- Activations remained full precision (FP32)
- Only tested on small datasets (CIFAR-10, SVHN)
- Noticeable accuracy drop vs. full-precision models

---

## 4.2 BinaryNet (2016): Binary Weights AND Activations

**Paper:** Courbariaux et al., "Binarized Neural Networks"

Extended BinaryConnect to also binarize activations.

### The Method

```python
def binarynet_forward(weight_fp, x):
    """BinaryNet: both weights and activations are binary."""
    # Binarize weights
    weight_binary = weight_fp.sign()  # {-1, +1}

    # Binarize activations (using sign function instead of ReLU)
    x_binary = x.sign()  # {-1, +1}

    # Matrix multiply: all values are ±1
    # This means multiply becomes XNOR, accumulate becomes popcount
    return x_binary @ weight_binary.T
```

### The XNOR-Popcount Trick

When both operands are binary {-1, +1}, encoded as bits {0, 1}:

```
Multiplication of two binary values = XNOR operation
  (+1) * (+1) = +1  →  1 XNOR 1 = 1
  (+1) * (-1) = -1  →  1 XNOR 0 = 0
  (-1) * (+1) = -1  →  0 XNOR 1 = 0
  (-1) * (-1) = +1  →  0 XNOR 0 = 1

Dot product = 2 * popcount(XNOR(a, b)) - n
  where n is the vector length
```

This means a dot product of 64 binary values can be computed with:
1. One 64-bit XNOR operation
2. One popcount (count set bits)
3. One multiply-add (2 * result - 64)

Compared to 64 floating-point multiply-adds. **Massive speedup potential.**

---

## 4.3 XNOR-Net (2016): Making It Work on ImageNet

**Paper:** Rastegari et al., "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks"

BinaryNet struggled on large datasets. XNOR-Net added **scaling factors** to make binary networks viable on ImageNet.

### The Key Insight

Don't just use `sign(W)`, use `sign(W) * alpha` where alpha preserves magnitude information:

```python
def xnor_net_quantize(weight):
    """XNOR-Net: binary weights with per-channel scaling."""
    # Binary weights
    weight_binary = weight.sign()  # {-1, +1}

    # Per-channel scaling factor (preserves magnitude)
    alpha = weight.abs().mean(dim=1, keepdim=True)  # Mean absolute value per row

    # Scaled binary weight approximation
    return weight_binary * alpha
    # W ≈ alpha * sign(W)
```

### Results

```
Model          ImageNet Top-1    Memory Savings    Speed
────────────────────────────────────────────────────────
ResNet-18      69.3%             -                 -
Binary ResNet  51.2%             32x               58x faster
XNOR ResNet    51.2%             32x               58x faster
(with scaling)
```

~18% accuracy drop on ImageNet — significant improvement over raw binarization, but still a large gap.

---

## 4.4 The XNOR-Bitcount Trick (In Detail)

This is the fundamental computation optimization for binary networks:

### Standard Matrix Multiply (FP32)

```
A (m x k) @ B (k x n) → C (m x n)

Operations: m * n * k multiplications + m * n * k additions
Each multiply: ~15 pJ energy
Each add: ~1 pJ energy
```

### Binary Matrix Multiply (XNOR + Popcount)

```
Encode weights as bits: +1 → 1, -1 → 0
Pack 64 binary weights into one uint64

For each output element C[i][j]:
  1. Load A_bits[i] and B_bits[j]  (64 binary values packed in uint64)
  2. result = XNOR(A_bits[i], B_bits[j])
  3. count = popcount(result)  # Count 1s = count of matching signs
  4. C[i][j] = 2 * count - 64  # Convert to actual dot product

Energy per operation: ~0.1 pJ (vs ~15 pJ for FP32 multiply)
Memory: 64x less (1 bit vs 32 bits per weight)
```

---

## 4.5 Why Binary Nets Struggled for Years

Despite the theoretical advantages, binary networks never achieved widespread adoption. Here's why:

### 1. Accuracy Gap
Binary networks consistently showed 10-20% accuracy drops on challenging tasks. For production use, this was unacceptable.

### 2. Training Instability
With only {-1, +1} for weights, the loss landscape is extremely jagged. Small gradient updates don't change quantized weights until they cross the sign boundary.

### 3. Limited Representational Power
Binary weights have very limited capacity. A binary linear layer with k inputs can only represent 2^k distinct functions (vs. infinite for continuous weights).

### 4. The Activation Problem
Binary activations ({-1, +1}) lose too much information. Every intermediate representation is binary, creating an information bottleneck.

### 5. No Scaling to Language Models
All binary network research was on **CNNs for image classification**. Language models have different requirements:
- Sequential processing (autoregressive generation)
- Very large models (billions of parameters)
- Attention mechanisms (different from convolutions)

---

## 4.6 The Gap: From Binary CNNs to Binary LLMs

### Timeline

```
2015: BinaryConnect (binary weights, small CNNs)
2016: BinaryNet (binary weights + activations)
2016: XNOR-Net (scaling factors, ImageNet)
2017-2022: Many incremental BNN improvements, mostly CNNs
            ─── Large gap: nobody tackled binary LLMs ───
2023: BitNet b1 (Microsoft) — First binary-weight Transformer LLM
2024: BitNet b1.58 — Ternary weights, matches full-precision LLMs
2025: BitNet b1.58 2B4T — First open-source 1-bit LLM model
```

### What Changed? Three Key Insights from BitNet

**1. Ternary instead of binary**
Adding `0` to the value set {-1, 0, 1} was transformative:
- Provides natural sparsity (30-50% of weights are zero)
- `log2(3) = 1.58` bits — barely more than 1 bit, but dramatically more expressive
- Zero means "this connection doesn't matter" → learned pruning

**2. Keep activations at higher precision**
BitNet doesn't binarize activations — it uses 8-bit integer activations:
- Avoids the information bottleneck that killed earlier binary nets
- 8-bit activations are well-understood and hardware-supported
- The quantization only affects weights, where the parameter count lives

**3. Scale matters**
At 3B+ parameters, ternary models match full-precision models:
- Larger models are more robust to quantization
- The redundancy in LLMs at scale allows ternary representation to work
- Smaller models (< 1B) still see accuracy drops

### The Conceptual Evolution

```
BinaryConnect:  Binary weights, FP32 activations, small CNNs
     ↓
BinaryNet:      Binary weights, binary activations, small CNNs
     ↓
XNOR-Net:       Binary weights + scaling, binary activations, ImageNet
     ↓
(many papers trying to close accuracy gap on CNNs)
     ↓
BitNet b1:      Binary weights {-1,1}, INT8 activations, Transformer LLMs
     ↓
BitNet b1.58:   Ternary weights {-1,0,1}, INT8 activations, LLMs
                 → MATCHES full-precision LLMs at scale
```

---

## Exercise 4: Binary Linear Layer with STE

See `../exercises/exercise_04_binary_linear.py`.

Tasks:
1. Implement a binary linear layer (weights constrained to {-1, +1})
2. Use STE for backpropagation
3. Train on a simple classification task
4. Compare accuracy vs. full-precision linear layer
5. Add scaling factors (XNOR-Net style)
6. Observe the accuracy gap and understand why ternary is better

---

## Key Takeaways

1. Binary neural networks started in 2015 with BinaryConnect
2. The XNOR-popcount trick replaces multiply-accumulate with bit operations
3. Binary CNNs had 10-20% accuracy gaps — too large for practical use
4. Three key insights enabled BitNet: **ternary weights**, **8-bit activations**, **scale**
5. The zero in {-1, 0, 1} provides natural sparsity — a form of learned pruning
6. At 3B+ parameters, ternary models match full-precision models
