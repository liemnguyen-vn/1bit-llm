# Module 1: Foundations

## 1.1 Linear Algebra Refresher

Everything in neural networks boils down to **matrix multiplication**. Before we can understand 1-bit LLMs, we need to be fluent in the basics.

### Vectors

A vector is an ordered list of numbers:

```
x = [3, 1, 4]     # A vector in R^3
```

In neural networks, vectors represent:
- Input features (e.g., a tokenized word → embedding vector)
- Hidden states
- Gradients

### Matrices

A matrix is a 2D grid of numbers. An `m x n` matrix has `m` rows and `n` columns:

```
W = [[1, 2, 3],
     [4, 5, 6]]    # 2x3 matrix
```

In neural networks, matrices represent:
- **Weight matrices** (the parameters we train)
- **Attention score matrices**
- **Batched inputs**

### Matrix Multiplication

The operation at the heart of ALL neural networks. Given matrix A (m x k) and B (k x n):

```
C = A @ B    # Result is m x n

C[i][j] = sum(A[i][p] * B[p][j] for p in range(k))
```

**Why this matters for 1-bit LLMs:** In a standard LLM, each element of A and B is a 16-bit floating point number. In BitNet, each weight in B is just {-1, 0, 1}. This means:
- **Multiplication becomes trivial:** `x * 1 = x`, `x * 0 = 0`, `x * (-1) = -x`
- **We replace multiply-accumulate with add/subtract/skip**
- **This is the entire insight behind 1-bit LLMs**

### Key Intuition

```
Standard: y = W @ x    →  millions of FP16 multiplications
BitNet:   y = W @ x    →  millions of additions/subtractions (no multiplications!)
```

---

## 1.2 Neural Network Fundamentals

### The Neuron

A single neuron computes:

```
output = activation(w1*x1 + w2*x2 + ... + wn*xn + bias)
       = activation(W @ x + b)
```

### Forward Pass

Data flows through layers:

```
Input → Linear(W1) → ReLU → Linear(W2) → ReLU → Linear(W3) → Output
```

Each `Linear` layer does: `y = W @ x + b`

In code:

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),   # W1: 784x256 = 200,704 parameters
    nn.ReLU(),
    nn.Linear(256, 128),   # W2: 256x128 = 32,768 parameters
    nn.ReLU(),
    nn.Linear(128, 10),    # W3: 128x10 = 1,280 parameters
)
# Total: 234,752 parameters, each stored as FP32 (4 bytes) = ~940 KB
# In 1-bit: each weight is {-1,0,1}, needing ~1.58 bits = ~46 KB
```

### Backpropagation

Training adjusts weights to minimize a loss function. The chain rule computes gradients:

```
dL/dW = dL/dy * dy/dW
```

**Critical for BitNet:** The quantization function `round()` has zero gradient almost everywhere. We solve this with the **Straight-Through Estimator** (covered in Module 3).

### Gradient Descent

```python
for each training step:
    y_pred = model(x)              # Forward pass
    loss = loss_fn(y_pred, y_true)  # Compute loss
    loss.backward()                 # Backward pass (compute gradients)
    optimizer.step()                # Update weights: W = W - lr * dL/dW
    optimizer.zero_grad()
```

---

## 1.3 Loss Functions and Optimization

### Cross-Entropy Loss (for language models)

LLMs predict the next token. Cross-entropy measures how wrong the prediction is:

```
L = -sum(y_true[i] * log(y_pred[i]))
```

For next-token prediction with vocabulary size V:
- `y_true` is one-hot (1 for correct token, 0 elsewhere)
- `y_pred` is a probability distribution over V tokens (after softmax)

### Perplexity

Perplexity = `exp(cross_entropy_loss)`. Lower is better.

- Perplexity 1 = perfect prediction
- Perplexity 10 = on average, the model is as confused as choosing between 10 equally likely options
- BitNet b1.58 matches LLaMA perplexity at 3B+ parameters

### Adam Optimizer

The go-to optimizer for LLMs. It adapts the learning rate per-parameter:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

BitNet uses Adam with standard settings. The trick is that shadow weights (full precision) are updated by Adam, then quantized for the forward pass.

---

## 1.4 Number Representation in Computers

This is **the core problem** that 1-bit LLMs solve.

### Floating Point Formats

| Format | Bits | Range | Precision | Memory per param |
|--------|------|-------|-----------|-----------------|
| FP32   | 32   | ±3.4e38 | ~7 decimal digits | 4 bytes |
| FP16   | 16   | ±65504 | ~3.3 digits | 2 bytes |
| BF16   | 16   | ±3.4e38 | ~2.4 digits | 2 bytes |
| INT8   | 8    | -128 to 127 | Exact integers | 1 byte |
| INT4   | 4    | -8 to 7 | Exact integers | 0.5 bytes |
| Binary | 1    | {-1, 1} | Two values | 0.125 bytes |
| Ternary| 1.58 | {-1, 0, 1} | Three values | ~0.2 bytes |

### The LLM Memory Problem

A 7B parameter LLM in FP16:
```
7,000,000,000 params * 2 bytes = 14 GB
```

The same model in 1.58-bit (ternary):
```
7,000,000,000 params * 0.2 bytes ≈ 1.4 GB
```

**That's a 10x reduction in memory.** This is why 1-bit LLMs matter.

### Why "1.58 bits"?

Information theory tells us how many bits we need to encode N distinct values:

```
bits = log2(N)

Binary  {-1, 1}:    log2(2) = 1.00 bits
Ternary {-1, 0, 1}: log2(3) = 1.58 bits
INT4    {-8..7}:     log2(16) = 4.00 bits
INT8    {-128..127}: log2(256) = 8.00 bits
```

---

## 1.5 Why Quantization Matters: The Memory Wall

### The Bottleneck

Modern GPUs are incredibly fast at computation, but moving data from memory to compute units is slow. This is the **memory wall**.

```
GPU Compute: ~300 TFLOPS (can do 300 trillion operations/second)
GPU Memory Bandwidth: ~2 TB/s (can move 2 trillion bytes/second)

For a 7B FP16 model:
- Memory needed: 14 GB
- Just LOADING weights: 14 GB / 2 TB/s = 7 ms
- Actual compute: << 1 ms

The model is MEMORY-BOUND, not compute-bound.
```

### How 1-bit LLMs Help

By reducing each parameter from 16 bits to 1.58 bits:
- **10x less memory** → smaller GPUs, or bigger models on the same GPU
- **10x less data movement** → faster inference
- **No multiplications** → simpler, faster hardware
- **Less energy** → 71x reduction in matrix multiply energy

### The Key Insight

In a standard LLM, each `y = W @ x` involves:
```
FP16 multiply: ~15 pJ per operation
FP16 add: ~1 pJ per operation
```

In a 1-bit LLM, weights are {-1, 0, 1}, so:
```
INT8 add: ~0.03 pJ per operation (no multiply needed!)
```

That's a ~500x energy reduction per operation.

---

## Exercise 1: Two-Layer MLP from Scratch

Build and train a 2-layer MLP on MNIST using only PyTorch primitives. See `../exercises/exercise_01_mlp.py`.

Key learning goals:
1. Implement matrix multiplication manually
2. Understand what `nn.Linear` does under the hood
3. See the weight matrices and understand their shapes
4. Train the model and watch the loss decrease

---

## Key Takeaways

1. **Matrix multiplication** is the core operation in all neural networks
2. Standard LLMs store weights in FP16 (16 bits per parameter)
3. 1-bit LLMs store weights as {-1, 0, 1} using only 1.58 bits per parameter
4. This gives ~10x memory savings and eliminates expensive floating-point multiplications
5. The challenge: how do you **train** a model when weights can only be {-1, 0, 1}?

That challenge is what the rest of this course answers.
