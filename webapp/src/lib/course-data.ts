import { Course } from "./types";

// === MODULE CONTENT (Markdown) — declared before COURSE to avoid TDZ ===

const MOD_01_CONTENT = `## Why This Module Matters

Everything in neural networks — and therefore everything in 1-bit LLMs — boils down to **matrix multiplication**. Before we can understand BitNet, we need fluency in the basics.

## Linear Algebra Refresher

### Vectors and Matrices

A **vector** is an ordered list of numbers. In neural networks, vectors represent token embeddings, hidden states, and gradients.

A **matrix** is a 2D grid of numbers. Neural network **weight matrices** are the parameters we train.

### Matrix Multiplication

The operation at the heart of ALL neural networks. Given matrix A (m×k) and B (k×n):

\`\`\`
C = A @ B    →    C[i][j] = Σ A[i][p] × B[p][j]
\`\`\`

**Why this matters for 1-bit LLMs:** In a standard LLM, each element is a 16-bit float. In BitNet, each weight is just {-1, 0, 1}:
- \`x × 1 = x\` (just use x)
- \`x × 0 = 0\` (skip entirely)
- \`x × (-1) = -x\` (negate x)

**No multiplications needed. Just additions, subtractions, and skips.**

## Neural Network Fundamentals

### The Neuron
\`\`\`
output = activation(W @ x + b)
\`\`\`

### Forward Pass
Data flows through layers: Input → Linear → Activation → Linear → Output

### Backpropagation
Training adjusts weights to minimize loss. The chain rule computes gradients. **Critical for BitNet:** The \`round()\` function has zero gradient — solved by the **Straight-Through Estimator** (Module 3).

## Number Representation

| Format | Bits | Memory/param | Use in BitNet |
|--------|------|-------------|---------------|
| FP32 | 32 | 4 bytes | Optimizer state |
| FP16 | 16 | 2 bytes | Standard LLM weights |
| INT8 | 8 | 1 byte | BitNet activations |
| Ternary | 1.58 | ~0.2 bytes | **BitNet weights** |

### The Memory Problem

A 7B parameter LLM in FP16: **14 GB**
The same in 1.58-bit: **~1.4 GB** → **10× reduction**

## The Memory Wall

Modern GPUs compute faster than they can read data. LLM inference is **memory-bound** — the bottleneck is loading weights from memory, not computing with them.

1-bit LLMs help by:
- **10× less memory** → fits on smaller devices
- **10× less data movement** → faster inference
- **No multiplications** → simpler hardware
- **71× less energy** in matrix multiply
`;

const MOD_02_CONTENT = `## The Transformer: Where BitNet Lives

Every modern LLM is built on the Transformer architecture. Understanding it is essential because BitNet modifies specific parts of it.

## Attention Mechanism

The key innovation: allowing every token to look at every other token.

\`\`\`
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
\`\`\`

- **Q** (Query): "What am I looking for?"
- **K** (Key): "What do I contain?"
- **V** (Value): "What information do I provide?"

## Self-Attention with Learned Projections

Q, K, V come from the same input, projected through **learned weight matrices**:

\`\`\`python
Q = x @ W_q    # These W_q, W_k, W_v matrices
K = x @ W_k    # are exactly what BitNet
V = x @ W_v    # quantizes to {-1, 0, 1}
\`\`\`

## Multi-Head Attention

Multiple attention "heads" attend to different aspects of the input. BitNet 2B4T uses 20 heads with 5 KV heads (Grouped Query Attention).

## Feed-Forward Network

After attention, each position passes through an FFN:
\`\`\`
FFN(x) = W2 × SquaredReLU(W1 × x)
\`\`\`

BitNet uses **Squared ReLU** (\`ReLU(x)²\`) which produces sparser activations.

## The Full Transformer Block

\`\`\`
x = x + Attention(RMSNorm(x))    ← residual + pre-norm
x = x + FFN(RMSNorm(x))          ← residual + pre-norm
\`\`\`

## Where Parameters Live

~95% of LLM parameters are in \`nn.Linear\` layers. BitNet replaces these with \`BitLinear\`:

| Component | Standard | BitNet |
|-----------|----------|--------|
| W_q, W_k, W_v, W_o | FP16 | **Ternary {-1,0,1}** |
| FFN w1, w2 | FP16 | **Ternary {-1,0,1}** |
| Embeddings | FP16 | FP16 (kept) |
| RMSNorm | FP32 | FP32 (kept) |
`;

const MOD_03_CONTENT = `## Quantization: The Bridge to 1-Bit

Quantization maps values from a large set (all floats) to a smaller set (e.g., 256 integers, or just 3 values).

## Symmetric Quantization

Zero maps to zero. Used by BitNet for both weights and activations.

\`\`\`
quantize(x) = round(x / scale)
dequantize(q) = q × scale
\`\`\`

## Granularity

| Level | Description | BitNet uses |
|-------|-------------|-------------|
| Per-tensor | One scale for all | **Weights** |
| Per-channel | One scale per row | — |
| Per-token | One scale per token | **Activations** |

## PTQ vs QAT

**Post-Training Quantization (PTQ):** Quantize after training. Quick but lossy at low bit widths.

**Quantization-Aware Training (QAT):** Include quantization in training. The model learns to work with it. **BitNet IS QAT** — the most extreme form, training with ternary weights from the start.

## The Straight-Through Estimator (STE)

**THE most important concept for training quantized models.**

### The Problem
\`round()\` has zero gradient almost everywhere. No gradient → no learning → training stops.

### The Solution
Pretend \`round()\` is the identity function during backpropagation:

\`\`\`
Forward:  w_q = round(w)           ← Use quantized weights
Backward: dL/dw ≈ dL/dw_q × 1    ← Treat round() as identity
\`\`\`

### PyTorch Implementation: The Detach Trick

\`\`\`python
w_quant = w + (weight_quant(w) - w).detach()
\`\`\`

- **Forward:** \`w + (wq - w) = wq\` (quantized value)
- **Backward:** \`.detach()\` makes \`(wq - w)\` a constant → gradient flows through \`w\` only

### The Shadow Weight Pattern

\`\`\`
1. shadow_weight (FP16):  [0.342, -0.891, 0.015]
2. Quantize (forward):     [1,     -1,     0]      (ternary)
3. Compute loss, backprop gradients to shadow_weight
4. Optimizer updates:       [0.341, -0.893, 0.013]
5. Re-quantize:            [1,     -1,     0]       (same — until boundary crossed)
\`\`\`
`;

const MOD_04_CONTENT = `## A Brief History of Binary Neural Networks

The path from 1-bit CNNs to 1-bit LLMs.

## BinaryConnect (2015)
First paper to constrain weights to {-1, +1}. Used stochastic binarization and STE for backprop. Only tested on small CNNs.

## BinaryNet (2016)
Extended BinaryConnect to also binarize activations. Both weights AND activations are binary.

## XNOR-Net (2016)
Added per-channel scaling factors to make binary nets work on ImageNet. Key insight: \`W ≈ α × sign(W)\`.

### The XNOR-Bitcount Trick
When both operands are binary, multiplication becomes XNOR, and dot products become popcount:
\`\`\`
dot_product = 2 × popcount(XNOR(a, b)) - n
\`\`\`
58× faster convolutions, 32× memory savings.

## Why Binary Nets Struggled
1. **10-20% accuracy gap** on challenging tasks
2. **Training instability** — jagged loss landscape
3. **Limited representational power** with only {-1, +1}
4. **Information bottleneck** from binary activations
5. **Never scaled to LLMs** — all research was on CNNs

## What Changed: Three Key Insights

### 1. Ternary Instead of Binary
Adding \`0\` to {-1, +1} was transformative:
- Provides natural sparsity (30-50% zeros)
- Only 0.58 bits more, but dramatically more expressive
- Zero means "this connection doesn't matter"

### 2. Keep Activations at Higher Precision
8-bit activations avoid the information bottleneck.

### 3. Scale Matters
At 3B+ parameters, ternary matches full-precision models.

## Timeline
\`\`\`
2015: BinaryConnect
2016: BinaryNet, XNOR-Net
2017-2022: Incremental BNN improvements (CNNs only)
2023: BitNet b1 — First binary Transformer LLM
2024: BitNet b1.58 — Ternary, MATCHES full precision
2025: BitNet b1.58 2B4T — First open-source 1-bit LLM
\`\`\`
`;

const MOD_05_CONTENT = `## BitNet Architecture: The Complete Picture

This is the heart of the course. Every detail of the BitLinear layer.

## BitNet b1 vs b1.58

| | BitNet b1 (2023) | BitNet b1.58 (2024) |
|---|---|---|
| Weights | Binary {-1, +1} | **Ternary {-1, 0, +1}** |
| Activations | INT8 | INT8 |
| Sparsity | None (all weights active) | **30-50% zeros** |
| vs. FP16 | Gap narrows with scale | **Matches at 3B+** |

## The BitLinear Layer

\`\`\`
Input x → RMSNorm → Activation Quant (INT8) ──┐
                                                ├── MatMul → Output
Weight W → Weight Quant (ternary) ─────────────┘
\`\`\`

## Weight Quantization: AbsMean

\`\`\`python
def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u
\`\`\`

**Step by step:**
1. \`scale = 1 / mean(|W|)\` — average absolute value as scaling factor
2. \`W × scale\` — scale so average magnitude ≈ 1
3. \`round()\` — snap to nearest integer
4. \`clamp(-1, 1)\` — constrain to ternary range
5. \`/ scale\` — dequantize back to original magnitude

**Why absmean?** More robust than absmax (not sensitive to outliers).

## Activation Quantization: AbsMax (Per-Token)

\`\`\`python
def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y
\`\`\`

Per-token ensures each token uses the full INT8 range [-128, 127].

## STE: The Detach Trick

\`\`\`python
w_quant = w + (weight_quant(w) - w).detach()
x_quant = x + (activation_quant(x) - x).detach()
\`\`\`

Forward = quantized values. Backward = gradients flow through shadow weights.

## Why 1.58 Bits?
\`log₂(3) = 1.58\` — the information-theoretic minimum bits for 3 values.

## The Role of Zero
~30-50% of trained BitNet weights are zero. This means:
- Those connections are **effectively pruned**
- **No computation needed** for zero weights
- The model **learns which connections don't matter**

## Benchmarks: BitNet b1.58 3B vs LLaMA 3B

| Metric | LLaMA 3B | BitNet 3B | Improvement |
|--------|----------|-----------|-------------|
| Perplexity | baseline | matches | — |
| Memory | 6.0 GB | 1.7 GB | **3.55×** less |
| Latency | 1× | 0.37× | **2.71×** faster |
| Energy | 1× | 0.014× | **71×** less |
`;

const MOD_06_CONTENT = `## Building BitNet from Scratch

Time to write code. We'll implement everything in PyTorch.

## Core: weight_quant()

\`\`\`python
def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u
\`\`\`

## Core: activation_quant()

\`\`\`python
def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y
\`\`\`

## The BitLinear Layer

\`\`\`python
class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features)

    def forward(self, x):
        w = self.weight
        x_norm = self.norm(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        return F.linear(x_quant, w_quant)
\`\`\`

## Assembling the Full Model

\`\`\`
MiniBitNet:
  ├── Token Embedding (FP16)
  ├── Position Embedding (FP16)
  ├── N × BitNetBlock:
  │   ├── RMSNorm → BitNetAttention (all BitLinear)
  │   └── RMSNorm → BitNetFFN (BitLinear + Squared ReLU)
  ├── Final RMSNorm
  └── Output Head (FP16, weight-tied with embedding)
\`\`\`

## Training Loop

The key insight: **the optimizer updates full-precision shadow weights**. Quantization only happens in the forward pass.

\`\`\`python
for batch in data:
    logits = model(x)                    # Quantization via STE
    loss = cross_entropy(logits, target)
    loss.backward()                       # Gradients → shadow weights
    clip_grad_norm_(params, 1.0)
    optimizer.step()                      # Update shadow weights
\`\`\`

## What to Monitor
- Weight distribution: expect ~30-50% zeros
- Quantization error: should be low relative to weight magnitude
- Loss/perplexity: should decrease, comparable to FP16 at scale

## Memory Savings

\`\`\`
Tiny model (~2M params):
  FP16: ~4 MB
  BitNet: ~0.7 MB (5.7× smaller)
\`\`\`

The exercises include complete runnable code — see \`bitnet_from_scratch.py\`.
`;

const MOD_07_CONTENT = `## Training 1-Bit LLMs at Scale

Moving from toy experiments to production-grade training.

## Training Recipe

\`\`\`
Architecture: Transformer decoder (LLaMA-like) with BitLinear
Optimizer: Adam (β1=0.9, β2=0.95, weight_decay=0.1)
LR: 1.5e-4 peak, cosine decay, 2000-step warmup
Batch: ~2M tokens effective
Gradient clipping: 1.0
\`\`\`

## Learning Rate Sensitivity

BitNet is more sensitive to LR than FP16 models. Too high → weights oscillate wildly around quantization boundaries.

## Lambda Scheduling (for Fine-tuning)

Applying full quantization to pre-trained weights immediately is catastrophic (loss: 2 → 13). Solution: **gradually introduce quantization**.

\`\`\`python
lambda_val = min(2 * step / total_steps, 1.0)  # Reach λ=1 at 50%
x_quant = x + lambda_val * (activation_quant(x) - x).detach()
w_quant = w + lambda_val * (weight_quant(w) - w).detach()
\`\`\`

| Scheduler | WikiText PPL (after 10B tokens) |
|-----------|-------------------------------|
| No warmup (λ=1) | 26-30 (poor) |
| **Linear** | **12.2 (best)** |
| Exponential (k=8) | ~15 |
| Sigmoid (k=100) | ~13.5 |

## Fine-tuning vs Training from Scratch

\`\`\`
Fine-tuning Llama3 8B → 1.58-bit (10B tokens):  MMLU 47.3
Training from scratch BitNet 7B (100B tokens):   MMLU 41.4
\`\`\`

Fine-tuning is more efficient — 10B tokens gets results that take 100B from scratch.

## Dataset Diversity Matters

Low-capacity models overfit narrow data:
- FineWeb-edu → WikiText PPL: 12 (good)
- TinyStories only → WikiText PPL: 42 (terrible)

## BitNet b1.58 2B4T

First open-source 1-bit LLM (Microsoft, 2025):
- 2.4B params, 4T training tokens
- 30 layers, hidden=2560, 20 heads (5 KV), FFN=6912
- Context: 4096, Squared ReLU, RoPE, no bias
`;

const MOD_08_CONTENT = `## Making 1-Bit LLMs Fast

Turning ternary weights into real speed improvements.

## Weight Packing: 4 Values per Byte

Each ternary value {-1, 0, 1} → 2-bit encoding:
\`\`\`
-1 → 00,  0 → 01,  1 → 10
4 values packed into 1 uint8 byte
\`\`\`

A 7B model: 14 GB (FP16) → ~1.75 GB (packed ternary)

## Pack-Store-Load-Unpack-Compute Pipeline

1. **Pack** (offline): Quantize + pack 4 values/byte
2. **Store**: Packed weights in GPU memory (8× smaller)
3. **Load**: Tiled loading (8× less bandwidth)
4. **Unpack**: On-the-fly in registers (bit-shift + mask)
5. **Compute**: INT8 accumulation (no FP multiply)

## Custom Kernels

Standard \`torch.mm\` doesn't exploit ternary structure. Custom Triton kernels unpack on-the-fly and use INT8 addition.

## bitnet.cpp (CPU Inference)

Microsoft's framework uses **lookup tables**: for each packed byte (4 ternary values, 256 possibilities), pre-compute the sum. One table lookup = 4 multiply-accumulates.

\`\`\`bash
git clone https://github.com/microsoft/BitNet.git
python setup_env.py --hf-repo microsoft/BitNet-b1.58-2B-4T-gguf -q i2_s
python run_inference.py -m models/.../ggml-model-i2_s.gguf -p "prompt" -n 100
\`\`\`

## Energy Efficiency

| Operation | Energy (pJ) |
|-----------|------------|
| FP16 multiply | 1.1 |
| FP16 add | 0.4 |
| INT8 add | 0.03 |

BitNet: ~0.015 pJ/op vs FP16: ~1.5 pJ/op → **~71× less energy**

## What This Enables

| Scenario | FP16 | BitNet |
|----------|------|--------|
| 7B on phone | Impossible (14GB) | **Possible (~1.4GB)** |
| 70B on laptop | 140GB RAM needed | **~14GB — feasible** |

## Future: Custom Hardware

Ideal BitNet accelerator: ternary weight memory + INT8 adders only + zero-skip logic. No FP multipliers needed. Research ongoing in FPGAs, ASICs, and in-memory computing.
`;

export const COURSE: Course = {
  id: "1bit-llm",
  title: "1-Bit LLMs from Scratch",
  description:
    "Master the theory and implementation of 1-bit Large Language Models (BitNet). Learn the math, build the code, and understand the most efficient frontier in LLM research.",
  modules: [
    {
      id: "mod-01",
      number: 1,
      title: "Foundations",
      description:
        "Linear algebra, neural network basics, number representation, and the memory wall problem.",
      content: MOD_01_CONTENT,
      exercises: [
        {
          id: "ex-01-1",
          title: "Matrix Multiplication Basics",
          description: "Understand the core operation of neural networks",
          type: "multiple-choice",
          question:
            "In a standard LLM, what operation does BitNet eliminate from matrix multiplication?",
          options: [
            "Addition",
            "Floating-point multiplication",
            "Subtraction",
            "Division",
          ],
          correctAnswer: "Floating-point multiplication",
          explanation:
            "BitNet weights are {-1, 0, 1}, so multiplication becomes trivial: x*1=x, x*0=0, x*(-1)=-x. Only additions and subtractions remain.",
        },
        {
          id: "ex-01-2",
          title: "Memory Calculation",
          description: "Calculate memory savings",
          type: "short-answer",
          question:
            "A 7B parameter LLM in FP16 uses 14 GB. How many GB would the same model use in 1.58-bit (ternary) representation? (Answer as a number)",
          correctAnswer: "1.4",
          explanation:
            "7B × 0.2 bytes per param ≈ 1.4 GB. Each ternary value needs ~1.58 bits ≈ 0.2 bytes, giving a ~10x memory reduction.",
        },
        {
          id: "ex-01-3",
          title: "Why 1.58 bits?",
          description: "Information theory of ternary encoding",
          type: "multiple-choice",
          question:
            'Why is BitNet b1.58 called "1.58-bit"?',
          options: [
            "It uses 1.58 bytes per parameter",
            "log₂(3) = 1.58 — the bits needed to encode 3 values {-1,0,1}",
            "It achieves 1.58x compression",
            "It was the 1.58th version of BitNet",
          ],
          correctAnswer:
            "log₂(3) = 1.58 — the bits needed to encode 3 values {-1,0,1}",
          explanation:
            "Information theory tells us we need log₂(N) bits to encode N distinct values. For ternary {-1, 0, 1}, N=3, so log₂(3) ≈ 1.58 bits.",
        },
      ],
      resources: [
        {
          id: "res-01-1",
          title: "3Blue1Brown — Neural Networks",
          type: "video",
          url: "https://www.youtube.com/watch?v=aircAruvnKk",
        },
        {
          id: "res-01-2",
          title: "Linear Algebra Review (Khan Academy)",
          type: "link",
          url: "https://www.khanacademy.org/math/linear-algebra",
        },
      ],
    },
    {
      id: "mod-02",
      number: 2,
      title: "Transformer Architecture",
      description:
        "Attention mechanism, multi-head attention, feed-forward networks, and the full decoder-only LLM.",
      content: MOD_02_CONTENT,
      exercises: [
        {
          id: "ex-02-1",
          title: "Attention Formula",
          description: "The core of the Transformer",
          type: "multiple-choice",
          question:
            "What is the correct attention formula?",
          options: [
            "Attention(Q,K,V) = softmax(Q × K^T / √d_k) × V",
            "Attention(Q,K,V) = sigmoid(Q × K^T) × V",
            "Attention(Q,K,V) = Q + K + V",
            "Attention(Q,K,V) = relu(Q × V^T) × K",
          ],
          correctAnswer: "Attention(Q,K,V) = softmax(Q × K^T / √d_k) × V",
          explanation:
            "Scaled dot-product attention: compute similarity scores (Q×K^T), scale by √d_k to prevent vanishing gradients in softmax, then weight the values V.",
        },
        {
          id: "ex-02-2",
          title: "BitNet and Linear Layers",
          description: "Where BitNet makes changes",
          type: "multiple-choice",
          question:
            "Which parts of a Transformer does BitNet quantize to ternary {-1, 0, 1}?",
          options: [
            "Only the embedding layer",
            "All nn.Linear layers inside transformer blocks (Q,K,V,O projections and FFN)",
            "Every single parameter including embeddings and norms",
            "Only the feed-forward network layers",
          ],
          correctAnswer:
            "All nn.Linear layers inside transformer blocks (Q,K,V,O projections and FFN)",
          explanation:
            "BitNet replaces nn.Linear with BitLinear in transformer blocks. Embeddings, output head, and normalization layers remain in full precision.",
        },
        {
          id: "ex-02-3",
          title: "BitNet Activation Function",
          description: "What activation does BitNet use?",
          type: "multiple-choice",
          question:
            "What activation function does BitNet b1.58 use in the FFN layers?",
          options: [
            "ReLU",
            "GELU",
            "Squared ReLU (ReLU(x)²)",
            "Sigmoid",
          ],
          correctAnswer: "Squared ReLU (ReLU(x)²)",
          explanation:
            "BitNet uses Squared ReLU which produces sparser activations that work well with ternary weights. It zeros out negative values and squares positive ones.",
        },
      ],
      resources: [
        {
          id: "res-02-1",
          title: "Attention Is All You Need (Original Paper)",
          type: "pdf",
          url: "https://arxiv.org/pdf/1706.03762",
        },
        {
          id: "res-02-2",
          title: "The Illustrated Transformer",
          type: "link",
          url: "https://jalammar.github.io/illustrated-transformer/",
        },
        {
          id: "res-02-3",
          title: "Andrej Karpathy — Let's Build GPT",
          type: "video",
          url: "https://www.youtube.com/watch?v=kCc8FmEb1nY",
        },
      ],
    },
    {
      id: "mod-03",
      number: 3,
      title: "Quantization Theory",
      description:
        "Symmetric vs asymmetric, PTQ vs QAT, and the Straight-Through Estimator — the key trick.",
      content: MOD_03_CONTENT,
      exercises: [
        {
          id: "ex-03-1",
          title: "STE Understanding",
          description: "The most important trick for 1-bit training",
          type: "multiple-choice",
          question:
            "What does the Straight-Through Estimator (STE) do?",
          options: [
            "Removes quantization during inference",
            "Forward pass uses quantized values; backward pass pretends quantization didn't happen (gradient ≈ 1)",
            "Trains a separate dequantization network",
            "Gradually increases the number of quantization levels",
          ],
          correctAnswer:
            "Forward pass uses quantized values; backward pass pretends quantization didn't happen (gradient ≈ 1)",
          explanation:
            "STE is the key trick: round() has zero gradient almost everywhere, so we pretend it's the identity function during backprop. This allows gradient descent to update the full-precision shadow weights.",
        },
        {
          id: "ex-03-2",
          title: "The Detach Trick",
          description: "How STE is implemented in PyTorch",
          type: "short-answer",
          question:
            "In PyTorch, STE is implemented as: w + (weight_quant(w) - w).______(). What method fills the blank?",
          correctAnswer: "detach",
          explanation:
            'The .detach() method stops gradient flow through (weight_quant(w) - w), making it a constant. Forward: w + const = quantized value. Backward: gradient flows through w only (the "straight-through" part).',
        },
        {
          id: "ex-03-3",
          title: "PTQ vs QAT",
          description: "Two approaches to quantization",
          type: "multiple-choice",
          question:
            "Why does BitNet use Quantization-Aware Training (QAT) instead of Post-Training Quantization (PTQ)?",
          options: [
            "QAT is faster to run",
            "PTQ at 1.58 bits would destroy the model; QAT lets the model learn to work with quantized weights from the start",
            "PTQ doesn't support ternary values",
            "QAT uses less memory",
          ],
          correctAnswer:
            "PTQ at 1.58 bits would destroy the model; QAT lets the model learn to work with quantized weights from the start",
          explanation:
            "At extreme quantization levels (1-2 bits), PTQ fails badly because the model was never trained to handle such low precision. QAT includes quantization in the training loop, so the model learns to be robust to it.",
        },
      ],
      resources: [
        {
          id: "res-03-1",
          title: "Quantization Deep Dive (HuggingFace)",
          type: "link",
          url: "https://huggingface.co/docs/optimum/concept_guides/quantization",
        },
      ],
    },
    {
      id: "mod-04",
      number: 4,
      title: "Binary Neural Networks History",
      description:
        "From BinaryConnect (2015) to XNOR-Net (2016) — the path to BitNet.",
      content: MOD_04_CONTENT,
      exercises: [
        {
          id: "ex-04-1",
          title: "XNOR-Bitcount Trick",
          description: "Hardware optimization for binary nets",
          type: "multiple-choice",
          question:
            "When both weights and activations are binary {-1, +1}, what replaces floating-point multiplication?",
          options: [
            "Integer addition",
            "XNOR operation followed by popcount",
            "Lookup tables",
            "Bit shifting",
          ],
          correctAnswer: "XNOR operation followed by popcount",
          explanation:
            "When operands are binary, multiplication of two values equals XNOR. A dot product becomes: 2 × popcount(XNOR(a,b)) - n. One 64-bit XNOR + popcount replaces 64 multiply-adds.",
        },
        {
          id: "ex-04-2",
          title: "Binary to Ternary",
          description: "The key insight of BitNet b1.58",
          type: "multiple-choice",
          question:
            "What are the THREE key insights that made BitNet succeed where earlier binary nets failed?",
          options: [
            "Bigger GPUs, more data, longer training",
            "Ternary weights {-1,0,1}, 8-bit activations (not binary), and scale (3B+ parameters)",
            "New optimizer, new loss function, new architecture",
            "Better data preprocessing, more augmentation, ensemble methods",
          ],
          correctAnswer:
            "Ternary weights {-1,0,1}, 8-bit activations (not binary), and scale (3B+ parameters)",
          explanation:
            "Three key insights: (1) Ternary {-1,0,1} adds sparsity via zero, (2) keeping activations at 8-bit avoids the information bottleneck that killed earlier binary nets, (3) at large scale (3B+), the model has enough redundancy for ternary weights to work.",
        },
      ],
      resources: [
        {
          id: "res-04-1",
          title: "Binary Neural Networks: A Survey",
          type: "pdf",
          url: "https://arxiv.org/pdf/2004.03333",
        },
        {
          id: "res-04-2",
          title: "XNOR-Net Paper",
          type: "pdf",
          url: "https://arxiv.org/pdf/1603.05279",
        },
      ],
    },
    {
      id: "mod-05",
      number: 5,
      title: "BitNet Architecture Deep Dive",
      description:
        "BitLinear layer, absmean/absmax quantization, STE implementation, and complete math.",
      content: MOD_05_CONTENT,
      exercises: [
        {
          id: "ex-05-1",
          title: "Weight Quantization Function",
          description: "AbsMean quantization for weights",
          type: "multiple-choice",
          question:
            "In BitNet's weight quantization, what is the scaling factor?",
          options: [
            "max(|W|) — the maximum absolute value",
            "mean(|W|) — the mean of absolute values",
            "std(W) — the standard deviation",
            "median(|W|) — the median absolute value",
          ],
          correctAnswer: "mean(|W|) — the mean of absolute values",
          explanation:
            "BitNet uses absmean: scale = 1/mean(|W|). This is more robust than absmax because it's not sensitive to outliers. The weights are then: round(W × scale), clamped to [-1, 1].",
        },
        {
          id: "ex-05-2",
          title: "Activation Quantization",
          description: "How activations are quantized",
          type: "multiple-choice",
          question:
            "BitNet quantizes activations to how many bits, and at what granularity?",
          options: [
            "1 bit, per-tensor",
            "4 bits, per-channel",
            "8 bits, per-token",
            "16 bits, per-layer",
          ],
          correctAnswer: "8 bits, per-token",
          explanation:
            "Activations are quantized to INT8 (8-bit) using absmax scaling per-token. Each token gets its own scale factor: scale = 127 / max(|x_token|). This ensures each token uses the full INT8 range.",
        },
        {
          id: "ex-05-3",
          title: "Zero Weight Meaning",
          description: "Sparsity in BitNet",
          type: "short-answer",
          question:
            "In a trained BitNet model, approximately what percentage of weights are zero (providing natural sparsity)? Answer as a range like 30-50.",
          correctAnswer: "30-50",
          explanation:
            "Trained BitNet models typically have 30-50% zero weights. This means those connections are effectively pruned — the model learned which connections don't matter, providing natural structured sparsity without a separate pruning step.",
        },
      ],
      resources: [
        {
          id: "res-05-1",
          title:
            "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits",
          type: "pdf",
          url: "https://arxiv.org/pdf/2402.17764",
        },
        {
          id: "res-05-2",
          title: "ArXiv Dive: BitNet 1.58",
          type: "link",
          url: "https://ghost.oxen.ai/arxiv-dives-bitnet-1-58/",
        },
      ],
    },
    {
      id: "mod-06",
      number: 6,
      title: "Hands-on Implementation",
      description:
        "Build BitLinear, assemble a Mini-BitNet Transformer, and train it from scratch in PyTorch.",
      content: MOD_06_CONTENT,
      exercises: [
        {
          id: "ex-06-1",
          title: "BitLinear Forward Pass",
          description: "Understand the forward pass order",
          type: "multiple-choice",
          question:
            "What is the correct order of operations in a BitLinear forward pass?",
          options: [
            "Quantize weights → Quantize activations → Linear → Normalize",
            "Normalize → Quantize activations (STE) → Quantize weights (STE) → F.linear",
            "F.linear → Normalize → Quantize",
            "Quantize weights → F.linear → Normalize → Quantize activations",
          ],
          correctAnswer:
            "Normalize → Quantize activations (STE) → Quantize weights (STE) → F.linear",
          explanation:
            "The order is: (1) RMSNorm the input, (2) quantize activations to INT8 with STE, (3) quantize weights to ternary with STE, (4) F.linear for the matrix multiply.",
        },
        {
          id: "ex-06-2",
          title: "Shadow Weights",
          description: "Why full-precision copies are needed",
          type: "multiple-choice",
          question:
            "Why does BitNet maintain full-precision 'shadow weights' during training?",
          options: [
            "For faster inference",
            "Because ternary weights can't store gradients — Adam updates the shadow weights, which are re-quantized each forward pass",
            "To compare with the quantized model",
            "Shadow weights are only used for validation",
          ],
          correctAnswer:
            "Because ternary weights can't store gradients — Adam updates the shadow weights, which are re-quantized each forward pass",
          explanation:
            "Ternary values {-1,0,1} can't meaningfully accumulate small gradient updates. Shadow weights (FP16) are what the optimizer actually updates. Each forward pass, they're re-quantized to ternary. Over time, shadow weights drift and cross quantization boundaries, causing ternary weights to flip.",
        },
        {
          id: "ex-06-3",
          title: "Weight Tying",
          description: "A common LLM technique",
          type: "multiple-choice",
          question:
            "In the MiniBitNet implementation, the output head shares weights with which layer?",
          options: [
            "The first BitLinear layer",
            "The token embedding layer",
            "The final RMSNorm layer",
            "All attention Q projections",
          ],
          correctAnswer: "The token embedding layer",
          explanation:
            "Weight tying: self.head.weight = self.tok_emb.weight. The output projection and input embedding share the same weight matrix. This reduces parameters and often improves quality.",
        },
      ],
      resources: [
        {
          id: "res-06-1",
          title: "Community PyTorch BitNet Implementation",
          type: "link",
          url: "https://github.com/kyegomez/BitNet",
        },
        {
          id: "res-06-2",
          title: "Pure PyTorch BitNet b1.58 2B4T",
          type: "link",
          url: "https://github.com/kevbuh/bitnet",
        },
      ],
    },
    {
      id: "mod-07",
      number: 7,
      title: "Training at Scale",
      description:
        "Training recipes, lambda scheduling for fine-tuning, hyperparameter selection, and benchmark analysis.",
      content: MOD_07_CONTENT,
      exercises: [
        {
          id: "ex-07-1",
          title: "Lambda Scheduling",
          description: "Gradual quantization for fine-tuning",
          type: "multiple-choice",
          question:
            "When fine-tuning a pre-trained model to 1.58-bit, what does lambda scheduling do?",
          options: [
            "Gradually increases the learning rate",
            "Gradually introduces quantization: λ goes from 0 (no quantization) to 1 (full quantization) during training",
            "Gradually decreases the model size",
            "Gradually removes layers from the model",
          ],
          correctAnswer:
            "Gradually introduces quantization: λ goes from 0 (no quantization) to 1 (full quantization) during training",
          explanation:
            "Without warmup, applying full ternary quantization to pre-trained weights destroys the model (loss: 2 → 13). Lambda scheduling lets the model gradually adapt: x_quant = x + λ*(quant(x) - x).detach(). Linear schedule reaching λ=1 at 50% of training works best.",
        },
        {
          id: "ex-07-2",
          title: "Training Speed",
          description: "Training cost of BitNet",
          type: "multiple-choice",
          question:
            "Compared to standard FP16 training, BitNet training per step is:",
          options: [
            "2x faster (less data to process)",
            "About the same speed",
            "20-30% slower (quantization overhead), but produces a 3-10x smaller/faster model",
            "10x slower (ternary is hard to optimize)",
          ],
          correctAnswer:
            "20-30% slower (quantization overhead), but produces a 3-10x smaller/faster model",
          explanation:
            "Training is slightly slower because of the quantization/dequantization in each forward pass. But the resulting model is dramatically smaller and faster at inference — a worthwhile trade-off.",
        },
      ],
      resources: [
        {
          id: "res-07-1",
          title: "HuggingFace: Fine-tuning LLMs to 1.58-bit",
          type: "link",
          url: "https://huggingface.co/blog/1_58_llm_extreme_quantization",
        },
        {
          id: "res-07-2",
          title: "BitNet b1.58 2B4T on HuggingFace",
          type: "link",
          url: "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T",
        },
      ],
    },
    {
      id: "mod-08",
      number: 8,
      title: "Inference & Hardware",
      description:
        "Weight packing, custom kernels, bitnet.cpp, energy efficiency, and future hardware.",
      content: MOD_08_CONTENT,
      exercises: [
        {
          id: "ex-08-1",
          title: "Weight Packing",
          description: "Efficient ternary storage",
          type: "multiple-choice",
          question:
            "How many ternary values can be packed into a single uint8 byte?",
          options: ["2", "4", "8", "16"],
          correctAnswer: "4",
          explanation:
            "Each ternary value needs 2 bits (encoding -1→00, 0→01, 1→10). A byte has 8 bits, so 8/2 = 4 ternary values per byte. This gives 4x memory reduction over int8-per-value storage.",
        },
        {
          id: "ex-08-2",
          title: "Energy Savings",
          description: "The efficiency advantage",
          type: "short-answer",
          question:
            "BitNet achieves approximately how many times less energy in matrix multiply compared to FP16 LLMs? (Answer as a number)",
          correctAnswer: "71",
          explanation:
            "BitNet achieves ~71.4x energy reduction in matrix multiplication. This is because INT8 additions (~0.03 pJ) replace FP16 multiplications (~1.1 pJ), and ~50% of operations are skipped due to zero weights.",
        },
        {
          id: "ex-08-3",
          title: "Inference Framework",
          description: "Running 1-bit models",
          type: "multiple-choice",
          question:
            "What technique does bitnet.cpp use for fast CPU inference of ternary models?",
          options: [
            "GPU offloading",
            "Lookup tables — pre-computed sums for all 256 possible byte values",
            "XNOR operations",
            "Float16 simulation",
          ],
          correctAnswer:
            "Lookup tables — pre-computed sums for all 256 possible byte values",
          explanation:
            "For each packed byte (4 ternary values), bitnet.cpp pre-computes a lookup table of 256 entries. A single table lookup replaces 4 multiply-accumulate operations. This is extremely fast on CPU.",
        },
      ],
      resources: [
        {
          id: "res-08-1",
          title: "Microsoft BitNet (Official Inference Framework)",
          type: "link",
          url: "https://github.com/microsoft/BitNet",
        },
        {
          id: "res-08-2",
          title: "BitNet b1.58 2B4T Technical Report",
          type: "pdf",
          url: "https://arxiv.org/pdf/2504.12285",
        },
      ],
    },
  ],
};
