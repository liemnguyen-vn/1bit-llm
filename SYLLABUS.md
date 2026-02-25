# 1-Bit LLMs from Scratch: Complete Course

## Course Overview

This course takes you from zero to fully understanding and implementing 1-bit Large Language Models (BitNet). You'll learn the math, the theory, the code, and the practical engineering behind the most efficient frontier in LLM research.

**Target audience:** Developers/ML practitioners who want deep understanding, not just surface-level awareness.

**Prerequisites:** Basic Python, some linear algebra exposure (we review everything needed).

**Time estimate:** ~40-60 hours of focused study + coding.

---

## Course Map

```
Module 1: Foundations
    |
Module 2: Transformer Architecture
    |
Module 3: Quantization Theory
    |
Module 4: Binary Neural Networks (History)
    |
Module 5: BitNet Architecture (Core Theory)
    |
Module 6: Hands-on Implementation (Code)
   / \
Module 7: Training at Scale    Module 8: Inference & Hardware
```

---

## Module 1: Foundations
**File:** `module-01-foundations/README.md`

- 1.1 Linear Algebra Refresher (vectors, matrices, matrix multiplication)
- 1.2 Neural Network Fundamentals (forward pass, backprop, gradient descent)
- 1.3 Loss Functions and Optimization
- 1.4 Number Representation in Computers (FP32, FP16, BF16, INT8)
- 1.5 Why Quantization Matters: The Memory Wall Problem

**Exercise:** Implement a 2-layer MLP from scratch in PyTorch.

---

## Module 2: Transformer Architecture Deep Dive
**File:** `module-02-transformers/README.md`

- 2.1 Attention Mechanism: Intuition and Math
- 2.2 Self-Attention: Q, K, V Projections
- 2.3 Multi-Head Attention
- 2.4 Feed-Forward Networks in Transformers
- 2.5 Positional Encoding (Sinusoidal + RoPE)
- 2.6 Layer Normalization (LayerNorm, RMSNorm)
- 2.7 The Full Transformer Block
- 2.8 From Transformer to LLM: Decoder-Only Architecture

**Exercise:** Build a minimal GPT from scratch (character-level).

---

## Module 3: Quantization Theory
**File:** `module-03-quantization/README.md`

- 3.1 What is Quantization? (Mapping continuous to discrete)
- 3.2 Symmetric vs. Asymmetric Quantization
- 3.3 Per-Tensor vs. Per-Channel vs. Per-Token Quantization
- 3.4 Post-Training Quantization (PTQ)
- 3.5 Quantization-Aware Training (QAT)
- 3.6 The Quantization Spectrum: FP32 -> FP16 -> INT8 -> INT4 -> Binary
- 3.7 Information Theory: Bits per Parameter (log2 of levels)
- 3.8 The Straight-Through Estimator (STE)

**Exercise:** Implement INT8 quantization and dequantization by hand.

---

## Module 4: Binary Neural Networks — A Brief History
**File:** `module-04-binary-neural-nets/README.md`

- 4.1 BinaryConnect (2015): The First Binary Weights
- 4.2 BinaryNet (2016): Binary Weights AND Activations
- 4.3 XNOR-Net (2016): Making It Work on ImageNet
- 4.4 The XNOR-Bitcount Trick
- 4.5 Why Binary Nets Struggled for Years
- 4.6 The Gap: From Binary CNNs to Binary LLMs

**Exercise:** Implement a simple binary linear layer with STE.

---

## Module 5: BitNet Architecture Deep Dive
**File:** `module-05-bitnet-architecture/README.md`

- 5.1 BitNet b1 (2023): The 1-Bit LLM Breakthrough
- 5.2 BitNet b1.58 (2024): "All LLMs Are in 1.58 Bits"
- 5.3 The BitLinear Layer — Complete Math
- 5.4 Weight Quantization: AbsMean Function
- 5.5 Activation Quantization: AbsMax Function (8-bit per-token)
- 5.6 STE in BitLinear: The Detach Trick
- 5.7 Why 1.58 Bits? Information Theory of Ternary {-1, 0, 1}
- 5.8 The Role of Zero: Sparsity as a Feature
- 5.9 Scaling Laws for 1-bit LLMs
- 5.10 Performance Benchmarks: BitNet vs. LLaMA

**Exercise:** Derive the BitLinear forward pass by hand on a 4x4 matrix.

---

## Module 6: Hands-on Implementation
**File:** `module-06-implementation/README.md`
**Code:** `module-06-implementation/bitnet_from_scratch.py`

- 6.1 Implementing `weight_quant()` from Scratch
- 6.2 Implementing `activation_quant()` from Scratch
- 6.3 Building the Full `BitLinear` Layer
- 6.4 Assembling a Mini-BitNet Transformer
- 6.5 Training Loop: STE + Adam Optimizer
- 6.6 Monitoring Quantization Quality During Training
- 6.7 Comparing BitLinear vs. nn.Linear on a Toy Task

**Exercise:** Train a tiny BitNet on a text generation task.

---

## Module 7: Training at Scale
**File:** `module-07-training-at-scale/README.md`

- 7.1 Training Recipe from Microsoft Research
- 7.2 Hyperparameters: LR, Batch Size, Warmup
- 7.3 Lambda Scheduling: Gradual Quantization Introduction
- 7.4 Fine-tuning Existing Models to 1.58-bit
- 7.5 Warmup Quantization: Linear, Exponential, Sigmoid Schedulers
- 7.6 Dataset Selection and Diversity
- 7.7 The BitNet b1.58 2B4T Model (2.4B params, 4T tokens)
- 7.8 Benchmark Results and Analysis

**Exercise:** Fine-tune a small model using warmup quantization.

---

## Module 8: Inference Optimization & Hardware
**File:** `module-08-inference-optimization/README.md`

- 8.1 Weight Packing: 4 Ternary Values per INT8
- 8.2 The Pack-Store-Load-Unpack-Compute Pipeline
- 8.3 Custom Triton Kernels for Ternary MatMul
- 8.4 bitnet.cpp: CPU Inference Framework
- 8.5 Energy Efficiency: 71x Reduction in MatMul Energy
- 8.6 INT8 Addition vs. FP16 Multiplication
- 8.7 Future: Custom Hardware for 1-bit LLMs
- 8.8 Deployment Considerations and Practical Tips

**Exercise:** Profile inference speed of BitNet vs. standard model.

---

## Key Papers (Reading List)

1. **BitNet: Scaling 1-bit Transformers for Large Language Models** (2023)
   - https://arxiv.org/abs/2310.11453
2. **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits** (2024)
   - https://arxiv.org/abs/2402.17764
3. **BitNet b1.58 2B4T Technical Report** (2025)
   - https://arxiv.org/abs/2504.12285
4. **XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks** (2016)
   - https://arxiv.org/abs/1603.05279
5. **Binary Neural Networks: A Survey** (2020)
   - https://arxiv.org/abs/2004.03333

---

## Code Resources

- Microsoft BitNet (official): https://github.com/microsoft/BitNet
- HuggingFace BitNet model: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
- Community PyTorch impl: https://github.com/kyegomez/BitNet
- Pure PyTorch impl: https://github.com/kevbuh/bitnet
- HuggingFace fine-tuning blog: https://huggingface.co/blog/1_58_llm_extreme_quantization
