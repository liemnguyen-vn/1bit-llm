# Module 7: Training at Scale

Moving from toy experiments to real-world 1-bit LLM training.

## 7.1 Training Recipe from Microsoft Research

Microsoft's BitNet training recipe has evolved through several iterations:

### Pre-training from Scratch

For training BitNet from randomly initialized weights:

```
Architecture: Transformer decoder-only (LLaMA-like)
  - Replace all nn.Linear with BitLinear
  - Keep embeddings and output head in FP16
  - Use RMSNorm (Sub-LayerNorm placement: before each sub-layer)
  - Use Squared ReLU in FFN
  - Use RoPE for positional encoding
  - No bias terms anywhere

Optimizer: Adam (or AdamW)
  - β1 = 0.9, β2 = 0.95 (same as LLaMA)
  - Weight decay: 0.1 (applied to shadow weights)
  - Gradient clipping: 1.0

Learning rate:
  - Peak LR: 1.5e-4 (for 3B model)
  - Warmup: 2000 steps (linear warmup)
  - Decay: cosine schedule to 1.5e-5

Batch size:
  - Effective batch size: ~2M tokens
  - Accumulated over gradient accumulation steps

Sequence length: 2048 or 4096 tokens
```

### Key Differences from Standard LLM Training

| Aspect | Standard (FP16) | BitNet |
|--------|----------------|--------|
| Weight init | Normal(0, 0.02) | Normal(0, 0.02) — same |
| Forward pass | FP16 matmul | Quantized matmul + STE |
| Optimizer state | FP32 | FP32 (shadow weights) |
| Training speed | 1x | ~0.7-0.8x (quantization overhead) |
| Convergence | Standard | May need more tokens |

Training a BitNet model is ~20-30% **slower** per step than FP16 (due to quantization overhead), but the resulting model is 3-10x **smaller** and **faster** at inference.

---

## 7.2 Hyperparameters: LR, Batch Size, Warmup

### Learning Rate Sensitivity

BitNet is sensitive to learning rate — too high causes oscillation around quantization boundaries:

```
LR too high (1e-3):
  Shadow weights: [0.45, -0.55, 0.12]
  Quantized:      [1,    -1,    0]
  After update:   [-0.10, 0.30, 0.60]  ← Weights flip wildly
  Quantized:      [0,     1,    1]     ← Unstable!

LR just right (1.5e-4):
  Shadow weights: [0.45, -0.55, 0.12]
  Quantized:      [1,    -1,    0]
  After update:   [0.44, -0.56, 0.13]  ← Small change
  Quantized:      [1,    -1,    0]     ← Stable, eventually flips when ready
```

### Recommended Hyperparameters by Model Size

```
Model Size    Peak LR     Warmup Steps    Batch (tokens)
─────────────────────────────────────────────────────────
125M          3e-4        1000            0.5M
350M          2e-4        1500            1M
1.3B          1.5e-4      2000            2M
3B            1.5e-4      2000            2M
7B            1e-4        2000            4M
```

---

## 7.3 Lambda Scheduling: Gradual Quantization Introduction

When fine-tuning an existing model to 1.58-bit, applying full quantization immediately is catastrophic:

```
Full-precision model: Loss = 2.0 (good)
Immediate quantization: Loss = 13.0 (destroyed!)
```

The solution: **gradually introduce quantization** using a lambda parameter.

### The Lambda-STE Pattern

```python
def lambda_ste_forward(x, w, lambda_val):
    """STE with gradual quantization introduction."""
    # lambda = 0: no quantization (full precision)
    # lambda = 1: full quantization (standard BitLinear)

    x_quant = x + lambda_val * (activation_quant(x) - x).detach()
    w_quant = w + lambda_val * (weight_quant(w) - w).detach()

    return F.linear(x_quant, w_quant)
```

### Lambda Schedulers

```python
# Linear scheduler (simplest, works well)
def linear_schedule(step, total_steps):
    """lambda goes from 0 to 1 linearly."""
    return min(step / total_steps, 1.0)

# Accelerated linear (reaches full quantization at 50% of training)
def accelerated_linear(step, total_steps):
    """Reach full quantization halfway through training."""
    return min(2 * step / total_steps, 1.0)

# Exponential scheduler
def exponential_schedule(step, total_steps, k=8):
    """Slow start, rapid finish."""
    normalized = step / total_steps
    return 1 - (1 - normalized) ** k

# Sigmoid scheduler
def sigmoid_schedule(step, total_steps, k=100):
    """S-curve: slow start, fast middle, slow finish."""
    import math
    normalized = step / total_steps
    return 1 / (1 + math.exp(-k * (normalized - 0.5)))
```

### Scheduler Comparison (Fine-tuning Llama3 8B → 1.58-bit)

```
Scheduler              WikiText Perplexity (after 10B tokens)
──────────────────────────────────────────────────────────────
No warmup (λ=1)        26-30 (poor)
Linear (λ: 0→1)        12.2 (good)
Accel Linear (2x)      ~12 (similar, faster convergence)
Exponential (k=8)      ~15 (worse — too abrupt at the end)
Sigmoid (k=100)        ~13.5 (okay)
```

**Winner: Linear or accelerated linear scheduler.**

---

## 7.4 Fine-tuning Existing Models to 1.58-bit

You don't always need to pre-train from scratch. You can take an existing FP16 model and fine-tune it to 1.58-bit.

### The Process

```python
def convert_to_bitnet(model):
    """Replace all nn.Linear in a pre-trained model with BitLinear."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Create BitLinear with same dimensions
            bit_layer = BitLinear(module.in_features, module.out_features)
            # Copy pre-trained weights as initial shadow weights
            bit_layer.weight.data = module.weight.data.clone()
            setattr(model, name, bit_layer)
        else:
            convert_to_bitnet(module)  # Recurse into sub-modules
    return model
```

### Fine-tuning Recipe

```
1. Load pre-trained model (e.g., Llama3 8B)
2. Replace nn.Linear → BitLinear (keep shadow weights = pre-trained)
3. Set lambda = 0 (no quantization initially)
4. Train with lambda scheduling:
   - Optimizer: Adam, LR = 1e-4
   - Dataset: Diverse text (FineWeb-edu recommended)
   - Tokens: 10B-100B
   - Lambda: linear schedule from 0 → 1 over training
5. Result: 1.58-bit model that retains pre-trained knowledge
```

### Results (HuggingFace experiments)

```
Model                    Tokens  MMLU   ARC-C  TruthfulQA
─────────────────────────────────────────────────────────
Llama3 8B (FP16)         15T     66.0   53.7   51.6
Fine-tuned 1.58-bit      10B     47.3   52.4   48.5
Fine-tuned 1.58-bit      100B    ~60    ~53    ~50
BitNet 7B (from scratch) 100B    41.4   45.6   43.5
```

Fine-tuning is more efficient than training from scratch — 10B tokens gets competitive results that took 100B tokens from scratch.

---

## 7.5 Warmup Quantization Deep Dive

### Why It Works

Pre-trained weights have a specific distribution (narrow, peaked). Ternary quantization maps this to {-1, 0, 1}. The model needs time to adjust its internal representations.

```
Pre-trained weight distribution:
  Mean: ~0, Std: ~0.013
  Most weights: [-0.05, 0.05]

Random init distribution:
  Mean: ~0, Std: ~0.02
  Wider spread

After ternary quantization:
  Only 3 values: {-scale, 0, +scale}
  Massive information loss from pre-trained distribution
```

Lambda scheduling lets the model gradually adapt:
- λ=0: Uses original weights, learns nothing about quantization
- λ=0.5: Half quantized, model starts adapting
- λ=1.0: Fully quantized, model has adapted

### Practical Tips

1. **Linear schedule is best** — simple and effective
2. **Reach λ=1 by 50% of training** — gives model time to optimize in fully quantized regime
3. **Don't warmup too slowly** — the model needs to actually train with quantization
4. **Monitor loss after λ reaches 1** — should continue decreasing

---

## 7.6 Dataset Selection and Diversity

### Why Dataset Matters More for BitNet

With only 3 weight values, the model has less capacity per parameter. It needs more diverse training data to generalize well.

### Recommended Datasets

```
Pre-training:
  - FineWeb-edu (HuggingFace): High-quality web text, filtered for educational content
  - RedPajama: Diverse web + books + code
  - The Pile: Mix of many domains

Fine-tuning:
  - FineWeb-edu: 10B-100B tokens
  - AVOID single-domain datasets (e.g., only TinyStories)
    → Causes poor generalization

Instruction tuning:
  - Standard instruction datasets work fine
  - UltraChat, OpenAssistant, etc.
```

### The Diversity Problem

Experiment from HuggingFace:
```
Fine-tuned on TinyStories only → WikiText PPL: 42 (terrible)
Fine-tuned on FineWeb-edu      → WikiText PPL: 12 (good)
```

Low-capacity models overfit to narrow distributions. Use diverse data.

---

## 7.7 The BitNet b1.58 2B4T Model

The first open-source 1-bit LLM, released by Microsoft in 2025.

### Specifications

```
Name: BitNet b1.58 2B4T
Parameters: 2,412,820,480 (2.4B)
Training tokens: 4,000,000,000,000 (4T)
Architecture:
  - Layers: 30
  - Hidden size: 2560
  - Attention heads: 20 (GQA with 5 KV heads)
  - FFN hidden: 6912
  - Vocab size: 128,256
  - Context length: 4,096
  - Activation: Squared ReLU
  - Norm: SubLayerNorm (RMSNorm before each sub-layer)
  - Position: RoPE
  - Bias: None (no bias anywhere)

Weight representation:
  - All Linear layers: ternary {-1, 0, 1}
  - Embedding: FP16
  - Output head: FP16
  - Norms: FP32
```

### Using It

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "microsoft/bitnet-b1.58-2B-4T"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "The key advantage of 1-bit LLMs is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 7.8 Benchmark Results and Analysis

### BitNet b1.58 2B4T vs. Comparable Models

```
Benchmark        BitNet 2B4T   Llama3.2 3B   Gemma3 1B   Qwen2.5 1.5B
─────────────────────────────────────────────────────────────────────────
MMLU (5-shot)    52.3          63.4          38.5        60.9
ARC-C (25-shot)  51.5          53.0          39.2        48.6
HellaSwag        63.2          73.7          57.0        67.2
Winogrande       62.8          67.8          56.7        64.3
TruthfulQA       51.2          44.2          40.3        42.5

Model size       ~0.4 GB*      6 GB          2 GB        3 GB
```

*0.4 GB with ternary packing vs. 4.8 GB in FP16

Key observations:
- Competitive with models 4-15x larger in memory
- Excels on TruthfulQA
- Gap on knowledge-heavy tasks (MMLU) — expected with fewer effective bits
- At equivalent memory budget, BitNet can afford a much larger model

---

## Exercise 7: Fine-tune with Warmup Quantization

See `../exercises/exercise_07_finetune.py`.

Tasks:
1. Start with a pre-trained small model
2. Replace nn.Linear with BitLinear
3. Implement linear lambda scheduling
4. Fine-tune on a text dataset
5. Compare perplexity at different lambda schedule rates
6. Generate text and evaluate quality

---

## Key Takeaways

1. BitNet training uses the same optimizer (Adam) and general recipe as standard LLMs
2. **Learning rate sensitivity** is higher — use lower LR than FP16 models
3. **Lambda scheduling** is crucial for fine-tuning: gradually introduce quantization
4. Linear schedule reaching λ=1 at 50% of training works best
5. **Diverse datasets** are critical — BitNet models overfit more on narrow data
6. The 2B4T model shows ternary LLMs are competitive at the 2-3B parameter range
7. Fine-tuning (10-100B tokens) is more efficient than pre-training from scratch
