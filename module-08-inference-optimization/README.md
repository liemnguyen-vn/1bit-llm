# Module 8: Inference Optimization & Hardware

Turning ternary weights into actual speed improvements.

## 8.1 Weight Packing: 4 Ternary Values per INT8

During training, shadow weights are stored in FP16. For inference, we pack ternary values densely.

### Encoding Scheme

Each ternary value {-1, 0, 1} needs 2 bits:

```
Encoding:
  -1 → 00
   0 → 01
   1 → 10
  (11 is unused)

Packing 4 ternary values into 1 byte (uint8):
  byte = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6)
```

### Implementation

```python
def pack_ternary_weights(weight_ternary):
    """
    Pack ternary weights {-1, 0, 1} into uint8.
    4 ternary values per byte.

    Args:
        weight_ternary: Tensor of {-1, 0, 1}, shape (out_feat, in_feat)

    Returns:
        packed: uint8 tensor, shape (out_feat, in_feat // 4)
    """
    # Map {-1, 0, 1} → {0, 1, 2}
    mapped = (weight_ternary + 1).to(torch.uint8)  # {0, 1, 2}

    # Reshape to groups of 4
    out_feat, in_feat = mapped.shape
    assert in_feat % 4 == 0
    mapped = mapped.view(out_feat, in_feat // 4, 4)

    # Pack: 4 values × 2 bits = 8 bits per byte
    packed = (mapped[:, :, 0]
            | (mapped[:, :, 1] << 2)
            | (mapped[:, :, 2] << 4)
            | (mapped[:, :, 3] << 6))

    return packed


def unpack_ternary_weights(packed):
    """
    Unpack uint8 back to ternary values.
    """
    val0 = (packed & 0x03) - 1         # bits 0-1
    val1 = ((packed >> 2) & 0x03) - 1  # bits 2-3
    val2 = ((packed >> 4) & 0x03) - 1  # bits 4-5
    val3 = ((packed >> 6) & 0x03) - 1  # bits 6-7

    return torch.stack([val0, val1, val2, val3], dim=-1).flatten(-2)
```

### Memory Savings

```
Model: 2.4B parameters
  FP16:    2.4B × 2 bytes = 4.8 GB
  Packed:  2.4B × 0.25 bytes = 0.6 GB (+ scales and non-quantized parts)
  Actual:  ~0.4 GB for ternary weights + ~0.5 GB for rest = ~0.9 GB total
```

---

## 8.2 The Pack-Store-Load-Unpack-Compute Pipeline

For GPU inference, the full pipeline is:

```
1. PACK (offline, once):
   - Quantize trained weights to ternary
   - Pack 4 values per uint8
   - Store packed weights in HBM (GPU memory)

2. STORE: Packed weights sit in HBM
   - 8x smaller than FP16 → less HBM needed
   - More cache-friendly → less memory bandwidth wasted

3. LOAD: Load packed weights from HBM to SM (streaming multiprocessor)
   - Tiled loading: load BLOCK_SIZE_K columns at a time
   - 8x less data to transfer → 8x faster loading

4. UNPACK: On-the-fly in registers
   - Extract 4 ternary values from each uint8
   - Bit-shift and mask operations (very fast)
   - No memory allocation needed

5. COMPUTE: INT8 accumulation
   - Activations are INT8, weights are {-1, 0, 1}
   - Multiply: x * 1 = x, x * 0 = 0, x * (-1) = -x
   - Accumulate in INT32 to avoid overflow
   - Final: convert to FP16 and scale
```

---

## 8.3 Custom Triton Kernels for Ternary MatMul

### Why Custom Kernels?

Standard `torch.mm` doesn't know weights are ternary. It still does full FP16 multiply-accumulate. Custom kernels exploit the ternary structure.

### Triton Kernel (Simplified)

```python
import triton
import triton.language as tl

@triton.jit
def bitnet_matmul_kernel(
    # Pointers to matrices
    a_ptr,      # INT8 activations, shape (M, K)
    b_ptr,      # Packed ternary weights, shape (K//4, N) as uint8
    c_ptr,      # Output, shape (M, N)
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes (tuned per GPU)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled matrix multiply with on-the-fly weight unpacking."""
    # Which tile are we computing?
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator (INT32 to avoid overflow)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Tile over K dimension, processing 4 values per packed byte
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load INT8 activations tile
        a = tl.load(a_ptr + offs_m[:, None] * stride_am
                           + offs_k[None, :] * stride_ak)

        # Load packed weights (K//4 because 4 values per byte)
        b_packed = tl.load(b_ptr + (offs_k[None, :] // 4) * stride_bk
                                 + offs_n[:, None] * stride_bn)

        # Unpack: extract the relevant 2-bit group
        shift = (offs_k[None, :] % 4) * 2
        b_ternary = ((b_packed >> shift) & 3).to(tl.int8) - 1  # {0,1,2} → {-1,0,1}

        # Accumulate: INT8 × ternary → INT32
        acc += tl.dot(a, b_ternary.trans(), out_dtype=tl.int32)

    # Store result (convert to FP16 with scaling)
    c = acc.to(tl.float16)
    tl.store(c_ptr + offs_m[:, None] * stride_cm
                    + offs_n[None, :] * stride_cn, c)
```

### Performance

From HuggingFace benchmarks:
```
Method           Throughput (relative to torch.compile)
─────────────────────────────────────────────────────────
torch.mm (FP16)  0.8x (baseline, not optimized for ternary)
torch.compile    1.0x (compiles but doesn't exploit ternary)
Custom Triton    ~1.0x (similar to compile for small models)
BitBlas library  1.15-1.25x (fastest, but requires compilation)
bitnet.cpp (CPU) Optimized for CPU-only inference
```

---

## 8.4 bitnet.cpp: CPU Inference Framework

Microsoft's official inference framework for 1-bit LLMs on CPU.

### Key Features

```
- Optimized for x86 (AVX2/AVX-512) and ARM (NEON) CPUs
- Uses lookup tables for ternary multiplication
- No GPU required — runs on consumer hardware
- Based on llama.cpp architecture
```

### How It Works

Instead of multiply-accumulate, uses **lookup tables**:

```
For each output element:
  result = 0
  for each weight group (4 ternary values packed in 1 byte):
    index = packed_byte    # 0-255, encoding 4 ternary values
    result += LUT[index]   # Pre-computed sum of selected activations

LUT is pre-built for each activation vector:
  LUT[index] = sum of (activation[i] * ternary[i]) for 4 values
             = precomputed for all 256 possible byte values
```

This is extremely fast on CPU — a single table lookup replaces 4 multiplications.

### Usage

```bash
# Clone and build
git clone https://github.com/microsoft/BitNet.git
cd BitNet
pip install -r requirements.txt

# Download and convert model
python setup_env.py --hf-repo microsoft/BitNet-b1.58-2B-4T-gguf -q i2_s

# Run inference
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -p "The key advantage of 1-bit LLMs is" \
    -n 100 -temp 0.8
```

---

## 8.5 Energy Efficiency: 71x Reduction in MatMul Energy

### The Energy Breakdown

Energy cost per operation (45nm process):

```
Operation          Energy (pJ)    Relative
─────────────────────────────────────────
FP32 multiply      3.7            1x
FP32 add           0.9            0.24x
FP16 multiply      1.1            0.30x
FP16 add           0.4            0.11x
INT8 multiply      0.2            0.05x
INT8 add           0.03           0.008x
```

### BitNet vs. Standard LLM

```
Standard LLM (FP16):
  Each element of y = W @ x requires:
  - 1 FP16 multiply (1.1 pJ)
  - 1 FP16 add (0.4 pJ)
  Total: 1.5 pJ per operation

BitNet (ternary weights, INT8 activations):
  Each element requires:
  - 0 multiplies (weight is {-1,0,1}: negate, skip, or keep)
  - 1 INT8 add (0.03 pJ) — for non-zero weights only
  - ~50% weights are zero → ~0.5 adds per element
  Total: ~0.015 pJ per operation

Ratio: 1.5 / 0.015 = ~100x energy reduction
Paper claims: 71.4x (conservative, real-world measurement)
```

### What This Enables

```
Model          Standard (FP16)    BitNet          Implication
─────────────────────────────────────────────────────────────
7B on phone    Impossible         ~1.4 GB RAM     Possible!
70B on laptop  Need 140GB RAM     ~14 GB RAM      Feasible
7B inference   ~10W GPU           ~0.14W          Battery-powered AI
```

---

## 8.6 INT8 Addition vs. FP16 Multiplication

The fundamental computational advantage explained.

### Standard LLM: FP16 Multiply-Accumulate

```python
# Each element of output y[i]
y[i] = sum(x[j] * W[i][j] for j in range(hidden_dim))
     = x[0]*W[i][0] + x[1]*W[i][1] + ... + x[d]*W[i][d]

# Each term: 1 FP16 multiply + 1 FP16 add
# For hidden_dim=4096: 4096 multiplies + 4096 adds
```

### BitNet: INT8 Add/Subtract/Skip

```python
# Each element of output y[i]
y[i] = sum(x_int8[j] * W_ternary[i][j] for j in range(hidden_dim))

# But W[i][j] is {-1, 0, 1}, so:
#   x * (+1) = +x  → just add x
#   x * (0)  = 0   → skip entirely
#   x * (-1) = -x  → subtract x

y[i] = 0
for j in range(hidden_dim):
    if W[i][j] == 1:
        y[i] += x_int8[j]     # INT8 addition
    elif W[i][j] == -1:
        y[i] -= x_int8[j]     # INT8 subtraction

# NO MULTIPLICATIONS AT ALL
# ~50% of iterations are skipped (weight = 0)
# Result: ~hidden_dim/2 INT8 additions
```

### Hardware Implication

```
FP16 multiplier: Large circuit, high power, moderate speed
INT8 adder:      Small circuit, low power, very fast

A chip designed for BitNet could:
- Remove FP multipliers entirely (for weight layers)
- Use only INT8 adders
- Pack more compute units in same chip area
- Use fraction of the power
```

---

## 8.7 Future: Custom Hardware for 1-bit LLMs

### Why Standard GPUs Are Suboptimal

Current GPUs (NVIDIA A100, H100) are designed for FP16/FP32 computation:
- Tensor cores do FP16 multiply-accumulate
- Memory hierarchy optimized for 16-bit data
- Can't skip zero-weight operations automatically

For BitNet, GPUs waste energy on unused multiplication circuits.

### What Ideal BitNet Hardware Looks Like

```
Custom 1-bit accelerator:
┌─────────────────────────────────┐
│ Weight memory: Packed ternary   │ ← 10x denser than FP16
│ (4 values per byte)             │
├─────────────────────────────────┤
│ Activation buffer: INT8         │ ← Standard
├─────────────────────────────────┤
│ Compute: INT8 adders only       │ ← No FP multipliers needed
│ + Zero-skip logic               │ ← Skip computation for W=0
├─────────────────────────────────┤
│ Accumulator: INT32              │ ← Prevent overflow
├─────────────────────────────────┤
│ Output scaling: FP16            │ ← Final dequantization
└─────────────────────────────────┘
```

### Research Directions

1. **FPGA prototypes:** Binary/ternary neural network accelerators on FPGA
2. **ASIC designs:** Custom chips for 1-bit inference
3. **In-memory computing:** Store ternary weights in memory cells, compute in-place
4. **Neuromorphic chips:** Event-driven computation naturally handles sparsity

---

## 8.8 Deployment Considerations and Practical Tips

### When to Use 1-bit LLMs

```
Good use cases:
  ✓ Edge deployment (phones, IoT)
  ✓ Cost-sensitive serving (high QPS)
  ✓ Memory-constrained environments
  ✓ Battery-powered devices
  ✓ Need largest possible model in limited memory

Not ideal (yet):
  ✗ Tasks requiring maximum accuracy (use FP16)
  ✗ Very small models (< 1B) — accuracy gap too large
  ✗ Tasks requiring fine-grained numerical reasoning
```

### Inference Optimization Checklist

```
1. Pack weights: Convert FP16 shadow weights → packed ternary uint8
2. Pre-compute scales: Store activation/weight scales
3. Choose runtime:
   - CPU: Use bitnet.cpp with lookup tables
   - GPU: Use BitBlas or custom Triton kernels
   - Mixed: Ternary matmul on GPU, other ops in FP16
4. Batch for throughput: Larger batches amortize weight loading
5. Profile: Check if you're memory-bound (likely) or compute-bound
```

### Quick Start Deployment

```python
# Using HuggingFace Transformers (simplest)
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/bitnet-b1.58-2B-4T",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
# Note: This loads in FP16 and simulates ternary.
# For actual speedup, use bitnet.cpp or custom kernels.
```

```bash
# Using bitnet.cpp (actual speedup on CPU)
git clone https://github.com/microsoft/BitNet.git
cd BitNet
python setup_env.py --hf-repo microsoft/BitNet-b1.58-2B-4T-gguf -q i2_s
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -p "Your prompt here" -n 100
```

---

## Exercise 8: Profile Inference Speed

See `../exercises/exercise_08_inference.py`.

Tasks:
1. Implement weight packing (ternary → uint8)
2. Implement unpacking
3. Verify pack/unpack roundtrip is lossless
4. Time standard FP16 matmul vs. simulated ternary matmul
5. Measure memory usage difference
6. (Bonus) Install bitnet.cpp and compare speed

---

## Key Takeaways

1. **Weight packing** stores 4 ternary values per byte → 8x memory reduction from FP16
2. **Custom kernels** unpack on-the-fly and use INT8 addition instead of FP16 multiplication
3. **Energy savings** of ~71x in matrix multiply operations
4. **bitnet.cpp** enables fast CPU inference using lookup tables
5. **Current GPUs** don't fully exploit ternary structure — custom hardware could unlock even more speedup
6. The key insight: **no multiplications** — only additions, subtractions, and skips
