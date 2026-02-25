"""
Exercise 8: Inference Optimization — Weight Packing & Speed
=============================================================

Implement ternary weight packing and measure the speed/memory
benefits of 1-bit representations.

Goals:
  - Implement ternary weight packing (4 values per byte)
  - Verify lossless pack/unpack roundtrip
  - Measure memory savings
  - Time standard vs. simulated ternary matrix multiply

Run:
    python exercises/exercise_08_inference.py
"""

import time
import torch
import torch.nn.functional as F


# ============================================================================
# Part 1: Weight Packing
# ============================================================================

def pack_ternary(weights: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary weights {-1, 0, 1} into uint8.
    4 ternary values per byte using 2-bit encoding:
        -1 → 00, 0 → 01, 1 → 10

    Args:
        weights: Tensor of values in {-1, 0, 1}, shape (..., K) where K % 4 == 0

    Returns:
        packed: uint8 tensor, shape (..., K // 4)
    """
    assert weights.shape[-1] % 4 == 0, "Last dimension must be divisible by 4"

    # Map {-1, 0, 1} → {0, 1, 2}
    mapped = (weights + 1).to(torch.uint8)

    # Reshape to groups of 4
    *leading, K = mapped.shape
    mapped = mapped.view(*leading, K // 4, 4)

    # Pack: 4 × 2-bit values into 1 byte
    packed = (mapped[..., 0]
              | (mapped[..., 1] << 2)
              | (mapped[..., 2] << 4)
              | (mapped[..., 3] << 6))

    return packed


def unpack_ternary(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack uint8 back to ternary {-1, 0, 1}.

    Args:
        packed: uint8 tensor, shape (..., K // 4)

    Returns:
        weights: int8 tensor, shape (..., K)
    """
    val0 = ((packed >> 0) & 0x03).to(torch.int8) - 1
    val1 = ((packed >> 2) & 0x03).to(torch.int8) - 1
    val2 = ((packed >> 4) & 0x03).to(torch.int8) - 1
    val3 = ((packed >> 6) & 0x03).to(torch.int8) - 1

    return torch.stack([val0, val1, val2, val3], dim=-1).flatten(-2)


def test_packing():
    """Verify that pack/unpack roundtrip is lossless."""
    print("=== Weight Packing Test ===\n")

    # Small example
    weights = torch.tensor([[-1, 0, 1, 1],
                            [0, -1, -1, 0],
                            [1, 1, 0, -1]], dtype=torch.int8)

    print(f"Original weights ({weights.shape}):")
    print(f"  {weights.tolist()}")

    packed = pack_ternary(weights)
    print(f"\nPacked ({packed.shape}, dtype={packed.dtype}):")
    print(f"  {packed.tolist()}")
    print(f"  Memory: {weights.numel()} bytes → {packed.numel()} bytes "
          f"({packed.numel() / weights.numel():.0%})")

    unpacked = unpack_ternary(packed)
    print(f"\nUnpacked ({unpacked.shape}):")
    print(f"  {unpacked.tolist()}")

    match = (weights == unpacked).all().item()
    print(f"\nRoundtrip lossless: {match}")
    assert match, "Pack/unpack roundtrip failed!"

    # Large-scale test
    print("\n--- Large-scale packing test ---")
    large_weights = torch.randint(-1, 2, (2048, 2048), dtype=torch.int8)
    large_packed = pack_ternary(large_weights)
    large_unpacked = unpack_ternary(large_packed)
    match = (large_weights == large_unpacked).all().item()
    print(f"  Shape: {large_weights.shape} → {large_packed.shape}")
    print(f"  Memory: {large_weights.numel() / 1e6:.1f} MB → "
          f"{large_packed.numel() / 1e6:.1f} MB (4x reduction)")
    print(f"  Lossless: {match}")
    print()


# ============================================================================
# Part 2: Memory Comparison
# ============================================================================

def memory_comparison():
    """Compare memory usage of different representations."""
    print("=== Memory Comparison ===\n")

    sizes = [
        ("125M params", 125_000_000),
        ("1.3B params", 1_300_000_000),
        ("2.4B params (BitNet 2B4T)", 2_400_000_000),
        ("7B params", 7_000_000_000),
        ("70B params", 70_000_000_000),
    ]

    print(f"{'Model':<30} {'FP16':>8} {'INT8':>8} {'INT4':>8} {'1.58b':>8}")
    print("-" * 70)

    for name, n_params in sizes:
        fp16 = n_params * 2 / 1e9
        int8 = n_params * 1 / 1e9
        int4 = n_params * 0.5 / 1e9
        ternary = n_params * 0.25 / 1e9  # 2 bits packed (practical)
        print(f"{name:<30} {fp16:>7.1f}G {int8:>7.1f}G "
              f"{int4:>7.1f}G {ternary:>7.1f}G")

    print()


# ============================================================================
# Part 3: Speed Comparison
# ============================================================================

def speed_comparison():
    """Compare matrix multiply speed: standard FP16 vs simulated ternary."""
    print("=== Speed Comparison ===\n")

    device = 'cpu'  # Use CPU for fair comparison
    sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]

    for M, K in sizes:
        N = M  # Square matrices

        # Standard FP16 matmul
        x_fp16 = torch.randn(M, K, dtype=torch.float32, device=device)
        w_fp16 = torch.randn(K, N, dtype=torch.float32, device=device)

        # Ternary weights (simulated as float for fair comparison)
        w_ternary = torch.randint(-1, 2, (K, N), dtype=torch.float32, device=device)

        # INT8 activations (simulated)
        x_int8 = (x_fp16 * 127 / x_fp16.abs().max()).round().clamp(-128, 127)

        # Warm up
        for _ in range(5):
            _ = x_fp16 @ w_fp16
            _ = x_int8 @ w_ternary

        # Time FP16 matmul
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = x_fp16 @ w_fp16
        fp16_time = (time.perf_counter() - start) / n_iters

        # Time ternary matmul (still float, but demonstrates concept)
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = x_int8 @ w_ternary
        ternary_time = (time.perf_counter() - start) / n_iters

        # Memory comparison
        fp16_mem = w_fp16.numel() * 4  # FP32 on CPU
        ternary_mem = w_ternary.numel() * 4  # FP32 (simulated)
        packed_mem = w_ternary.numel() // 4  # Actual packed size

        print(f"  [{M}x{K}] FP32: {fp16_time*1000:.2f}ms | "
              f"Ternary: {ternary_time*1000:.2f}ms | "
              f"Weight mem: {fp16_mem/1024:.0f}KB → {packed_mem/1024:.0f}KB "
              f"(packed)")

    print()
    print("  Note: Both use float32 on CPU. Real speedup requires custom")
    print("  kernels (Triton/CUDA) that exploit the ternary structure.")
    print("  The key win is MEMORY reduction, which helps on memory-bound")
    print("  workloads (which LLM inference almost always is).")
    print()


# ============================================================================
# Part 4: Ternary MatMul — The Add/Subtract/Skip Pattern
# ============================================================================

def ternary_matmul_demo():
    """
    Demonstrate that ternary matmul only needs add/subtract/skip.
    No multiplications!
    """
    print("=== Ternary MatMul Demo ===\n")

    # Small example
    x = torch.tensor([3.0, -2.0, 5.0, 1.0])
    W = torch.tensor([[ 1,  0, -1,  1],
                       [-1,  1,  0,  0],
                       [ 0, -1,  1,  1]], dtype=torch.float32)

    print(f"Input x: {x.tolist()}")
    print(f"Weight W (ternary):")
    for i, row in enumerate(W.tolist()):
        ops = []
        for j, (w, xi) in enumerate(zip(row, x.tolist())):
            if w == 1:
                ops.append(f"+{xi}")
            elif w == -1:
                ops.append(f"-{xi}")
            else:
                ops.append("skip")
        result = sum(w * xi for w, xi in zip(row, x.tolist()))
        print(f"  y[{i}] = {' '.join(f'{op:>6s}' for op in ops)} = {result:.1f}")

    y_standard = W @ x
    print(f"\nResult (standard matmul):  {y_standard.tolist()}")

    # Manual add/subtract (NO multiplications)
    y_manual = torch.zeros(3)
    n_adds = 0
    n_skips = 0
    for i in range(3):
        for j in range(4):
            if W[i, j] == 1:
                y_manual[i] += x[j]
                n_adds += 1
            elif W[i, j] == -1:
                y_manual[i] -= x[j]
                n_adds += 1
            else:
                n_skips += 1

    print(f"Result (add/subtract only): {y_manual.tolist()}")
    print(f"  Operations: {n_adds} additions, {n_skips} skipped (zeros)")
    print(f"  Multiplications: 0")
    print(f"\n  Match: {torch.allclose(y_standard, y_manual)}")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 8: Inference Optimization")
    print("=" * 60)
    print()

    test_packing()
    memory_comparison()
    ternary_matmul_demo()
    speed_comparison()

    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("1. Ternary weights pack 4:1 into uint8 (lossless)")
    print("2. A 7B model: 14GB (FP16) → 1.75GB (packed ternary)")
    print("3. MatMul becomes add/subtract/skip — zero multiplications")
    print("4. Real speedup needs custom kernels (Triton, bitnet.cpp)")
    print("5. The main win is memory bandwidth, not raw compute")
