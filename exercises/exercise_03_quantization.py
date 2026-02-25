"""
Exercise 3: Quantization and the Straight-Through Estimator
============================================================

Build quantization functions from scratch and understand STE,
the key trick that makes training 1-bit models possible.

Goals:
  - Implement INT8 symmetric quantization
  - Implement ternary quantization {-1, 0, 1}
  - Understand and implement the Straight-Through Estimator
  - Verify that gradients flow through STE correctly

Run:
    python exercises/exercise_03_quantization.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Part 1: INT8 Symmetric Quantization
# ============================================================================

def quantize_int8(x: torch.Tensor) -> tuple:
    """
    Quantize a tensor to INT8 using symmetric absmax quantization.

    Returns:
        (quantized_int8, scale) tuple
    """
    # Find the maximum absolute value
    max_abs = x.abs().max().clamp(min=1e-5)

    # Scale to INT8 range [-127, 127] (symmetric, so we don't use -128)
    scale = 127.0 / max_abs

    # Quantize: scale, round, clamp
    x_q = (x * scale).round().clamp(-127, 127).to(torch.int8)

    return x_q, scale


def dequantize_int8(x_q: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequantize INT8 back to float."""
    return x_q.float() / scale


def test_int8_quantization():
    """Test INT8 quantization roundtrip."""
    print("=== INT8 Quantization Test ===\n")

    x = torch.tensor([0.5, -1.2, 0.3, 0.8, -0.1, 2.0])
    print(f"Original:     {x.tolist()}")

    x_q, scale = quantize_int8(x)
    print(f"Quantized:    {x_q.tolist()} (scale={scale:.3f})")

    x_deq = dequantize_int8(x_q, scale)
    print(f"Dequantized:  {x_deq.tolist()}")

    error = (x - x_deq).abs()
    print(f"Error:        {error.tolist()}")
    print(f"Max error:    {error.max():.6f}")
    print(f"Relative err: {(error / x.abs().clamp(min=1e-5)).mean():.4%}")
    print()


# ============================================================================
# Part 2: Ternary Quantization {-1, 0, 1}
# ============================================================================

def quantize_ternary(w: torch.Tensor) -> tuple:
    """
    Quantize weights to ternary {-1, 0, 1} using absmean scaling.

    This is the actual BitNet weight quantization function.

    Returns:
        (ternary_values, scale) tuple
    """
    scale = w.abs().mean().clamp(min=1e-5)
    ternary = (w / scale).round().clamp(-1, 1)
    return ternary, scale


def dequantize_ternary(ternary: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequantize ternary back to float."""
    return ternary.float() * scale


def test_ternary_quantization():
    """Test ternary quantization and analyze distribution."""
    print("=== Ternary Quantization Test ===\n")

    # Simulate a weight matrix (similar to what a trained model has)
    torch.manual_seed(42)
    W = torch.randn(8, 8) * 0.02  # Typical LLM weight std

    print(f"Original weight matrix (8x8):")
    print(f"  Mean: {W.mean():.4f}")
    print(f"  Std:  {W.std():.4f}")
    print(f"  Range: [{W.min():.4f}, {W.max():.4f}]")

    ternary, scale = quantize_ternary(W)

    n_neg = (ternary == -1).sum().item()
    n_zero = (ternary == 0).sum().item()
    n_pos = (ternary == 1).sum().item()
    total = ternary.numel()

    print(f"\nAfter ternary quantization:")
    print(f"  Scale: {scale:.4f}")
    print(f"  -1: {n_neg}/{total} ({n_neg/total:.1%})")
    print(f"   0: {n_zero}/{total} ({n_zero/total:.1%})")
    print(f"  +1: {n_pos}/{total} ({n_pos/total:.1%})")

    W_deq = dequantize_ternary(ternary, scale)
    error = (W - W_deq).abs()
    print(f"\n  Quantization error: {error.mean():.6f} (mean)")
    print(f"  Relative error: {(error / W.abs().clamp(min=1e-5)).mean():.1%}")

    # Show the actual ternary matrix
    print(f"\nTernary weight matrix:")
    for row in ternary.int().tolist():
        print(f"  {['  '.join(f'{v:+d}' for v in row)]}")
    print()


# ============================================================================
# Part 3: The Straight-Through Estimator (STE)
# ============================================================================

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """
    Round with straight-through gradient.

    Forward: returns round(x)
    Backward: gradient is 1 (as if round didn't happen)
    """
    return x + (x.round() - x).detach()


def test_ste():
    """Demonstrate that STE allows gradients through rounding."""
    print("=== Straight-Through Estimator Test ===\n")

    # Test 1: Regular round() blocks gradients
    print("Test 1: Regular round() has ZERO gradients")
    x1 = torch.tensor([0.3, 0.7, 1.5, -0.4], requires_grad=True)
    y1 = x1.round()
    loss1 = y1.sum()
    loss1.backward()
    print(f"  Input:    {x1.data.tolist()}")
    print(f"  Rounded:  {y1.data.tolist()}")
    print(f"  Gradient: {x1.grad.tolist()}")
    print(f"  All zero? {(x1.grad == 0).all().item()}")
    print()

    # Test 2: STE round() preserves gradients
    print("Test 2: STE round() preserves gradients")
    x2 = torch.tensor([0.3, 0.7, 1.5, -0.4], requires_grad=True)
    y2 = ste_round(x2)
    loss2 = y2.sum()
    loss2.backward()
    print(f"  Input:    {x2.data.tolist()}")
    print(f"  Rounded:  {y2.data.tolist()}")
    print(f"  Gradient: {x2.grad.tolist()}")
    print(f"  All ones? {(x2.grad == 1).all().item()}")
    print()

    # Test 3: STE in the full BitNet pattern
    print("Test 3: Full BitNet STE pattern")
    w = torch.tensor([0.42, -0.13, 0.87, -0.55, 0.03, -0.71],
                     requires_grad=True)

    # Forward: quantize to ternary
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    w_quant = w + ((w * scale).round().clamp(-1, 1) / scale - w).detach()

    print(f"  Shadow weights:   {[f'{v:.2f}' for v in w.data.tolist()]}")
    print(f"  Quantized values: {[f'{v:.2f}' for v in w_quant.data.tolist()]}")

    # Backward: gradient flows through
    loss = (w_quant ** 2).sum()
    loss.backward()
    print(f"  Gradients:        {[f'{v:.2f}' for v in w.grad.tolist()]}")
    print(f"  Gradients are non-zero: {(w.grad != 0).all().item()}")
    print()


# ============================================================================
# Part 4: Training with STE — Does it actually work?
# ============================================================================

def test_ste_training():
    """Train a simple model with ternary weights to show STE works."""
    print("=== STE Training Test ===\n")

    # Simple task: learn y = sign(W @ x) for a known W
    torch.manual_seed(42)
    d = 16
    W_true = torch.randn(d, d)

    # Generate data
    X = torch.randn(500, d)
    Y = (X @ W_true.T).sign()  # Target: sign of linear transformation

    # Model with ternary weights
    class TernaryLinear(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(d_out, d_in) * 0.1)

        def forward(self, x):
            w = self.weight
            scale = 1.0 / w.abs().mean().clamp(min=1e-5)
            w_q = w + ((w * scale).round().clamp(-1, 1) / scale - w).detach()
            return F.linear(x, w_q)

    model = TernaryLinear(d, d)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training ternary model with STE:")
    for epoch in range(1, 101):
        pred = model(X)
        loss = F.mse_loss(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            # Check accuracy: does sign(pred) match Y?
            acc = ((pred.sign() == Y).float().mean() * 100).item()
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"Sign accuracy: {acc:.1f}%")

    # Show final ternary weights
    w = model.weight.data
    scale = w.abs().mean()
    ternary = (w / scale).round().clamp(-1, 1)
    n_neg = (ternary == -1).sum().item()
    n_zero = (ternary == 0).sum().item()
    n_pos = (ternary == 1).sum().item()
    total = ternary.numel()
    print(f"\nFinal weight distribution: "
          f"-1={n_neg/total:.1%}, 0={n_zero/total:.1%}, +1={n_pos/total:.1%}")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    test_int8_quantization()
    test_ternary_quantization()
    test_ste()
    test_ste_training()

    print("=" * 50)
    print("KEY TAKEAWAYS:")
    print("=" * 50)
    print("1. INT8 quantization: small error (~0.4% relative)")
    print("2. Ternary quantization: only 3 values, ~30-50% zeros")
    print("3. STE: round() has zero grad, but STE trick gives grad=1")
    print("4. STE training works: ternary models CAN learn via gradient descent")
    print("5. The detach trick: x + (quant(x) - x).detach()")
