"""
Exercise 4: Binary vs Ternary Linear Layer
============================================

Implement binary and ternary linear layers, train them on the same task,
and observe why ternary {-1,0,1} is better than binary {-1,+1}.

Goals:
  - Implement binary linear layer (BinaryConnect style)
  - Implement ternary linear layer (BitNet style)
  - Compare accuracy on a classification task
  - Understand the role of zero (sparsity)

Run:
    python exercises/exercise_04_binary_linear.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# Standard Linear (Baseline)
# ============================================================================

class StandardLinear(nn.Linear):
    """Normal full-precision linear layer."""
    pass


# ============================================================================
# Binary Linear {-1, +1}
# ============================================================================

class BinaryLinear(nn.Linear):
    """
    Binary linear layer: weights are {-1, +1}.

    Uses sign() for binarization with STE.
    Similar to BinaryConnect / XNOR-Net.
    """

    def __init__(self, *args, **kwargs):
        kwargs['bias'] = False
        super().__init__(*args, **kwargs)

    def forward(self, x):
        w = self.weight
        # Binary quantization with STE
        w_binary = w + (w.sign() - w).detach()
        # Scale factor (XNOR-Net style) for better accuracy
        scale = w.abs().mean()
        w_scaled = w_binary * scale
        return F.linear(x, w_scaled)


# ============================================================================
# Ternary Linear {-1, 0, +1}
# ============================================================================

class TernaryLinear(nn.Linear):
    """
    Ternary linear layer: weights are {-1, 0, +1}.

    Uses absmean quantization with STE.
    This is the BitNet approach.
    """

    def __init__(self, *args, **kwargs):
        kwargs['bias'] = False
        super().__init__(*args, **kwargs)

    def forward(self, x):
        w = self.weight
        # Ternary quantization with STE
        scale = 1.0 / w.abs().mean().clamp(min=1e-5)
        w_ternary = w + ((w * scale).round().clamp(-1, 1) / scale - w).detach()
        return F.linear(x, w_ternary)


# ============================================================================
# Models
# ============================================================================

def make_model(layer_class, input_dim=64, hidden_dim=128, output_dim=10):
    """Create a 2-layer classifier using the given layer type."""
    return nn.Sequential(
        layer_class(input_dim, hidden_dim),
        nn.ReLU(),
        layer_class(hidden_dim, output_dim),
    )


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_and_eval(model, X_train, y_train, X_test, y_test,
                   name="Model", epochs=100, lr=0.01):
    """Train and evaluate a model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 25 == 0:
            model.eval()
            with torch.no_grad():
                train_acc = (model(X_train).argmax(1) == y_train).float().mean()
                test_acc = (model(X_test).argmax(1) == y_test).float().mean()
            history.append((epoch, train_acc.item(), test_acc.item()))

    return history


def analyze_weights(model, name):
    """Analyze the weight distribution."""
    print(f"\n  {name} weight analysis:")
    for i, module in enumerate(model):
        if hasattr(module, 'weight'):
            w = module.weight.data
            if isinstance(module, BinaryLinear):
                w_q = w.sign()
                n_neg = (w_q == -1).sum().item()
                n_pos = (w_q == 1).sum().item()
                total = w_q.numel()
                print(f"    Layer {i}: -1={n_neg/total:.1%}, +1={n_pos/total:.1%}")
            elif isinstance(module, TernaryLinear):
                scale = w.abs().mean()
                w_q = (w / scale).round().clamp(-1, 1)
                n_neg = (w_q == -1).sum().item()
                n_zero = (w_q == 0).sum().item()
                n_pos = (w_q == 1).sum().item()
                total = w_q.numel()
                print(f"    Layer {i}: -1={n_neg/total:.1%}, "
                      f"0={n_zero/total:.1%}, +1={n_pos/total:.1%}")
            else:
                print(f"    Layer {i}: FP32, range=[{w.min():.3f}, {w.max():.3f}]")


# ============================================================================
# Main Comparison
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 4: Binary vs Ternary Linear Layer Comparison")
    print("=" * 60)

    # Generate data
    torch.manual_seed(42)
    n_train, n_test = 3000, 1000
    d, n_classes = 64, 10
    W_true = torch.randn(d, n_classes)

    X_train = torch.randn(n_train, d)
    y_train = (X_train @ W_true).argmax(dim=-1)
    X_test = torch.randn(n_test, d)
    y_test = (X_test @ W_true).argmax(dim=-1)

    # Train all three models
    models = {
        "Standard (FP32)": make_model(StandardLinear),
        "Binary {-1,+1}":  make_model(BinaryLinear),
        "Ternary {-1,0,1}": make_model(TernaryLinear),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        history = train_and_eval(model, X_train, y_train, X_test, y_test,
                                 name=name, epochs=100)
        results[name] = history

        analyze_weights(model, name)

    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Train Acc':>10} {'Test Acc':>10} {'Bits/param':>12}")
    print("-" * 60)
    for name, history in results.items():
        _, train_acc, test_acc = history[-1]
        bits = {"Standard (FP32)": 32, "Binary {-1,+1}": 1, "Ternary {-1,0,1}": 1.58}
        print(f"{name:<25} {train_acc:>9.1%} {test_acc:>9.1%} {bits[name]:>10.2f}")

    print("\n--- Key Observations ---")
    print("1. Standard FP32: Best accuracy (baseline)")
    print("2. Binary {-1,+1}: Noticeable accuracy drop")
    print("3. Ternary {-1,0,1}: Much closer to FP32!")
    print("4. The zero in ternary provides sparsity and better expressiveness")
    print("5. At larger scale (3B+ params), ternary matches FP32 completely")
