"""
Exercise 1: Two-Layer MLP from Scratch
=======================================

Build a simple MLP to classify handwritten digits, understanding
the fundamentals of neural networks before moving to BitNet.

Goals:
  - Understand matrix multiplication as the core NN operation
  - See weight matrices and their shapes
  - Train with gradient descent
  - Prepare for understanding what BitNet changes

Run:
    python exercises/exercise_01_mlp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def create_synthetic_data(n_samples=5000, n_features=64, n_classes=10):
    """Create synthetic classification data (no need to download MNIST)."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, n_features)
    # Create labels based on a random projection (learnable pattern)
    W_true = torch.randn(n_features, n_classes)
    logits = X @ W_true
    y = logits.argmax(dim=-1)
    return X, y


# ============================================================================
# TODO 1: Implement matrix multiplication manually
# ============================================================================

def manual_matmul(A, B):
    """
    Multiply matrices A (m x k) and B (k x n) without using @ or torch.mm.

    This is the operation that happens billions of times in an LLM.
    In BitNet, we'll replace the multiplication with addition.
    """
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, f"Incompatible shapes: {A.shape} and {B.shape}"

    C = torch.zeros(m, n)
    # YOUR CODE HERE:
    # Hint: C[i][j] = sum(A[i][p] * B[p][j] for p in range(k))
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
    return C


# ============================================================================
# TODO 2: Build a 2-layer MLP
# ============================================================================

class SimpleMLP(nn.Module):
    """
    A simple 2-layer MLP for classification.

    Architecture:
        Input (64) → Linear (64 → 128) → ReLU → Linear (128 → 10) → Output
    """

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=10):
        super().__init__()
        # YOUR CODE HERE: Define two Linear layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # YOUR CODE HERE: Forward pass with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================================
# TODO 3: Training loop
# ============================================================================

def train_mlp(model, X_train, y_train, epochs=50, lr=0.01, batch_size=64):
    """Train the MLP and observe the loss decrease."""
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == y_batch).sum().item()
            total += len(y_batch)

        if epoch % 10 == 0:
            acc = correct / total * 100
            print(f"Epoch {epoch:3d} | Loss: {total_loss/len(loader):.4f} | "
                  f"Accuracy: {acc:.1f}%")


# ============================================================================
# TODO 4: Examine weights and understand memory usage
# ============================================================================

def examine_weights(model):
    """Look at the weight matrices and understand their sizes."""
    print("\n=== Weight Analysis ===\n")

    total_params = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        print(f"{name}:")
        print(f"  Shape: {list(param.shape)}")
        print(f"  Params: {n:,}")
        print(f"  FP32 size: {n * 4 / 1024:.1f} KB")
        print(f"  FP16 size: {n * 2 / 1024:.1f} KB")
        print(f"  1.58-bit size: {n * 0.2 / 1024:.1f} KB")
        print(f"  Value range: [{param.data.min():.3f}, {param.data.max():.3f}]")
        print(f"  Mean abs value: {param.data.abs().mean():.4f}")
        print()

    print(f"Total parameters: {total_params:,}")
    print(f"FP32 total: {total_params * 4 / 1024:.1f} KB")
    print(f"1.58-bit total: {total_params * 0.2 / 1024:.1f} KB")
    print(f"Memory savings: {4 / 0.2:.0f}x")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Exercise 1: Two-Layer MLP from Scratch")
    print("=" * 50)

    # Test manual matmul
    print("\n--- Testing manual matrix multiplication ---")
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    C_manual = manual_matmul(A, B)
    C_torch = A @ B
    error = (C_manual - C_torch).abs().max().item()
    print(f"Max error vs torch: {error:.2e}")
    assert error < 1e-5, "Manual matmul is incorrect!"
    print("Manual matmul: CORRECT")

    # Create data and train
    print("\n--- Training MLP ---")
    X, y = create_synthetic_data()
    model = SimpleMLP()
    train_mlp(model, X, y)

    # Examine weights
    examine_weights(model)

    print("\n--- Key Insight ---")
    print("Every weight you see above would become {-1, 0, 1} in BitNet.")
    print("The fc1.weight (64x128 = 8,192 params) goes from 32 KB to 1.6 KB.")
    print("And matrix multiplication becomes just addition and subtraction!")
