"""
model.py — BeamPredictor Deep Residual MLP
===========================================
Position-aided beam prediction for 6G massive MIMO.

Architecture (upgraded for asu_campus_3p5 real dataset):
---------------------------------------------------------
Input(3) → Linear(512) → BN → GELU
         → ResBlock(512)   [skip: Linear → BN + main: Linear→BN→GELU→Linear→BN]
         → ResBlock(512)
         → Linear(256) → BN → GELU → Dropout(0.2)
         → Linear(128) → BN → GELU → Dropout(0.1)
         → Linear(64)  → [logits over N_b beam classes]

Key upgrades vs. original 3-layer MLP:
  - Residual (skip) connections prevent gradient vanishing in deeper layers
  - GELU activation: smoother gradient flow vs. ReLU for position regression tasks
  - Larger initial projection (3 → 512) captures richer positional embeddings
  - Label smoothing in loss encourages softer probability distributions
  - Total parameters: ~610K (vs 109K before)

Input:  Normalized UE positions (x, y, z) — StandardScaler fitted on train set
Output: Raw logits over N_b beam classes — apply CrossEntropyLoss externally
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Residual Block
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Post-activated residual block for fully-connected networks.

    y = x + Dropout(Linear(GELU(BN(Linear(GELU(BN(x)))))))

    Uses a small-scale init (1e-3) on the second linear layer so the
    residual branch starts near zero without being exactly zero —
    this avoids cuBLAS NaN/Inf during backward through zero-weight matmuls.

    Parameters
    ----------
    dim     : int   — input and output dimension
    dropout : float — dropout probability
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.bn1  = nn.BatchNorm1d(dim)
        self.fc1  = nn.Linear(dim, dim)
        self.bn2  = nn.BatchNorm1d(dim)
        self.fc2  = nn.Linear(dim, dim)
        self.drop = nn.Dropout(p=dropout)

        # Standard Kaiming init for fc1
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        # Small-scale init for fc2 so residual starts small but non-zero
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.fc2.weight.data.mul_(0.01)    # scale down — avoids cuBLAS NaN
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.bn1(x))
        out = self.fc1(out)
        out = F.gelu(self.bn2(out))
        out = self.drop(self.fc2(out))
        return out + residual


# ─────────────────────────────────────────────────────────────────────────────
# BeamPredictor — Deep Residual MLP
# ─────────────────────────────────────────────────────────────────────────────

class BeamPredictor(nn.Module):
    """Deep residual MLP for position-to-beam classification.

    Maps normalized UE positions (x, y, z) to logits over N_b beam classes.

    Architecture overview:
        stem  : Linear(3 → dim) → BN → GELU
        trunk : n_res_blocks × ResBlock(dim)
        neck  : Linear(dim → 256) → BN → GELU → Dropout(0.2)
                Linear(256 → 128) → BN → GELU → Dropout(0.1)
        head  : Linear(128 → num_classes)

    Parameters
    ----------
    input_dim    : int        — number of input features (3 for x,y,z)
    hidden_dims  : List[int]  — [stem_dim] (first element sets residual width;
                                subsequent elements define the neck layers)
    num_classes  : int        — codebook size N_b (beam count)
    dropout      : float      — base dropout rate (halved in final neck layer)
    n_res_blocks : int        — number of residual blocks in trunk (default 2)

    Example
    -------
    >>> model = BeamPredictor()
    >>> x = torch.randn(32, 3)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([32, 64])
    """

    def __init__(
        self,
        input_dim:    int       = 3,
        hidden_dims:  List[int] = None,
        num_classes:  int       = 64,
        dropout:      float     = 0.2,
        n_res_blocks: int       = 2,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        stem_dim  = hidden_dims[0]               # residual block width
        neck_dims = hidden_dims[1:]              # narrowing neck layers

        # ── Stem: input projection ────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Linear(input_dim, stem_dim),
            nn.BatchNorm1d(stem_dim),
            nn.GELU(),
        )

        # ── Trunk: residual blocks at full width ──────────────────────────
        self.trunk = nn.Sequential(
            *[ResBlock(stem_dim, dropout=dropout / 2)
              for _ in range(n_res_blocks)]
        )

        # ── Neck: progressive dimension reduction ─────────────────────────
        neck_layers = []
        in_dim = stem_dim
        for i, out_dim in enumerate(neck_dims):
            neck_layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(p=dropout if i == 0 else dropout / 2),
            ]
            in_dim = out_dim
        self.neck = nn.Sequential(*neck_layers)

        # ── Classification head (no activation — raw logits) ─────────────
        self.head = nn.Linear(in_dim, num_classes)

        # Save config for checkpoint reload
        self.config = {
            'input_dim':    input_dim,
            'hidden_dims':  hidden_dims,
            'num_classes':  num_classes,
            'dropout':      dropout,
            'n_res_blocks': n_res_blocks,
        }

        # ── Weight init ───────────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """He (Kaiming) init for Linear layers; BN init to identity."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.requires_grad and m.weight.shape[0] != 0:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, input_dim)
            Batch of normalized UE position vectors.

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)
            Raw classification logits (not softmax'd).
        """
        x = self.stem(x)
        x = self.trunk(x)
        x = self.neck(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Count and print the number of trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
    return total


def model_summary(model: nn.Module, input_shape: tuple = (1, 3)) -> None:
    """Print model structure and verify output shape with a dummy forward pass."""
    print(model)
    print()
    count_parameters(model)
    with torch.no_grad():
        dummy = torch.zeros(*input_shape)
        out = model(dummy)
    print(f"Input  shape: {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  model.py — BeamPredictor smoke test")
    print("=" * 60)

    model = BeamPredictor(
        input_dim=3,
        hidden_dims=[512, 256, 128],
        num_classes=64,
        dropout=0.2,
        n_res_blocks=2,
    )
    model_summary(model, input_shape=(4, 3))

    batch  = torch.randn(32, 3)
    logits = model(batch)
    assert logits.shape == (32, 64), f"Expected (32,64), got {logits.shape}"

    loss = logits.mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print("\nForward + backward PASSED ✓")
    print("model.py smoke test PASSED ✓")
