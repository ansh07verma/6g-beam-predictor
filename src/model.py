"""
model.py — BeamPredictor Deep Residual MLP
===========================================
Position-aided beam prediction for 6G massive MIMO.

Architecture:
    Input(6) → Linear(512) → BN → GELU
             → 2× ResBlock(512)
             → Linear(256) → BN → GELU → Dropout(0.2)
             → Linear(128) → BN → GELU → Dropout(0.1)
             → Linear(64) → [logits]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ResBlock(nn.Module):
    """Pre-activated residual block with small-scale init for stability."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.fc2.weight.data.mul_(0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(x))
        out = self.fc1(out)
        out = F.gelu(self.bn2(out))
        out = self.drop(self.fc2(out))
        return out + x


class BeamPredictor(nn.Module):
    """Deep residual MLP for position-to-beam classification."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = None,
        num_classes: int = 64,
        dropout: float = 0.2,
        n_res_blocks: int = 2,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        stem_dim = hidden_dims[0]
        neck_dims = hidden_dims[1:]

        self.stem = nn.Sequential(
            nn.Linear(input_dim, stem_dim),
            nn.BatchNorm1d(stem_dim),
            nn.GELU(),
        )

        self.trunk = nn.Sequential(
            *[ResBlock(stem_dim, dropout=dropout / 2) for _ in range(n_res_blocks)]
        )

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

        self.head = nn.Linear(in_dim, num_classes)

        self.config = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'num_classes': num_classes,
            'dropout': dropout,
            'n_res_blocks': n_res_blocks,
        }

        self._init_weights()

    def _init_weights(self) -> None:
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
        x = self.stem(x)
        x = self.trunk(x)
        x = self.neck(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
    return total


if __name__ == '__main__':
    model = BeamPredictor(input_dim=6, hidden_dims=[512, 256, 128], num_classes=64)
    count_parameters(model)
    x = torch.randn(32, 6)
    logits = model(x)
    assert logits.shape == (32, 64)
    print("model.py smoke test PASSED")
