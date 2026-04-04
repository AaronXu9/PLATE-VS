"""Prediction head: MLP that maps fused representation to pK value."""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """MLP prediction head for binding affinity regression."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Map fused representation to scalar pK prediction.

        Args:
            fused: [B, input_dim]

        Returns:
            pred: [B, 1] predicted pK (scaled [0, 1])
        """
        return self.head(fused)
