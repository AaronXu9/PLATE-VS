"""Protein encoder: projects precomputed ESM2 + ANKH embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProteinEncoder(nn.Module):
    """Projects precomputed per-residue protein embeddings to shared dim.

    The input embeddings are pre-extracted from GEMS B6AEPL datasets:
      - ANKH-base: 320 dims
      - ESM2-t6: 768 dims
      - Total: 1088 dims per residue

    These are frozen (no gradient flows back to ESM2/ANKH).
    """

    def __init__(self, esm_dim: int = 1088, proj_dim: int = 256):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(esm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(
        self, prot_emb: torch.Tensor, prot_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            prot_emb: [B, max_res, esm_dim] padded embeddings
            prot_mask: [B, max_res] boolean mask (True = real residue)

        Returns:
            prot_features: [B, max_res, proj_dim]
            prot_mask: [B, max_res] passthrough
        """
        prot_features = self.proj(prot_emb)
        return prot_features, prot_mask
