"""Bidirectional cross-attention fusion for protein-ligand interaction."""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """Single layer of bidirectional cross-attention.

    Protein attends to ligand, then ligand attends to protein.
    Pre-norm style with residual connections.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Protein attends to Ligand
        self.prot_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Ligand attends to Protein
        self.lig_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward blocks
        self.prot_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.lig_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        # Pre-norm LayerNorms
        self.prot_norm1 = nn.LayerNorm(d_model)
        self.prot_norm2 = nn.LayerNorm(d_model)
        self.lig_norm1 = nn.LayerNorm(d_model)
        self.lig_norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        prot_feat: torch.Tensor,
        lig_feat: torch.Tensor,
        prot_mask: torch.Tensor,
        lig_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            prot_feat: [B, N_res, d_model]
            lig_feat: [B, N_atoms, d_model]
            prot_mask: [B, N_res] (True = valid)
            lig_mask: [B, N_atoms] (True = valid)

        Returns:
            Updated (prot_feat, lig_feat) with same shapes.
        """
        # Protein attends to Ligand
        prot_normed = self.prot_norm1(prot_feat)
        prot_cross, _ = self.prot_cross_attn(
            query=prot_normed,
            key=lig_feat,
            value=lig_feat,
            key_padding_mask=~lig_mask,  # True = IGNORE in PyTorch MHA
        )
        prot_feat = prot_feat + prot_cross
        prot_feat = prot_feat + self.prot_ffn(self.prot_norm2(prot_feat))

        # Ligand attends to Protein
        lig_normed = self.lig_norm1(lig_feat)
        lig_cross, _ = self.lig_cross_attn(
            query=lig_normed,
            key=prot_feat,
            value=prot_feat,
            key_padding_mask=~prot_mask,
        )
        lig_feat = lig_feat + lig_cross
        lig_feat = lig_feat + self.lig_ffn(self.lig_norm2(lig_feat))

        return prot_feat, lig_feat


class CrossAttentionFusion(nn.Module):
    """Stack of cross-attention layers with gated pooling.

    Produces a fixed-size fused representation from variable-length
    protein and ligand sequences.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [CrossAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Gating mechanism for final pooled representations
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.Sigmoid(),
        )
        self.final_norm = nn.LayerNorm(d_model * 2)

    def forward(
        self,
        prot_feat: torch.Tensor,
        lig_feat: torch.Tensor,
        prot_mask: torch.Tensor,
        lig_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            prot_feat: [B, N_res, d_model]
            lig_feat: [B, N_atoms, d_model]
            prot_mask: [B, N_res]
            lig_mask: [B, N_atoms]

        Returns:
            fused: [B, d_model * 2] batch-level fused representation
        """
        for layer in self.layers:
            prot_feat, lig_feat = layer(prot_feat, lig_feat, prot_mask, lig_mask)

        # Masked mean pooling
        prot_mask_f = prot_mask.unsqueeze(-1).float()
        lig_mask_f = lig_mask.unsqueeze(-1).float()

        prot_pooled = (prot_feat * prot_mask_f).sum(1) / prot_mask_f.sum(1).clamp(min=1)
        lig_pooled = (lig_feat * lig_mask_f).sum(1) / lig_mask_f.sum(1).clamp(min=1)

        # Concatenate + gating
        combined = torch.cat([prot_pooled, lig_pooled], dim=-1)
        gate_weights = self.gate(combined)
        fused = self.final_norm(combined * gate_weights)

        return fused
