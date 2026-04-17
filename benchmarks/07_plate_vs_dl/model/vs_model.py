"""Virtual screening model — classification variant of BindingAffinityModel.

Same architecture (ligand encoder + protein encoder + cross-attention),
but with BCE loss and sigmoid output for active/inactive classification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Reuse components from the binding affinity model
SCRIPT_DIR = Path(__file__).resolve().parent
BA_MODEL_DIR = SCRIPT_DIR.parent.parent / "06_binding_affinity_model"
sys.path.insert(0, str(BA_MODEL_DIR))

from model.ligand_encoder import LigandEncoder
from model.protein_encoder import ProteinEncoder
from model.cross_attention import CrossAttentionFusion
from data.collate import scatter_to_padded


class VSPredictionHead(nn.Module):
    """Classification head for virtual screening."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.2):
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
        """Returns raw logits (apply sigmoid externally for probability)."""
        return self.head(fused)


class VirtualScreeningModel(nn.Module):
    """Dual-encoder model adapted for virtual screening classification."""

    def __init__(
        self,
        esm_dim: int = 320,
        proj_dim: int = 256,
        et_layers: int = 4,
        et_heads: int = 8,
        et_rbf: int = 64,
        et_cutoff: float = 10.0,
        cross_attn_layers: int = 3,
        cross_attn_heads: int = 8,
        dropout: float = 0.1,
        ligand_backend: str = "auto",
    ):
        super().__init__()

        self.protein_encoder = ProteinEncoder(esm_dim, proj_dim)
        self.ligand_encoder = LigandEncoder(
            hidden_channels=proj_dim,
            num_layers=et_layers,
            num_heads=et_heads,
            num_rbf=et_rbf,
            cutoff=et_cutoff,
            proj_dim=proj_dim,
            backend=ligand_backend,
        )
        self.fusion = CrossAttentionFusion(
            d_model=proj_dim,
            n_heads=cross_attn_heads,
            n_layers=cross_attn_layers,
            dropout=dropout,
        )
        self.predictor = VSPredictionHead(
            input_dim=proj_dim * 2,
            hidden_dim=proj_dim,
            dropout=dropout * 2,
        )

    def forward(self, batch) -> torch.Tensor:
        """Forward pass.

        Returns:
            logits: [B, 1] raw classification logits
        """
        # Protein
        prot_feat, prot_mask = self.protein_encoder(
            batch.prot_padded, batch.prot_mask
        )

        # Ligand
        lig_feat_flat, lig_batch = self.ligand_encoder(
            batch.z, batch.pos, batch.batch
        )
        lig_feat_padded, lig_mask = scatter_to_padded(lig_feat_flat, lig_batch)

        # Fusion
        fused = self.fusion(prot_feat, lig_feat_padded, prot_mask, lig_mask)

        # Classification
        logits = self.predictor(fused)
        return logits
