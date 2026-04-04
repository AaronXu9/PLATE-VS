"""Full binding affinity model: dual-encoder with cross-attention fusion.

Architecture:
    LigandEncoder (TorchMD-NET ET) → per-atom features
    ProteinEncoder (ESM2 projection) → per-residue features
    CrossAttentionFusion → fused representation
    PredictionHead → pK prediction
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.cross_attention import CrossAttentionFusion
from model.ligand_encoder import LigandEncoder
from model.prediction_head import PredictionHead
from model.protein_encoder import ProteinEncoder

from data.collate import scatter_to_padded


class BindingAffinityModel(nn.Module):
    """Dual-encoder binding affinity prediction model."""

    def __init__(
        self,
        esm_dim: int = 1088,
        proj_dim: int = 256,
        et_layers: int = 6,
        et_heads: int = 8,
        et_rbf: int = 64,
        et_cutoff: float = 5.0,
        cross_attn_layers: int = 3,
        cross_attn_heads: int = 8,
        dropout: float = 0.1,
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
        )
        self.fusion = CrossAttentionFusion(
            d_model=proj_dim,
            n_heads=cross_attn_heads,
            n_layers=cross_attn_layers,
            dropout=dropout,
        )
        self.predictor = PredictionHead(
            input_dim=proj_dim * 2,
            hidden_dim=proj_dim,
            dropout=dropout * 2,
        )

    def forward(self, batch) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: Custom-collated batch with:
                - z: [N_total_atoms] atomic numbers
                - pos: [N_total_atoms, 3] coordinates
                - batch (attr): [N_total_atoms] batch index
                - prot_padded: [B, max_res, esm_dim]
                - prot_mask: [B, max_res]
                - y: [B, 1] target pK (scaled)

        Returns:
            pred: [B, 1] predicted pK (scaled)
        """
        # Protein branch
        prot_feat, prot_mask = self.protein_encoder(
            batch.prot_padded, batch.prot_mask
        )

        # Ligand branch
        lig_feat_flat, lig_batch = self.ligand_encoder(
            batch.z, batch.pos, batch.batch
        )

        # Convert flat ligand features to padded batch form
        lig_feat_padded, lig_mask = scatter_to_padded(lig_feat_flat, lig_batch)

        # Fusion
        fused = self.fusion(prot_feat, lig_feat_padded, prot_mask, lig_mask)

        # Prediction
        pred = self.predictor(fused)
        return pred
