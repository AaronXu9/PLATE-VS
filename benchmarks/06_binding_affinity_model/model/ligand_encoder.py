"""Ligand encoder using equivariant 3D molecular models.

Supports two backends:
  - TorchMD-NET ET (preferred, needs torchmd-net package)
  - SchNet from PyG (fallback, always available)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _try_import_torchmd_et():
    try:
        from torchmdnet.models.torchmd_et import TorchMD_ET
        return TorchMD_ET
    except (ImportError, OSError):
        return None


class SchNetEncoder(nn.Module):
    """SchNet-based ligand encoder (PyG built-in, always available)."""

    def __init__(self, hidden_channels=256, num_layers=6, num_gaussians=50,
                 cutoff=5.0, proj_dim=256):
        super().__init__()
        from torch_geometric.nn import SchNet
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=hidden_channels,
            num_interactions=num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout="add",  # We'll use per-atom features, not the readout
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_channels, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
        )

    def forward(self, z, pos, batch):
        # SchNet processes atoms and produces per-atom features internally
        # We need to access the intermediate representations
        h = self.schnet.embedding(z)
        edge_index, edge_weight, edge_attr = self._get_edges(pos, batch)
        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        return self.proj(h), batch

    def _get_edges(self, pos, batch):
        from torch_geometric.nn.models.schnet import InteractionBlock
        from torch_geometric.nn import radius_graph
        edge_index = radius_graph(pos, r=self.schnet.cutoff, batch=batch, max_num_neighbors=32)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.schnet.distance_expansion(dist)
        return edge_index, dist, edge_attr


class TorchMDNETEncoder(nn.Module):
    """TorchMD-NET Equivariant Transformer encoder (preferred)."""

    def __init__(self, hidden_channels=256, num_layers=6, num_heads=8,
                 num_rbf=64, cutoff=5.0, max_z=100, proj_dim=256):
        super().__init__()
        TorchMD_ET = _try_import_torchmd_et()
        if TorchMD_ET is None:
            raise ImportError("torchmd-net not available")
        self.et = TorchMD_ET(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_rbf=num_rbf,
            cutoff_lower=0.0,
            cutoff_upper=cutoff,
            max_z=max_z,
            attn_activation="silu",
            distance_influence="both",
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_channels, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
        )

    def forward(self, z, pos, batch):
        x_scalar, _x_vector, _z_out, _pos_out, batch_out = self.et(
            z=z, pos=pos, batch=batch
        )
        return self.proj(x_scalar), batch_out


class LigandEncoder(nn.Module):
    """Ligand encoder that auto-selects the best available backend.

    Tries TorchMD-NET ET first, falls back to SchNet.
    """

    def __init__(self, hidden_channels=256, num_layers=6, num_heads=8,
                 num_rbf=64, cutoff=5.0, max_z=100, proj_dim=256,
                 backend="auto"):
        super().__init__()

        if backend == "auto":
            TorchMD_ET = _try_import_torchmd_et()
            backend = "torchmd_et" if TorchMD_ET is not None else "schnet"

        if backend == "torchmd_et":
            self.encoder = TorchMDNETEncoder(
                hidden_channels, num_layers, num_heads, num_rbf,
                cutoff, max_z, proj_dim,
            )
            self.backend = "torchmd_et"
        else:
            self.encoder = SchNetEncoder(
                hidden_channels, num_layers, num_gaussians=num_rbf,
                cutoff=cutoff, proj_dim=proj_dim,
            )
            self.backend = "schnet"

        print(f"  LigandEncoder backend: {self.backend}")

    def forward(self, z, pos, batch):
        return self.encoder(z, pos, batch)
