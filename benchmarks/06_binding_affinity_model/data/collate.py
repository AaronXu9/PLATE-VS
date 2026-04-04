"""Custom collation for binding affinity model batches.

Handles variable-size protein and ligand graphs:
  - Ligand: batched via PyG Batch (z, pos, batch index)
  - Protein: padded to [B, max_res, embed_dim] with boolean mask
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data


def _get_int(val) -> int:
    """Extract int from a scalar tensor or int."""
    if isinstance(val, torch.Tensor):
        return val.item()
    return int(val)


class AffinityBatch:
    """Simple batch container for the binding affinity model.

    Holds PyG Batch for ligand graph + padded protein embeddings.
    """

    def __init__(self, ligand_batch: Batch, prot_padded: torch.Tensor,
                 prot_mask: torch.Tensor, y: torch.Tensor, pdb_ids: list[str]):
        self.ligand_batch = ligand_batch
        self.prot_padded = prot_padded
        self.prot_mask = prot_mask
        self.y = y
        self.pdb_ids = pdb_ids

    @property
    def z(self):
        return self.ligand_batch.z

    @property
    def pos(self):
        return self.ligand_batch.pos

    @property
    def batch(self):
        return self.ligand_batch.batch

    @property
    def pdb_id(self):
        return self.pdb_ids

    def to(self, device):
        """Move all tensors to device."""
        self.ligand_batch = self.ligand_batch.to(device)
        self.prot_padded = self.prot_padded.to(device)
        self.prot_mask = self.prot_mask.to(device)
        self.y = self.y.to(device)
        return self


def custom_collate(data_list: list[Data]) -> AffinityBatch:
    """Collate a list of Data objects into an AffinityBatch."""
    B = len(data_list)

    # Extract protein embeddings and metadata
    prot_embs = [d.prot_emb.clone() for d in data_list]
    pocket_sizes = [_get_int(d.num_pocket_res) for d in data_list]
    pdb_ids = []
    y_list = []

    # Build stripped Data objects for PyG batching (ligand only)
    stripped = []
    for d in data_list:
        d_new = Data(z=d.z, pos=d.pos)
        stripped.append(d_new)
        y_list.append(d.y.flatten())
        pid = d.pdb_id
        if isinstance(pid, (list, tuple)):
            pid = pid[0] if len(pid) == 1 else str(pid)
        elif isinstance(pid, torch.Tensor):
            pid = pid.item() if pid.ndim == 0 else str(pid)
        pdb_ids.append(str(pid))

    ligand_batch = Batch.from_data_list(stripped)
    y = torch.stack(y_list)  # [B, 1] or [B]

    # Pad protein embeddings
    max_res = max(pocket_sizes)
    embed_dim = prot_embs[0].shape[-1]

    prot_padded = torch.zeros(B, max_res, embed_dim, dtype=torch.float32)
    prot_mask = torch.zeros(B, max_res, dtype=torch.bool)

    for i in range(B):
        n = pocket_sizes[i]
        prot_padded[i, :n] = prot_embs[i][:n]
        prot_mask[i, :n] = True

    return AffinityBatch(ligand_batch, prot_padded, prot_mask, y, pdb_ids)


def scatter_to_padded(features: torch.Tensor, batch_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert flat per-atom features to padded batch format.

    Args:
        features: [N_total, d] per-atom features from the ligand encoder
        batch_idx: [N_total] batch assignment for each atom

    Returns:
        padded: [B, max_atoms, d] zero-padded features
        mask: [B, max_atoms] boolean mask (True = real atom)
    """
    B = batch_idx.max().item() + 1
    d = features.shape[-1]
    device = features.device

    counts = torch.bincount(batch_idx, minlength=B)
    max_atoms = counts.max().item()

    padded = torch.zeros(B, max_atoms, d, device=device)
    mask = torch.zeros(B, max_atoms, dtype=torch.bool, device=device)

    # Compute within-batch offsets
    sorted_indices = torch.argsort(batch_idx, stable=True)
    sorted_batch = batch_idx[sorted_indices]
    sorted_features = features[sorted_indices]

    group_starts = torch.zeros(B + 1, dtype=torch.long, device=device)
    group_starts[1:] = counts.cumsum(0)

    within_group = torch.arange(len(batch_idx), device=device)
    within_group -= group_starts[sorted_batch]

    padded[sorted_batch, within_group] = sorted_features
    mask[sorted_batch, within_group] = True

    return padded, mask
