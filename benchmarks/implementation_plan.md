# Implementation Plan: Dual-Encoder Binding Affinity Model
## ESM2 + TorchMD-NET ET with Cross-Attention Fusion on PDBbind CleanSplit

---

## Overview

**Goal:** Build a deep learning model for protein-ligand binding affinity prediction that outperforms GEMS (Test R = 0.815) on the PDBbind CleanSplit benchmark.

**Architecture:**
- Protein encoder: Frozen ESM2 (precomputed features from CleanSplit download)
- Ligand encoder: TorchMD-NET Equivariant Transformer (trained from scratch)
- Fusion: Bidirectional cross-attention
- Loss: Composite (Huber + pairwise ranking margin)

**Evaluation:** CASF-2016 benchmark (Pearson R, R², RMSE, Spearman ρ)

---

## Phase 0: Data Audit & Preparation (Week 1)

### 0.1 Verify the CleanSplit download

The GEMS Zenodo dataset provides precomputed PyTorch datasets (`.pt` files) containing interaction graphs. Each graph has:
- Node features including ESM2 embeddings (per-residue for protein atoms, per-atom for ligand atoms)
- The split file: `PDBbind_data/PDBbind_data_split_cleansplit.json`

**Critical check — pocket-level vs full-protein ESM2:**

```python
import torch
import json

# Load the precomputed dataset
dataset = torch.load("path/to/cleansplit_dataset.pt")

# Inspect a single sample
sample = dataset[0]
print(f"Keys: {sample.keys()}")
print(f"Number of nodes: {sample.num_nodes}")
print(f"Node feature dim: {sample.x.shape}")

# Check if protein embeddings exist and their scope
if hasattr(sample, 'protein_embeddings'):
    print(f"Protein embedding shape: {sample.protein_embeddings.shape}")
    # Shape [N_residues, embed_dim] — check N_residues
    # If N_residues << full protein length, it's pocket-level

# Load the split
with open("PDBbind_data/PDBbind_data_split_cleansplit.json") as f:
    split = json.load(f)

print(f"Train set size: {len(split['train'])}")
print(f"Test set size: {len(split['test'])}")
# CleanSplit train should be ~11,000-13,000 complexes
# CASF-2016 test = 285 complexes
```

**If ESM2 features are full-protein (not pocket-level):**
You'll need to extract pocket residues yourself. The standard approach:

```python
from Bio.PDB import PDBParser
import numpy as np

def extract_pocket_residues(protein_pdb, ligand_sdf, cutoff=10.0):
    """Extract residues within cutoff Angstroms of any ligand atom."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", protein_pdb)

    # Get ligand atom coordinates (from SDF/MOL2)
    ligand_coords = get_ligand_coords(ligand_sdf)  # shape [N_lig_atoms, 3]

    pocket_residue_indices = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':
                    continue  # skip heteroatoms
                ca = residue['CA'] if 'CA' in residue else None
                if ca is None:
                    continue
                ca_coord = ca.get_vector().get_array()
                # Min distance to any ligand atom
                dists = np.linalg.norm(ligand_coords - ca_coord, axis=1)
                if dists.min() < cutoff:
                    pocket_residue_indices.append(residue.id[1])

    return pocket_residue_indices
```

### 0.2 Prepare ligand 3D conformers

TorchMD-NET ET requires atom types (`z`) and 3D coordinates (`pos`). Extract these from the SDF files in PDBbind:

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

def ligand_to_graph(sdf_path):
    """Convert SDF to (z, pos) tensors for TorchMD-NET."""
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next(supplier)

    if mol is None:
        raise ValueError(f"Cannot parse {sdf_path}")

    conf = mol.GetConformer()
    positions = conf.GetPositions()  # [N_atoms, 3]
    atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    return {
        'z': torch.tensor(atomic_nums, dtype=torch.long),
        'pos': torch.tensor(positions, dtype=torch.float32),
        'num_atoms': len(atomic_nums),
    }
```

### 0.3 Build the unified dataset

```python
from torch_geometric.data import Data, InMemoryDataset
import os

class BindingAffinityDataset(InMemoryDataset):
    """
    Combined dataset with:
    - Protein: precomputed ESM2 pocket-residue embeddings [N_res, 1280]
    - Ligand: atom types + 3D coordinates for TorchMD-NET
    - Target: pK value (pKd/pKi/pIC50)
    """
    def __init__(self, root, split='train', transform=None):
        self.split = split
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.split}_data.pt']

    def process(self):
        # Load CleanSplit split file
        with open(os.path.join(self.root, 'cleansplit.json')) as f:
            split_dict = json.load(f)

        pdb_ids = split_dict[self.split]
        data_list = []

        for pdb_id in pdb_ids:
            try:
                # --- Protein features (precomputed ESM2) ---
                esm2_emb = load_esm2_embeddings(pdb_id)  # [N_res, 1280]
                pocket_mask = load_pocket_mask(pdb_id)     # boolean [N_res]
                prot_emb = esm2_emb[pocket_mask]           # [N_pocket, 1280]

                # --- Ligand features ---
                lig_data = ligand_to_graph(
                    os.path.join(self.root, 'structures', f'{pdb_id}_ligand.sdf')
                )

                # --- Target ---
                pk_value = affinity_dict[pdb_id]  # float

                # --- Build combined Data object ---
                data = Data(
                    # Ligand (for TorchMD-NET)
                    z=lig_data['z'],
                    pos=lig_data['pos'],
                    num_lig_atoms=lig_data['num_atoms'],

                    # Protein (precomputed ESM2)
                    prot_emb=prot_emb,
                    num_pocket_res=prot_emb.shape[0],

                    # Target
                    y=torch.tensor([pk_value], dtype=torch.float32),

                    # Metadata
                    pdb_id=pdb_id,
                )
                data_list.append(data)

            except Exception as e:
                print(f"Skipping {pdb_id}: {e}")
                continue

        self.save(data_list, self.processed_paths[0])
```

### 0.4 Custom collation

Because protein and ligand have variable sizes and different representations, you'll need a custom collate function:

```python
from torch_geometric.data import Batch

def custom_collate(data_list):
    """
    Batches ligand graphs (via PyG Batch) and pads protein
    embeddings to the same length within the batch.
    """
    batch = Batch.from_data_list(data_list)

    # Pad protein embeddings to max pocket size in batch
    max_res = max(d.num_pocket_res for d in data_list)
    prot_padded = torch.zeros(len(data_list), max_res, 1280)
    prot_mask = torch.zeros(len(data_list), max_res, dtype=torch.bool)

    for i, d in enumerate(data_list):
        n = d.num_pocket_res
        prot_padded[i, :n] = d.prot_emb
        prot_mask[i, :n] = True

    batch.prot_padded = prot_padded       # [B, max_res, 1280]
    batch.prot_mask = prot_mask           # [B, max_res]

    return batch
```

---

## Phase 1: Model Architecture (Week 2-3)

### 1.1 Ligand encoder: TorchMD-NET Equivariant Transformer

We use TorchMD-NET's ET as a **feature extractor** (not for energy prediction). This means we take the per-atom representations from the last layer rather than the aggregated scalar output.

```python
# Install: pip install torchmd-net

from torchmdnet.models.torchmd_et import EquivariantTransformer

class LigandEncoder(torch.nn.Module):
    """
    Wraps TorchMD-NET Equivariant Transformer to produce
    per-atom scalar representations for the ligand.
    """
    def __init__(
        self,
        hidden_channels=256,
        num_layers=6,
        num_heads=8,
        num_rbf=64,
        cutoff=5.0,        # Angstroms — ligands are small
        max_z=100,
        proj_dim=256,       # final projection dimension
    ):
        super().__init__()

        self.et = EquivariantTransformer(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_rbf=num_rbf,
            cutoff_lower=0.0,
            cutoff_upper=cutoff,
            max_z=max_z,
            attn_activation='silu',
            num_heads_scalar=num_heads,      # TorchMD-NET 2.0 param
            distance_influence='both',
        )

        # Project scalar features to shared dimension
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, proj_dim),
            torch.nn.LayerNorm(proj_dim),
            torch.nn.SiLU(),
        )

    def forward(self, z, pos, batch):
        """
        Args:
            z: [N_total_atoms] atom types (int)
            pos: [N_total_atoms, 3] coordinates
            batch: [N_total_atoms] batch assignment index

        Returns:
            lig_features: [N_total_atoms, proj_dim] per-atom features
            lig_batch: [N_total_atoms] batch index (passthrough)
        """
        # TorchMD-NET ET forward — returns scalar and vector features
        # We extract only the scalar (invariant) features for fusion
        x_scalar, x_vector, z_out, pos_out, batch_out = self.et(
            z=z, pos=pos, batch=batch
        )
        # x_scalar: [N_total_atoms, hidden_channels]

        lig_features = self.proj(x_scalar)
        return lig_features, batch_out
```

**Important implementation notes for the ET:**
- `cutoff=5.0` Å is appropriate for intramolecular ligand interactions (most bonds are <2 Å, non-bonded interactions within 5 Å). You may want to test 6.0 and 8.0 as well.
- The ET natively handles variable-size molecules via the `batch` index tensor.
- We discard the vector (equivariant) features at the fusion stage since binding affinity is a scalar (invariant) quantity. The vector features are still used internally within the ET layers for equivariant message passing.

### 1.2 Protein encoder: Frozen ESM2 projection

Since ESM2 embeddings are precomputed (dim=320 for ESM2-t6 from GEMS, or dim=1280 if you recompute with ESM2-t33-650M), we just need a projection:

```python
class ProteinEncoder(torch.nn.Module):
    """
    Projects precomputed ESM2 residue embeddings to the shared dimension.
    Frozen — no gradient flows back to ESM2.
    """
    def __init__(self, esm_dim=320, proj_dim=256):
        super().__init__()
        # Note: esm_dim=320 for esm2_t6, 1280 for esm2_t33_650M
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(esm_dim, proj_dim),
            torch.nn.LayerNorm(proj_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(proj_dim, proj_dim),
            torch.nn.LayerNorm(proj_dim),
        )

    def forward(self, prot_emb, prot_mask):
        """
        Args:
            prot_emb: [B, max_res, esm_dim] padded ESM2 embeddings
            prot_mask: [B, max_res] boolean mask (True = real residue)

        Returns:
            prot_features: [B, max_res, proj_dim]
            prot_mask: [B, max_res] passthrough
        """
        prot_features = self.proj(prot_emb)
        return prot_features, prot_mask
```

### 1.3 Cross-attention fusion module

This is the core innovation. Bidirectional cross-attention lets each modality attend to the other:

```python
class CrossAttentionLayer(torch.nn.Module):
    """Single layer of bidirectional cross-attention."""
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()

        # Protein attends to Ligand
        self.prot_cross_attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Ligand attends to Protein
        self.lig_cross_attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward blocks
        self.prot_ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model * 4, d_model),
        )
        self.lig_ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model * 4, d_model),
        )

        # Layer norms (pre-norm style)
        self.prot_norm1 = torch.nn.LayerNorm(d_model)
        self.prot_norm2 = torch.nn.LayerNorm(d_model)
        self.lig_norm1 = torch.nn.LayerNorm(d_model)
        self.lig_norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, prot_feat, lig_feat, prot_mask, lig_mask):
        """
        Args:
            prot_feat: [B, N_res, d_model]
            lig_feat:  [B, N_atoms, d_model]
            prot_mask: [B, N_res]   (True = valid)
            lig_mask:  [B, N_atoms] (True = valid)

        Returns:
            updated prot_feat, lig_feat (same shapes)
        """
        # --- Protein attends to Ligand ---
        prot_normed = self.prot_norm1(prot_feat)
        # key_padding_mask: True means IGNORE, so invert
        prot_cross, _ = self.prot_cross_attn(
            query=prot_normed,
            key=lig_feat,
            value=lig_feat,
            key_padding_mask=~lig_mask,
        )
        prot_feat = prot_feat + prot_cross
        prot_feat = prot_feat + self.prot_ffn(self.prot_norm2(prot_feat))

        # --- Ligand attends to Protein ---
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


class CrossAttentionFusion(torch.nn.Module):
    """Stack of cross-attention layers + gated pooling."""
    def __init__(self, d_model=256, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()

        self.layers = torch.nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Gating mechanism for final pooled representations
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(d_model * 2, d_model * 2),
            torch.nn.Sigmoid(),
        )
        self.final_norm = torch.nn.LayerNorm(d_model * 2)

    def forward(self, prot_feat, lig_feat, prot_mask, lig_mask):
        """
        Returns:
            fused: [B, d_model * 2] batch-level fused representation
        """
        for layer in self.layers:
            prot_feat, lig_feat = layer(
                prot_feat, lig_feat, prot_mask, lig_mask
            )

        # Masked mean pooling
        prot_mask_f = prot_mask.unsqueeze(-1).float()  # [B, N_res, 1]
        lig_mask_f = lig_mask.unsqueeze(-1).float()    # [B, N_atoms, 1]

        prot_pooled = (prot_feat * prot_mask_f).sum(1) / prot_mask_f.sum(1).clamp(min=1)
        lig_pooled = (lig_feat * lig_mask_f).sum(1) / lig_mask_f.sum(1).clamp(min=1)

        # Concatenate + gating
        combined = torch.cat([prot_pooled, lig_pooled], dim=-1)  # [B, d*2]
        gate_weights = self.gate(combined)
        fused = self.final_norm(combined * gate_weights)

        return fused
```

**Note on converting ligand features to batched form for cross-attention:**

The ET returns per-atom features with a flat batch index. We need to pad them into `[B, max_atoms, d]` for `nn.MultiheadAttention`:

```python
def scatter_to_padded(features, batch_idx, max_size=None):
    """
    Convert [N_total, d] + batch_idx -> [B, max_size, d] padded tensor.
    Returns (padded_features, mask).
    """
    B = batch_idx.max().item() + 1
    d = features.shape[-1]

    # Count atoms per molecule
    counts = torch.bincount(batch_idx, minlength=B)
    if max_size is None:
        max_size = counts.max().item()

    padded = torch.zeros(B, max_size, d, device=features.device)
    mask = torch.zeros(B, max_size, dtype=torch.bool, device=features.device)

    # Fill in
    offsets = torch.zeros(B, dtype=torch.long, device=features.device)
    for i in range(features.shape[0]):
        b = batch_idx[i].item()
        idx = offsets[b].item()
        if idx < max_size:
            padded[b, idx] = features[i]
            mask[b, idx] = True
            offsets[b] += 1

    return padded, mask
```

**Vectorized (faster) version:**

```python
def scatter_to_padded_fast(features, batch_idx):
    """Vectorized version — no Python loop."""
    B = batch_idx.max().item() + 1
    d = features.shape[-1]

    counts = torch.bincount(batch_idx, minlength=B)
    max_size = counts.max().item()

    padded = torch.zeros(B, max_size, d, device=features.device)
    mask = torch.zeros(B, max_size, dtype=torch.bool, device=features.device)

    # Compute within-batch indices
    sort_idx = torch.argsort(batch_idx, stable=True)
    sorted_batch = batch_idx[sort_idx]
    sorted_features = features[sort_idx]

    # Cumulative count within each batch element
    offsets = torch.zeros_like(batch_idx)
    for b in range(B):
        group_mask = (sorted_batch == b)
        offsets[group_mask] = torch.arange(
            group_mask.sum(), device=features.device
        )

    padded[sorted_batch, offsets] = sorted_features
    mask[sorted_batch, offsets] = True

    return padded, mask
```


### 1.4 Prediction head

```python
class PredictionHead(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, fused):
        """fused: [B, input_dim] -> [B, 1] predicted pK"""
        return self.head(fused)
```

### 1.5 Full model assembly

```python
class BindingAffinityModel(torch.nn.Module):
    def __init__(
        self,
        esm_dim=320,          # 320 for esm2_t6, 1280 for esm2_t33_650M
        proj_dim=256,          # shared embedding dimension
        et_layers=6,
        et_heads=8,
        et_rbf=64,
        et_cutoff=5.0,
        cross_attn_layers=3,
        cross_attn_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        # Encoders
        self.protein_encoder = ProteinEncoder(esm_dim, proj_dim)
        self.ligand_encoder = LigandEncoder(
            hidden_channels=proj_dim,
            num_layers=et_layers,
            num_heads=et_heads,
            num_rbf=et_rbf,
            cutoff=et_cutoff,
            proj_dim=proj_dim,
        )

        # Fusion
        self.fusion = CrossAttentionFusion(
            d_model=proj_dim,
            n_heads=cross_attn_heads,
            n_layers=cross_attn_layers,
            dropout=dropout,
        )

        # Prediction
        self.predictor = PredictionHead(
            input_dim=proj_dim * 2,
            hidden_dim=proj_dim,
            dropout=dropout * 2,  # heavier dropout in head
        )

    def forward(self, batch):
        """
        Args:
            batch: custom-collated batch with:
                - z, pos, batch (ligand graph - PyG format)
                - prot_padded [B, max_res, esm_dim]
                - prot_mask [B, max_res]
                - y [B, 1]

        Returns:
            pred_pk: [B, 1] predicted pK values
        """
        # --- Protein branch ---
        prot_feat, prot_mask = self.protein_encoder(
            batch.prot_padded, batch.prot_mask
        )
        # prot_feat: [B, max_res, proj_dim]

        # --- Ligand branch ---
        lig_feat_flat, lig_batch = self.ligand_encoder(
            batch.z, batch.pos, batch.batch
        )
        # lig_feat_flat: [N_total_atoms, proj_dim]

        # Convert flat ligand features to padded batch form
        lig_feat_padded, lig_mask = scatter_to_padded_fast(
            lig_feat_flat, lig_batch
        )
        # lig_feat_padded: [B, max_atoms, proj_dim]

        # --- Fusion ---
        fused = self.fusion(prot_feat, lig_feat_padded, prot_mask, lig_mask)
        # fused: [B, proj_dim * 2]

        # --- Prediction ---
        pred_pk = self.predictor(fused)
        return pred_pk
```

---

## Phase 2: Loss Function (Week 2)

### 2.1 Composite loss implementation

```python
class CompositeLoss(torch.nn.Module):
    """
    L_total = L_huber + lambda_rank * L_rank

    - L_huber: Smooth L1 loss (robust to affinity outliers)
    - L_rank: Pairwise margin ranking loss
    """
    def __init__(
        self,
        lambda_rank=0.1,
        huber_delta=1.0,
        rank_margin=0.5,
        rank_sample_size=64,  # pairs per batch for efficiency
    ):
        super().__init__()
        self.lambda_rank = lambda_rank
        self.huber_delta = huber_delta
        self.rank_margin = rank_margin
        self.rank_sample_size = rank_sample_size

        self.huber = torch.nn.SmoothL1Loss(beta=huber_delta)

    def pairwise_ranking_loss(self, pred, target):
        """
        For all pairs (i,j) where target_i > target_j,
        enforce pred_i > pred_j with a margin.
        """
        pred = pred.squeeze(-1)       # [B]
        target = target.squeeze(-1)   # [B]
        B = pred.shape[0]

        if B < 2:
            return torch.tensor(0.0, device=pred.device)

        # Sample pairs for efficiency
        n_pairs = min(self.rank_sample_size, B * (B - 1) // 2)
        idx_i = torch.randint(0, B, (n_pairs,), device=pred.device)
        idx_j = torch.randint(0, B, (n_pairs,), device=pred.device)

        # Ensure different indices
        mask_diff = idx_i != idx_j
        idx_i = idx_i[mask_diff]
        idx_j = idx_j[mask_diff]

        if idx_i.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device)

        # Target differences
        target_diff = target[idx_i] - target[idx_j]    # positive if i > j
        pred_diff = pred[idx_i] - pred[idx_j]

        # For pairs where i has higher affinity
        # We want pred_i - pred_j > margin (scaled by target diff)
        sign = torch.sign(target_diff)

        # Adaptive margin: larger affinity differences demand larger gaps
        adaptive_margin = self.rank_margin * target_diff.abs().clamp(min=0.1)

        loss = torch.relu(adaptive_margin - sign * pred_diff)
        return loss.mean()

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1] predicted pK
            target: [B, 1] true pK

        Returns:
            total_loss, dict of individual losses
        """
        l_huber = self.huber(pred, target)
        l_rank = self.pairwise_ranking_loss(pred, target)

        total = l_huber + self.lambda_rank * l_rank

        return total, {
            'huber': l_huber.item(),
            'rank': l_rank.item(),
            'total': total.item(),
        }
```

---

## Phase 3: Training Pipeline (Week 3-4)

### 3.1 Training configuration

```python
# config.yaml (or argparse)
config = {
    # Data
    'data_root': '/path/to/pdbbind_cleansplit/',
    'batch_size': 16,         # limited by cross-attention memory
    'num_workers': 4,

    # Model
    'esm_dim': 320,           # 320 for esm2_t6 precomputed
    'proj_dim': 256,
    'et_layers': 6,
    'et_heads': 8,
    'et_cutoff': 5.0,
    'cross_attn_layers': 3,
    'cross_attn_heads': 8,
    'dropout': 0.1,

    # Loss
    'lambda_rank': 0.1,
    'huber_delta': 1.0,

    # Optimizer
    'lr': 3e-4,
    'weight_decay': 1e-5,
    'warmup_steps': 500,
    'max_epochs': 100,
    'patience': 15,           # early stopping

    # Infrastructure
    'device': 'cuda',
    'seed': 42,
    'grad_clip': 1.0,
}
```

### 3.2 Training loop

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.stats import pearsonr, spearmanr
import numpy as np

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    all_pred, all_target = [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        loss, loss_dict = criterion(pred, batch.y.unsqueeze(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss_dict['total'] * batch.y.shape[0]
        all_pred.append(pred.detach().cpu())
        all_target.append(batch.y.cpu())

    all_pred = torch.cat(all_pred).numpy().flatten()
    all_target = torch.cat(all_target).numpy().flatten()

    r, _ = pearsonr(all_pred, all_target)
    return total_loss / len(all_pred), r


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_pred, all_target = [], []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss, loss_dict = criterion(pred, batch.y.unsqueeze(-1))

        total_loss += loss_dict['total'] * batch.y.shape[0]
        all_pred.append(pred.cpu())
        all_target.append(batch.y.cpu())

    all_pred = torch.cat(all_pred).numpy().flatten()
    all_target = torch.cat(all_target).numpy().flatten()

    r_pearson, _ = pearsonr(all_pred, all_target)
    r_spearman, _ = spearmanr(all_pred, all_target)
    rmse = np.sqrt(np.mean((all_pred - all_target) ** 2))
    r2 = 1 - np.sum((all_pred - all_target)**2) / np.sum(
        (all_target - all_target.mean())**2
    )

    return {
        'loss': total_loss / len(all_pred),
        'pearson_r': r_pearson,
        'spearman_rho': r_spearman,
        'rmse': rmse,
        'r2': r2,
    }
```

### 3.3 Learning rate scheduling

Use warmup + cosine annealing:

```python
from torch.optim.lr_scheduler import LambdaLR
import math

def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)
```

### 3.4 Cross-validation on CleanSplit train set

GEMS uses an 80/20 train/val split within the CleanSplit training set, with optional 5-fold CV. Follow the same protocol for fair comparison:

```python
from sklearn.model_selection import KFold

def run_cross_validation(config, n_folds=5):
    results = []

    # Load full training set PDB IDs
    with open(os.path.join(config['data_root'], 'cleansplit.json')) as f:
        split = json.load(f)
    train_ids = split['train']

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config['seed'])

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_ids)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")

        fold_train_ids = [train_ids[i] for i in train_idx]
        fold_val_ids = [train_ids[i] for i in val_idx]

        # Build fold-specific datasets and dataloaders
        # ... (subset the InMemoryDataset by pdb_id)

        model = BindingAffinityModel(**model_kwargs).to(config['device'])
        # ... train loop ...

        fold_result = evaluate(model, val_loader, criterion, config['device'])
        results.append(fold_result)

    # Report mean ± std across folds
    for key in results[0]:
        vals = [r[key] for r in results]
        print(f"{key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
```

---

## Phase 4: Evaluation on CASF-2016 (Week 5)

### 4.1 Scoring power

```python
def evaluate_casf2016(model, casf_loader, device):
    """
    CASF-2016 evaluation: scoring, ranking, docking, screening powers.
    Here we focus on scoring power (Pearson R) for direct comparison with GEMS.
    """
    metrics = evaluate(model, casf_loader, criterion, device)

    print("=== CASF-2016 Scoring Power ===")
    print(f"  Pearson R:    {metrics['pearson_r']:.3f}")
    print(f"  Spearman rho: {metrics['spearman_rho']:.3f}")
    print(f"  RMSE:         {metrics['rmse']:.3f}")
    print(f"  R²:           {metrics['r2']:.3f}")

    return metrics
```

### 4.2 Ranking power (per-cluster Spearman correlation)

CASF-2016 has 57 clusters of similar targets. Ranking power measures per-cluster Spearman ρ:

```python
def ranking_power(predictions, targets, cluster_assignments):
    """
    predictions: dict {pdb_id: predicted_pK}
    targets: dict {pdb_id: true_pK}
    cluster_assignments: dict {pdb_id: cluster_id}
    """
    from collections import defaultdict

    clusters = defaultdict(list)
    for pdb_id in predictions:
        cid = cluster_assignments[pdb_id]
        clusters[cid].append((predictions[pdb_id], targets[pdb_id]))

    spearman_per_cluster = []
    for cid, pairs in clusters.items():
        if len(pairs) < 3:
            continue
        preds = [p[0] for p in pairs]
        trues = [p[1] for p in pairs]
        rho, _ = spearmanr(preds, trues)
        spearman_per_cluster.append(rho)

    return {
        'mean_spearman': np.mean(spearman_per_cluster),
        'median_spearman': np.median(spearman_per_cluster),
        'std_spearman': np.std(spearman_per_cluster),
    }
```

---

## Phase 5: Ablation Studies (Week 5-6)

### 5.1 Ablation matrix

| Experiment | Protein | Ligand | Fusion | Expected insight |
|---|---|---|---|---|
| **A1 (baseline)** | ESM2 frozen | ET 6-layer | Cross-attn 3L | Main model |
| A2 | ESM2 frozen | ET 4-layer | Cross-attn 3L | ET depth |
| A3 | ESM2 frozen | ET 6-layer | Late fusion (concat) | Value of cross-attn |
| A4 | ESM2 frozen | ET 6-layer | Cross-attn 1L | Cross-attn depth |
| A5 | ESM2 frozen | MPNN (SchNet) | Cross-attn 3L | Value of equivariance |
| A6 | ESM2 frozen | ET 6-layer | Cross-attn 3L, no ranking loss | Value of ranking loss |
| A7 | No protein features | ET 6-layer | — (ligand only) | Is protein info helping? |
| A8 | Ankh + ESM2 concat | ET 6-layer | Cross-attn 3L | Multiple PLM benefit |

### 5.2 Comparison targets

| Model | CleanSplit Test R | Source |
|---|---|---|
| GEMS (pretrained, reported) | 0.815 | Graber et al. 2025 |
| Your SVM (FP + prot emb) | 0.722 | Your results |
| Your GBM (FP + prot emb) | 0.715 | Your results |
| **Your model (target)** | **> 0.82** | This work |

---

## Dependencies & Environment

```bash
# Core
conda create -n binding_affinity python=3.10
conda activate binding_affinity

# PyTorch + CUDA
conda install pytorch=2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# PyTorch Geometric
pip install torch-geometric

# TorchMD-NET (for ET architecture)
# Option A: install full package
pip install torchmd-net

# Option B: extract just the ET module (lighter)
# Clone repo and import torchmdnet.models.torchmd_et directly

# Molecular tools
conda install -c conda-forge rdkit biopython

# ML utilities
pip install scipy scikit-learn wandb  # wandb for experiment tracking

# ESM2 (only if you need to recompute embeddings)
pip install fair-esm
```

---

## Key Risk Areas & Mitigations

**Risk 1: Memory blowup in cross-attention.**
The cross-attention computes O(N_res × N_atoms) attention matrices. For large pockets (>100 residues) and large ligands (>80 atoms), this can be substantial.
→ Mitigation: Cap pocket to 80 residues (closest to ligand). Cap ligand at 100 atoms. Use gradient checkpointing on cross-attn layers.

**Risk 2: ESM2-t6 embeddings may be too low-dimensional (dim=320).**
The GEMS precomputed features use the small ESM2-t6 model. This was sufficient for their GNN but cross-attention may need richer representations.
→ Mitigation: If results plateau, recompute embeddings with ESM2-t33-650M (dim=1280). This is ~2-3 hours for the full PDBbind on a single GPU.

**Risk 3: TorchMD-NET ET overfitting on small PDBbind.**
The ET has ~1M+ parameters and CleanSplit training set is ~11K complexes.
→ Mitigation: Use dropout (0.1 in ET, 0.2 in head), weight decay, early stopping, and the ranking loss acts as implicit regularization. Consider also pretraining the ET on the SPICE dataset for molecular potentials, then fine-tuning.

**Risk 4: The precomputed GEMS data may not separate pocket vs full protein cleanly.**
The interaction graphs in GEMS are built from binding-site atoms within a distance cutoff, so embeddings on those graphs are inherently pocket-level. But the raw ESM2 features may have been computed on the full sequence.
→ Mitigation: Inspect the data carefully in Phase 0. If pocket selection is unclear, compute your own.

---

## Timeline Summary

| Week | Phase | Deliverable |
|---|---|---|
| 1 | Phase 0 | Dataset audit, pocket check, data pipeline |
| 2 | Phase 1 + 2 | Model architecture + loss function implemented |
| 3 | Phase 3 | Training pipeline, first run on 1-fold |
| 4 | Phase 3 | 5-fold CV results, hyperparameter tuning |
| 5 | Phase 4 + 5 | CASF-2016 evaluation + ablation studies |
| 6 | Phase 5 | Ablation analysis, paper-ready figures |
