"""
Lazy-loading PyTorch dataset for PLATE-VS virtual screening.

Loads samples on-the-fly from the registry CSV, generating 3D ligand
conformers from SMILES and looking up precomputed protein embeddings.

This avoids loading 2.9M samples into memory at once.

Usage:
    dataset = PlateVSDataset(
        registry_path="training_data_full/registry_2d_split.csv",
        protein_emb_path="data/plate_vs_protein_embeddings/esm2_embeddings.npz",
        split="train",
        similarity_threshold="0p7",
        include_decoys=True,
        max_decoys_per_target=50,
    )
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")


class PlateVSDataset(Dataset):
    """Lazy-loading dataset for PLATE-VS virtual screening.

    Each sample is a (protein, ligand) pair with a binary activity label.
    Ligand 3D conformers are generated on-the-fly from SMILES.
    Protein embeddings are precomputed and cached in memory (826 proteins).
    """

    def __init__(
        self,
        registry_path: str,
        protein_emb_path: str,
        split: str = "train",
        similarity_threshold: str = "0p7",
        include_decoys: bool = True,
        max_decoys_per_target: int | None = None,
        max_ligand_atoms: int = 100,
        max_pocket_res: int = 80,
        protein_partition: str | None = None,
    ):
        super().__init__()
        self.max_ligand_atoms = max_ligand_atoms
        self.max_pocket_res = max_pocket_res

        # Load protein embeddings (826 proteins, fits in memory)
        print(f"  Loading protein embeddings from {protein_emb_path}...")
        emb_data = np.load(protein_emb_path, allow_pickle=False)
        self.protein_embs = {k: emb_data[k] for k in emb_data.files}
        print(f"  {len(self.protein_embs)} proteins loaded")

        # Load and filter registry
        print(f"  Loading registry {split}/{similarity_threshold}...")
        self.samples = self._load_registry(
            registry_path, split, similarity_threshold,
            include_decoys, max_decoys_per_target, protein_partition,
        )
        print(f"  {len(self.samples)} samples")

        # Cache for generated conformers (LRU-style, keeps recent N)
        self._conformer_cache = {}
        self._cache_max = 10000

    def _load_registry(
        self,
        registry_path: str,
        split: str,
        similarity_threshold: str,
        include_decoys: bool,
        max_decoys_per_target: int | None,
        protein_partition: str | None,
    ) -> list[dict]:
        """Load and filter registry CSV into list of sample dicts."""
        samples = []
        decoy_counts = {}

        with open(registry_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                is_active = row["is_active"] == "True"
                row_split = row["split"]
                row_thresh = row["similarity_threshold"]

                # Filter actives by split and threshold
                if is_active:
                    if row_split != split:
                        continue
                    if row_thresh != similarity_threshold:
                        continue
                else:
                    # Decoys
                    if not include_decoys:
                        continue
                    # Apply protein partition filter if specified
                    if protein_partition and row.get("protein_partition") != protein_partition:
                        continue
                    # Cap decoys per target
                    uid = row["uniprot_id"]
                    if max_decoys_per_target:
                        decoy_counts[uid] = decoy_counts.get(uid, 0) + 1
                        if decoy_counts[uid] > max_decoys_per_target:
                            continue

                # Skip if no protein embeddings
                uid = row["uniprot_id"]
                if uid not in self.protein_embs:
                    continue

                samples.append({
                    "uniprot_id": uid,
                    "smiles": row["smiles"],
                    "is_active": 1 if is_active else 0,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        sample = self.samples[idx]
        smiles = sample["smiles"]
        uid = sample["uniprot_id"]

        # Ligand 3D from SMILES
        lig_data = self._get_conformer(smiles)
        if lig_data is None:
            # Return a dummy sample (will be filtered in collate)
            lig_data = {
                "z": torch.tensor([6], dtype=torch.long),
                "pos": torch.zeros(1, 3, dtype=torch.float32),
            }

        # Protein embedding
        prot_emb = torch.tensor(
            self.protein_embs[uid][:self.max_pocket_res],
            dtype=torch.float32,
        )

        return Data(
            z=lig_data["z"][:self.max_ligand_atoms],
            pos=lig_data["pos"][:self.max_ligand_atoms],
            prot_emb=prot_emb,
            num_lig_atoms=min(lig_data["z"].shape[0], self.max_ligand_atoms),
            num_pocket_res=prot_emb.shape[0],
            y=torch.tensor([sample["is_active"]], dtype=torch.float32),
            uniprot_id=uid,
        )

    def _get_conformer(self, smiles: str) -> dict | None:
        """Generate or retrieve cached 3D conformer."""
        if smiles in self._conformer_cache:
            return self._conformer_cache[smiles]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            if AllChem.EmbedMolecule(mol) != 0:
                return None

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

        mol = Chem.RemoveHs(mol)
        if mol.GetNumConformers() == 0:
            return None

        conf = mol.GetConformer()
        result = {
            "z": torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long),
            "pos": torch.tensor(conf.GetPositions(), dtype=torch.float32),
        }

        # Cache management
        if len(self._conformer_cache) >= self._cache_max:
            # Remove oldest entry
            oldest = next(iter(self._conformer_cache))
            del self._conformer_cache[oldest]
        self._conformer_cache[smiles] = result

        return result
