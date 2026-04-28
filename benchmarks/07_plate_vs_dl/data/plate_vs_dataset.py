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
    Ligand 3D conformers are loaded from precomputed .npz or generated on-the-fly.
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
        conformer_path: str | None = None,
        split_column: str = "split",
    ):
        super().__init__()
        self.max_ligand_atoms = max_ligand_atoms
        self.max_pocket_res = max_pocket_res

        # Load precomputed conformers if available
        self._precomputed_conformers = None
        if conformer_path and os.path.exists(conformer_path):
            import pickle
            print(f"  Loading precomputed conformers from {conformer_path}...")
            with open(conformer_path, "rb") as f:
                self._precomputed_conformers = pickle.load(f)
            print(f"  {len(self._precomputed_conformers):,} precomputed conformers loaded")

        # Load protein embeddings (826 proteins, fits in memory)
        print(f"  Loading protein embeddings from {protein_emb_path}...")
        emb_data = np.load(protein_emb_path, allow_pickle=False)
        self.protein_embs = {k: emb_data[k] for k in emb_data.files}
        print(f"  {len(self.protein_embs)} proteins loaded")

        # Load and filter registry
        print(f"  Loading registry {split}/{similarity_threshold} (split_column={split_column})...")
        self.samples = self._load_registry(
            registry_path, split, similarity_threshold,
            include_decoys, max_decoys_per_target, protein_partition,
            split_column,
        )
        print(f"  {len(self.samples)} samples")

        # Cache for on-the-fly conformers (only used when no precomputed data)
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
        split_column: str = "split",
    ) -> list[dict]:
        """Load and filter registry CSV into list of sample dicts.

        When split_column="protein_partition" (soft split), actives are filtered
        by the protein_partition column. Decoys (which all have protein_partition=
        'train') are included for any protein that has actives in the target split.
        """
        # Pass 1: collect actives and identify which proteins are in target split
        samples = []
        target_proteins = set()
        decoy_rows = []

        with open(registry_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                is_active = row["is_active"] == "True"
                uid = row["uniprot_id"]

                if uid not in self.protein_embs:
                    continue

                if is_active:
                    row_split = row.get(split_column, row["split"])
                    row_thresh = row["similarity_threshold"]
                    if row_split != split or row_thresh != similarity_threshold:
                        continue
                    target_proteins.add(uid)
                    samples.append({
                        "uniprot_id": uid,
                        "smiles": row["smiles"],
                        "is_active": 1,
                    })
                else:
                    if include_decoys:
                        decoy_rows.append(row)

        # Pass 2: add decoys for target proteins
        decoy_counts = {}
        for row in decoy_rows:
            uid = row["uniprot_id"]
            if uid not in target_proteins:
                continue
            if max_decoys_per_target:
                decoy_counts[uid] = decoy_counts.get(uid, 0) + 1
                if decoy_counts[uid] > max_decoys_per_target:
                    continue
            samples.append({
                "uniprot_id": uid,
                "smiles": row["smiles"],
                "is_active": 0,
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
            # WARNING: dummy single-atom fallback. This is a SENTINEL — the model
            # will trivially separate dummy from real samples. Caller must ensure
            # all SMILES have valid conformers (precompute or RDKit fallback).
            # Counter so we can monitor unexpected fallbacks.
            self._n_failed = getattr(self, "_n_failed", 0) + 1
            if self._n_failed in (1, 10, 100, 1000) or self._n_failed % 10000 == 0:
                print(f"  WARNING: {self._n_failed} samples failed to get conformer "
                      f"(returning dummy). Latest SMILES: {smiles[:80]}")
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
        """Load precomputed conformer; fall back to on-the-fly RDKit if missing."""
        # Try precomputed first
        if self._precomputed_conformers is not None:
            entry = self._precomputed_conformers.get(smiles)
            if entry is not None:
                return {
                    "z": torch.tensor(entry["z"], dtype=torch.long),
                    "pos": torch.tensor(entry["pos"], dtype=torch.float32),
                }
            # Cache miss — fall through to on-the-fly generation

        # On-the-fly generation with LRU cache
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
            oldest = next(iter(self._conformer_cache))
            del self._conformer_cache[oldest]
        self._conformer_cache[smiles] = result

        return result
