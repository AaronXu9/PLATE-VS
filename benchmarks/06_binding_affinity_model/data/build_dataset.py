"""
Build PyG InMemoryDataset for binding affinity model training.

Combines:
  - Ligand 3D conformers generated from SMILES via RDKit
  - Per-residue protein embeddings extracted from GEMS B6AEPL datasets
  - pK labels from PDBbind_data_dict.json

Usage:
    python benchmarks/06_binding_affinity_model/data/build_dataset.py \
        --gems-dir data/pdbbind_cleansplit/preprocessed/GEMS_pytorch_datasets \
        --output-dir data/pdbbind_cleansplit/binding_affinity_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Data, InMemoryDataset

RDLogger.DisableLog("rdApp.*")

# Add GEMS to path for torch.load (append, not insert, to avoid shadowing)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
GEMS_REPO = PROJECT_ROOT / "external" / "GEMS"
if GEMS_REPO.exists() and str(GEMS_REPO) not in sys.path:
    sys.path.append(str(GEMS_REPO))

# GEMS node feature layout (B6AEPL variant)
BASE_FEAT_DIM = 60  # First 60 dims are base atomic features
# After base features: [60:380] = ANKH (320d), [380:1148] = ESM2-t6 (768d)
PROT_EMB_START = BASE_FEAT_DIM
PROT_EMB_DIM = 1088  # 320 + 768


def load_crystal_sdf(sdf_path: str) -> dict | None:
    """Load ligand z and pos from a crystal SDF file.

    Returns dict with 'z' (atomic numbers) and 'pos' (3D coordinates),
    or None if loading fails.
    """
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)
    mol = next(supplier, None)
    if mol is None or mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer()
    pos = conf.GetPositions()
    # Check for degenerate coords (all zeros)
    if abs(pos).max() < 0.01:
        return None

    return {
        "z": torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long),
        "pos": torch.tensor(pos, dtype=torch.float32),
    }


def smiles_to_3d(smiles: str, max_attempts: int = 3) -> dict | None:
    """Generate 3D conformer from SMILES using RDKit.

    Returns dict with 'z' (atomic numbers) and 'pos' (3D coordinates),
    or None if conformer generation fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Try ETKDGv3 first, fall back to random coords
    for attempt in range(max_attempts):
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 + attempt
        result = AllChem.EmbedMolecule(mol, params)
        if result == 0:
            break
    else:
        # Last resort: use random coordinates
        result = AllChem.EmbedMolecule(mol, AllChem.EmbedParameters())
        if result != 0:
            return None

    # Optimize geometry
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass  # Use unoptimized coords if MMFF fails

    mol = Chem.RemoveHs(mol)
    if mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)

    return {"z": z, "pos": pos}


def extract_protein_residue_embeddings(gems_data) -> torch.Tensor | None:
    """Extract per-residue protein embeddings from a GEMS Data object.

    Returns tensor of shape [n_residues, PROT_EMB_DIM] or None.
    """
    n_total, n_lig, n_prot = gems_data.n_nodes.tolist()
    if n_prot == 0:
        return None

    # Protein node features: skip base features, take ANKH + ESM2
    prot_emb = gems_data.x[n_lig : n_lig + n_prot, PROT_EMB_START:]
    return prot_emb.clone()


def build_gems_index(gems_dir: str, variant: str = "B6AEPL") -> dict:
    """Build PDB ID → GEMS Data object lookup from .pt datasets."""
    gems_path = Path(gems_dir)
    index = {}

    for split in [
        "train_cleansplit", "casf2016", "casf2013",
        "casf2016_indep", "casf2013_indep",
    ]:
        dataset_path = gems_path / f"{variant}_{split}.pt"
        if not dataset_path.exists():
            print(f"  [skip] {dataset_path.name}")
            continue

        print(f"  Loading {dataset_path.name}...")
        dataset = torch.load(str(dataset_path), map_location="cpu", weights_only=False)
        for data in dataset:
            pdb_id = data.id.split("_")[0] if "_" in data.id else data.id
            pdb_id = pdb_id.lower()
            if pdb_id not in index:
                index[pdb_id] = data

    print(f"  GEMS index: {len(index)} complexes")
    return index


class BindingAffinityDataset(InMemoryDataset):
    """Combined dataset for binding affinity prediction.

    Each sample contains:
        - z: [N_lig] int64 atomic numbers
        - pos: [N_lig, 3] float32 ligand 3D coordinates
        - prot_emb: [N_res, 1088] float32 per-residue protein embeddings
        - num_lig_atoms: int
        - num_pocket_res: int
        - y: [1] float32 pK value (scaled to [0,1] via /16)
        - pdb_id: str
    """

    PK_MAX = 16.0  # Same scaling as GEMS

    def __init__(self, root, split="train", fold=0, transform=None):
        self.split = split
        self.fold = fold
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.split in ("casf2016", "casf2013"):
            return [f"{self.split}_data.pt"]
        return [f"{self.split}_f{self.fold}_data.pt"]

    def process(self):
        raise RuntimeError(
            "Call build_and_save() instead of relying on automatic processing."
        )

    @classmethod
    def build_and_save(
        cls,
        pdb_ids: list[str],
        smiles_lookup: dict[str, str],
        labels_dict: dict,
        gems_index: dict,
        output_path: str,
        max_ligand_atoms: int = 100,
        max_pocket_res: int = 80,
        crystal_dir: str | None = None,
    ) -> int:
        """Build dataset from components and save to disk.

        If crystal_dir is provided, loads ligand 3D from crystal SDF files
        (falling back to RDKit conformer if crystal SDF is missing).

        Returns number of successfully built samples.
        """
        data_list = []
        n_skip_smiles = 0
        n_skip_3d = 0
        n_skip_prot = 0
        n_crystal = 0
        n_rdkit = 0

        crystal_path = Path(crystal_dir) if crystal_dir else None

        for i, pdb_id in enumerate(pdb_ids):
            pid = pdb_id.lower()

            # Get label
            entry = labels_dict.get(pid) or labels_dict.get(pid.upper())
            if entry is None or entry.get("log_kd_ki") is None:
                continue
            pk = float(entry["log_kd_ki"])

            # Get ligand 3D: try crystal SDF first, fall back to RDKit
            lig_data = None
            if crystal_path:
                sdf_file = crystal_path / f"{pid}_ligand.sdf"
                if sdf_file.exists():
                    lig_data = load_crystal_sdf(str(sdf_file))
                    if lig_data is not None:
                        n_crystal += 1

            if lig_data is None:
                smiles = smiles_lookup.get(pid)
                if smiles is None:
                    n_skip_smiles += 1
                    continue
                lig_data = smiles_to_3d(smiles)
                if lig_data is not None:
                    n_rdkit += 1

            if lig_data is None:
                n_skip_3d += 1
                continue

            # Cap ligand size
            if lig_data["z"].shape[0] > max_ligand_atoms:
                n_skip_3d += 1
                continue

            # Get protein embeddings from GEMS
            gems_data = gems_index.get(pid)
            if gems_data is None:
                n_skip_prot += 1
                continue

            prot_emb = extract_protein_residue_embeddings(gems_data)
            if prot_emb is None or prot_emb.shape[0] == 0:
                n_skip_prot += 1
                continue

            # Cap pocket size
            if prot_emb.shape[0] > max_pocket_res:
                prot_emb = prot_emb[:max_pocket_res]

            # Scale pK to [0, 1] like GEMS
            y_scaled = pk / cls.PK_MAX

            data = Data(
                z=lig_data["z"],
                pos=lig_data["pos"],
                prot_emb=prot_emb,
                num_lig_atoms=lig_data["z"].shape[0],
                num_pocket_res=prot_emb.shape[0],
                y=torch.tensor([y_scaled], dtype=torch.float32),
                pdb_id=pid,
            )
            data_list.append(data)

            if (i + 1) % 2000 == 0:
                print(f"    Processed {i+1}/{len(pdb_ids)}, built {len(data_list)}")

        if data_list:
            # Save using PyG collation
            data, slices = InMemoryDataset.collate(data_list)
            torch.save((data, slices), output_path)

        print(
            f"    Built {len(data_list)} samples "
            f"(crystal: {n_crystal}, rdkit: {n_rdkit}, "
            f"skip: {n_skip_smiles} no SMILES, {n_skip_3d} 3D fail, "
            f"{n_skip_prot} no protein)"
        )
        return len(data_list)


def load_dataset(pt_path: str) -> list[Data]:
    """Load a saved dataset .pt file back into a list of Data objects."""
    data, slices = torch.load(pt_path, weights_only=False)
    n_samples = slices["z"].shape[0] - 1
    dataset = []
    for i in range(n_samples):
        d = Data()
        for key in slices:
            s = slices[key]
            d[key] = data[key][s[i] : s[i + 1]]
        dataset.append(d)
    return dataset


def load_smiles_lookup(smiles_csv: str) -> dict[str, str]:
    """Load PDB ID → SMILES mapping."""
    lookup = {}
    with open(smiles_csv) as f:
        for row in csv.DictReader(f):
            lookup[row["pdb_id"].lower()] = row["smiles"]
    return lookup


def main():
    parser = argparse.ArgumentParser(description="Build binding affinity dataset")
    parser.add_argument(
        "--gems-dir",
        default="data/pdbbind_cleansplit/preprocessed/GEMS_pytorch_datasets",
    )
    parser.add_argument("--gems-variant", default="B6AEPL")
    parser.add_argument(
        "--smiles-csv",
        default="data/pdbbind_cleansplit/smiles/pdbbind_smiles.csv",
    )
    parser.add_argument(
        "--labels",
        default="data/pdbbind_cleansplit/labels/PDBbind_data_dict.json",
    )
    parser.add_argument(
        "--splits",
        default="data/pdbbind_cleansplit/labels/PDBbind_data_split_cleansplit.json",
    )
    parser.add_argument(
        "--cv-dir",
        default="data/pdbbind_cleansplit/labels",
        help="Directory containing PDBbind_cleansplit_train_val_split_f*.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/pdbbind_cleansplit/binding_affinity_dataset",
    )
    parser.add_argument("--max-ligand-atoms", type=int, default=100)
    parser.add_argument("--max-pocket-res", type=int, default=80)
    parser.add_argument(
        "--crystal-dir",
        type=str,
        default=None,
        help="Directory with crystal ligand SDFs (e.g., data/pdbbind_cleansplit/crystal_ligands)",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated fold indices to build",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    # Load shared data
    print("Loading SMILES...")
    smiles_lookup = load_smiles_lookup(args.smiles_csv)
    print(f"  {len(smiles_lookup)} SMILES entries")

    print("Loading labels...")
    with open(args.labels) as f:
        labels_dict = json.load(f)

    print("Loading splits...")
    with open(args.splits) as f:
        split_dict = json.load(f)

    print("Building GEMS index...")
    gems_index = build_gems_index(args.gems_dir, args.gems_variant)

    # Build test sets (CASF-2016 and CASF-2013)
    for test_set in ["casf2016", "casf2013"]:
        pdb_ids = [pid.lower() for pid in split_dict.get(test_set, [])]
        if not pdb_ids:
            continue
        out_path = processed_dir / f"{test_set}_data.pt"
        if out_path.exists():
            print(f"\n[skip] {test_set} already exists")
            continue
        print(f"\nBuilding {test_set} ({len(pdb_ids)} complexes)...")
        BindingAffinityDataset.build_and_save(
            pdb_ids, smiles_lookup, labels_dict, gems_index, str(out_path),
            args.max_ligand_atoms, args.max_pocket_res, args.crystal_dir,
        )

    # Build train/val folds
    folds = [int(f) for f in args.folds.split(",")]
    for fold in folds:
        cv_path = Path(args.cv_dir) / f"PDBbind_cleansplit_train_val_split_f{fold}.json"
        if not cv_path.exists():
            print(f"\n[skip] CV fold {fold}: {cv_path} not found")
            continue

        with open(cv_path) as f:
            fold_split = json.load(f)

        for subset in ["train", "validation"]:
            pdb_ids = [pid.lower() for pid in fold_split.get(subset, [])]
            if not pdb_ids:
                continue
            label = "val" if subset == "validation" else "train"
            out_path = processed_dir / f"{label}_f{fold}_data.pt"
            if out_path.exists():
                print(f"\n[skip] {label}_f{fold} already exists")
                continue
            print(f"\nBuilding {label}_f{fold} ({len(pdb_ids)} complexes)...")
            BindingAffinityDataset.build_and_save(
                pdb_ids, smiles_lookup, labels_dict, gems_index, str(out_path),
                args.max_ligand_atoms, args.max_pocket_res, args.crystal_dir,
            )

    print(f"\nDatasets saved to {output_dir}")


if __name__ == "__main__":
    main()
