"""
Build dataset using HiQBind refined crystal structures.

HiQBind provides curated, refined ligand SDFs with correct bond orders
and protonation states. Falls back to RDKit conformers for complexes
not in HiQBind.

Usage (on CARC):
    python benchmarks/06_binding_affinity_model/data/build_hiqbind_dataset.py \
        --hiqbind-dir /project2/katritch_223/aoxu/data/hiqbind \
        --output-dir data/pdbbind_cleansplit/binding_affinity_hiqbind_dataset
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
GEMS_REPO = PROJECT_ROOT / "external" / "GEMS"
if GEMS_REPO.exists() and str(GEMS_REPO) not in sys.path:
    sys.path.append(str(GEMS_REPO))

from build_dataset import (
    smiles_to_3d,
    extract_protein_residue_embeddings,
    build_gems_index,
    load_smiles_lookup,
    load_dataset,
    BindingAffinityDataset,
    PROT_EMB_START,
)


def find_hiqbind_sdf(hiqbind_sm_dir: Path, pdb_id: str, het_code: str = None) -> Path | None:
    """Find the refined ligand SDF in HiQBind for a given PDB ID.

    HiQBind structure: {pdb_id}/{pdb_id}_{het}_{chain}_{resnum}/{...}_ligand_refined.sdf
    """
    pdb_dir = hiqbind_sm_dir / pdb_id.lower()
    if not pdb_dir.exists():
        return None

    # If het_code given, prefer matching subdir
    for subdir in sorted(pdb_dir.iterdir()):
        if not subdir.is_dir():
            continue
        sdf = subdir / f"{subdir.name}_ligand_refined.sdf"
        if sdf.exists():
            if het_code and het_code.upper() in subdir.name.upper():
                return sdf

    # Fallback: take first available refined SDF
    for subdir in sorted(pdb_dir.iterdir()):
        if not subdir.is_dir():
            continue
        sdf = subdir / f"{subdir.name}_ligand_refined.sdf"
        if sdf.exists():
            return sdf

    return None


def load_hiqbind_sdf(sdf_path: str) -> dict | None:
    """Load ligand z and pos from a HiQBind refined SDF file."""
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)
    mol = next(supplier, None)
    if mol is None or mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer()
    pos = conf.GetPositions()
    if abs(pos).max() < 0.01:
        return None

    return {
        "z": torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long),
        "pos": torch.tensor(pos, dtype=torch.float32),
    }


def main():
    parser = argparse.ArgumentParser(description="Build dataset with HiQBind refined structures")
    parser.add_argument("--hiqbind-dir", required=True, help="Path to HiQBind dataset root")
    parser.add_argument("--gems-dir", default="data/pdbbind_cleansplit/preprocessed/GEMS_pytorch_datasets")
    parser.add_argument("--gems-variant", default="B6AEPL")
    parser.add_argument("--smiles-csv", default="data/pdbbind_cleansplit/smiles/pdbbind_smiles.csv")
    parser.add_argument("--labels", default="data/pdbbind_cleansplit/labels/PDBbind_data_dict.json")
    parser.add_argument("--splits", default="data/pdbbind_cleansplit/labels/PDBbind_data_split_cleansplit.json")
    parser.add_argument("--cv-dir", default="data/pdbbind_cleansplit/labels")
    parser.add_argument("--output-dir", default="data/pdbbind_cleansplit/binding_affinity_hiqbind_dataset")
    parser.add_argument("--max-ligand-atoms", type=int, default=100)
    parser.add_argument("--max-pocket-res", type=int, default=80)
    parser.add_argument("--folds", type=str, default="0")
    args = parser.parse_args()

    hiqbind_sm_dir = Path(args.hiqbind_dir) / "raw_data_hiq_sm"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    print("Loading SMILES...")
    smiles_lookup = load_smiles_lookup(args.smiles_csv)
    print(f"  {len(smiles_lookup)} entries")

    print("Loading labels...")
    with open(args.labels) as f:
        labels_dict = json.load(f)
    with open(args.splits) as f:
        split_dict = json.load(f)

    print("Building GEMS index...")
    gems_index = build_gems_index(args.gems_dir, args.gems_variant)

    def build_split(pdb_ids, out_path, split_name):
        if out_path.exists():
            print(f"\n[skip] {split_name} already exists")
            return

        data_list = []
        n_hiqbind = 0
        n_rdkit = 0
        n_skip = 0

        for i, pdb_id in enumerate(pdb_ids):
            pid = pdb_id.lower()
            entry = labels_dict.get(pid) or labels_dict.get(pid.upper())
            if entry is None or entry.get("log_kd_ki") is None:
                continue
            pk = float(entry["log_kd_ki"])
            het_code = entry.get("ligand_name", "")

            # Try HiQBind refined SDF first
            lig_data = None
            hiq_sdf = find_hiqbind_sdf(hiqbind_sm_dir, pid, het_code)
            if hiq_sdf:
                lig_data = load_hiqbind_sdf(str(hiq_sdf))
                if lig_data is not None:
                    n_hiqbind += 1

            # Fallback to RDKit conformer
            if lig_data is None:
                smiles = smiles_lookup.get(pid)
                if smiles:
                    lig_data = smiles_to_3d(smiles)
                    if lig_data is not None:
                        n_rdkit += 1

            if lig_data is None:
                n_skip += 1
                continue

            if lig_data["z"].shape[0] > args.max_ligand_atoms:
                n_skip += 1
                continue

            # Protein embeddings from GEMS
            gems_data = gems_index.get(pid)
            if gems_data is None:
                n_skip += 1
                continue

            prot_emb = extract_protein_residue_embeddings(gems_data)
            if prot_emb is None or prot_emb.shape[0] == 0:
                n_skip += 1
                continue
            if prot_emb.shape[0] > args.max_pocket_res:
                prot_emb = prot_emb[:args.max_pocket_res]

            y_scaled = pk / 16.0

            data = Data(
                z=lig_data["z"], pos=lig_data["pos"], prot_emb=prot_emb,
                num_lig_atoms=lig_data["z"].shape[0],
                num_pocket_res=prot_emb.shape[0],
                y=torch.tensor([y_scaled], dtype=torch.float32),
                pdb_id=pid,
            )
            data_list.append(data)

            if (i + 1) % 2000 == 0:
                print(f"    {i+1}/{len(pdb_ids)}, built {len(data_list)}")

        if data_list:
            data, slices = InMemoryDataset.collate(data_list)
            torch.save((data, slices), str(out_path))

        print(f"  {split_name}: {len(data_list)} samples "
              f"(hiqbind: {n_hiqbind}, rdkit: {n_rdkit}, skip: {n_skip})")

    # Build test sets
    for test_set in ["casf2016", "casf2013"]:
        pdb_ids = [pid.lower() for pid in split_dict.get(test_set, [])]
        if pdb_ids:
            print(f"\nBuilding {test_set} ({len(pdb_ids)})...")
            build_split(pdb_ids, processed_dir / f"{test_set}_data.pt", test_set)

    # Build train/val folds
    for fold in [int(f) for f in args.folds.split(",")]:
        cv_path = Path(args.cv_dir) / f"PDBbind_cleansplit_train_val_split_f{fold}.json"
        if not cv_path.exists():
            continue
        with open(cv_path) as f:
            fold_split = json.load(f)
        for subset in ["train", "validation"]:
            pdb_ids = [pid.lower() for pid in fold_split.get(subset, [])]
            label = "val" if subset == "validation" else "train"
            if pdb_ids:
                print(f"\nBuilding {label}_f{fold} ({len(pdb_ids)})...")
                build_split(pdb_ids, processed_dir / f"{label}_f{fold}_data.pt", f"{label}_f{fold}")

    print(f"\nDone. Datasets at {output_dir}")


if __name__ == "__main__":
    main()
