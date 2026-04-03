"""
Extract SMILES strings from raw PDBbind files for classical ML training.

Reads ligand mol2/sdf files from the raw PDBbind directory, converts to
canonical SMILES, and joins with pK labels and split assignments from
the GEMS CleanSplit data.

Output: CSV with columns (pdb_id, smiles, pK, split)

Usage:
    python extract_pdbbind_smiles.py \
        --raw-dir data/pdbbind_cleansplit/raw \
        --labels data/pdbbind_cleansplit/labels/PDBbind_data_dict.json \
        --splits data/pdbbind_cleansplit/labels/PDBbind_data_split_cleansplit.json \
        --output data/pdbbind_cleansplit/smiles/pdbbind_smiles.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

# Suppress RDKit warnings during batch conversion
RDLogger.DisableLog("rdApp.*")


def find_ligand_file(pdb_dir: Path) -> Path | None:
    """Find the ligand file in a PDBbind complex directory.

    PDBbind uses various naming conventions:
      - {pdb_id}_ligand.sdf
      - {pdb_id}_ligand.mol2
    """
    pdb_id = pdb_dir.name
    for suffix in [".sdf", ".mol2"]:
        candidate = pdb_dir / f"{pdb_id}_ligand{suffix}"
        if candidate.exists():
            return candidate
    # Fallback: any .sdf or .mol2 file
    for suffix in ["*.sdf", "*.mol2"]:
        matches = list(pdb_dir.glob(suffix))
        if matches:
            return matches[0]
    return None


def ligand_to_smiles(ligand_path: Path) -> str | None:
    """Convert a ligand file (SDF or mol2) to canonical SMILES."""
    suffix = ligand_path.suffix.lower()

    mol = None
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(ligand_path), removeHs=True)
        for m in supplier:
            if m is not None:
                mol = m
                break
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(ligand_path), removeHs=True)
    else:
        return None

    if mol is None:
        return None

    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def build_split_lookup(split_dict: dict) -> dict[str, str]:
    """Build PDB ID → split name mapping.

    CleanSplit JSON has keys like: train, casf2016, casf2013, casf2016_indep, casf2013_indep
    We map to simpler names for the CSV.
    """
    lookup = {}
    for split_name, pdb_ids in split_dict.items():
        for pdb_id in pdb_ids:
            lookup[pdb_id.lower()] = split_name
    return lookup


def main():
    parser = argparse.ArgumentParser(
        description="Extract SMILES from raw PDBbind ligand files"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/pdbbind_cleansplit/raw",
        help="Directory containing PDBbind complex subdirectories",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/pdbbind_cleansplit/labels/PDBbind_data_dict.json",
        help="Path to PDBbind_data_dict.json",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="data/pdbbind_cleansplit/labels/PDBbind_data_split_cleansplit.json",
        help="Path to PDBbind_data_split_cleansplit.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/pdbbind_cleansplit/smiles/pdbbind_smiles.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    if not raw_dir.exists():
        print(f"ERROR: Raw data directory not found: {raw_dir}")
        print("Download PDBbind v.2020 from http://www.pdbbind.org.cn/")
        sys.exit(1)

    with open(args.labels) as f:
        data_dict = json.load(f)
    with open(args.splits) as f:
        split_dict = json.load(f)

    split_lookup = build_split_lookup(split_dict)

    # Collect all PDB IDs from splits
    all_split_ids = set()
    for pdb_ids in split_dict.values():
        all_split_ids.update(pid.lower() for pid in pdb_ids)

    print(f"Total PDB IDs in CleanSplit: {len(all_split_ids)}")
    print(f"Scanning raw directory: {raw_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_success = 0
    n_missing_file = 0
    n_failed_smiles = 0
    n_no_label = 0

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["pdb_id", "smiles", "pK", "split"])

        for pdb_id_lower in sorted(all_split_ids):
            # Check if we have a label
            # data_dict keys may be lowercase 4-char PDB codes
            label_entry = data_dict.get(pdb_id_lower) or data_dict.get(
                pdb_id_lower.upper()
            )
            if label_entry is None:
                n_no_label += 1
                continue

            pk_value = label_entry.get("log_kd_ki")
            if pk_value is None:
                n_no_label += 1
                continue

            # Find raw directory
            pdb_dir = raw_dir / pdb_id_lower
            if not pdb_dir.exists():
                pdb_dir = raw_dir / pdb_id_lower.upper()
            if not pdb_dir.exists():
                n_missing_file += 1
                continue

            ligand_path = find_ligand_file(pdb_dir)
            if ligand_path is None:
                n_missing_file += 1
                continue

            smiles = ligand_to_smiles(ligand_path)
            if smiles is None:
                n_failed_smiles += 1
                continue

            split_name = split_lookup.get(pdb_id_lower, "unknown")
            writer.writerow([pdb_id_lower, smiles, pk_value, split_name])
            n_success += 1

    print(f"\nResults:")
    print(f"  Extracted SMILES: {n_success}")
    print(f"  Missing ligand files: {n_missing_file}")
    print(f"  Failed SMILES conversion: {n_failed_smiles}")
    print(f"  No pK label: {n_no_label}")
    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    main()
