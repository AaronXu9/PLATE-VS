"""
Precompute 3D conformers for all unique SMILES in the PLATE-VS registry.

Saves as .pkl dict: {smiles: {"z": np.ndarray, "pos": np.ndarray}}.
Uses the same ETKDGv3 + MMFF pipeline as plate_vs_dataset.py.

Usage:
    python benchmarks/07_plate_vs_dl/data/precompute_conformers.py \
        --registry training_data_full/registry_2d_split.csv \
        --output data/plate_vs_conformers/conformers.pkl \
        --threshold 0p7 \
        --num-workers 8
"""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def generate_conformer(smiles: str) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    """Generate 3D conformer for a SMILES string.

    Returns (smiles, z_array, pos_array) or (smiles, None, None) on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, None, None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol) != 0:
            return smiles, None, None

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass

    mol = Chem.RemoveHs(mol)
    if mol.GetNumConformers() == 0:
        return smiles, None, None

    conf = mol.GetConformer()
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    pos = np.array(conf.GetPositions(), dtype=np.float32)
    return smiles, z, pos


def collect_unique_smiles(registry_path: str, threshold: str | None) -> list[str]:
    """Read unique SMILES from registry CSV, optionally filtering by threshold."""
    smiles_set = set()
    with open(registry_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if threshold and row.get("similarity_threshold") != threshold:
                continue
            smi = row["smiles"]
            if smi:
                smiles_set.add(smi)
    return sorted(smiles_set)


def main():
    parser = argparse.ArgumentParser(description="Precompute ligand conformers")
    parser.add_argument("--registry", required=True, help="Path to registry CSV")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--threshold", default=None, help="Filter by similarity_threshold (e.g. 0p7). None=all.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=5000, help="Progress reporting batch size")
    args = parser.parse_args()

    # Collect unique SMILES
    print(f"Reading SMILES from {args.registry} (threshold={args.threshold})...")
    smiles_list = collect_unique_smiles(args.registry, args.threshold)
    print(f"  {len(smiles_list):,} unique SMILES")

    # Generate conformers
    print(f"Generating conformers with {args.num_workers} worker(s)...")
    t0 = time.time()
    results = {}
    n_success = 0
    n_fail = 0

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            for i, (smi, z, pos) in enumerate(pool.imap_unordered(generate_conformer, smiles_list, chunksize=200)):
                if z is not None:
                    results[smi] = {"z": z, "pos": pos}
                    n_success += 1
                else:
                    n_fail += 1
                if (i + 1) % args.batch_size == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(smiles_list) - i - 1) / rate
                    print(f"  {i+1:,}/{len(smiles_list):,} ({n_success:,} ok, {n_fail:,} fail) "
                          f"[{rate:.0f} mol/s, ETA {eta/60:.0f}min]")
    else:
        for i, smi in enumerate(smiles_list):
            smi, z, pos = generate_conformer(smi)
            if z is not None:
                results[smi] = {"z": z, "pos": pos}
                n_success += 1
            else:
                n_fail += 1
            if (i + 1) % args.batch_size == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(smiles_list) - i - 1) / rate
                print(f"  {i+1:,}/{len(smiles_list):,} ({n_success:,} ok, {n_fail:,} fail) "
                      f"[{rate:.0f} mol/s, ETA {eta/60:.0f}min]")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min — {n_success:,} succeeded, {n_fail:,} failed "
          f"({n_fail/(n_success+n_fail)*100:.1f}% failure rate)")

    # Save as pickle
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_path}...")
    with open(out_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved {size_mb:.0f} MB ({n_success:,} conformers)")


if __name__ == "__main__":
    main()
