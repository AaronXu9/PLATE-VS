"""
Prepare ligand files for GNINA docking.

For each selected target:
  - Actives: read from existing SDF file (already have 3D coords), filter to
    test-split compounds for this target and threshold.
  - Decoys: generate 3D conformers from SMILES using RDKit ETKDG (no MMFF —
    GNINA refines poses internally so initial conformer quality is secondary).
    Embedding is parallelised across CPU cores for speed.
  - Combine into a single SDF with an 'is_active' property per molecule.

Usage (from project root):
    conda run -n rdkit_env python3 benchmarks/04_docking/prepare_ligands.py \
        --config benchmarks/04_docking/configs/gnina_config.yaml
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _embed_single(args: tuple) -> bytes | None:
    """
    Worker function for multiprocessing: embed one molecule and return as
    molblock bytes (picklable). Returns None on failure.
    Args: (smiles, compound_id, seed)
    """
    smiles, compound_id, seed = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    ret = AllChem.EmbedMolecule(mol, params)
    if ret == -1:
        params.useRandomCoords = True
        ret = AllChem.EmbedMolecule(mol, params)
    if ret == -1:
        return None
    # Skip MMFF — GNINA performs its own pose refinement internally
    mol = Chem.RemoveHs(mol)
    mol.SetProp("_Name", compound_id)
    mol.SetProp("is_active", "0")
    return Chem.MolToMolBlock(mol).encode()


def embed_smiles_batch(smiles_id_pairs: list, n_workers: int = 4) -> list:
    """
    Generate 3D conformers for a list of (smiles, compound_id) pairs in parallel.
    Returns list of Chem.Mol (or None for failures), in input order.
    """
    args = [(smi, cid, 42 + i) for i, (smi, cid) in enumerate(smiles_id_pairs)]
    mols = []
    with multiprocessing.Pool(processes=n_workers) as pool:
        for molblock in pool.imap(_embed_single, args, chunksize=50):
            if molblock is None:
                mols.append(None)
            else:
                mols.append(Chem.MolFromMolBlock(molblock.decode(), removeHs=False))
    return mols


def load_active_mols_from_sdf(sdf_path: Path) -> list:
    """
    Load ALL molecules from an SDF file as actives (is_active=1).
    We don't filter by compound_id because compound_ids are often NaN in the registry.
    """
    mols = []
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        mol.SetProp("is_active", "1")
        if not (mol.HasProp("_Name") and mol.GetProp("_Name").strip()):
            mol.SetProp("_Name", f"active_{i}")
        mols.append(mol)
    return mols


def embed_actives_from_smiles(smiles_col: list, cid_col: list, n_workers: int = 4) -> list:
    """
    Generate 3D conformers for actives from SMILES column.
    smiles_col / cid_col are aligned lists from the registry.
    Returns list of Mol with is_active=1, skipping NaN SMILES.
    """
    pairs = []
    for smi, cid in zip(smiles_col, cid_col):
        smi_str = str(smi).strip()
        if smi_str.lower() == "nan" or not smi_str:
            continue
        cid_str = str(cid) if str(cid).lower() != "nan" else f"active_{len(pairs)}"
        pairs.append((smi_str, cid_str))
    if not pairs:
        return []
    mols = embed_smiles_batch(pairs, n_workers=n_workers)
    result = []
    for mol in mols:
        if mol is not None:
            mol.SetProp("is_active", "1")
            result.append(mol)
    return result


def parse_decoy_smiles(raw_smiles: str) -> str | None:
    """
    Decoy SMILES in the registry are stored as 'reference_smiles decoy_smiles'
    (DeepCoy format). Extract just the decoy (second) SMILES.
    Returns None if the string is NaN or invalid.
    """
    s = str(raw_smiles).strip()
    if s.lower() == "nan" or not s:
        return None
    parts = s.split(" ")
    # Take the last part — in case there's only one SMILES, use it directly
    return parts[-1] if parts else None


def prepare_ligands_for_target(
    target: dict,
    df_target: pd.DataFrame,
    threshold: str,
    ligand_dir: Path,
    workdir: Path,
    n_workers: int = 4,
    max_decoys: int = None,
) -> dict:
    """Prepare combined actives+decoys SDF for one target."""
    uniprot = target["uniprot_id"]
    ligand_dir.mkdir(parents=True, exist_ok=True)

    out_sdf = ligand_dir / f"{uniprot}_all_ligands.sdf"
    actives_sdf = ligand_dir / f"{uniprot}_actives.sdf"
    decoys_sdf = ligand_dir / f"{uniprot}_decoys.sdf"

    # ---- Actives ----
    test_actives = df_target[
        (df_target["split"] == "test")
        & (df_target["similarity_threshold"] == threshold)
        & (df_target["is_active"] == True)
    ]
    active_mols = []

    # Try to load from the SDF file (has 3D coords, no compound_id filtering needed)
    sdf_path_col = test_actives["sdf_path"].dropna()
    if len(sdf_path_col) > 0:
        raw_sdf = str(sdf_path_col.iloc[0])
        sdf_path = Path(raw_sdf) if Path(raw_sdf).is_absolute() else workdir / raw_sdf
        if sdf_path.exists():
            active_mols = load_active_mols_from_sdf(sdf_path)

    # If SDF loading got 0 mols, fall back to generating 3D from SMILES
    if not active_mols:
        smiles_col = test_actives["smiles"].tolist()
        cid_col = test_actives["compound_id"].tolist()
        print(f"  [{uniprot}] SDF load yielded 0 actives; generating 3D from SMILES ({len(smiles_col)} compounds)...")
        active_mols = embed_actives_from_smiles(smiles_col, cid_col, n_workers=n_workers)

    # Write actives SDF
    writer = Chem.SDWriter(str(actives_sdf))
    for mol in active_mols:
        writer.write(mol)
    writer.close()

    # ---- Decoys ----
    decoy_rows = df_target[df_target["split"] == "decoy"]
    if max_decoys is not None and len(decoy_rows) > max_decoys:
        decoy_rows = decoy_rows.sample(n=max_decoys, random_state=42)
    # Parse decoy SMILES: registry stores "reference_smiles decoy_smiles" (DeepCoy format)
    decoy_smiles_ids = []
    for idx, (_, row) in enumerate(decoy_rows.iterrows()):
        smi = parse_decoy_smiles(row["smiles"])
        if smi:
            cid = str(row["compound_id"]) if str(row["compound_id"]).lower() != "nan" else f"decoy_{idx}"
            decoy_smiles_ids.append((smi, cid))

    print(f"  [{uniprot}] Embedding {len(decoy_smiles_ids)} decoys with {n_workers} workers ...")
    raw_decoy_mols = embed_smiles_batch(decoy_smiles_ids, n_workers=n_workers)
    decoy_mols = []
    failed = len(decoy_rows) - len(decoy_smiles_ids)  # count NaN SMILES as failed
    for mol in raw_decoy_mols:
        if mol is None:
            failed += 1
        else:
            mol.SetProp("is_active", "0")
            decoy_mols.append(mol)

    # Write decoys SDF
    writer = Chem.SDWriter(str(decoys_sdf))
    for mol in decoy_mols:
        writer.write(mol)
    writer.close()

    # ---- Combine ----
    writer = Chem.SDWriter(str(out_sdf))
    for mol in active_mols:
        writer.write(mol)
    for mol in decoy_mols:
        writer.write(mol)
    writer.close()

    return {
        "n_actives": len(active_mols),
        "n_decoys": len(decoy_mols),
        "n_decoy_embed_failed": failed,
        "output_sdf": str(out_sdf),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare ligand SDF files for GNINA docking")
    parser.add_argument("--config", required=True, help="Path to gnina_config.yaml")
    parser.add_argument("--workdir", default=".", help="Project root directory")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Parallel workers for 3D embedding (default: CPU count)")
    parser.add_argument("--max-decoys", type=int, default=None,
                        help="Maximum decoys per target (random subsample for speed)")
    parser.add_argument("--targets", nargs="*",
                        help="Only process these UniProt IDs (default: all selected)")
    args = parser.parse_args()

    config = load_config(args.config)
    workdir = Path(args.workdir).resolve()
    threshold = config["data"]["similarity_threshold"]
    ligand_dir = workdir / config["paths"]["ligand_dir"]
    output_dir = workdir / config["paths"]["output_dir"]
    n_workers = args.n_workers or multiprocessing.cpu_count()

    targets_path = output_dir / "selected_targets.json"
    if not targets_path.exists():
        print(f"ERROR: {targets_path} not found. Run select_targets.py first.")
        sys.exit(1)

    with open(targets_path) as f:
        targets = json.load(f)

    registry_path = workdir / config["data"]["registry"]
    print(f"Loading registry from {registry_path} ...")
    df = pd.read_csv(registry_path, low_memory=False)

    if args.targets:
        targets = [t for t in targets if t["uniprot_id"] in args.targets]
        print(f"Filtered to {len(targets)} specified targets")

    print(f"Preparing ligands for {len(targets)} targets (n_workers={n_workers}, max_decoys={args.max_decoys}) ...\n")

    # Load existing results to avoid re-processing completed targets
    results_path = output_dir / "ligand_prep_results.json"
    results = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    for target in targets:
        uniprot = target["uniprot_id"]
        df_target = df[df["uniprot_id"] == uniprot]

        n_actives_expected = target["n_actives"]
        n_decoys_expected = target["n_decoys"]
        print(
            f"[{uniprot}] expected: {n_actives_expected} actives, "
            f"{n_decoys_expected} decoys"
            + (f" (capping at {args.max_decoys})" if args.max_decoys else "")
        )

        result = prepare_ligands_for_target(
            target, df_target, threshold, ligand_dir, workdir,
            n_workers=n_workers, max_decoys=args.max_decoys,
        )
        results[uniprot] = result

        print(
            f"  [{uniprot}] actives={result['n_actives']}  "
            f"decoys={result['n_decoys']}  "
            f"embed_failed={result['n_decoy_embed_failed']}"
        )
        if result["n_decoy_embed_failed"] > 0:
            fail_pct = 100 * result["n_decoy_embed_failed"] / max(1, n_decoys_expected)
            print(f"  [{uniprot}] Warning: {fail_pct:.1f}% of decoys failed 3D embedding")
        print()

    prep_path = output_dir / "ligand_prep_results.json"
    with open(prep_path, "w") as f:
        json.dump(results, f, indent=2)

    total_actives = sum(r["n_actives"] for r in results.values())
    total_decoys = sum(r["n_decoys"] for r in results.values())
    print(f"Ligand preparation complete:")
    print(f"  Total actives: {total_actives}")
    print(f"  Total decoys:  {total_decoys}")
    print(f"Results saved to {prep_path}")


if __name__ == "__main__":
    main()
