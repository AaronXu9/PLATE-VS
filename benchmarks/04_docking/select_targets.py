"""
Select a representative subset of test targets for GNINA docking benchmark.

Reads registry_2d_split.csv and selects n_targets diverse protein targets from
the test partition that have at least min_actives_per_target test actives at the
configured similarity threshold.

Usage (from project root):
    conda run -n rdkit_env python3 benchmarks/04_docking/select_targets.py \
        --config benchmarks/04_docking/configs/gnina_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_cif_path(cif_path_str: str, registry_dir: Path, workdir: Path) -> Path:
    """
    Resolve a cif_path from the registry.
    The registry stores paths relative to its own directory (e.g. ../../plate-vs/...).
    Fall back to treating the path as relative to the project root if not found.
    """
    # Try resolving from registry directory
    p = (registry_dir / cif_path_str).resolve()
    if p.exists():
        return p
    # Strip leading ../../ and try from project root
    stripped = cif_path_str.lstrip("./").lstrip("../")
    # Remove all leading ../ components
    parts = Path(cif_path_str).parts
    # Drop leading '..' parts
    clean_parts = []
    skipping = True
    for part in parts:
        if skipping and part in ("..", "."):
            continue
        skipping = False
        clean_parts.append(part)
    p2 = (workdir / Path(*clean_parts)).resolve() if clean_parts else workdir
    if p2.exists():
        return p2
    return p  # return original (non-existent) path so caller can warn


def select_targets(config: dict, workdir: str = ".") -> list[dict]:
    workdir = Path(workdir)
    registry_path = workdir / config["data"]["registry"]
    protein_refs_path = workdir / config["data"]["protein_references"]
    threshold = config["data"]["similarity_threshold"]
    n_targets = config["data"]["n_targets"]
    min_actives = config["data"]["min_actives_per_target"]

    print(f"Loading registry from {registry_path} ...")
    df = pd.read_csv(registry_path, low_memory=False)
    registry_dir = registry_path.parent

    # Test actives: protein_partition=test, split=test, correct threshold, is_active
    test_actives = df[
        (df["protein_partition"] == "test")
        & (df["split"] == "test")
        & (df["similarity_threshold"] == threshold)
        & (df["is_active"] == True)
    ]
    print(f"Found {len(test_actives)} test actives across {test_actives['uniprot_id'].nunique()} proteins")

    # Count actives per target
    active_counts = test_actives.groupby("uniprot_id").size().rename("n_actives")

    # Filter by minimum actives
    valid_targets = active_counts[active_counts >= min_actives].index.tolist()
    print(f"{len(valid_targets)} targets have >= {min_actives} test actives")

    # Load protein references for quality info
    with open(protein_refs_path) as f:
        protein_refs = json.load(f)

    # Build target metadata
    target_meta = []
    for uniprot in valid_targets:
        ref = protein_refs.get(uniprot, {})
        # Get cif_path from registry (take first row for this target)
        row = df[df["uniprot_id"] == uniprot].iloc[0]
        n_actives = int(active_counts[uniprot])

        # Count decoys for this target
        n_decoys = int(df[(df["uniprot_id"] == uniprot) & (df["split"] == "decoy")]["sample_id"].count())

        raw_cif = str(row.get("cif_path", ""))
        resolved_cif = resolve_cif_path(raw_cif, registry_dir, workdir)

        target_meta.append(
            {
                "uniprot_id": uniprot,
                "pdb_id": row.get("pdb_id", ref.get("pdb_id", "")),
                "cif_path": str(resolved_cif),
                "resolution": float(row.get("resolution", 99.0) or 99.0),
                "quality_score": float(row.get("quality_score", 0.0) or 0.0),
                "chosen_ligand": ref.get("chosen_ligand", ""),
                "n_actives": n_actives,
                "n_decoys": n_decoys,
            }
        )

    if not target_meta:
        print("ERROR: No valid targets found. Check registry and config settings.")
        sys.exit(1)

    # Sort by resolution (lower = better), then quality_score descending
    target_meta.sort(key=lambda x: (x["resolution"], -x["quality_score"]))

    # Diverse selection: bin by active count (log-scale) and sample uniformly
    if len(target_meta) <= n_targets:
        selected = target_meta
    else:
        counts = np.array([t["n_actives"] for t in target_meta])
        log_counts = np.log1p(counts)
        bins = np.linspace(log_counts.min(), log_counts.max(), n_targets + 1)
        selected = []
        rng = np.random.default_rng(42)
        for i in range(n_targets):
            lo, hi = bins[i], bins[i + 1]
            candidates = [
                t for t, lc in zip(target_meta, log_counts) if lo <= lc <= hi
            ]
            if not candidates:
                # Fall back to any not yet selected
                chosen_ids = {t["uniprot_id"] for t in selected}
                candidates = [t for t in target_meta if t["uniprot_id"] not in chosen_ids]
            if candidates:
                # Prefer best resolution within the bin
                candidates.sort(key=lambda x: (x["resolution"], -x["quality_score"]))
                selected.append(candidates[0])

        # Deduplicate (bins can overlap at edges)
        seen = set()
        deduped = []
        for t in selected:
            if t["uniprot_id"] not in seen:
                seen.add(t["uniprot_id"])
                deduped.append(t)
        selected = deduped[:n_targets]

    # Skip targets with no chosen_ligand (can't define pocket)
    before = len(selected)
    selected = [t for t in selected if t["chosen_ligand"]]
    if len(selected) < before:
        print(f"Warning: dropped {before - len(selected)} targets with no chosen_ligand")

    # Check CIF files actually exist
    checked = []
    for t in selected:
        cif = workdir / t["cif_path"]
        if not cif.exists():
            print(f"Warning: CIF not found for {t['uniprot_id']}: {cif}")
        else:
            checked.append(t)
    selected = checked

    print(f"\nSelected {len(selected)} targets:")
    for t in selected:
        print(
            f"  {t['uniprot_id']:12s}  PDB={t['pdb_id']}  res={t['resolution']:.1f}Å  "
            f"actives={t['n_actives']:4d}  decoys={t['n_decoys']:6d}  ligand={t['chosen_ligand']}"
        )

    return selected


def main():
    parser = argparse.ArgumentParser(description="Select representative GNINA benchmark targets")
    parser.add_argument("--config", required=True, help="Path to gnina_config.yaml")
    parser.add_argument("--workdir", default=".", help="Project root directory")
    args = parser.parse_args()

    config = load_config(args.config)
    workdir = Path(args.workdir)

    selected = select_targets(config, workdir)

    output_dir = workdir / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "selected_targets.json"

    with open(output_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
