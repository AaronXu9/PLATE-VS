"""Split combined ligand SDFs into per-ligand SDFs + batch.txt for UniDock2 -lb.

For each selected target, reads the gnina-prepared `{uniprot}_all_ligands.sdf`
multi-record SDF, writes one single-record SDF per ligand into
`<unidock2_inputs_dir>/<uniprot>/ligands/`, and writes a `batch.txt` listing
those paths (one per line) for UniDock2's `-lb` flag.

Also derives the docking box from `{uniprot}_ref_ligand.sdf` (centroid plus
bounding-box plus 2x box_padding_per_side, floored at box_size_min) and writes
`box.json`. An `index.json` maps ligand title -> {is_active, path} so the
collector can rejoin scores to labels.

Usage:
    /home/aoxu/miniconda3/envs/rdkit_env/bin/python \\
        benchmarks/04_docking/build_unidock2_batches.py \\
        --config benchmarks/04_docking/configs/unidock2_config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol | None:
    """Return the largest connected fragment (by heavy-atom count).

    UniDock2's antechamber/GAFF pipeline crashes on multi-component ligands
    (e.g. quaternary-ammonium SMILES paired with a [Cl-]/[I-] counter-ion in
    PLATE-VS). We strip salts here before writing per-ligand SDFs.
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    return max(frags, key=lambda f: f.GetNumHeavyAtoms())


def derive_box(
    ref_ligand_sdf: Path, padding: float, size_min: float
) -> tuple[float, float, float, float, float, float]:
    # sanitize=False: some co-crystal ligands (e.g. tetra-valent ammonium N
    # without an explicit charge in the SDF) trip strict RDKit sanitization,
    # but we only need atom coordinates for the box.
    suppl = Chem.SDMolSupplier(str(ref_ligand_sdf), removeHs=False, sanitize=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError(f"Cannot read ref ligand from {ref_ligand_sdf}")
    conf = mol.GetConformer()
    xyz = np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ]
    )
    cmin, cmax = xyz.min(0), xyz.max(0)
    centre = (cmin + cmax) / 2.0
    bbox = (cmax - cmin) + 2 * padding
    size = np.maximum(bbox, size_min)
    return (
        float(centre[0]),
        float(centre[1]),
        float(centre[2]),
        float(size[0]),
        float(size[1]),
        float(size[2]),
    )


def build_target_inputs(
    uniprot: str,
    all_ligands_sdf: Path,
    ref_ligand_sdf: Path,
    out_dir: Path,
    padding: float,
    size_min: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ligands_dir = out_dir / "ligands"
    ligands_dir.mkdir(exist_ok=True)

    # sanitize=True so RDKit drops malformed molecules before they reach
    # UniDock2's antechamber/GAFF pipeline, which crashes the whole batch on
    # single-atom ions or weird valences.
    suppl = Chem.SDMolSupplier(str(all_ligands_sdf), removeHs=False, sanitize=True)
    index: dict[str, dict] = {}
    batch_lines: list[str] = []
    n_skipped = 0
    n_dropped_small = 0
    n_dropped_unsanitized = 0
    n_desalted = 0
    for i, mol in enumerate(suppl):
        if mol is None:
            n_dropped_unsanitized += 1
            n_skipped += 1
            continue
        # Strip counter-ions / co-formers — antechamber crashes on
        # multi-component ligands.
        n_frags = len(Chem.GetMolFrags(mol))
        if n_frags > 1:
            mol = _largest_fragment(mol)
            if mol is None:
                n_skipped += 1
                continue
            n_desalted += 1
        # Drop ions / single atoms / diatomics — antechamber crashes on these.
        if mol.GetNumHeavyAtoms() < 5:
            n_dropped_small += 1
            n_skipped += 1
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"{uniprot}_lig_{i}"
        safe = name.replace("/", "_").replace(" ", "_")
        if safe in index:
            safe = f"{safe}_{i}"
        is_active = int(mol.GetProp("is_active")) if mol.HasProp("is_active") else 0
        out_sdf = ligands_dir / f"{safe}.sdf"
        try:
            w = Chem.SDWriter(str(out_sdf))
            w.write(mol)
            w.close()
        except Exception:
            n_skipped += 1
            continue
        index[safe] = {"is_active": is_active, "path": str(out_sdf), "title": name}
        batch_lines.append(str(out_sdf))

    batch_path = out_dir / "batch.txt"
    batch_path.write_text("\n".join(batch_lines) + "\n")

    cx, cy, cz, sx, sy, sz = derive_box(ref_ligand_sdf, padding=padding, size_min=size_min)
    box = {"center": [cx, cy, cz], "size": [sx, sy, sz]}
    (out_dir / "box.json").write_text(json.dumps(box, indent=2))
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))

    return {
        "uniprot": uniprot,
        "n_ligands": len(batch_lines),
        "n_skipped": n_skipped,
        "n_dropped_small": n_dropped_small,
        "n_dropped_unsanitized": n_dropped_unsanitized,
        "n_desalted": n_desalted,
        "batch_path": str(batch_path),
        "box": box,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--targets", nargs="*", help="Subset of UniProt IDs to prepare")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    selected = json.loads((paths["output_dir"] / "selected_targets.json").read_text())

    pad = cfg["unidock2"]["box_padding_per_side"]
    smin = cfg["unidock2"]["box_size_min"]

    summary = {}
    for tgt in selected:
        u = tgt["uniprot_id"]
        if args.targets and u not in args.targets:
            continue
        all_lig = paths["ligand_dir"] / f"{u}_all_ligands.sdf"
        ref = paths["receptor_dir"] / f"{u}_ref_ligand.sdf"
        out_dir = paths["unidock2_inputs_dir"] / u
        if not all_lig.exists() or not ref.exists():
            summary[u] = {"status": "missing_inputs"}
            continue
        info = build_target_inputs(u, all_lig, ref, out_dir, padding=pad, size_min=smin)
        info["status"] = "ok"
        summary[u] = info
        print(
            f"[{u}] {info['n_ligands']} ligands, "
            f"box centre={info['box']['center']}, "
            f"size={info['box']['size']}, "
            f"skipped={info['n_skipped']}"
        )

    out_json = paths["output_dir"] / "unidock2_targets.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
