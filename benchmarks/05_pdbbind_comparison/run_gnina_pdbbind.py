"""
Run GNINA redocking on PDBbind CleanSplit test complexes.

For each CASF-2016/2013 complex, extracts the protein and ligand from
the raw PDBbind files, re-docks the ligand with GNINA, and extracts
the CNNaffinity score as a pK prediction.

Prerequisites:
  - Raw PDBbind data in data/pdbbind_cleansplit/raw/<pdb_id>/
  - GNINA binary
  - conda env 'rdkit_env'

Usage:
    conda run -n rdkit_env python benchmarks/05_pdbbind_comparison/run_gnina_pdbbind.py \
        --config benchmarks/05_pdbbind_comparison/configs/gnina_pdbbind_config.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_complex_files(raw_dir: Path, pdb_id: str) -> dict | None:
    """Find protein and ligand files for a PDBbind complex."""
    pdb_dir = raw_dir / pdb_id
    if not pdb_dir.exists():
        pdb_dir = raw_dir / pdb_id.upper()
    if not pdb_dir.exists():
        return None

    protein_file = None
    ligand_file = None

    # Protein: {pdb_id}_protein.pdb
    for suffix in [".pdb", ".pdbqt"]:
        candidate = pdb_dir / f"{pdb_id}_protein{suffix}"
        if candidate.exists():
            protein_file = candidate
            break

    # Ligand: {pdb_id}_ligand.sdf or .mol2
    for suffix in [".sdf", ".mol2"]:
        candidate = pdb_dir / f"{pdb_id}_ligand{suffix}"
        if candidate.exists():
            ligand_file = candidate
            break

    if protein_file and ligand_file:
        return {"protein": protein_file, "ligand": ligand_file}
    return None


def prepare_receptor_pdbqt(protein_pdb: Path, output_pdbqt: Path) -> bool:
    """Convert protein PDB to PDBQT using obabel."""
    if output_pdbqt.exists():
        return True
    try:
        subprocess.run(
            ["obabel", str(protein_pdb), "-O", str(output_pdbqt), "-xr"],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  [ERROR] obabel conversion failed: {exc}")
        return False


def dock_complex(
    pdb_id: str,
    receptor_pdbqt: Path,
    ligand_file: Path,
    ref_ligand: Path,
    out_sdf: Path,
    gnina_config: dict,
) -> dict:
    """Run GNINA redocking for one complex."""
    cmd = [
        gnina_config["binary"],
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_file),
        "--autobox_ligand", str(ref_ligand),
        "--autobox_add", str(gnina_config["autobox_add"]),
        "--num_modes", str(gnina_config["num_modes"]),
        "--exhaustiveness", str(gnina_config["exhaustiveness"]),
        "--cnn_scoring", str(gnina_config["cnn_scoring"]),
        "--out", str(out_sdf),
        "--cpu", str(gnina_config["cpu"]),
        "--device", str(gnina_config["device"]),
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            return {
                "status": "error",
                "reason": f"GNINA exited with code {result.returncode}",
                "elapsed_s": elapsed,
            }
        return {"status": "ok", "elapsed_s": elapsed}

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed_s": 3600}
    except FileNotFoundError:
        return {"status": "error", "reason": "GNINA binary not found"}


def main():
    parser = argparse.ArgumentParser(
        description="Run GNINA redocking on PDBbind CleanSplit test complexes"
    )
    parser.add_argument("--config", required=True, help="Path to gnina_pdbbind_config.yaml")
    parser.add_argument("--test-set", type=str, default=None, help="Override test set")
    args = parser.parse_args()

    config = load_config(args.config)
    config_dir = Path(args.config).resolve().parent

    raw_dir = (config_dir / config["data"]["raw_dir"]).resolve()
    split_path = (config_dir / config["data"]["split_path"]).resolve()
    labels_path = (config_dir / config["data"]["labels_path"]).resolve()
    test_set = args.test_set or config["data"]["test_set"]

    receptor_dir = (config_dir / config["paths"]["receptor_dir"]).resolve()
    docking_dir = (config_dir / config["paths"]["docking_dir"]).resolve()
    output_dir = (config_dir / config["paths"]["output_dir"]).resolve()

    receptor_dir.mkdir(parents=True, exist_ok=True)
    docking_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"ERROR: Raw data directory not found: {raw_dir}")
        print("Download PDBbind v.2020 from http://www.pdbbind.org.cn/")
        sys.exit(1)

    with open(split_path) as f:
        split_dict = json.load(f)
    with open(labels_path) as f:
        labels_dict = json.load(f)

    test_pdb_ids = [pid.lower() for pid in split_dict.get(test_set, [])]
    print(f"Test set: {test_set} ({len(test_pdb_ids)} complexes)")
    print(f"Raw data: {raw_dir}")

    gnina_config = config["gnina"]
    docking_results = {}

    # Load existing results to allow resuming
    results_json = output_dir / "docking_results.json"
    if results_json.exists():
        with open(results_json) as f:
            docking_results = json.load(f)

    n_docked = 0
    n_skipped = 0
    n_failed = 0

    for i, pdb_id in enumerate(test_pdb_ids):
        prefix = f"[{i+1}/{len(test_pdb_ids)} {pdb_id}]"

        # Skip if already docked
        if pdb_id in docking_results and docking_results[pdb_id].get("status") == "ok":
            n_skipped += 1
            continue

        # Find complex files
        files = find_complex_files(raw_dir, pdb_id)
        if files is None:
            print(f"{prefix} Missing raw files, skipping")
            docking_results[pdb_id] = {"status": "missing_files"}
            n_failed += 1
            continue

        # Prepare receptor PDBQT
        receptor_pdbqt = receptor_dir / f"{pdb_id}_receptor.pdbqt"
        if not prepare_receptor_pdbqt(files["protein"], receptor_pdbqt):
            docking_results[pdb_id] = {"status": "prep_failed"}
            n_failed += 1
            continue

        # Dock
        out_sdf = docking_dir / f"{pdb_id}_docked.sdf"
        print(f"{prefix} Docking...", end=" ", flush=True)

        result = dock_complex(
            pdb_id=pdb_id,
            receptor_pdbqt=receptor_pdbqt,
            ligand_file=files["ligand"],
            ref_ligand=files["ligand"],  # Self-dock: reference is the native pose
            out_sdf=out_sdf,
            gnina_config=gnina_config,
        )

        docking_results[pdb_id] = result
        if result["status"] == "ok":
            n_docked += 1
            print(f"OK ({result['elapsed_s']:.1f}s)")
        else:
            n_failed += 1
            print(f"FAILED: {result.get('reason', result['status'])}")

        # Save intermediate results
        if (i + 1) % 10 == 0:
            with open(results_json, "w") as f:
                json.dump(docking_results, f, indent=2)

    # Save final results
    with open(results_json, "w") as f:
        json.dump(docking_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Docking complete: {n_docked} docked, {n_skipped} skipped, {n_failed} failed")
    print(f"  Results: {results_json}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
