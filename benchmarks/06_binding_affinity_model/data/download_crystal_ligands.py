"""
Download crystal-pose ligand SDF files from RCSB PDB.

For each PDBbind complex, downloads the mmCIF file and extracts the
bound ligand with its crystal 3D coordinates as an SDF file.

This provides experimentally-determined binding conformations instead
of RDKit-generated approximate conformers.

Usage:
    python benchmarks/06_binding_affinity_model/data/download_crystal_ligands.py \
        --output-dir data/pdbbind_cleansplit/crystal_ligands \
        --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

RCSB_LIGAND_URL = (
    "https://models.rcsb.org/v1/{pdb_id}/ligand"
    "?auth_comp_id={het_code}&encoding=sdf&copy_all_categories=false"
)

RCSB_IDEAL_URL = (
    "https://files.rcsb.org/ligands/download/{het_code}_ideal.sdf"
)


def download_ligand_sdf(pdb_id: str, het_code: str, output_dir: Path,
                         retries: int = 2) -> dict:
    """Download bound ligand SDF from RCSB for a specific PDB entry.

    Tries the model API first (gives crystal coordinates for the specific
    PDB entry), falls back to ideal coordinates if that fails.

    Returns dict with status and path info.
    """
    out_path = output_dir / f"{pdb_id}_ligand.sdf"
    if out_path.exists():
        # Validate existing file
        mol = next(Chem.SDMolSupplier(str(out_path), removeHs=True), None)
        if mol is not None and mol.GetNumConformers() > 0:
            return {"status": "exists", "path": str(out_path), "pdb_id": pdb_id}

    # Try model API (crystal coordinates)
    url = RCSB_LIGAND_URL.format(pdb_id=pdb_id.upper(), het_code=het_code.upper())

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                sdf_content = resp.read().decode("utf-8")

            # Validate SDF content
            if "$$$$" not in sdf_content or len(sdf_content) < 50:
                break  # Empty/invalid response

            # Check if RDKit can parse it and it has 3D coords
            supplier = Chem.SDMolSupplier()
            supplier.SetData(sdf_content)
            mol = next(supplier, None)
            if mol is None or mol.GetNumConformers() == 0:
                break

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            # Check it's not all zeros (sometimes happens)
            if abs(pos).max() < 0.01:
                break

            with open(out_path, "w") as f:
                f.write(sdf_content)

            return {
                "status": "ok",
                "path": str(out_path),
                "pdb_id": pdb_id,
                "n_atoms": mol.GetNumAtoms(),
                "source": "crystal",
            }

        except urllib.error.HTTPError as e:
            if e.code == 404:
                break  # Not found, try fallback
            if attempt < retries:
                time.sleep(0.5)
        except Exception:
            if attempt < retries:
                time.sleep(0.5)

    # Fallback: ideal coordinates (not crystal, but better than RDKit)
    url_ideal = RCSB_IDEAL_URL.format(het_code=het_code.upper())
    try:
        req = urllib.request.Request(url_ideal)
        with urllib.request.urlopen(req, timeout=10) as resp:
            sdf_content = resp.read().decode("utf-8")

        if "$$$$" in sdf_content:
            supplier = Chem.SDMolSupplier()
            supplier.SetData(sdf_content)
            mol = next(supplier, None)
            if mol is not None and mol.GetNumConformers() > 0:
                with open(out_path, "w") as f:
                    f.write(sdf_content)
                return {
                    "status": "ok",
                    "path": str(out_path),
                    "pdb_id": pdb_id,
                    "n_atoms": mol.GetNumAtoms(),
                    "source": "ideal",
                }
    except Exception:
        pass

    return {"status": "failed", "pdb_id": pdb_id}


def main():
    parser = argparse.ArgumentParser(
        description="Download crystal-pose ligand SDFs from RCSB PDB"
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
        "--output-dir",
        default="data/pdbbind_cleansplit/crystal_ligands",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-downloaded files",
    )
    args = parser.parse_args()

    with open(args.labels) as f:
        data_dict = json.load(f)
    with open(args.splits) as f:
        split_dict = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all PDB IDs from CleanSplit
    all_ids = set()
    for pdb_ids in split_dict.values():
        all_ids.update(pid.lower() for pid in pdb_ids)

    # Build download tasks
    tasks = []
    for pid in sorted(all_ids):
        entry = data_dict.get(pid) or data_dict.get(pid.upper())
        if not entry:
            continue
        het_code = entry.get("ligand_name", "")
        if not het_code or "&" in het_code:
            continue
        tasks.append((pid, het_code))

    print(f"Total complexes: {len(tasks)}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.workers}")

    n_ok = 0
    n_crystal = 0
    n_ideal = 0
    n_failed = 0
    n_skipped = 0
    t0 = time.time()

    # Status file for tracking
    status_path = output_dir / "download_status.json"
    status = {}
    if args.resume and status_path.exists():
        with open(status_path) as f:
            status = json.load(f)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for pid, het_code in tasks:
            if args.resume and pid in status and status[pid].get("status") == "ok":
                n_skipped += 1
                continue
            futures[executor.submit(download_ligand_sdf, pid, het_code, output_dir)] = pid

        for future in as_completed(futures):
            pid = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {"status": "error", "pdb_id": pid, "error": str(e)}

            status[pid] = result
            if result["status"] == "ok":
                n_ok += 1
                if result.get("source") == "crystal":
                    n_crystal += 1
                else:
                    n_ideal += 1
            elif result["status"] == "exists":
                n_ok += 1
            else:
                n_failed += 1

            total = n_ok + n_failed + n_skipped
            if total % 500 == 0:
                elapsed = time.time() - t0
                rate = (n_ok + n_failed) / max(elapsed, 1)
                print(
                    f"  Progress: {total}/{len(tasks)} | "
                    f"OK: {n_ok} (crystal: {n_crystal}, ideal: {n_ideal}) | "
                    f"Failed: {n_failed} | {rate:.0f}/s"
                )
                with open(status_path, "w") as f:
                    json.dump(status, f)

    # Save final status
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Downloaded: {n_ok} (crystal: {n_crystal}, ideal: {n_ideal})")
    print(f"  Skipped: {n_skipped}")
    print(f"  Failed: {n_failed}")
    print(f"  Status: {status_path}")

    # Report per-split coverage
    for split_name, pdb_ids in split_dict.items():
        ids = [pid.lower() for pid in pdb_ids]
        n_have = sum(1 for pid in ids if status.get(pid, {}).get("status") in ("ok", "exists"))
        print(f"  {split_name}: {n_have}/{len(ids)}")


if __name__ == "__main__":
    main()
