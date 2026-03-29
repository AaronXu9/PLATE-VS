"""
Prepare protein receptor structures for GNINA docking.

For each selected target:
  1. Convert CIF → PDB (via obabel)
  2. Extract the co-crystal reference ligand as a separate SDF (for autobox_ligand)
  3. Strip the co-crystal ligand and waters from the receptor
  4. Convert the cleaned receptor to PDBQT (via obabel, rigid receptor mode)

Usage (from project root):
    conda run -n rdkit_env python3 benchmarks/04_docking/prepare_structures.py \
        --config benchmarks/04_docking/configs/gnina_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_chosen_ligand(chosen_ligand_str: str):
    """
    Parse chosen_ligand field like 'ANP@A:500(heavy=31)'.
    Returns (res_name, chain, resid) or None on failure.
    """
    m = re.match(r"([^@]+)@([^:]+):(\d+)", chosen_ligand_str)
    if m:
        return m.group(1).strip(), m.group(2).strip(), int(m.group(3))
    return None


def run_obabel(args: list[str]) -> tuple[int, str, str]:
    """Run obabel and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["obabel"] + args,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def cif_to_pdb(cif_path: Path, out_pdb: Path) -> bool:
    rc, _, err = run_obabel([str(cif_path), "-O", str(out_pdb)])
    if rc != 0 or not out_pdb.exists():
        print(f"  ERROR: obabel CIF→PDB failed: {err.strip()}")
        return False
    return True


def read_pdb_lines(pdb_path: Path) -> list[str]:
    with open(pdb_path) as f:
        return f.readlines()


def get_residue_name(line: str) -> str | None:
    """Return the 3-letter residue name from a PDB ATOM/HETATM line, or None."""
    if len(line) < 20:
        return None
    record = line[:6].strip()
    if record not in ("HETATM", "ATOM"):
        return None
    return line[17:20].strip()


def extract_ref_ligand_lines(pdb_lines: list[str], res_name: str) -> list[str]:
    """
    Extract all ATOM/HETATM lines whose residue name matches res_name.
    obabel may change chain IDs and residue numbers, so we match by name only.
    """
    return [line for line in pdb_lines if get_residue_name(line) == res_name]


def strip_residue_and_waters(pdb_lines: list[str], res_name: str) -> list[str]:
    """
    Remove all ATOM/HETATM lines for res_name and all HOH water records.
    Keeps protein ATOM records and all non-coordinate lines.
    """
    # Standard amino acid 3-letter codes — never strip these
    AA_NAMES = {
        "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
        "HID","HIE","HIP","CYX","MSE","SEC","PYL",
    }
    clean = []
    for line in pdb_lines:
        rname = get_residue_name(line)
        if rname is None:
            clean.append(line)
            continue
        # Remove co-crystal ligand (by name, skip if it's an amino acid)
        if rname == res_name and rname not in AA_NAMES:
            continue
        # Remove waters
        if rname in ("HOH", "WAT", "H2O"):
            continue
        clean.append(line)
    return clean


def pdb_to_sdf(pdb_path: Path, sdf_path: Path) -> bool:
    rc, _, err = run_obabel([str(pdb_path), "-O", str(sdf_path)])
    if rc != 0 or not sdf_path.exists():
        print(f"  ERROR: obabel PDB→SDF failed: {err.strip()}")
        return False
    return True


def strip_conect_records(pdb_path: Path) -> Path:
    """
    Strip CONECT records from PDB to avoid obabel conversion failures.
    Returns path to a cleaned temp file (same dir, _nc suffix).
    """
    nc_path = pdb_path.parent / (pdb_path.stem + "_nc.pdb")
    with open(pdb_path) as fin, open(nc_path, "w") as fout:
        for line in fin:
            if not line.startswith("CONECT"):
                fout.write(line)
    return nc_path


def strip_invalid_element_atoms(pdb_path: Path) -> Path:
    """
    Remove ATOM/HETATM records whose element field (cols 77-78) is '*'.
    obabel sometimes writes '*' for atoms it can't classify; GNINA crashes on these.
    Returns path to a fixed temp file (same dir, _fixed suffix).
    """
    fixed_path = pdb_path.parent / (pdb_path.stem + "_fixed.pdb")
    with open(pdb_path) as fin, open(fixed_path, "w") as fout:
        for line in fin:
            record = line[:6].strip()
            if record in ("ATOM", "HETATM"):
                elem = line[76:78].strip() if len(line) >= 78 else ""
                if elem == "*":
                    continue
            fout.write(line)
    return fixed_path


def pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> bool:
    """Convert receptor PDB to PDBQT (rigid, Gasteiger charges)."""
    # Strip CONECT records — obabel fails silently on bad CONECT lines
    nc_pdb = strip_conect_records(pdb_path)
    rc, _, err = run_obabel(
        [str(nc_pdb), "-O", str(pdbqt_path), "-xr", "--partialcharge", "gasteiger"]
    )
    # Clean up temp file
    nc_pdb.unlink(missing_ok=True)
    if rc != 0 or not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
        print(f"  ERROR: obabel PDB→PDBQT failed (rc={rc}): {err.strip()[:200]}")
        return False
    return True


def prepare_receptor(target: dict, receptor_dir: Path, workdir: Path) -> dict:
    """
    Full receptor preparation pipeline for one target.
    Returns dict with status and output paths.
    """
    uniprot = target["uniprot_id"]
    # cif_path stored in selected_targets.json is already resolved to absolute path
    cif_path = Path(target["cif_path"])
    if not cif_path.is_absolute():
        cif_path = workdir / cif_path
    chosen_ligand = target["chosen_ligand"]

    parsed = parse_chosen_ligand(chosen_ligand)
    if parsed is None:
        return {"status": "error", "reason": f"Could not parse chosen_ligand: {chosen_ligand}"}
    res_name, chain, resid = parsed

    receptor_dir.mkdir(parents=True, exist_ok=True)

    raw_pdb = receptor_dir / f"{uniprot}_raw.pdb"
    ref_lig_pdb = receptor_dir / f"{uniprot}_ref_ligand.pdb"
    ref_lig_sdf = receptor_dir / f"{uniprot}_ref_ligand.sdf"
    clean_pdb = receptor_dir / f"{uniprot}_clean.pdb"
    pdbqt = receptor_dir / f"{uniprot}.pdbqt"

    # Step 1: CIF → PDB
    print(f"  [{uniprot}] CIF → PDB ...")
    if not cif_to_pdb(cif_path, raw_pdb):
        return {"status": "error", "reason": "CIF→PDB conversion failed"}

    pdb_lines = read_pdb_lines(raw_pdb)

    # Step 2: Extract reference ligand
    print(f"  [{uniprot}] Extracting reference ligand {res_name} (from {chosen_ligand}) ...")
    ref_lines = extract_ref_ligand_lines(pdb_lines, res_name)
    if not ref_lines:
        return {
            "status": "error",
            "reason": f"Reference ligand residue '{res_name}' not found in converted PDB (original: {chosen_ligand})",
        }
    print(f"  [{uniprot}] Found {len(ref_lines)} atoms for {res_name}")

    with open(ref_lig_pdb, "w") as f:
        f.writelines(["REMARK Reference ligand for autobox\n"] + ref_lines + ["END\n"])

    if not pdb_to_sdf(ref_lig_pdb, ref_lig_sdf):
        return {"status": "error", "reason": "Reference ligand PDB→SDF conversion failed"}

    # Step 3: Strip co-crystal and waters
    print(f"  [{uniprot}] Stripping co-crystal and waters ...")
    clean_lines = strip_residue_and_waters(pdb_lines, res_name)

    with open(clean_pdb, "w") as f:
        f.writelines(clean_lines)

    # Step 4: Clean PDB → PDBQT (fall back to raw PDB if obabel fails)
    print(f"  [{uniprot}] PDB → PDBQT ...")
    if pdb_to_pdbqt(clean_pdb, pdbqt):
        receptor_path = str(pdbqt)
        receptor_format = "pdbqt"
    else:
        # obabel kekulization failure — GNINA accepts PDB directly.
        # Also strip '*'-element records that crash GNINA's parser.
        print(f"  [{uniprot}] PDBQT conversion failed; falling back to clean PDB as receptor")
        fixed_pdb = strip_invalid_element_atoms(clean_pdb)
        receptor_path = str(fixed_pdb)
        receptor_format = "pdb"

    return {
        "status": "ok",
        "pdbqt": receptor_path,
        "receptor_format": receptor_format,
        "ref_ligand_sdf": str(ref_lig_sdf),
        "n_ref_atoms": len(ref_lines),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare receptor structures for GNINA docking")
    parser.add_argument("--config", required=True, help="Path to gnina_config.yaml")
    parser.add_argument("--workdir", default=".", help="Project root directory")
    parser.add_argument("--targets", nargs="*", help="Only process these UniProt IDs (default: all)")
    args = parser.parse_args()

    config = load_config(args.config)
    workdir = Path(args.workdir).resolve()
    receptor_dir = workdir / config["paths"]["receptor_dir"]
    output_dir = workdir / config["paths"]["output_dir"]

    targets_path = output_dir / "selected_targets.json"
    if not targets_path.exists():
        print(f"ERROR: {targets_path} not found. Run select_targets.py first.")
        sys.exit(1)

    with open(targets_path) as f:
        targets = json.load(f)

    if args.targets:
        targets = [t for t in targets if t["uniprot_id"] in args.targets]
        print(f"Filtered to {len(targets)} specified targets")

    print(f"Preparing receptors for {len(targets)} targets ...\n")

    # Load existing results to merge
    prep_path = output_dir / "receptor_prep_results.json"
    results = {}
    if prep_path.exists():
        with open(prep_path) as f:
            results = json.load(f)

    ok_count = sum(1 for r in results.values() if r.get("status") == "ok")
    for target in targets:
        uniprot = target["uniprot_id"]
        print(f"[{uniprot}] PDB={target['pdb_id']}  ligand={target['chosen_ligand']}")
        result = prepare_receptor(target, receptor_dir, workdir)
        results[uniprot] = result
        if result["status"] == "ok":
            ok_count += 1
            print(f"  [{uniprot}] OK — PDBQT: {result['pdbqt']}")
        else:
            print(f"  [{uniprot}] FAILED: {result['reason']}")
        print()

    # Save preparation results (merges with existing)
    with open(prep_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Receptor preparation complete: {ok_count}/{len(targets)} succeeded")
    print(f"Results saved to {prep_path}")


if __name__ == "__main__":
    main()
