"""
Run GNINA docking benchmark across selected targets (parallelized).

For each target, docks the prepared ligand SDF against the prepared PDBQT
receptor using the co-crystal reference ligand to define the binding box.

Usage (from project root):
    conda run -n rdkit_env python3 benchmarks/04_docking/run_gnina_benchmark.py \
        --config benchmarks/04_docking/configs/gnina_config.yaml \
        [--n-workers 4]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def dock_target(
    uniprot: str,
    receptor_pdbqt: str,
    ligand_sdf: str,
    ref_ligand_sdf: str,
    out_sdf: str,
    log_file: str,
    gnina_binary: str,
    num_modes: int,
    exhaustiveness: int,
    cnn_scoring: str,
    autobox_add: float,
    cpu: int,
    device: int = 0,
) -> dict:
    """Run GNINA for one target. Returns result dict."""
    cmd = [
        gnina_binary,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_sdf,
        "--autobox_ligand", ref_ligand_sdf,
        "--autobox_add", str(autobox_add),
        "--num_modes", str(num_modes),
        "--exhaustiveness", str(exhaustiveness),
        "--cnn_scoring", cnn_scoring,
        "--out", out_sdf,
        "--cpu", str(cpu),
        "--device", str(device),
    ]

    start = time.time()
    try:
        with open(log_file, "w") as log:
            result = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=14400,  # 4-hour timeout per target
            )
        elapsed = time.time() - start
        if result.returncode != 0:
            return {
                "status": "error",
                "reason": f"GNINA exited with code {result.returncode}",
                "elapsed_s": elapsed,
            }
        if not Path(out_sdf).exists():
            return {
                "status": "error",
                "reason": "GNINA produced no output SDF",
                "elapsed_s": elapsed,
            }
        return {"status": "ok", "elapsed_s": elapsed, "out_sdf": out_sdf}

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "reason": "exceeded 4-hour limit", "elapsed_s": 14400}
    except Exception as e:
        return {"status": "error", "reason": str(e), "elapsed_s": time.time() - start}


def _run_job(job: dict) -> dict:
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    p = job["gnina_params"]
    return dock_target(
        uniprot=job["uniprot"],
        receptor_pdbqt=job["receptor_pdbqt"],
        ligand_sdf=job["ligand_sdf"],
        ref_ligand_sdf=job["ref_ligand_sdf"],
        out_sdf=job["out_sdf"],
        log_file=job["log_file"],
        gnina_binary=p["gnina_binary"],
        num_modes=p["num_modes"],
        exhaustiveness=p["exhaustiveness"],
        cnn_scoring=p["cnn_scoring"],
        autobox_add=p["autobox_add"],
        cpu=p["cpu"],
        device=p.get("device", 0),
    )


def main():
    parser = argparse.ArgumentParser(description="Run GNINA docking benchmark")
    parser.add_argument("--config", required=True, help="Path to gnina_config.yaml")
    parser.add_argument("--workdir", default=".", help="Project root directory")
    parser.add_argument("--n-workers", type=int, default=None, help="Parallel targets (overrides config)")
    parser.add_argument("--targets", nargs="*", help="Subset of UniProt IDs to dock (default: all selected)")
    args = parser.parse_args()

    config = load_config(args.config)
    workdir = Path(args.workdir).resolve()
    output_dir = workdir / config["paths"]["output_dir"]
    receptor_dir = workdir / config["paths"]["receptor_dir"]
    ligand_dir = workdir / config["paths"]["ligand_dir"]
    docking_dir = workdir / config["paths"]["docking_dir"]
    docking_dir.mkdir(parents=True, exist_ok=True)

    gnina_cfg = config["gnina"]
    gnina_binary = gnina_cfg["binary"]
    if not Path(gnina_binary).exists():
        print(f"ERROR: GNINA binary not found at {gnina_binary}")
        sys.exit(1)

    n_workers = args.n_workers or config["parallelism"]["n_workers"]

    # Load selected targets
    targets_path = output_dir / "selected_targets.json"
    if not targets_path.exists():
        print(f"ERROR: {targets_path} not found. Run select_targets.py first.")
        sys.exit(1)
    with open(targets_path) as f:
        targets = json.load(f)

    # Load prep results to check which receptors/ligands are ready
    receptor_prep_path = output_dir / "receptor_prep_results.json"
    ligand_prep_path = output_dir / "ligand_prep_results.json"

    receptor_results = {}
    if receptor_prep_path.exists():
        with open(receptor_prep_path) as f:
            receptor_results = json.load(f)

    ligand_results = {}
    if ligand_prep_path.exists():
        with open(ligand_prep_path) as f:
            ligand_results = json.load(f)

    # Build job list
    jobs = []
    skipped = []
    for target in targets:
        uniprot = target["uniprot_id"]

        # Filter to requested targets if specified
        if args.targets and uniprot not in args.targets:
            continue

        rec_info = receptor_results.get(uniprot, {})
        lig_info = ligand_results.get(uniprot, {})

        # Use receptor path from prep results (may be .pdb fallback when PDBQT conversion failed)
        receptor_pdbqt = rec_info.get("pdbqt", str(receptor_dir / f"{uniprot}.pdbqt"))
        ref_ligand_sdf = str(receptor_dir / f"{uniprot}_ref_ligand.sdf")
        ligand_sdf = str(ligand_dir / f"{uniprot}_all_ligands.sdf")
        out_sdf = str(docking_dir / f"{uniprot}_docked.sdf")
        log_file = str(docking_dir / f"{uniprot}_gnina.log")

        # Validate inputs exist (also check receptor is non-empty)
        missing = []
        for label, path in [("receptor", receptor_pdbqt), ("ref ligand SDF", ref_ligand_sdf), ("ligand SDF", ligand_sdf)]:
            if not Path(path).exists():
                missing.append(f"{label}: {path}")

        if missing:
            print(f"[{uniprot}] Skipping — missing inputs: {missing}")
            skipped.append(uniprot)
            continue

        n_ligands = lig_info.get("n_actives", 0) + lig_info.get("n_decoys", 0)
        if n_ligands == 0:
            print(f"[{uniprot}] Skipping — 0 ligands prepared")
            skipped.append(uniprot)
            continue

        jobs.append(
            {
                "uniprot": uniprot,
                "receptor_pdbqt": receptor_pdbqt,
                "ligand_sdf": ligand_sdf,
                "ref_ligand_sdf": ref_ligand_sdf,
                "out_sdf": out_sdf,
                "log_file": log_file,
                "n_ligands": n_ligands,
            }
        )

    if not jobs:
        print("No jobs to run. Check that prepare_structures.py and prepare_ligands.py completed successfully.")
        sys.exit(1)

    total_ligands = sum(j["n_ligands"] for j in jobs)
    print(f"\nDocking {len(jobs)} targets ({total_ligands} total ligands) with {n_workers} workers")
    device = gnina_cfg.get("device", 0)
    print(f"GNINA: num_modes={gnina_cfg['num_modes']}  exhaustiveness={gnina_cfg['exhaustiveness']}  cnn_scoring={gnina_cfg['cnn_scoring']}  device={device}\n")

    # Attach fixed GNINA params to each job dict so the module-level worker can access them
    gnina_params = {
        "gnina_binary": gnina_binary,
        "num_modes": gnina_cfg["num_modes"],
        "exhaustiveness": gnina_cfg["exhaustiveness"],
        "cnn_scoring": gnina_cfg["cnn_scoring"],
        "autobox_add": gnina_cfg["autobox_add"],
        "cpu": gnina_cfg["cpu"],
        "device": gnina_cfg.get("device", 0),  # GPU device index; 0 = first GPU
    }
    for job in jobs:
        job["gnina_params"] = gnina_params

    docking_results = {}

    if n_workers == 1:
        # Sequential for debugging
        for job in jobs:
            print(f"[{job['uniprot']}] Docking {job['n_ligands']} ligands ...")
            result = _run_job(job)
            docking_results[job["uniprot"]] = result
            status = result["status"]
            elapsed = result.get("elapsed_s", 0)
            if status == "ok":
                print(f"  [{job['uniprot']}] Done in {elapsed:.0f}s")
            else:
                print(f"  [{job['uniprot']}] {status.upper()}: {result.get('reason', '')}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_job = {executor.submit(_run_job, job): job for job in jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                uniprot = job["uniprot"]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"status": "error", "reason": str(e)}
                docking_results[uniprot] = result
                status = result["status"]
                elapsed = result.get("elapsed_s", 0)
                if status == "ok":
                    print(f"[{uniprot}] Done in {elapsed:.0f}s")
                else:
                    print(f"[{uniprot}] {status.upper()}: {result.get('reason', '')}")

    # Save docking results
    docking_results_path = output_dir / "docking_results.json"
    with open(docking_results_path, "w") as f:
        json.dump(docking_results, f, indent=2)

    ok = sum(1 for r in docking_results.values() if r["status"] == "ok")
    print(f"\nDocking complete: {ok}/{len(jobs)} targets succeeded")
    if skipped:
        print(f"Skipped: {skipped}")
    print(f"Results saved to {docking_results_path}")


if __name__ == "__main__":
    main()
