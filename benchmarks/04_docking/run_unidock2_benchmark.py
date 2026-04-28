"""Run UniDock2 over the 15 selected targets using the prepared batch inputs.

One UniDock2 invocation per target using `-lb` (batch ligand list). Box derived
from the reference co-crystal ligand. GPU device chosen by config.

Usage:
    /home/aoxu/miniconda3/envs/rdkit_env/bin/python \\
        benchmarks/04_docking/run_unidock2_benchmark.py \\
        --config benchmarks/04_docking/configs/unidock2_config.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _write_target_yaml(
    cfg_path: Path,
    size: list[float],
    num_pose: int,
    seed: int,
    energy_range: float,
    search_mode: str,
    gpu_id: int,
) -> None:
    body = (
        "Settings:\n"
        f"    size: [{size[0]}, {size[1]}, {size[2]}]\n"
        f"    search_mode: {search_mode}\n"
        "    task: screen\n"
        "Advanced:\n"
        f"    seed: {seed}\n"
        f"    num_pose: {num_pose}\n"
        f"    energy_range: {energy_range}\n"
        "Hardware:\n"
        f"    gpu_device_id: {gpu_id}\n"
    )
    cfg_path.write_text(body)


def dock_target(
    uniprot: str,
    receptor_pdb: Path,
    batch_txt: Path,
    box: dict,
    out_sdf: Path,
    log_file: Path,
    ud_cfg: dict,
) -> dict:
    target_yaml = out_sdf.parent / f"{uniprot}_unidock2.yaml"
    _write_target_yaml(
        target_yaml,
        box["size"],
        ud_cfg["num_pose"],
        ud_cfg["seed"],
        ud_cfg["energy_range"],
        ud_cfg["search_mode"],
        ud_cfg["gpu_device_id"],
    )

    cx, cy, cz = box["center"]
    cmd = [
        ud_cfg["binary"],
        "docking",
        "-r", str(receptor_pdb),
        "-lb", str(batch_txt),
        "-c", f"{cx}", f"{cy}", f"{cz}",
        "-cf", str(target_yaml),
        "-o", str(out_sdf),
    ]
    start = time.time()
    try:
        with open(log_file, "w") as log:
            res = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=ud_cfg["per_target_timeout_s"],
            )
        elapsed = time.time() - start
        if res.returncode != 0:
            return {
                "status": "error",
                "reason": f"unidock2 exited {res.returncode}",
                "elapsed_s": elapsed,
            }
        if not out_sdf.exists():
            return {"status": "error", "reason": "no output sdf", "elapsed_s": elapsed}
        return {"status": "ok", "elapsed_s": elapsed, "out_sdf": str(out_sdf)}
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "reason": "exceeded timeout",
            "elapsed_s": ud_cfg["per_target_timeout_s"],
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "reason": str(e), "elapsed_s": time.time() - start}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--targets", nargs="*", help="Subset of UniProt IDs to dock")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    paths["docking_dir"].mkdir(parents=True, exist_ok=True)
    ud = cfg["unidock2"]
    if not Path(ud["binary"]).exists():
        print(f"ERROR: unidock2 binary not at {ud['binary']}")
        sys.exit(1)

    targets = json.loads((paths["output_dir"] / "selected_targets.json").read_text())
    prep_path = paths["output_dir"] / "unidock2_targets.json"
    if not prep_path.exists():
        print(f"ERROR: {prep_path} not found. Run build_unidock2_batches.py first.")
        sys.exit(1)
    prep = json.loads(prep_path.read_text())

    results = {}
    for tgt in targets:
        u = tgt["uniprot_id"]
        if args.targets and u not in args.targets:
            continue
        info = prep.get(u, {})
        if info.get("status") != "ok":
            results[u] = {"status": "skipped", "reason": info.get("status", "missing")}
            continue
        receptor_pdb = paths["receptor_dir"] / f"{u}_clean.pdb"
        batch_txt = Path(info["batch_path"])
        out_sdf = paths["docking_dir"] / f"{u}_docked.sdf"
        log_file = paths["docking_dir"] / f"{u}_unidock2.log"
        if not receptor_pdb.exists():
            results[u] = {"status": "skipped", "reason": f"receptor missing: {receptor_pdb}"}
            continue
        print(f"[{u}] docking {info['n_ligands']} ligands ...", flush=True)
        r = dock_target(u, receptor_pdb, batch_txt, info["box"], out_sdf, log_file, ud)
        results[u] = r
        msg = f"  [{u}] {r['status']} in {r.get('elapsed_s', 0):.0f}s"
        if r["status"] != "ok":
            msg += f"  err={r.get('reason')}"
        print(msg, flush=True)

    (paths["output_dir"] / "unidock2_results.json").write_text(json.dumps(results, indent=2))
    ok = sum(1 for v in results.values() if v["status"] == "ok")
    print(f"\nDone: {ok}/{len(results)} targets succeeded")


if __name__ == "__main__":
    main()
