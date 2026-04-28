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


def _skip_target(uniprot: str, out_sdf: Path, prior: dict) -> dict | None:
    """Return a reusable prior result entry if this target is already done.

    A target is "done" when (a) its docked SDF is present on disk and
    (b) the prior unidock2_results.json has a status of ok or ok_partial.
    Otherwise return None and the runner re-docks.
    """
    if not out_sdf.exists():
        return None
    rec = prior.get(uniprot)
    if not rec:
        return None
    if rec.get("status") not in ("ok", "ok_partial"):
        return None
    return rec


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


def _run_unidock2_once(
    receptor_pdb: Path,
    batch_lines: list[str],
    box: dict,
    out_sdf: Path,
    target_yaml: Path,
    log_handle,
    ud_cfg: dict,
    timeout_s: int,
) -> tuple[int, float]:
    """Single UniDock2 invocation on a batch of ligands. Returns (rc, elapsed)."""
    tmp_batch = out_sdf.parent / f"{out_sdf.stem}_batch.txt"
    tmp_batch.write_text("\n".join(batch_lines) + "\n")
    cx, cy, cz = box["center"]
    cmd = [
        "conda", "run", "-n", ud_cfg["conda_env"], "--no-capture-output",
        "unidock2", "docking",
        "-r", str(receptor_pdb),
        "-lb", str(tmp_batch),
        "-c", f"{cx}", f"{cy}", f"{cz}",
        "-cf", str(target_yaml),
        "-o", str(out_sdf),
    ]
    start = time.time()
    try:
        log_handle.write(f"\n>>> UniDock2 invocation on {len(batch_lines)} ligands\n")
        log_handle.flush()
        res = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, timeout=timeout_s)
        return res.returncode, time.time() - start
    except subprocess.TimeoutExpired:
        return 124, time.time() - start


def _concat_sdf(parts: list[Path], dest: Path) -> None:
    with open(dest, "w") as out:
        for p in parts:
            if not p.exists():
                continue
            with open(p) as f:
                out.write(f.read())


def dock_target(
    uniprot: str,
    receptor_pdb: Path,
    batch_txt: Path,
    box: dict,
    out_sdf: Path,
    log_file: Path,
    ud_cfg: dict,
) -> dict:
    """Dock all ligands listed in batch_txt. If the full batch fails, retry
    with chunks (default 200 ligands/chunk); chunks that still fail are dropped
    and reported. Final per-target SDF is the concatenation of successful
    chunks. UniDock2's GAFF parameteriser raises an unrecoverable KeyError on
    a few exotic ligand chemistries — chunking keeps the failure local.
    """
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

    all_lines = [ln.strip() for ln in batch_txt.read_text().splitlines() if ln.strip()]
    full_timeout = ud_cfg["per_target_timeout_s"]
    chunk_size = ud_cfg.get("chunk_size", 200)
    bisect_max_depth = ud_cfg.get("bisect_max_depth", 3)

    overall_start = time.time()
    with open(log_file, "w") as log:
        # Attempt 1: full batch.
        rc, elapsed = _run_unidock2_once(
            receptor_pdb, all_lines, box, out_sdf, target_yaml, log, ud_cfg, full_timeout
        )
        if rc == 0 and out_sdf.exists():
            return {
                "status": "ok",
                "elapsed_s": elapsed,
                "out_sdf": str(out_sdf),
                "n_input": len(all_lines),
                "n_chunks": 1,
                "n_chunks_failed": 0,
            }
        log.write(f"\n>>> Full batch returned rc={rc}; falling back to {chunk_size}-ligand chunks\n")
        log.flush()

        # Fallback: chunked + bisection. If a chunk fails, halve it and
        # recurse — the offending ligand ends up isolated in a chunk of size 1
        # while the rest still get docked.
        chunk_dir = out_sdf.parent / f"{uniprot}_chunks"
        chunk_dir.mkdir(exist_ok=True)
        chunk_outputs: list[Path] = []
        n_dropped = 0
        chunk_idx = [0]

        def _dock_chunk(lines: list[str], depth: int = 0) -> None:
            nonlocal n_dropped
            if not lines:
                return
            remaining = full_timeout - int(time.time() - overall_start)
            if remaining < 60:
                log.write(f"\n>>> {len(lines)} ligands dropped: only {remaining}s budget left\n")
                n_dropped += len(lines)
                return
            c = chunk_idx[0]
            chunk_idx[0] += 1
            chunk_out = chunk_dir / f"chunk_{c:04d}.sdf"
            chunk_yaml = chunk_dir / f"chunk_{c:04d}.yaml"
            _write_target_yaml(
                chunk_yaml,
                box["size"],
                ud_cfg["num_pose"],
                ud_cfg["seed"],
                ud_cfg["energy_range"],
                ud_cfg["search_mode"],
                ud_cfg["gpu_device_id"],
            )
            chunk_timeout = min(remaining, max(120, len(lines) * 3 + 60))
            rc_c, _ = _run_unidock2_once(
                receptor_pdb, lines, box, chunk_out, chunk_yaml, log, ud_cfg, chunk_timeout
            )
            if rc_c == 0 and chunk_out.exists():
                chunk_outputs.append(chunk_out)
                return
            # Failure path: bisect up to bisect_max_depth, otherwise drop the
            # whole chunk so a chemistry-killer doesn't blow the time budget.
            if depth >= bisect_max_depth:
                log.write(
                    f"\n>>> chunk of {len(lines)} FAILED at depth {depth} "
                    f"(>= cap {bisect_max_depth}) — dropping {len(lines)} ligands\n"
                )
                n_dropped += len(lines)
                return
            if len(lines) <= 1:
                log.write(f"\n>>> singleton ligand FAILED — dropping: {lines}\n")
                n_dropped += 1
                return
            log.write(f"\n>>> chunk of {len(lines)} FAILED, bisecting (depth {depth + 1})\n")
            mid = len(lines) // 2
            _dock_chunk(lines[:mid], depth=depth + 1)
            _dock_chunk(lines[mid:], depth=depth + 1)

        for start in range(0, len(all_lines), chunk_size):
            _dock_chunk(all_lines[start : start + chunk_size])

        if not chunk_outputs:
            return {
                "status": "error",
                "reason": "all chunks failed",
                "elapsed_s": time.time() - overall_start,
                "n_input": len(all_lines),
                "n_dropped": n_dropped,
            }
        _concat_sdf(chunk_outputs, out_sdf)
        return {
            "status": "ok_partial" if n_dropped > 0 else "ok",
            "elapsed_s": time.time() - overall_start,
            "out_sdf": str(out_sdf),
            "n_input": len(all_lines),
            "n_dropped": n_dropped,
            "n_chunks_run": chunk_idx[0],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--targets", nargs="*", help="Subset of UniProt IDs to dock")
    ap.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-dock every target even if results exist (default: skip).",
    )
    ap.add_argument("--shard-id", type=int, default=None, help="0-indexed shard id (default: no sharding)")
    ap.add_argument("--n-shards", type=int, default=None, help="Total number of shards")
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

    # Load prior results so SLURM restarts can skip already-completed targets.
    results_path = paths["output_dir"] / "unidock2_results.json"
    prior: dict = {}
    if results_path.exists() and not args.no_skip_existing:
        try:
            prior = json.loads(results_path.read_text())
        except (json.JSONDecodeError, OSError):
            prior = {}

    targets_sorted = sorted(targets, key=lambda t: t["uniprot_id"])

    results = {}
    for idx, tgt in enumerate(targets_sorted):
        if args.shard_id is not None and args.n_shards:
            if idx % args.n_shards != args.shard_id:
                continue
        u = tgt["uniprot_id"]
        if args.targets and u not in args.targets:
            continue
        info = prep.get(u, {})
        if info.get("status") != "ok":
            results[u] = {"status": "skipped", "reason": info.get("status", "missing")}
            continue
        receptor_pdb = paths["receptor_dir"] / f"{u}_clean.pdb"
        out_sdf = paths["docking_dir"] / f"{u}_docked.sdf"
        log_file = paths["docking_dir"] / f"{u}_unidock2.log"

        reuse = _skip_target(u, out_sdf, prior)
        if reuse is not None:
            print(f"[{u}] skip-existing: reusing prior result ({reuse.get('status')})", flush=True)
            results[u] = reuse
            continue

        if not receptor_pdb.exists():
            results[u] = {"status": "skipped", "reason": f"receptor missing: {receptor_pdb}"}
            continue
        batch_txt = Path(info["batch_path"])
        print(f"[{u}] docking {info['n_ligands']} ligands ...", flush=True)
        r = dock_target(u, receptor_pdb, batch_txt, info["box"], out_sdf, log_file, ud)
        results[u] = r
        msg = f"  [{u}] {r['status']} in {r.get('elapsed_s', 0):.0f}s"
        if r["status"] not in ("ok", "ok_partial"):
            msg += f"  err={r.get('reason')}"
        print(msg, flush=True)

    (paths["output_dir"] / "unidock2_results.json").write_text(json.dumps(results, indent=2))
    ok = sum(1 for v in results.values() if v["status"] in ("ok", "ok_partial"))
    print(f"\nDone: {ok}/{len(results)} targets succeeded")


if __name__ == "__main__":
    main()
