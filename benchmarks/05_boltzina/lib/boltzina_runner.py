# benchmarks/05_boltzina/lib/boltzina_runner.py
"""Boltzina config generation and subprocess runner."""
import json
import subprocess
from pathlib import Path


def write_boltzina_config(uid, work_dir, vina_config, receptor_pdb,
                           ligand_files, output_dir):
    """Write boltzina config.json for one protein.

    Args:
        uid: UniProt ID (used as fname, must match boltz predict YAML stem)
        work_dir: path to boltz predict output directory
        vina_config: path to Vina config file
        receptor_pdb: path to predicted receptor PDB
        ligand_files: list of ligand PDB file paths (actives + decoys)
        output_dir: where boltzina writes its results CSV

    Returns:
        path to written config.json (str)
    """
    config = {
        'work_dir': str(work_dir),
        'vina_config': str(vina_config),
        'fname': uid,
        'input_ligand_name': 'UNL',
        'output_dir': str(output_dir),
        'receptor_pdb': str(receptor_pdb),
        'ligand_files': [str(f) for f in ligand_files],
    }
    config_path = Path(output_dir) / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return str(config_path)


def run_boltzina(config_path, boltzina_submodule_path, boltzina_env='boltzina_env',
                 num_workers=16, vina_cpu=2, skip_docking=False, batch_size=4):
    """Run boltzina via conda run.

    Args:
        config_path: path to boltzina config.json
        boltzina_submodule_path: path to external/boltzina/ (where run.py lives)
        boltzina_env: conda environment name
        num_workers: parallel workers (docking pool + structure prep + DataLoader)
        vina_cpu: CPUs per Vina docking process (ignored when skip_docking=True)
        skip_docking: skip Vina docking phase (use when Uni-Dock already docked)
        batch_size: Boltz-2 scoring batch size (higher = faster, more VRAM)
    """
    run_py = Path(boltzina_submodule_path) / 'run.py'
    cmd = [
        'conda', 'run', '-n', boltzina_env,
        'python', str(run_py), str(config_path),
        '--num_workers', str(num_workers),
        '--vina_cpu', str(vina_cpu),
        '--batch_size', str(batch_size),
    ]
    if skip_docking:
        cmd.append('--skip_docking')
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result


def collect_ligand_paths(ligands_dir, uid):
    """Return sorted list of all PDB paths for actives and decoys.

    Args:
        ligands_dir: root ligands directory (contains {uid}/actives/ and {uid}/decoys/)
        uid: UniProt ID

    Returns:
        list of absolute path strings, actives first then decoys.
    """
    actives_dir = Path(ligands_dir) / uid / 'actives'
    decoys_dir = Path(ligands_dir) / uid / 'decoys'
    actives = sorted(actives_dir.glob('*.pdb'))
    decoys = sorted(decoys_dir.glob('*.pdb'))
    return [str(p) for p in actives + decoys]
