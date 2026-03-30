# benchmarks/05_boltzina/lib/unidock_docking.py
"""GPU-accelerated docking with Uni-Dock + parallel post-processing."""
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path


def parse_vina_config(config_path):
    """Parse Vina config file into dict of floats/ints."""
    config = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                try:
                    config[key] = int(val)
                except ValueError:
                    config[key] = float(val)
    return config


def _convert_one_pdb_to_pdbqt(args):
    """Worker: convert single PDB to PDBQT via obabel."""
    pdb_path, pdbqt_path, boltzina_env = args
    if Path(pdbqt_path).exists():
        return pdbqt_path
    try:
        subprocess.run(
            ['conda', 'run', '-n', boltzina_env,
             'obabel', str(pdb_path), '-O', str(pdbqt_path)],
            check=True, capture_output=True, text=True,
        )
    except Exception as e:
        print(f'[warn] obabel failed for {pdb_path}: {e}')
        return None
    return pdbqt_path if Path(pdbqt_path).exists() else None


def batch_convert_pdb_to_pdbqt(pdb_paths, output_dir, boltzina_env,
                                num_workers=16):
    """Convert PDB ligand files to PDBQT in parallel.

    Returns list of (idx, pdbqt_path) for successful conversions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i, pdb in enumerate(pdb_paths):
        pdbqt = output_dir / f'ligand_{i:05d}.pdbqt'
        tasks.append((str(pdb), str(pdbqt), boltzina_env))
    with Pool(num_workers) as pool:
        results = pool.map(_convert_one_pdb_to_pdbqt, tasks)
    return [(i, r) for i, r in enumerate(results) if r is not None]


def run_unidock_batch(receptor_pdbqt, pdbqt_paths, output_dir, config,
                      unidock_env='unidock2', batch_size=200):
    """Run Uni-Dock GPU batch docking.

    Args:
        receptor_pdbqt: path to receptor PDBQT
        pdbqt_paths: list of (idx, pdbqt_path) tuples
        output_dir: directory for docked outputs
        config: dict from parse_vina_config (center_x/y/z, size_x/y/z, etc.)
        unidock_env: conda env with unidock
        batch_size: max ligands per GPU call (memory limit ~200)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chunk ligands into batches
    for chunk_start in range(0, len(pdbqt_paths), batch_size):
        chunk = pdbqt_paths[chunk_start:chunk_start + batch_size]
        paths = [p for _, p in chunk]

        # Write ligand index file
        index_file = output_dir / f'_unidock_batch_{chunk_start}.txt'
        index_file.write_text('\n'.join(paths) + '\n')

        cmd = [
            'conda', 'run', '-n', unidock_env, 'unidock',
            '--receptor', str(receptor_pdbqt),
            '--ligand_index', str(index_file),
            '--dir', str(output_dir),
            '--center_x', str(config['center_x']),
            '--center_y', str(config['center_y']),
            '--center_z', str(config['center_z']),
            '--size_x', str(config['size_x']),
            '--size_y', str(config['size_y']),
            '--size_z', str(config['size_z']),
            '--num_modes', str(config.get('num_modes', 1)),
            '--seed', str(config.get('seed', 1)),
            '--search_mode', 'fast',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'[warn] Uni-Dock batch {chunk_start} failed: {result.stderr[-500:]}')

        # Clean up index file
        try:
            index_file.unlink()
        except OSError:
            pass


def setup_boltzina_layout(idx_pdbqt_pairs, unidock_output_dir, boltzina_output_dir):
    """Map Uni-Dock outputs to boltzina's expected out/{idx}/docked.pdbqt layout.

    Uni-Dock writes {stem}_out.pdbqt in the output dir.
    """
    unidock_dir = Path(unidock_output_dir)
    boltzina_dir = Path(boltzina_output_dir)

    for idx, pdbqt_path in idx_pdbqt_pairs:
        stem = Path(pdbqt_path).stem  # e.g. ligand_00042
        unidock_out = unidock_dir / f'{stem}_out.pdbqt'

        ligand_dir = boltzina_dir / 'out' / str(idx)
        ligand_dir.mkdir(parents=True, exist_ok=True)
        target = ligand_dir / 'docked.pdbqt'

        if target.exists():
            continue
        if unidock_out.exists():
            shutil.copy2(str(unidock_out), str(target))
        else:
            print(f'[warn] missing Uni-Dock output for idx={idx}: {unidock_out}')


def _postprocess_single(args):
    """Worker: post-process one docked ligand (obabel split + pdb tools + maxit).

    Replicates boltzina_main._preprocess_docked_structures + _process_pose.
    """
    (idx, output_dir, receptor_pdb, ligand_chain_id,
     input_ligand_name, base_ligand_name, pose_idxs, boltzina_env) = args

    ligand_dir = Path(output_dir) / 'out' / str(idx)
    docked_pdbqt = ligand_dir / 'docked.pdbqt'
    done_file = ligand_dir / 'done'

    if done_file.exists():
        return True
    if not docked_pdbqt.exists():
        return False

    docked_ligands_dir = ligand_dir / 'docked_ligands'
    docked_ligands_dir.mkdir(exist_ok=True)

    # Check if all complex_fix.cif files already exist
    all_exist = all(
        (docked_ligands_dir / f'docked_ligand_{pi}_{ligand_chain_id}_complex_fix.cif').exists()
        for pi in pose_idxs
    )
    if all_exist:
        done_file.touch()
        return True

    def _run(cmd, **kwargs):
        return subprocess.run(
            ['conda', 'run', '-n', boltzina_env] + cmd,
            capture_output=True, text=True, **kwargs,
        )

    def _shell(cmd):
        return subprocess.run(
            f'conda run -n {boltzina_env} bash -c {repr(cmd)}',
            shell=True, capture_output=True, text=True,
        )

    try:
        # Split docked PDBQT into individual PDB files
        _run(['obabel', str(docked_pdbqt), '-m', '-O',
              str(docked_ligands_dir / 'docked_ligand_.pdb')])

        for pose_idx in pose_idxs:
            pdb_file = docked_ligands_dir / f'docked_ligand_{pose_idx}.pdb'
            if not pdb_file.exists():
                continue

            base_name = f'docked_ligand_{pose_idx}'
            prep_file = docked_ligands_dir / f'{base_name}_prep.pdb'
            complex_file = docked_ligands_dir / f'{base_name}_{ligand_chain_id}_complex.pdb'
            complex_cif = docked_ligands_dir / f'{base_name}_{ligand_chain_id}_complex.cif'
            complex_fix = docked_ligands_dir / f'{base_name}_{ligand_chain_id}_complex_fix.cif'

            if complex_fix.exists():
                continue

            # pdb_chain + pdb_rplresname + pdb_tidy
            if input_ligand_name != base_ligand_name:
                pipe = (f'pdb_chain -{ligand_chain_id} {pdb_file} '
                        f'| pdb_rplresname -"{input_ligand_name}":{base_ligand_name} '
                        f'| pdb_tidy > {prep_file}')
            else:
                pipe = f'pdb_chain -{ligand_chain_id} {pdb_file} | pdb_tidy > {prep_file}'
            _shell(pipe)

            # Merge with receptor
            _shell(f'pdb_merge {receptor_pdb} {prep_file} | pdb_tidy > {complex_file}')

            # PDB → CIF
            _run(['maxit', '-input', str(complex_file),
                  '-output', str(complex_cif), '-o', '1'])

            # Fix CIF
            _run(['maxit', '-input', str(complex_cif),
                  '-output', str(complex_fix), '-o', '8'])

        done_file.touch()
        return True
    except Exception as e:
        print(f'[warn] postprocess failed for idx={idx}: {e}')
        if done_file.exists():
            done_file.unlink(missing_ok=True)
        return False


def batch_postprocess(output_dir, receptor_pdb, boltzina_env,
                      ligand_chain_id='B', input_ligand_name='UNL',
                      base_ligand_name='LIG', pose_idxs=('1',),
                      num_workers=16):
    """Post-process all docked ligands in parallel."""
    output_dir = Path(output_dir)
    out_dir = output_dir / 'out'
    if not out_dir.exists():
        return

    # Find all ligand indices that have docked.pdbqt but no done file
    indices = sorted(
        int(d.name) for d in out_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    tasks = [
        (idx, str(output_dir), str(receptor_pdb), ligand_chain_id,
         input_ligand_name, base_ligand_name, pose_idxs, boltzina_env)
        for idx in indices
    ]
    if not tasks:
        return

    print(f'  Post-processing {len(tasks)} docked ligands ({num_workers} workers)...')
    with Pool(num_workers) as pool:
        results = pool.map(_postprocess_single, tasks)
    n_ok = sum(1 for r in results if r)
    n_fail = sum(1 for r in results if not r)
    print(f'  Post-processing complete: {n_ok} ok, {n_fail} failed')


def run_unidock_pipeline(receptor_pdbqt, receptor_pdb, ligand_files,
                         vina_config, output_dir,
                         unidock_env='unidock2', boltzina_env='boltzina_env',
                         batch_size=200, num_workers=16):
    """Full Uni-Dock docking pipeline: convert → dock → layout → postprocess.

    Replaces boltzina's Phase 1 (Vina docking) entirely.
    """
    output_dir = Path(output_dir)
    pdbqt_dir = output_dir / '_unidock_pdbqts'
    unidock_out_dir = output_dir / '_unidock_results'

    config = parse_vina_config(vina_config)

    # 1. Convert PDB ligands to PDBQT (parallel)
    print(f'  Converting {len(ligand_files)} ligands to PDBQT...')
    idx_pdbqt = batch_convert_pdb_to_pdbqt(
        ligand_files, str(pdbqt_dir), boltzina_env, num_workers=num_workers)
    print(f'  Converted {len(idx_pdbqt)}/{len(ligand_files)} ligands')

    # 2. GPU batch docking with Uni-Dock
    print(f'  Running Uni-Dock batch docking ({len(idx_pdbqt)} ligands)...')
    run_unidock_batch(
        str(receptor_pdbqt), idx_pdbqt, str(unidock_out_dir),
        config, unidock_env=unidock_env, batch_size=batch_size)

    # 3. Map Uni-Dock outputs to boltzina directory layout
    setup_boltzina_layout(idx_pdbqt, str(unidock_out_dir), str(output_dir))

    # 4. Parallel post-processing (obabel split, pdb_chain, pdb_merge, maxit)
    batch_postprocess(
        str(output_dir), str(receptor_pdb), boltzina_env,
        num_workers=num_workers)

    # 5. Cleanup temp dirs
    try:
        shutil.rmtree(str(pdbqt_dir), ignore_errors=True)
        shutil.rmtree(str(unidock_out_dir), ignore_errors=True)
    except Exception:
        pass
