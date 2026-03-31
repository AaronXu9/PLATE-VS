# benchmarks/05_boltzina/04_run_boltzina.py
"""Stage 04: Run boltzina docking + Boltz-2 scoring per protein."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from boltz_prep import get_boltz_results_dir, get_cif_path, prep_receptor_pdbqt, _exp_cif_to_pdb
from boltzina_runner import collect_ligand_paths, write_boltzina_config, run_boltzina
from unidock_docking import run_unidock_pipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poc-proteins', required=True)
    p.add_argument('--results-dir', required=True)
    p.add_argument('--boltzina-dir', default='external/boltzina',
                   help='Path to cloned boltzina submodule')
    p.add_argument('--boltzina-env', default='boltzina_env')
    p.add_argument('--ligands-dir', default=None,
                   help='Override ligands root dir (default: {results-dir}/ligands)')
    p.add_argument('--base-dir', default=None,
                   help='Repo root for resolving experimental CIF paths')
    p.add_argument('--use-unidock', action='store_true', default=True,
                   help='Use GPU Uni-Dock instead of CPU Vina (default)')
    p.add_argument('--no-unidock', dest='use_unidock', action='store_false',
                   help='Use CPU Vina docking (slower)')
    p.add_argument('--unidock-env', default='unidock2',
                   help='Conda env with Uni-Dock')
    p.add_argument('--unidock-batch-size', type=int, default=200,
                   help='Ligands per Uni-Dock GPU batch')
    p.add_argument('--batch-size', type=int, default=4,
                   help='Boltz-2 scoring batch size')
    p.add_argument('--num-workers', type=int, default=16,
                   help='Parallel workers for docking/prep')
    args = p.parse_args()

    with open(args.poc_proteins) as f:
        proteins = json.load(f)

    results_dir = Path(args.results_dir)

    for i, protein in enumerate(proteins, 1):
        uid = protein['uniprot_id']
        raw_dir = results_dir / 'raw_results' / uid

        if (raw_dir / 'boltzina_results.csv').exists():
            print(f'[{i}/{len(proteins)}] {uid}: skip (results.csv exists)')
            continue

        work_dir = results_dir / 'work_dirs' / uid
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Pre-generate receptor.pdbqt from experimental CIF (avoids Meeko failures
        # on boltz-2 predicted structures with structural clashes)
        if args.base_dir and protein.get('pdb_id'):
            exp_cif = get_cif_path(uid, protein['pdb_id'], args.base_dir)
            try:
                prep_receptor_pdbqt(exp_cif, str(raw_dir / 'receptor.pdbqt'), args.boltzina_env)
            except Exception as e:
                print(f'  [warn] {uid}: receptor prep failed: {e}')

        boltz_results = get_boltz_results_dir(str(work_dir), uid)
        vina_config = work_dir / 'vina_config.txt'

        # Generate receptor PDB from experimental CIF (for pdb_merge in post-processing)
        receptor_pdb_path = raw_dir / 'receptor.pdb'
        if not receptor_pdb_path.exists() and args.base_dir and protein.get('pdb_id'):
            exp_cif = get_cif_path(uid, protein['pdb_id'], args.base_dir)
            _exp_cif_to_pdb(exp_cif, str(receptor_pdb_path))
        receptor_pdb = str(receptor_pdb_path)
        ligands_root = args.ligands_dir if args.ligands_dir else str(results_dir / 'ligands')
        ligand_files = collect_ligand_paths(ligands_root, uid)

        print(f'[{i}/{len(proteins)}] {uid}: {len(ligand_files)} ligands...')

        # Phase 1: Docking (Uni-Dock GPU or Vina CPU)
        if args.use_unidock:
            run_unidock_pipeline(
                receptor_pdbqt=str(raw_dir / 'receptor.pdbqt'),
                receptor_pdb=receptor_pdb,
                ligand_files=ligand_files,
                vina_config=str(vina_config),
                output_dir=str(raw_dir),
                unidock_env=args.unidock_env,
                boltzina_env=args.boltzina_env,
                batch_size=args.unidock_batch_size,
                num_workers=args.num_workers,
            )

        # Phase 2+3: Structure prep + Boltz-2 scoring
        config_path = write_boltzina_config(
            uid=uid,
            work_dir=str(boltz_results),
            vina_config=str(vina_config),
            receptor_pdb=receptor_pdb,
            ligand_files=ligand_files,
            output_dir=str(raw_dir),
        )
        run_boltzina(
            config_path, args.boltzina_dir, args.boltzina_env,
            num_workers=args.num_workers, vina_cpu=2,
            skip_docking=args.use_unidock,
            batch_size=args.batch_size,
        )
        print(f'  ✓ {uid}: boltzina complete')

    print('Stage 04 complete.')


if __name__ == '__main__':
    main()
