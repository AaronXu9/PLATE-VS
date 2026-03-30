# benchmarks/05_boltzina/04_run_boltzina.py
"""Stage 04: Run boltzina docking + Boltz-2 scoring per protein."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from boltz_prep import get_receptor_pdb, get_boltz_results_dir, get_cif_path, prep_receptor_pdbqt
from boltzina_runner import collect_ligand_paths, write_boltzina_config, run_boltzina


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
        receptor_pdb = get_receptor_pdb(str(work_dir), uid)
        ligands_root = args.ligands_dir if args.ligands_dir else str(results_dir / 'ligands')
        ligand_files = collect_ligand_paths(ligands_root, uid)

        print(f'[{i}/{len(proteins)}] {uid}: {len(ligand_files)} ligands...')
        config_path = write_boltzina_config(
            uid=uid,
            work_dir=str(boltz_results),
            vina_config=str(vina_config),
            receptor_pdb=receptor_pdb,
            ligand_files=ligand_files,
            output_dir=str(raw_dir),
        )
        run_boltzina(config_path, args.boltzina_dir, args.boltzina_env)
        print(f'  ✓ {uid}: boltzina complete')

    print('Stage 04 complete.')


if __name__ == '__main__':
    main()
