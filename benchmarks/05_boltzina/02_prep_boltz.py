# benchmarks/05_boltzina/02_prep_boltz.py
"""Stage 02: Run boltz predict per protein → results/work_dirs/{uid}/"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from boltz_prep import prep_protein


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poc-proteins', required=True,
                   help='Path to poc_proteins.json from Stage 01')
    p.add_argument('--registry', required=True,
                   help='Path to registry_soft_split_regression.csv')
    p.add_argument('--results-dir', required=True,
                   help='Root results directory (e.g. benchmarks/05_boltzina/results)')
    p.add_argument('--base-dir', required=True,
                   help='Repo root (for resolving CIF paths)')
    p.add_argument('--boltzina-env', default='boltzina_env')
    args = p.parse_args()

    with open(args.poc_proteins) as f:
        proteins = json.load(f)

    print(f'Loading registry...')
    registry_df = pd.read_csv(
        args.registry,
        usecols=['uniprot_id', 'is_active', 'smiles'],
    )

    print(f'Running boltz predict for {len(proteins)} proteins...')
    for i, protein in enumerate(proteins, 1):
        print(f'[{i}/{len(proteins)}] {protein["uniprot_id"]}')
        prep_protein(protein, registry_df, args.results_dir,
                     args.base_dir, args.boltzina_env)

    print('Stage 02 complete.')


if __name__ == '__main__':
    main()
