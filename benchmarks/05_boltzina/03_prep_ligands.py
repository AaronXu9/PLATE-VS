# benchmarks/05_boltzina/03_prep_ligands.py
"""Stage 03: Convert SMILES to 3D PDB files for actives and sampled decoys."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from ligands import prep_protein_ligands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poc-proteins', required=True)
    p.add_argument('--registry', required=True)
    p.add_argument('--results-dir', required=True)
    p.add_argument('--base-dir', required=True)
    args = p.parse_args()

    with open(args.poc_proteins) as f:
        proteins = json.load(f)

    print('Loading registry...')
    registry_df = pd.read_csv(
        args.registry,
        usecols=['uniprot_id', 'compound_id', 'smiles', 'similarity_threshold',
                 'protein_partition', 'split', 'is_active'],
    )

    for i, protein in enumerate(proteins, 1):
        uid = protein['uniprot_id']
        print(f'[{i}/{len(proteins)}] {uid}: preparing ligands...')
        n_act, n_dec, n_fail = prep_protein_ligands(
            protein, registry_df, args.results_dir, args.base_dir)
        print(f'  actives={n_act}, decoys={n_dec}, failed={n_fail}')

    print('Stage 03 complete.')


if __name__ == '__main__':
    main()
