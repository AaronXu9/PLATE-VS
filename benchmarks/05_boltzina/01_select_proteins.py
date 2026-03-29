# benchmarks/05_boltzina/01_select_proteins.py
"""Stage 01: Select top-N test proteins from regression registry."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from protein_select import select_proteins


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', required=True,
                   help='Path to registry_soft_split_regression.csv')
    p.add_argument('--output', required=True,
                   help='Path to write poc_proteins.json')
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--min-actives', type=int, default=50)
    p.add_argument('--min-pchembl-coverage', type=float, default=0.80)
    args = p.parse_args()

    print(f"Loading registry from {args.registry}...")
    df = pd.read_csv(args.registry)
    proteins = select_proteins(df, args.n, args.min_actives, args.min_pchembl_coverage)
    print(f"Selected {len(proteins)} proteins:")
    for prot in proteins:
        print(f"  {prot['uniprot_id']} ({prot['pdb_id']}): "
              f"{prot['n_actives']} actives, "
              f"quality={prot['quality_score']:.0f}, "
              f"pchembl_coverage={prot['pchembl_coverage']:.2f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(proteins, f, indent=2)
    print(f"Written to {args.output}")


if __name__ == '__main__':
    main()
