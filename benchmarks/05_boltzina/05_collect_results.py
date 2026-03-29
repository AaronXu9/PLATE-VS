# benchmarks/05_boltzina/05_collect_results.py
"""Stage 05: Parse boltzina CSVs → compute metrics → write training_summary.json."""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from metrics import parse_boltzina_csv, compute_vs_metrics, aggregate_results, write_training_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poc-proteins', required=True)
    p.add_argument('--registry', required=True)
    p.add_argument('--results-dir', required=True)
    p.add_argument('--affinity-col', default=None,
                   help='Name of boltz affinity column in results CSV (auto-detected if omitted)')
    args = p.parse_args()

    with open(args.poc_proteins) as f:
        proteins = json.load(f)

    # Load pchembl values for joining
    registry_df = pd.read_csv(
        args.registry,
        usecols=['uniprot_id', 'compound_id', 'pchembl', 'similarity_threshold',
                 'protein_partition', 'split', 'is_active'],
    )

    results_dir = Path(args.results_dir)
    per_protein = []
    per_protein_rows = []
    t0 = time.time()
    total_n = 0

    for protein in proteins:
        uid = protein['uniprot_id']
        csv_path = results_dir / 'raw_results' / uid / 'results.csv'
        if not csv_path.exists():
            print(f'  [missing] {uid}: no results.csv, skipping')
            continue

        df = parse_boltzina_csv(str(csv_path), affinity_col=args.affinity_col)
        total_n += len(df)

        # Build pchembl array aligned with actives
        active_df = df[df['is_active'] == 1].copy()
        active_df['compound_id'] = (
            active_df['ligand_file']
            .apply(lambda x: Path(x).stem)
        )
        pchembl_map = (
            registry_df[
                (registry_df['uniprot_id'] == uid) &
                (registry_df['is_active'] == True) &
                (registry_df['similarity_threshold'] == '0p7')
            ]
            .drop_duplicates('compound_id')
            .set_index('compound_id')['pchembl']
            .to_dict()
        )

        # Combine for metrics
        scores = df['boltz_affinity'].tolist()
        labels = df['is_active'].tolist()
        pchembl_full = [None] * len(df)
        for j, (idx, row) in enumerate(df.iterrows()):
            if row['is_active'] == 1:
                cid = Path(row['ligand_file']).stem
                pchembl_full[j] = pchembl_map.get(cid)

        metrics = compute_vs_metrics(scores, labels, pchembl_values=pchembl_full)
        per_protein.append(metrics)
        per_protein_rows.append({
            'uniprot_id': uid,
            'n_actives': int(df['is_active'].sum()),
            'n_decoys': int((df['is_active'] == 0).sum()),
            **metrics,
        })
        print(f'  {uid}: roc_auc={metrics["roc_auc"]}, '
              f'ef1pct={metrics["ef1pct"]}, spearman_r={metrics["spearman_r"]}')

    elapsed = time.time() - t0
    agg = aggregate_results(per_protein)
    print(f'\nAggregate ({len(per_protein)} proteins):')
    print(f'  ROC-AUC={agg["roc_auc"]}, EF1%={agg["ef1pct"]}, Spearman r={agg["spearman_r"]}')

    summary_path = results_dir / 'boltzina_training_summary.json'
    write_training_summary(agg, str(summary_path), n_test=total_n, elapsed_s=elapsed)
    print(f'Written: {summary_path}')

    per_protein_csv = results_dir / 'boltzina_per_protein_results.csv'
    pd.DataFrame(per_protein_rows).to_csv(per_protein_csv, index=False)
    print(f'Written: {per_protein_csv}')


if __name__ == '__main__':
    main()
