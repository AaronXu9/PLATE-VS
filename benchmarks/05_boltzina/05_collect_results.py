# benchmarks/05_boltzina/05_collect_results.py
"""Stage 05: Parse boltzina CSVs → compute metrics → write training_summary.json."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from metrics import (
    parse_boltzina_csv, compute_vs_metrics, aggregate_results,
    write_training_summary, SCORE_CONFIGS,
)


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
    per_protein_metrics = []
    per_protein_rows = []
    t0 = time.time()
    total_n = 0

    for protein in proteins:
        uid = protein['uniprot_id']
        csv_path = results_dir / 'raw_results' / uid / 'boltzina_results.csv'
        if not csv_path.exists():
            print(f'  [missing] {uid}: no boltzina_results.csv, skipping')
            continue

        df = parse_boltzina_csv(str(csv_path), affinity_col=args.affinity_col)
        total_n += len(df)

        # Build pchembl array aligned with rows
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
        labels = df['is_active'].tolist()
        pchembl_full = [None] * len(df)
        for j, (_, row) in enumerate(df.iterrows()):
            if row['is_active'] == 1:
                cid = Path(row['ligand_file']).stem
                pchembl_full[j] = pchembl_map.get(cid)

        # Evaluate each score independently
        row_metrics = {
            'uniprot_id': uid,
            'n_actives': int(df['is_active'].sum()),
            'n_decoys': int((df['is_active'] == 0).sum()),
        }
        for score_name, csv_col, negate in SCORE_CONFIGS:
            if csv_col not in df.columns or df[csv_col].isna().all():
                continue
            raw_scores = df[csv_col].astype(float).tolist()
            scores = [-s for s in raw_scores] if negate else raw_scores
            m = compute_vs_metrics(scores, labels, pchembl_values=pchembl_full)
            for mk, mv in m.items():
                row_metrics[f'{score_name}_{mk}'] = mv

        per_protein_metrics.append(row_metrics)
        per_protein_rows.append(row_metrics)

        # Print summary line
        aff_auc = row_metrics.get('affinity_pred_value_roc_auc', '—')
        dock_auc = row_metrics.get('docking_score_roc_auc', '—')
        print(f'  {uid}: affinity_auc={aff_auc}, docking_auc={dock_auc}')

    elapsed = time.time() - t0

    # Aggregate (only metric columns, not uid/n_actives/n_decoys)
    metric_only = [
        {k: v for k, v in m.items() if k not in ('uniprot_id', 'n_actives', 'n_decoys')}
        for m in per_protein_metrics
    ]
    agg = aggregate_results(metric_only)

    print(f'\nAggregate ({len(per_protein_metrics)} proteins):')
    for score_name, _, _ in SCORE_CONFIGS:
        auc = agg.get(f'{score_name}_roc_auc', '—')
        ef1 = agg.get(f'{score_name}_ef1pct', '—')
        ef5 = agg.get(f'{score_name}_ef5pct', '—')
        print(f'  {score_name}: ROC-AUC={auc}, EF1%={ef1}, EF5%={ef5}')

    summary_path = results_dir / 'boltzina_training_summary.json'
    write_training_summary(agg, str(summary_path), n_test=total_n, elapsed_s=elapsed)
    print(f'Written: {summary_path}')

    per_protein_csv = results_dir / 'boltzina_per_protein_results.csv'
    pd.DataFrame(per_protein_rows).to_csv(per_protein_csv, index=False)
    print(f'Written: {per_protein_csv}')


if __name__ == '__main__':
    main()
