# benchmarks/05_boltzina/lib/protein_select.py
import pandas as pd


def select_proteins(df, n=10, min_actives=50, min_pchembl_coverage=0.80):
    """Select top-n test proteins from registry DataFrame.

    Args:
        df: Full registry DataFrame (all rows, all columns).
        n: Number of proteins to return.
        min_actives: Minimum active compounds required per protein.
        min_pchembl_coverage: Minimum fraction of actives with pchembl value.

    Returns:
        List of dicts, sorted by quality_score DESC, each with keys:
        uniprot_id, pdb_id, cif_path, n_actives, n_decoys_to_sample,
        pchembl_coverage, quality_score.
    """
    test = df[
        (df['similarity_threshold'] == '0p7') &
        (df['protein_partition'] == 'test') &
        (df['is_active'] == True)
    ].copy()

    stats = test.groupby('uniprot_id').agg(
        pdb_id=('pdb_id', 'first'),
        cif_path=('cif_path', 'first'),
        quality_score=('quality_score', 'first'),
        n_actives=('is_active', 'count'),
        pchembl_coverage=('pchembl', lambda x: x.notna().mean()),
    ).reset_index()

    filtered = stats[
        (stats['n_actives'] >= min_actives) &
        (stats['pchembl_coverage'] >= min_pchembl_coverage)
    ]

    top = filtered.sort_values('quality_score', ascending=False).head(n)

    result = []
    for _, row in top.iterrows():
        n_actives = int(row['n_actives'])
        result.append({
            'uniprot_id': row['uniprot_id'],
            'pdb_id': row['pdb_id'],
            'cif_path': str(row['cif_path']),
            'n_actives': n_actives,
            'n_decoys_to_sample': min(n_actives * 50, 2500),
            'pchembl_coverage': round(float(row['pchembl_coverage']), 3),
            'quality_score': float(row['quality_score']),
        })
    return result
