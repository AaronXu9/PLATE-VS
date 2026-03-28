# benchmarks/01_preprocessing/enrich_pchembl.py
"""
Enrich registry_soft_split.csv with pChEMBL values and assay metadata from
filtered_chembl_affinity.parquet.

Source:    data/filtered_chembl_affinity.parquet — per-protein quality-filtered ChEMBL data
           (standard_flag=1, no duplicates, valid measurements only; all rows have pchembl_value).

Join key:  (registry.uniprot_id, registry.smiles_canonical)
           <-> (activities.source_uniprot_id, activities.canonical_smiles)

Strategy:  For each (protein, compound) pair across all binding assays:
           - pchembl      : median pChEMBL (log-space aggregation)
           - affinity_value: back-calculated as 10^(9 - pchembl_median) [nM]
                            Guaranteed consistent with pchembl: pchembl = 9 - log10(affinity_value)
                            For single-measurement rows this equals the original value exactly.
           - affinity_type: mode of standard_type (IC50, Ki, Kd, …)
           - assay_type   : mode of assay_type (B=binding, F=functional, …)
           - document_year: most recent publication year
           - n_measurements: number of raw measurements aggregated (1 → original value kept)

Output:    new CSV at --output path (registry_soft_split.csv unchanged).
"""
import argparse
import logging
from pathlib import Path

import ctypes as _ctypes
try:
    _ctypes.CDLL('libstdc++.so.6')  # ensure conda's libstdc++ is loaded before rdkit
except OSError:
    pass

import numpy as np
import pandas as pd

VALID_ASSAY_TYPES: set = {'B'}
VALID_STANDARD_TYPES: set = {'IC50', 'Ki', 'Kd', 'EC50', 'Potency'}
PCHEMBL_MIN: float = 4.0
PCHEMBL_MAX: float = 12.0

_FILTERED_COLS = [
    'source_uniprot_id', 'canonical_smiles', 'pchembl_value',
    'assay_type', 'standard_type', 'standard_value', 'document_year',
]


def _canonicalize_smiles_series(smiles: pd.Series) -> pd.Series:
    """Canonicalize a Series of SMILES strings using RDKit. Returns a Series of same length."""
    from rdkit import Chem

    def _canon(smi):
        if not isinstance(smi, str):
            return None
        try:
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol) if mol is not None else None
        except Exception:
            return None

    return smiles.map(_canon)


def build_protein_pchembl_map(
    activities: pd.DataFrame,
    assay_types: set = VALID_ASSAY_TYPES,
    standard_types: set = VALID_STANDARD_TYPES,
    pchembl_min: float = PCHEMBL_MIN,
    pchembl_max: float = PCHEMBL_MAX,
) -> pd.Series:
    """
    Build a (source_uniprot_id, canonical_smiles) -> median_pChEMBL mapping.

    Returns a Series with a MultiIndex (source_uniprot_id, canonical_smiles).
    """
    if activities.empty:
        return pd.Series(dtype=float)

    pchembl_numeric = pd.to_numeric(activities['pchembl_value'], errors='coerce')
    mask = (
        activities['assay_type'].isin(assay_types)
        & activities['standard_type'].isin(standard_types)
        & pchembl_numeric.notna()
        & (pchembl_numeric >= pchembl_min)
        & (pchembl_numeric <= pchembl_max)
        & activities['canonical_smiles'].notna()
        & activities['source_uniprot_id'].notna()
    )
    filtered = activities.loc[mask].copy()
    filtered['pchembl_value'] = pchembl_numeric[mask]
    if filtered.empty:
        return pd.Series(dtype=float)

    return filtered.groupby(
        ['source_uniprot_id', 'canonical_smiles']
    )['pchembl_value'].median()


def build_protein_assay_metadata(
    activities: pd.DataFrame,
    assay_types: set = VALID_ASSAY_TYPES,
    standard_types: set = VALID_STANDARD_TYPES,
    pchembl_min: float = PCHEMBL_MIN,
    pchembl_max: float = PCHEMBL_MAX,
) -> pd.DataFrame:
    """
    Build a per-(protein, compound) metadata table for assay-level columns.

    Returns a DataFrame indexed by (source_uniprot_id, canonical_smiles) with columns:
      - pchembl          : median pChEMBL
      - affinity_value   : 10^(9 - pchembl), guaranteed consistent with pchembl
      - affinity_type    : mode of standard_type (e.g. IC50, Ki)
      - assay_type       : mode of assay_type (e.g. B, F)
      - document_year    : most recent publication year
      - n_measurements   : number of raw measurements aggregated
    """
    if activities.empty:
        cols = ['pchembl', 'affinity_value', 'affinity_type',
                'assay_type', 'document_year', 'n_measurements']
        return pd.DataFrame(columns=cols)

    needed = ['source_uniprot_id', 'canonical_smiles', 'pchembl_value',
              'assay_type', 'standard_type', 'document_year']
    act = activities[needed].copy()
    pchembl_numeric = pd.to_numeric(act['pchembl_value'], errors='coerce')
    mask = (
        act['assay_type'].isin(assay_types)
        & act['standard_type'].isin(standard_types)
        & pchembl_numeric.notna()
        & (pchembl_numeric >= pchembl_min)
        & (pchembl_numeric <= pchembl_max)
        & act['canonical_smiles'].notna()
        & act['source_uniprot_id'].notna()
    )
    act = act.loc[mask].copy()
    act['pchembl_value'] = pchembl_numeric[mask]

    if act.empty:
        cols = ['pchembl', 'affinity_value', 'affinity_type',
                'assay_type', 'document_year', 'n_measurements']
        return pd.DataFrame(columns=cols)

    grp = act.groupby(['source_uniprot_id', 'canonical_smiles'])

    # pchembl: median in log space
    pchembl_med = grp['pchembl_value'].median().rename('pchembl')

    # affinity_value: back-calculated from pchembl — guaranteed consistent
    # For single-measurement rows: equals the original measured value exactly
    # For multi-measurement rows: geometric mean (correct for log-normal distributions)
    affinity_val = (10 ** (9 - pchembl_med)).rename('affinity_value').round(4)

    # affinity_type: most frequent standard_type
    affinity_type = grp['standard_type'].agg(
        lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan
    ).rename('affinity_type')

    # assay_type: most frequent assay_type
    assay_type_mode = grp['assay_type'].agg(
        lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan
    ).rename('assay_type_agg')

    # document_year: most recent publication
    doc_year = grp['document_year'].max().rename('document_year')

    # n_measurements: how many rows were aggregated
    n_meas = grp.size().rename('n_measurements')

    meta = pd.concat(
        [pchembl_med, affinity_val, affinity_type, assay_type_mode, doc_year, n_meas],
        axis=1,
    )
    meta.columns = ['pchembl', 'affinity_value', 'affinity_type',
                    'assay_type_agg', 'document_year', 'n_measurements']
    return meta


def aggregate_pchembl(
    activities: pd.DataFrame,
    assay_types: set = VALID_ASSAY_TYPES,
    standard_types: set = VALID_STANDARD_TYPES,
    pchembl_min: float = PCHEMBL_MIN,
    pchembl_max: float = PCHEMBL_MAX,
) -> pd.Series:
    """
    Filter activities to binding assays with valid pChEMBL, then aggregate
    to one median pChEMBL per molecule_chembl_id.

    Returns a Series indexed by molecule_chembl_id.
    """
    if activities.empty:
        return pd.Series(dtype=float)

    pchembl_numeric = pd.to_numeric(activities['pchembl_value'], errors='coerce')
    mask = (
        activities['assay_type'].isin(assay_types)
        & activities['standard_type'].isin(standard_types)
        & pchembl_numeric.notna()
        & (pchembl_numeric >= pchembl_min)
        & (pchembl_numeric <= pchembl_max)
    )
    filtered = activities.loc[mask].copy()
    filtered['pchembl_value'] = pchembl_numeric[mask]
    if filtered.empty:
        return pd.Series(dtype=float)

    return filtered.groupby('molecule_chembl_id')['pchembl_value'].median()


def enrich_registry(registry: pd.DataFrame, agg: pd.Series) -> pd.DataFrame:
    """
    Map aggregated pChEMBL onto registry rows via compound_id.
    Always overwrites the pchembl column.
    Returns a new DataFrame (does not mutate registry).
    """
    out = registry.copy()
    out['pchembl'] = out['compound_id'].map(agg).astype(float)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description='Enrich soft-split registry with pChEMBL values and assay metadata')
    parser.add_argument('--registry', required=True,
                        help='Path to registry_soft_split.csv')
    parser.add_argument('--activities', required=True,
                        help='Path to filtered_chembl_affinity.parquet (has source_uniprot_id)')
    parser.add_argument('--output', required=True,
                        help='Output path for enriched registry CSV')
    parser.add_argument('--assay-types', nargs='+', default=list(VALID_ASSAY_TYPES))
    parser.add_argument('--standard-types', nargs='+', default=list(VALID_STANDARD_TYPES))
    parser.add_argument('--pchembl-min', type=float, default=PCHEMBL_MIN)
    parser.add_argument('--pchembl-max', type=float, default=PCHEMBL_MAX)
    args = parser.parse_args()

    filter_kwargs = dict(
        assay_types=set(args.assay_types),
        standard_types=set(args.standard_types),
        pchembl_min=args.pchembl_min,
        pchembl_max=args.pchembl_max,
    )

    logging.info(f'Loading registry: {args.registry}')
    registry = pd.read_csv(args.registry, low_memory=False)
    logging.info(f'  {len(registry):,} rows, {registry["uniprot_id"].nunique():,} unique proteins')

    logging.info(f'Loading activities: {args.activities}')
    activities = pd.read_parquet(args.activities, columns=_FILTERED_COLS)
    logging.info(f'  {len(activities):,} rows, {activities["source_uniprot_id"].nunique():,} proteins')

    # Build protein-consistent (uniprot_id, canonical_smiles) -> metadata table
    logging.info('Building (protein, SMILES) -> pChEMBL + assay metadata...')
    meta = build_protein_assay_metadata(activities, **filter_kwargs)
    logging.info(f'  {len(meta):,} (protein, SMILES) pairs with valid measurements')

    # Canonicalize unique registry SMILES (ChEMBL rows only — decoys have no measurements)
    chembl_mask = registry['source'] == 'chembl'
    reg_chembl = registry.loc[chembl_mask, ['uniprot_id', 'smiles']].copy()
    unique_smiles = reg_chembl['smiles'].dropna().unique()
    logging.info(f'  Canonicalizing {len(unique_smiles):,} unique registry SMILES...')

    canon_series = _canonicalize_smiles_series(pd.Series(unique_smiles))
    canon_map = dict(zip(unique_smiles, canon_series))
    n_failed = sum(1 for v in canon_map.values() if v is None)
    if n_failed:
        logging.warning(f'  {n_failed:,} SMILES failed RDKit canonicalization — those rows will have NaN metadata')

    # Join: look up metadata for each (uniprot_id, canonical_smiles) pair
    reg_chembl = reg_chembl.copy()
    reg_chembl['canon_smiles'] = reg_chembl['smiles'].map(canon_map)

    def _lookup(uid, csmi):
        if csmi is None or pd.isna(csmi):
            return (np.nan,) * 6
        key = (uid, csmi)
        if key in meta.index:
            row = meta.loc[key]
            return (row['pchembl'], row['affinity_value'], row['affinity_type'],
                    row['assay_type_agg'], row['document_year'], row['n_measurements'])
        return (np.nan,) * 6

    looked_up = [
        _lookup(uid, csmi)
        for uid, csmi in zip(reg_chembl['uniprot_id'], reg_chembl['canon_smiles'])
    ]
    lu_df = pd.DataFrame(
        looked_up,
        columns=['pchembl', 'affinity_value', 'affinity_type',
                 'assay_type_agg', 'document_year', 'n_measurements'],
        index=reg_chembl.index,
    )

    enriched = registry.copy()

    # Columns that already exist in registry — overwrite
    for col in ['pchembl', 'affinity_value', 'affinity_type']:
        enriched[col] = np.nan
        enriched.loc[chembl_mask, col] = lu_df[col]

    # New columns — add after existing schema
    for col in ['assay_type_agg', 'document_year', 'n_measurements']:
        enriched[col] = np.nan
        enriched.loc[chembl_mask, col] = lu_df[col]

    # Rename assay_type_agg to match schema intent
    enriched = enriched.rename(columns={'assay_type_agg': 'assay_type_enriched'})

    # Consistency check: verify pchembl = 9 - log10(affinity_value) for all enriched rows
    enriched_rows = enriched['pchembl'].notna() & enriched['affinity_value'].notna()
    pchembl_check = 9 - np.log10(enriched.loc[enriched_rows, 'affinity_value'].astype(float))
    max_diff = (enriched.loc[enriched_rows, 'pchembl'].astype(float) - pchembl_check).abs().max()
    if max_diff > 1e-6:
        logging.warning(f'  Consistency check: max |pchembl - (9 - log10(affinity_value))| = {max_diff:.2e} (expected ~0)')
    else:
        logging.info(f'  Consistency check PASSED: pchembl and affinity_value are fully consistent (max diff={max_diff:.2e})')

    n_enriched = int(enriched['pchembl'].notna().sum())
    pchembl_vals = pd.to_numeric(enriched['pchembl'], errors='coerce').dropna()
    logging.info(f'  {n_enriched:,} rows enriched ({n_enriched/len(enriched)*100:.1f}%)')
    if n_enriched > 0:
        logging.info(f'  pChEMBL range: {pchembl_vals.min():.2f} – {pchembl_vals.max():.2f}')
        logging.info(f'  pChEMBL mean:  {pchembl_vals.mean():.2f}')
        enriched_proteins = enriched.loc[enriched['pchembl'].notna(), 'uniprot_id'].nunique()
        logging.info(f'  Proteins with ≥1 enriched compound: {enriched_proteins}')
        logging.info(f'  affinity_type breakdown:\n{enriched["affinity_type"].value_counts().to_string()}')
        logging.info(f'  n_measurements stats: {enriched["n_measurements"].describe().to_dict()}')

    logging.info(f'Writing to {args.output}')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    logging.info('Done.')


if __name__ == '__main__':
    main()
