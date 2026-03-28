# benchmarks/01_preprocessing/enrich_pchembl.py
"""
Enrich registry_soft_split.csv with pChEMBL values from chembl_activities_enriched.parquet.

Primary join key:  registry.compound_id  <->  activities.molecule_chembl_id
Fallback join key: registry.smiles (canonicalized) <-> activities.canonical_smiles
                   used when compound_id is null (as in registry_soft_split.csv).

Strategy:  median pChEMBL per compound across binding assays with valid measurements.

Output:    new CSV at --output path (registry_soft_split.csv unchanged unless output
           path is the same).
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

_ACTIVITIES_COLS = ['molecule_chembl_id', 'pchembl_value', 'assay_type', 'standard_type']
_ACTIVITIES_COLS_SMILES = ['molecule_chembl_id', 'canonical_smiles', 'pchembl_value', 'assay_type', 'standard_type']


def _canonicalize_smiles_series(smiles: pd.Series, n_jobs: int = 4) -> pd.Series:
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


def build_smiles_to_pchembl(
    activities: pd.DataFrame,
    assay_types: set = VALID_ASSAY_TYPES,
    standard_types: set = VALID_STANDARD_TYPES,
    pchembl_min: float = PCHEMBL_MIN,
    pchembl_max: float = PCHEMBL_MAX,
) -> pd.Series:
    """
    Build a canonical_smiles -> median_pChEMBL mapping from activities.
    Uses canonical_smiles as the join key instead of molecule_chembl_id.

    Returns a Series indexed by canonical_smiles.
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
    )
    filtered = activities.loc[mask].copy()
    filtered['pchembl_value'] = pchembl_numeric[mask]
    if filtered.empty:
        return pd.Series(dtype=float)

    return filtered.groupby('canonical_smiles')['pchembl_value'].median()


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
    parser = argparse.ArgumentParser(description='Enrich soft-split registry with pChEMBL values')
    parser.add_argument('--registry', required=True, help='Path to registry_soft_split.csv')
    parser.add_argument('--activities', required=True, help='Path to chembl_activities_enriched.parquet')
    parser.add_argument('--output', required=True, help='Output path for enriched registry CSV')
    parser.add_argument('--assay-types', nargs='+', default=list(VALID_ASSAY_TYPES))
    parser.add_argument('--standard-types', nargs='+', default=list(VALID_STANDARD_TYPES))
    parser.add_argument('--pchembl-min', type=float, default=PCHEMBL_MIN)
    parser.add_argument('--pchembl-max', type=float, default=PCHEMBL_MAX)
    parser.add_argument('--n-jobs', type=int, default=4, help='Parallel workers for SMILES canonicalization')
    args = parser.parse_args()

    filter_kwargs = dict(
        assay_types=set(args.assay_types),
        standard_types=set(args.standard_types),
        pchembl_min=args.pchembl_min,
        pchembl_max=args.pchembl_max,
    )

    logging.info(f'Loading registry: {args.registry}')
    registry = pd.read_csv(args.registry, low_memory=False)
    logging.info(f'  {len(registry):,} rows, {registry["compound_id"].nunique():,} unique compounds')

    # Determine join strategy: use compound_id only when CHEMBL IDs are present
    non_null_ids = registry['compound_id'].dropna()
    chembl_id_frac = non_null_ids.str.startswith('CHEMBL').mean() if len(non_null_ids) > 0 else 0.0
    use_smiles_join = chembl_id_frac < 0.5
    if use_smiles_join:
        logging.info(f'  compound_id contains no CHEMBL IDs ({chembl_id_frac*100:.0f}% start with CHEMBL) — using SMILES-based join')
    else:
        logging.info(f'  Using compound_id join ({chembl_id_frac*100:.0f}% of IDs are CHEMBL IDs)')

    logging.info(f'Loading activities: {args.activities}')
    if use_smiles_join:
        activities = pd.read_parquet(args.activities, columns=_ACTIVITIES_COLS_SMILES)
    else:
        activities = pd.read_parquet(args.activities, columns=_ACTIVITIES_COLS)
    logging.info(f'  {len(activities):,} rows')

    if use_smiles_join:
        logging.info('Building canonical_smiles -> pChEMBL mapping...')
        smiles_to_pchembl = build_smiles_to_pchembl(activities, **filter_kwargs)
        logging.info(f'  {len(smiles_to_pchembl):,} unique canonical SMILES with pChEMBL')

        # Canonicalize registry SMILES for the chembl rows only (decoys have no SMILES to enrich)
        chembl_mask = registry['source'] == 'chembl'
        reg_chembl_smiles = registry.loc[chembl_mask, 'smiles'].copy()
        unique_smiles = reg_chembl_smiles.dropna().unique()
        logging.info(f'  Canonicalizing {len(unique_smiles):,} unique registry SMILES (n_jobs={args.n_jobs})...')

        canon_map = {}
        if len(unique_smiles) > 0:
            canon_series = _canonicalize_smiles_series(pd.Series(unique_smiles), n_jobs=args.n_jobs)
            canon_map = dict(zip(unique_smiles, canon_series))
            n_failed = sum(1 for v in canon_map.values() if v is None)
            if n_failed:
                logging.warning(f'  {n_failed:,} SMILES failed RDKit canonicalization — those rows will have pchembl=NaN')

        # Map: registry smiles -> canonical -> pchembl
        reg_canon = reg_chembl_smiles.map(canon_map)
        pchembl_values = reg_canon.map(smiles_to_pchembl).astype(float)

        enriched = registry.copy()
        enriched['pchembl'] = np.nan
        enriched.loc[chembl_mask, 'pchembl'] = pchembl_values

    else:
        logging.info('Aggregating pChEMBL...')
        agg = aggregate_pchembl(activities, **filter_kwargs)
        logging.info(f'  {len(agg):,} compounds with pChEMBL after filtering')
        enriched = enrich_registry(registry, agg)

    n_enriched = enriched['pchembl'].notna().sum()
    logging.info(f'  {n_enriched:,} rows enriched ({n_enriched/len(enriched)*100:.1f}%)')
    if n_enriched > 0:
        logging.info(f'  pChEMBL range: {enriched["pchembl"].min():.2f} – {enriched["pchembl"].max():.2f}')

    logging.info(f'Writing to {args.output}')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    logging.info('Done.')


if __name__ == '__main__':
    main()
