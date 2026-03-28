# benchmarks/01_preprocessing/enrich_pchembl.py
"""
Enrich registry_soft_split.csv with pChEMBL values from filtered_chembl_affinity.parquet.

Source:    data/filtered_chembl_affinity.parquet — per-protein quality-filtered ChEMBL data
           (standard_flag=1, no duplicates, valid measurements only; all rows have pchembl_value).

Join key:  (registry.uniprot_id, registry.smiles_canonical)
           <-> (activities.source_uniprot_id, activities.canonical_smiles)

Strategy:  median pChEMBL per (protein, compound) pair across all binding assays.
           Protein-consistent: a compound's IC50 against protein A is never assigned to
           a registry entry for protein B.

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

_FILTERED_COLS = ['source_uniprot_id', 'canonical_smiles', 'pchembl_value',
                  'assay_type', 'standard_type']


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

    Uses the protein-specific source_uniprot_id from filtered_chembl_affinity.parquet
    so that pChEMBL values are only assigned to registry entries for the matching protein.

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

    # Build protein-consistent (uniprot_id, canonical_smiles) -> median pChEMBL map
    logging.info('Building (protein, SMILES) -> pChEMBL map...')
    protein_pchembl = build_protein_pchembl_map(activities, **filter_kwargs)
    logging.info(f'  {len(protein_pchembl):,} (protein, SMILES) pairs with valid pChEMBL')

    # Canonicalize unique registry SMILES (ChEMBL rows only — decoys have no measurements)
    chembl_mask = registry['source'] == 'chembl'
    reg_chembl = registry.loc[chembl_mask, ['uniprot_id', 'smiles']].copy()
    unique_smiles = reg_chembl['smiles'].dropna().unique()
    logging.info(f'  Canonicalizing {len(unique_smiles):,} unique registry SMILES...')

    canon_series = _canonicalize_smiles_series(pd.Series(unique_smiles))
    canon_map = dict(zip(unique_smiles, canon_series))
    n_failed = sum(1 for v in canon_map.values() if v is None)
    if n_failed:
        logging.warning(f'  {n_failed:,} SMILES failed RDKit canonicalization — those rows will have pchembl=NaN')

    # Join: look up pChEMBL for each (uniprot_id, canonical_smiles) pair
    reg_chembl = reg_chembl.copy()
    reg_chembl['canon_smiles'] = reg_chembl['smiles'].map(canon_map)
    reg_chembl['pchembl'] = [
        protein_pchembl.get((uid, csmi), np.nan)
        for uid, csmi in zip(reg_chembl['uniprot_id'], reg_chembl['canon_smiles'])
    ]

    enriched = registry.copy()
    enriched['pchembl'] = np.nan
    enriched.loc[chembl_mask, 'pchembl'] = reg_chembl['pchembl'].values

    n_enriched = int(enriched['pchembl'].notna().sum())
    pchembl_vals = pd.to_numeric(enriched['pchembl'], errors='coerce').dropna()
    logging.info(f'  {n_enriched:,} rows enriched ({n_enriched/len(enriched)*100:.1f}%)')
    if n_enriched > 0:
        logging.info(f'  pChEMBL range: {pchembl_vals.min():.2f} – {pchembl_vals.max():.2f}')
        logging.info(f'  pChEMBL mean:  {pchembl_vals.mean():.2f}')
        # Protein coverage
        enriched_proteins = enriched.loc[enriched['pchembl'].notna(), 'uniprot_id'].nunique()
        logging.info(f'  Proteins with ≥1 enriched compound: {enriched_proteins}')

    logging.info(f'Writing to {args.output}')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    logging.info('Done.')


if __name__ == '__main__':
    main()
