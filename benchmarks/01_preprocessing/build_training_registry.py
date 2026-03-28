#!/usr/bin/env python3
"""
Create a flexible training data registry for the plate-vs dataset.

Dataset Structure:
plate-vs/VLS_benchmark/chembl_affinity/
└── uniprot_{ID}/
    ├── {ID}_active.sdf                                    # All actives
    ├── {ID}_active.pkl                                    # SMILES only
    ├── {ID}_chembl_activities_filtered.parquet            # Affinity values
    ├── sdf_filtered_by_ligand_similarity/
    │   ├── {ID}_active_0p3.sdf                           # Train split (0.3 similarity)
    │   ├── {ID}_testing_0p3.sdf                          # Test split
    │   ├── {ID}_active_0p7.sdf                           # Train split (0.7 similarity)
    │   └── {ID}_testing_0p7.sdf                          # Test split
    └── deepcoy_output/
        └── decoys...

Registry tracks all data locations and supports:
- Multiple similarity-based splits
- Active compounds with affinities
- Decoy compounds (negative samples)
- Both SDF (3D) and SMILES formats
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from tqdm import tqdm
import duckdb
from rdkit import Chem


@dataclass
class CompoundReference:
    """Reference to a single compound with all available data paths."""
    compound_id: str
    uniprot_id: str
    
    # Ligand data
    smiles: str
    sdf_path: Optional[str] = None
    
    # Label and affinity
    is_active: bool = True
    affinity_value: Optional[float] = None
    affinity_type: Optional[str] = None
    affinity_unit: Optional[str] = 'nM'
    
    # Data split info
    similarity_threshold: Optional[str] = None  # e.g., "0p3", "0p7"
    split: str = 'train'  # 'train', 'test', 'decoy'
    
    # Source info
    source: str = 'chembl'  # 'chembl', 'deepcoy'


@dataclass
class ProteinReference:
    """Reference to protein structure and features."""
    uniprot_id: str
    pdb_id: str
    
    # Protein data paths
    cif_path: str
    sequence: str
    
    # Structure quality
    resolution: float
    quality_score: float
    method: str
    
    # Pocket info
    chosen_ligand: str
    pocket_residue_count: int
    pocket_completeness: float


class RegistryBuilder:
    """Build training data registry from plate-vs dataset."""
    
    def __init__(
        self,
        affinity_base_dir: str,
        structures_csv: str,
        output_dir: str = "training_data"
    ):
        self.affinity_base = Path(affinity_base_dir)
        self.structures_df = pd.read_csv(structures_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Filter valid structures
        self.structures_df = self.structures_df[
            (self.structures_df['error'].fillna('') == '') &
            (self.structures_df['quality_score'].astype(float) > 250)
        ]
        
    def build_registry(
        self,
        similarity_thresholds: List[str] = ['0p3', '0p5', '0p7'],
        include_decoys: bool = True,
        max_uniprots: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Build complete registry by scanning plate-vs structure.
        
        Args:
            similarity_thresholds: List of similarity thresholds to include
            include_decoys: Whether to include decoy compounds
            max_uniprots: Limit number of UniProts (for testing)
        
        Returns:
            DataFrame with registry entries
        """
        all_entries = []
        stats = {
            'total_uniprots': 0,
            'success': 0,
            'no_affinity': 0,
            'no_splits': 0,
            'total_actives': 0,
            'total_decoys': 0,
        }
        
        structures = self.structures_df.head(max_uniprots) if max_uniprots else self.structures_df
        
        print(f"Building registry for {len(structures)} UniProts...")
        print(f"Similarity thresholds: {similarity_thresholds}")
        print(f"Include decoys: {include_decoys}\n")
        
        for idx, struct_row in tqdm(structures.iterrows(), total=len(structures), desc="Processing"):
            uniprot_id = struct_row['uniprot_id']
            stats['total_uniprots'] += 1
            
            uniprot_dir = self.affinity_base / f"uniprot_{uniprot_id}"
            if not uniprot_dir.exists():
                stats['no_affinity'] += 1
                continue
            
            # Get protein reference
            protein_ref = self._make_protein_reference(struct_row)
            
            # Load affinity data
            affinity_df = self._load_affinity_data(uniprot_id)
            if affinity_df is None:
                stats['no_affinity'] += 1
                continue
            
            # Process each similarity threshold
            for sim_thresh in similarity_thresholds:
                # Process actives (train)
                train_entries = self._process_split(
                    uniprot_id, protein_ref, affinity_df,
                    similarity_threshold=sim_thresh,
                    split_type='train'
                )
                all_entries.extend(train_entries)
                stats['total_actives'] += len(train_entries)
                
                # Process actives (test)
                test_entries = self._process_split(
                    uniprot_id, protein_ref, affinity_df,
                    similarity_threshold=sim_thresh,
                    split_type='test'
                )
                all_entries.extend(test_entries)
                stats['total_actives'] += len(test_entries)
            
            # Process decoys (if available)
            if include_decoys:
                decoy_entries = self._process_decoys(uniprot_id, protein_ref)
                all_entries.extend(decoy_entries)
                stats['total_decoys'] += len(decoy_entries)
            
            if train_entries or test_entries or decoy_entries:
                stats['success'] += 1
        
        # Create DataFrame
        registry_df = pd.DataFrame([e for e in all_entries])
        
        # Save registry
        registry_path = self.output_dir / "registry.csv"
        registry_df.to_csv(registry_path, index=False)
        
        # Save protein references
        protein_refs = {}
        for struct_row in structures.iterrows():
            uniprot_id = struct_row[1]['uniprot_id']
            protein_refs[uniprot_id] = self._make_protein_reference(struct_row[1])
        
        protein_refs_path = self.output_dir / "protein_references.json"
        with open(protein_refs_path, 'w') as f:
            json.dump(
                {k: asdict(v) for k, v in protein_refs.items()},
                f, indent=2
            )
        
        # Print statistics
        self._print_stats(stats, registry_df)
        
        return registry_df
    
    def _make_protein_reference(self, struct_row: pd.Series) -> ProteinReference:
        """Create protein reference from structure row."""
        return ProteinReference(
            uniprot_id=struct_row['uniprot_id'],
            pdb_id=struct_row['pdb_id'],
            cif_path=struct_row['cif_path'],
            sequence='',  # Will be loaded on demand
            resolution=float(struct_row['resolution']) if struct_row['resolution'] else 0.0,
            quality_score=float(struct_row['quality_score']),
            method=struct_row['method'],
            chosen_ligand=struct_row['chosen_ligand'],
            pocket_residue_count=int(struct_row['pocket_residue_count']),
            pocket_completeness=float(struct_row['pocket_completeness']),
        )
    
    def _load_affinity_data(self, uniprot_id: str) -> Optional[pd.DataFrame]:
        """Load affinity data for a UniProt."""
        affinity_path = self.affinity_base / f"uniprot_{uniprot_id}" / f"{uniprot_id}_chembl_activities_filtered.parquet"
        
        if not affinity_path.exists():
            return None
        
        try:
            df = duckdb.read_parquet(str(affinity_path)).df()
            return df
        except:
            return None
    
    def _process_split(
        self,
        uniprot_id: str,
        protein_ref: ProteinReference,
        affinity_df: pd.DataFrame,
        similarity_threshold: str,
        split_type: str  # 'train' or 'test'
    ) -> List[Dict]:
        """Process a specific data split.
        
        For train: use {uniprot}_active_{threshold}.sdf
        For test: use ALL actives minus train compounds
        """
        entries = []
        
        # Construct paths
        uniprot_dir = self.affinity_base / f"uniprot_{uniprot_id}"
        split_dir = uniprot_dir / "sdf_filtered_by_ligand_similarity"
        
        if split_type == 'train':
            sdf_path = split_dir / f"{uniprot_id}_active_{similarity_threshold}.sdf"
            pkl_path = split_dir / f"{uniprot_id}_active_{similarity_threshold}.pkl"
        else:  # test = all actives - train
            # Load all actives
            sdf_path = uniprot_dir / f"{uniprot_id}_active.sdf"
            pkl_path = uniprot_dir / f"{uniprot_id}_dict_smi_rdmol_active.pkl"
            
            # Get train SMILES to exclude
            train_sdf = split_dir / f"{uniprot_id}_active_{similarity_threshold}.sdf"
            train_smiles = set()
            if train_sdf.exists():
                try:
                    train_suppl = Chem.SDMolSupplier(str(train_sdf), sanitize=False, removeHs=False)
                    for mol in train_suppl:
                        if mol is not None:
                            try:
                                Chem.SanitizeMol(mol)
                                train_smiles.add(Chem.MolToSmiles(mol))
                            except:
                                pass
                except:
                    pass
        
        # Check if split exists
        if not sdf_path.exists():
            return entries
        
        # Read SDF file
        try:
            supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
            
            for mol_idx, mol in enumerate(supplier):
                if mol is None:
                    continue
                
                try:
                    Chem.SanitizeMol(mol)
                except:
                    continue
                
                # Get SMILES
                smiles = Chem.MolToSmiles(mol)
                
                # For test split, skip compounds that are in train set
                if split_type == 'test' and smiles in train_smiles:
                    continue
                
                # Get SMILES
                smiles = Chem.MolToSmiles(mol)
                
                # For test split, skip compounds that are in train set
                if split_type == 'test' and smiles in train_smiles:
                    continue
                
                # Get compound ID
                compound_id = mol.GetProp('_Name') if mol.HasProp('_Name') else f"{uniprot_id}_{split_type}_{mol_idx}"
                
                # Try to match with affinity data
                affinity_value = None
                affinity_type = None
                
                # Look for SMILES column
                smiles_col = None
                for col in affinity_df.columns:
                    if 'smiles' in col.lower():
                        smiles_col = col
                        break
                
                if smiles_col:
                    match = affinity_df[affinity_df[smiles_col] == smiles]
                    if len(match) > 0:
                        # Get affinity value
                        for col in affinity_df.columns:
                            if 'standard_value' in col.lower():
                                affinity_value = match[col].iloc[0]
                                break
                        for col in affinity_df.columns:
                            if 'standard_type' in col.lower():
                                affinity_type = match[col].iloc[0]
                                break
                
                # Create entry
                entry = {
                    'sample_id': f"{uniprot_id}_{compound_id}_{similarity_threshold}_{split_type}",
                    'uniprot_id': uniprot_id,
                    'pdb_id': protein_ref.pdb_id,
                    'compound_id': compound_id,
                    
                    # Protein references
                    'cif_path': protein_ref.cif_path,
                    'resolution': protein_ref.resolution,
                    'quality_score': protein_ref.quality_score,
                    
                    # Ligand data
                    'smiles': smiles,
                    'sdf_path': str(sdf_path),
                    'pkl_path': str(pkl_path) if pkl_path.exists() else None,
                    
                    # Label and affinity
                    'is_active': True,
                    'affinity_value': affinity_value,
                    'affinity_type': affinity_type,
                    
                    # Split info
                    'similarity_threshold': similarity_threshold,
                    'split': split_type,
                    'source': 'chembl',
                }
                
                entries.append(entry)
        
        except Exception as e:
            print(f"Error processing {sdf_path}: {e}")
        
        return entries
    
    def _process_decoys(
        self,
        uniprot_id: str,
        protein_ref: ProteinReference
    ) -> List[Dict]:
        """Process decoy compounds from deepcoy output."""
        entries = []
        
        decoy_dir = self.affinity_base / f"uniprot_{uniprot_id}" / "deepcoy_output"
        if not decoy_dir.exists():
            return entries
        
        # First try .txt file (SMILES format)
        decoy_txt = decoy_dir / f"{uniprot_id}_generated_decoys.txt"
        if decoy_txt.exists():
            try:
                with open(decoy_txt, 'r') as f:
                    for mol_idx, line in enumerate(f):
                        smiles = line.strip()
                        if not smiles or smiles.startswith('#'):
                            continue
                        
                        # Validate SMILES
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            continue
                        
                        compound_id = f"decoy_{mol_idx}"
                        
                        entry = {
                            'sample_id': f"{uniprot_id}_{compound_id}_decoy",
                            'uniprot_id': uniprot_id,
                            'pdb_id': protein_ref.pdb_id,
                            'compound_id': compound_id,
                            
                            # Protein references
                            'cif_path': protein_ref.cif_path,
                            'resolution': protein_ref.resolution,
                            'quality_score': protein_ref.quality_score,
                            
                            # Ligand data
                            'smiles': smiles,
                            'sdf_path': None,  # No 3D structure
                            'pkl_path': None,
                            
                            # Label and affinity
                            'is_active': False,
                            'affinity_value': None,
                            'affinity_type': None,
                            
                            # Split info
                            'similarity_threshold': None,
                            'split': 'decoy',
                            'source': 'deepcoy',
                        }
                        
                        entries.append(entry)
                
                return entries  # If txt file processed successfully, return
                
            except Exception as e:
                print(f"Error reading {decoy_txt}: {e}")
        
        # Fallback: look for decoy SDF files
        decoy_files = list(decoy_dir.glob("*.sdf"))
        
        for decoy_file in decoy_files:
            try:
                supplier = Chem.SDMolSupplier(str(decoy_file), sanitize=False, removeHs=False)
                
                for mol_idx, mol in enumerate(supplier):
                    if mol is None:
                        continue
                    
                    try:
                        Chem.SanitizeMol(mol)
                    except:
                        continue
                    
                    compound_id = mol.GetProp('_Name') if mol.HasProp('_Name') else f"{uniprot_id}_decoy_{mol_idx}"
                    smiles = Chem.MolToSmiles(mol)
                    
                    entry = {
                        'sample_id': f"{uniprot_id}_{compound_id}_decoy",
                        'uniprot_id': uniprot_id,
                        'pdb_id': protein_ref.pdb_id,
                        'compound_id': compound_id,
                        
                        # Protein references
                        'cif_path': protein_ref.cif_path,
                        'resolution': protein_ref.resolution,
                        'quality_score': protein_ref.quality_score,
                        
                        # Ligand data
                        'smiles': smiles,
                        'sdf_path': str(decoy_file),
                        'pkl_path': None,
                        
                        # Label (decoy = inactive)
                        'is_active': False,
                        'affinity_value': None,
                        'affinity_type': None,
                        
                        # Split info
                        'similarity_threshold': None,
                        'split': 'decoy',
                        'source': 'deepcoy',
                    }
                    
                    entries.append(entry)
            
            except Exception as e:
                print(f"Error processing decoys {decoy_file}: {e}")
        
        return entries
    
    def _print_stats(self, stats: Dict, registry_df: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "="*80)
        print("Registry Building Complete")
        print("="*80)
        print(f"\nUniProts processed: {stats['total_uniprots']}")
        print(f"  Success: {stats['success']}")
        print(f"  No affinity data: {stats['no_affinity']}")
        
        print(f"\nTotal samples: {len(registry_df):,}")
        print(f"  Active compounds: {stats['total_actives']:,}")
        print(f"  Decoy compounds: {stats['total_decoys']:,}")
        
        if len(registry_df) > 0:
            print(f"\nSamples by split:")
            for split in ['train', 'test', 'decoy']:
                count = len(registry_df[registry_df['split'] == split])
                pct = count / len(registry_df) * 100
                print(f"  {split}: {count:,} ({pct:.1f}%)")
            
            print(f"\nSamples by similarity threshold:")
            for thresh in registry_df['similarity_threshold'].dropna().unique():
                count = len(registry_df[registry_df['similarity_threshold'] == thresh])
                pct = count / len(registry_df[registry_df['split'] != 'decoy']) * 100
                print(f"  {thresh}: {count:,} ({pct:.1f}%)")
            
            print(f"\nAffinity data:")
            has_affinity = registry_df['affinity_value'].notna().sum()
            print(f"  Samples with affinity: {has_affinity:,} ({has_affinity/len(registry_df)*100:.1f}%)")


def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="Build training data registry")
    ap.add_argument("--affinity_dir", required=True, help="Base affinity data directory")
    ap.add_argument("--structures_csv", required=True, help="Selected structures CSV")
    ap.add_argument("--output_dir", default="training_data", help="Output directory")
    ap.add_argument("--similarity_thresholds", nargs='+', default=['0p3', '0p5', '0p7'],
                    help="Similarity thresholds to include")
    ap.add_argument("--include_decoys", action='store_true', help="Include decoy compounds")
    ap.add_argument("--max_uniprots", type=int, help="Limit number of UniProts (for testing)")
    args = ap.parse_args()
    
    builder = RegistryBuilder(
        affinity_base_dir=args.affinity_dir,
        structures_csv=args.structures_csv,
        output_dir=args.output_dir
    )
    
    registry_df = builder.build_registry(
        similarity_thresholds=args.similarity_thresholds,
        include_decoys=args.include_decoys,
        max_uniprots=args.max_uniprots
    )
    
    print(f"\n✓ Registry saved to: {Path(args.output_dir) / 'registry.csv'}")


if __name__ == "__main__":
    main()
