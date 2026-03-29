#!/usr/bin/env python3
"""
Flexible dataloaders for different model architectures.

Supports:
- RF/XGBoost: Fingerprints + sequence features
- GNN: Molecular/protein graphs  
- Transformer: Sequences
- 3D CNN: Voxelized structures
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle

import torch
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import gemmi


class BaseDataLoader(Dataset, ABC):
    """Base class for all dataloaders."""
    
    def __init__(
        self,
        registry_csv: str,
        split: str = 'train',
        similarity_threshold: str = '0p7',
        include_decoys: bool = False,
        cache_features: bool = True,
        protein_refs_json: Optional[str] = None
    ):
        """
        Args:
            registry_csv: Path to registry CSV
            split: 'train', 'test', or 'all'
            similarity_threshold: e.g., '0p3', '0p5', '0p7'
            include_decoys: Whether to include decoy samples
            cache_features: Cache computed features in memory
            protein_refs_json: Path to protein references JSON
        """
        self.registry_df = pd.read_csv(registry_csv)
        self.cache = {} if cache_features else None
        
        # Filter by split
        if split != 'all':
            if include_decoys and split == 'train':
                # Include decoys in training
                self.df = self.registry_df[
                    ((self.registry_df['split'] == split) & 
                     (self.registry_df['similarity_threshold'] == similarity_threshold)) |
                    (self.registry_df['split'] == 'decoy')
                ]
            else:
                self.df = self.registry_df[
                    (self.registry_df['split'] == split) &
                    (self.registry_df['similarity_threshold'] == similarity_threshold)
                ]
        else:
            self.df = self.registry_df
        
        # Load protein references
        self.protein_refs = {}
        if protein_refs_json:
            with open(protein_refs_json, 'r') as f:
                self.protein_refs = json.load(f)
        
        print(f"Loaded {len(self.df)} samples for {split} split (threshold: {similarity_threshold})")
        print(f"  Actives: {(self.df['is_active'] == True).sum()}")
        if include_decoys:
            print(f"  Decoys: {(self.df['is_active'] == False).sum()}")
    
    def __len__(self):
        return len(self.df)
    
    @abstractmethod
    def __getitem__(self, idx):
        pass
    
    def _get_protein_sequence(self, uniprot_id: str, cif_path: str) -> str:
        """Extract or retrieve protein sequence."""
        # Check protein refs first
        if uniprot_id in self.protein_refs and 'sequence' in self.protein_refs[uniprot_id]:
            seq = self.protein_refs[uniprot_id]['sequence']
            if seq:
                return seq
        
        # Extract from CIF
        try:
            struct = gemmi.read_structure(cif_path)
            if len(struct) == 0:
                return ""
            
            model = struct[0]
            for chain in model:
                seq = []
                for res in chain:
                    if res.entity_type == gemmi.EntityType.Polymer:
                        aa = gemmi.find_tabulated_residue(res.name)
                        if aa.is_amino_acid():
                            seq.append(aa.one_letter_code)
                if seq:
                    return ''.join(seq)
        except:
            pass
        
        return ""


class FingerprintDataLoader(BaseDataLoader):
    """
    For RF/XGBoost models.
    Returns: (protein_features, ligand_fingerprint) -> affinity/label
    """
    
    def __init__(
        self,
        registry_csv: str,
        fingerprint_type: str = 'ecfp4',
        fp_radius: int = 2,
        fp_bits: int = 2048,
        protein_feature_type: str = 'aac',  # 'aac', 'esm', 'onehot'
        **kwargs
    ):
        super().__init__(registry_csv, **kwargs)
        self.fp_type = fingerprint_type
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self.protein_feature_type = protein_feature_type
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Check cache
        cache_key = f"{idx}_{self.fp_type}_{self.protein_feature_type}"
        if self.cache is not None and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Ligand fingerprint
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None:
            # Return dummy
            fp_array = np.zeros(self.fp_bits)
        else:
            if self.fp_type == 'ecfp4':
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_bits
                )
            elif self.fp_type == 'maccs':
                fp = AllChem.GetMACCSKeysFingerprint(mol)
            else:
                raise ValueError(f"Unknown fingerprint type: {self.fp_type}")
            
            fp_array = np.array(fp, dtype=np.float32)
        
        # Protein features
        if self.protein_feature_type == 'aac':
            sequence = self._get_protein_sequence(row['uniprot_id'], row['cif_path'])
            prot_feat = self._amino_acid_composition(sequence)
        else:
            raise NotImplementedError(f"Protein feature type: {self.protein_feature_type}")
        
        # Target
        if row['is_active']:
            if pd.notna(row['affinity_value']):
                # Regression: pIC50
                target = -np.log10(row['affinity_value'] * 1e-9)  # Convert nM to M
            else:
                # Classification: active but no affinity
                target = 1.0
        else:
            # Decoy
            target = 0.0
        
        # Combine features
        features = np.concatenate([prot_feat, fp_array]).astype(np.float32)
        
        sample = {
            'features': features,
            'target': target,
            'sample_id': row['sample_id'],
            'is_active': row['is_active']
        }
        
        if self.cache is not None:
            self.cache[cache_key] = sample
        
        return sample
    
    def _amino_acid_composition(self, sequence: str) -> np.ndarray:
        """Compute amino acid composition."""
        if not sequence:
            return np.zeros(20, dtype=np.float32)
        
        aa_order = 'ACDEFGHIKLMNPQRSTVWY'
        counts = np.array([sequence.count(aa) / len(sequence) for aa in aa_order], dtype=np.float32)
        return counts
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        aa_order = 'ACDEFGHIKLMNPQRSTVWY'
        protein_names = [f'AA_{aa}' for aa in aa_order]
        ligand_names = [f'FP_{i}' for i in range(self.fp_bits)]
        return protein_names + ligand_names


class GraphDataLoader(BaseDataLoader):
    """
    For GNN models.
    Returns: (protein_graph, ligand_graph) -> affinity/label
    """
    
    def __init__(
        self,
        registry_csv: str,
        use_3d: bool = True,
        max_nodes: int = 500,
        **kwargs
    ):
        super().__init__(registry_csv, **kwargs)
        self.use_3d = use_3d
        self.max_nodes = max_nodes
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load ligand
        if self.use_3d and row['sdf_path']:
            ligand_graph = self._mol_from_sdf_to_graph(row['sdf_path'], row['smiles'])
        else:
            ligand_graph = self._smiles_to_graph(row['smiles'])
        
        # Load protein (pocket only for efficiency)
        protein_graph = self._pocket_to_graph(row['cif_path'])
        
        # Target
        if row['is_active']:
            if pd.notna(row['affinity_value']):
                target = -np.log10(row['affinity_value'] * 1e-9)
            else:
                target = 1.0
        else:
            target = 0.0
        
        return {
            'protein_graph': protein_graph,
            'ligand_graph': ligand_graph,
            'target': torch.tensor(target, dtype=torch.float32),
            'sample_id': row['sample_id'],
            'is_active': row['is_active']
        }
    
    def _smiles_to_graph(self, smiles: str) -> Dict:
        """Convert SMILES to molecular graph."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Return dummy graph
            return {
                'node_features': torch.zeros(1, 10),
                'edge_index': torch.zeros(2, 0, dtype=torch.long),
                'edge_features': torch.zeros(0, 5),
            }
        
        # Node features: atom type, degree, etc.
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization(),
                atom.GetIsAromatic(),
            ]
            node_features.append(features)
        
        # Edge features
        edge_index = []
        edge_features = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])  # Undirected
            
            bond_feat = [
                bond.GetBondTypeAsDouble(),
                bond.GetIsAromatic(),
                bond.IsInRing(),
            ]
            edge_features.extend([bond_feat, bond_feat])
        
        return {
            'node_features': torch.tensor(node_features, dtype=torch.float32),
            'edge_index': torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros(2, 0, dtype=torch.long),
            'edge_features': torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.zeros(0, 3),
        }
    
    def _mol_from_sdf_to_graph(self, sdf_path: str, fallback_smiles: str) -> Dict:
        """Convert SDF to graph with 3D coordinates."""
        try:
            suppl = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
            mol = next(suppl)
            if mol is None:
                return self._smiles_to_graph(fallback_smiles)
            
            Chem.SanitizeMol(mol)
            
            # Same as _smiles_to_graph but add 3D coords
            graph = self._smiles_to_graph(Chem.MolToSmiles(mol))
            
            # Add 3D coordinates
            conf = mol.GetConformer()
            coords = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            
            graph['node_coords'] = torch.tensor(coords, dtype=torch.float32)
            return graph
        
        except:
            return self._smiles_to_graph(fallback_smiles)
    
    def _pocket_to_graph(self, cif_path: str) -> Dict:
        """Convert protein pocket to residue graph."""
        # Placeholder: implement residue-level graph
        # Nodes = residues, edges = contacts (<8Å)
        return {
            'node_features': torch.zeros(1, 20),  # amino acid one-hot
            'edge_index': torch.zeros(2, 0, dtype=torch.long),
        }


class SequenceDataLoader(BaseDataLoader):
    """
    For transformer models (ESM-2, ChemBERTa).
    Returns: (protein_sequence, ligand_smiles) -> affinity/label
    """
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        sequence = self._get_protein_sequence(row['uniprot_id'], row['cif_path'])
        
        # Target
        if row['is_active']:
            if pd.notna(row['affinity_value']):
                target = -np.log10(row['affinity_value'] * 1e-9)
            else:
                target = 1.0
        else:
            target = 0.0
        
        return {
            'protein_sequence': sequence,
            'ligand_smiles': row['smiles'],
            'target': target,
            'sample_id': row['sample_id'],
            'is_active': row['is_active']
        }


def collate_fingerprint(batch: List[Dict]) -> Dict:
    """Collate function for fingerprint dataloader."""
    return {
        'features': torch.tensor(np.stack([b['features'] for b in batch])),
        'target': torch.tensor(np.array([b['target'] for b in batch]), dtype=torch.float32),
        'sample_id': [b['sample_id'] for b in batch],
        'is_active': torch.tensor([b['is_active'] for b in batch])
    }


def collate_graph(batch: List[Dict]) -> Dict:
    """Collate function for graph dataloader."""
    # Implement graph batching (PyG style)
    pass


if __name__ == "__main__":
    # Test dataloaders
    print("Testing FingerprintDataLoader...")
    
    loader = FingerprintDataLoader(
        registry_csv="training_data/registry.csv",
        split='train',
        similarity_threshold='0p7',
        include_decoys=True
    )
    
    print(f"Dataset size: {len(loader)}")
    
    # Test one sample
    sample = loader[0]
    print(f"\nSample 0:")
    print(f"  Features shape: {sample['features'].shape}")
    print(f"  Target: {sample['target']:.3f}")
    print(f"  Is active: {sample['is_active']}")
    print(f"  Sample ID: {sample['sample_id']}")
