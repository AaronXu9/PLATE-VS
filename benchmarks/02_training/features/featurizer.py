"""
Molecular Featurizer Module.

This module provides utilities for converting molecular SMILES strings
into numerical features for machine learning.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from tqdm import tqdm

if TYPE_CHECKING:
    from .feature_cache import FeatureCache


class MorganFingerprintFeaturizer:
    """
    Generate Morgan (ECFP) fingerprints from SMILES strings.
    
    Morgan fingerprints are circular fingerprints that capture structural
    features around each atom in the molecule.
    """
    
    def __init__(self, radius: int = 2, n_bits: int = 2048, use_features: bool = False):
        """
        Initialize the featurizer.
        
        Args:
            radius: Radius of the circular fingerprint (default: 2)
            n_bits: Number of bits in the fingerprint (default: 2048)
            use_features: Use pharmacophoric features instead of atom types (default: False)
        """
        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features
        self.name = f"morgan_r{radius}_b{n_bits}"
        
    def smiles_to_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """
        Convert a SMILES string to a Morgan fingerprint.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Fingerprint as numpy array or None if conversion fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            if self.use_features:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.n_bits, useFeatures=True
                )
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.n_bits
                )
            
            return np.array(fp, dtype=np.float32)
        except Exception as e:
            print(f"Error converting SMILES to fingerprint: {smiles[:50]}... Error: {e}")
            return None
    
    def featurize(self, smiles_list: List[str], show_progress: bool = True,
                  cache: Optional["FeatureCache"] = None) -> np.ndarray:
        """
        Convert a list of SMILES to fingerprints.

        Args:
            smiles_list: List of SMILES strings
            show_progress: Show progress bar (default: True)
            cache: Optional FeatureCache to load/store precomputed features

        Returns:
            2D numpy array of fingerprints (n_molecules, n_bits)
        """
        if cache is not None:
            return cache.featurize_with_cache(
                smiles_list,
                lambda keys: self._compute_fingerprints(keys, show_progress),
                show_progress=show_progress,
            )
        return self._compute_fingerprints(smiles_list, show_progress)

    def _compute_fingerprints(self, smiles_list: List[str], show_progress: bool) -> tuple:
        invalid_indices = []
        # Pre-allocate float32 array to avoid list→array conversion doubling memory
        fingerprints = np.zeros((len(smiles_list), self.n_bits), dtype=np.float32)

        iterator = tqdm(enumerate(smiles_list), total=len(smiles_list),
                       desc="Generating fingerprints") if show_progress else enumerate(smiles_list)

        for idx, smiles in iterator:
            fp = self.smiles_to_fingerprint(smiles)
            if fp is not None:
                fingerprints[idx] = fp
            else:
                invalid_indices.append(idx)
                # Row stays as zeros for invalid molecules

        if invalid_indices:
            print(f"Warning: {len(invalid_indices)} molecules could not be converted")

        return fingerprints, invalid_indices
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the featurizer configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'type': 'morgan_fingerprint',
            'radius': self.radius,
            'n_bits': self.n_bits,
            'use_features': self.use_features
        }


class DescriptorFeaturizer:
    """
    Generate molecular descriptors from SMILES strings.
    
    Computes various physicochemical descriptors like MW, LogP, etc.
    """
    
    def __init__(self, descriptor_names: Optional[List[str]] = None):
        """
        Initialize the descriptor featurizer.
        
        Args:
            descriptor_names: List of descriptor names to compute.
                            If None, uses a default set of common descriptors.
        """
        if descriptor_names is None:
            self.descriptor_names = [
                'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
                'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
                'FractionCSP3', 'NumSaturatedRings', 'NumAliphaticRings'
            ]
        else:
            self.descriptor_names = descriptor_names
        
        self.name = f"descriptors_{len(self.descriptor_names)}"
    
    def smiles_to_descriptors(self, smiles: str) -> Optional[np.ndarray]:
        """
        Convert a SMILES string to molecular descriptors.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Descriptor array or None if conversion fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            descriptors = []
            for desc_name in self.descriptor_names:
                desc_func = getattr(Descriptors, desc_name)
                descriptors.append(desc_func(mol))
            
            return np.array(descriptors)
        except Exception as e:
            print(f"Error computing descriptors: {smiles[:50]}... Error: {e}")
            return None
    
    def featurize(self, smiles_list: List[str], show_progress: bool = True,
                  cache: Optional["FeatureCache"] = None) -> np.ndarray:
        """
        Convert a list of SMILES to descriptors.

        Args:
            smiles_list: List of SMILES strings
            show_progress: Show progress bar (default: True)
            cache: Optional FeatureCache to load/store precomputed features

        Returns:
            2D numpy array of descriptors (n_molecules, n_descriptors)
        """
        if cache is not None:
            return cache.featurize_with_cache(
                smiles_list,
                lambda keys: self._compute_descriptors(keys, show_progress),
                show_progress=show_progress,
            )
        return self._compute_descriptors(smiles_list, show_progress)

    def _compute_descriptors(self, smiles_list: List[str], show_progress: bool) -> tuple:
        invalid_indices = []
        n_desc = len(self.descriptor_names)
        # Pre-allocate float32 array to avoid list→array conversion doubling memory
        descriptors = np.zeros((len(smiles_list), n_desc), dtype=np.float32)

        iterator = tqdm(enumerate(smiles_list), total=len(smiles_list),
                       desc="Computing descriptors") if show_progress else enumerate(smiles_list)

        for idx, smiles in iterator:
            desc = self.smiles_to_descriptors(smiles)
            if desc is not None:
                descriptors[idx] = desc
            else:
                invalid_indices.append(idx)
                # Row stays as zeros for invalid molecules

        if invalid_indices:
            print(f"Warning: {len(invalid_indices)} molecules could not be converted")

        return descriptors, invalid_indices
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the featurizer configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'type': 'molecular_descriptors',
            'descriptor_names': self.descriptor_names
        }


def get_featurizer(config: Dict[str, Any]) -> Any:
    """
    Factory function to get the appropriate featurizer based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized featurizer object
    """
    feature_type = config.get('type', 'morgan_fingerprint')
    
    if feature_type == 'morgan_fingerprint':
        return MorganFingerprintFeaturizer(
            radius=config.get('radius', 2),
            n_bits=config.get('n_bits', 2048),
            use_features=config.get('use_features', False)
        )
    elif feature_type == 'molecular_descriptors':
        return DescriptorFeaturizer(
            descriptor_names=config.get('descriptor_names', None)
        )
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
