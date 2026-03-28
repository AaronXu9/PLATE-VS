"""
Combined Featurizer Module.

This module combines ligand and protein features into a unified
feature representation for protein-ligand binding affinity prediction.
"""

import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional
from tqdm import tqdm

from .featurizer import get_featurizer as get_ligand_featurizer
from .protein_featurizer import get_protein_featurizer

if TYPE_CHECKING:
    from .feature_cache import FeatureCache


class CombinedFeaturizer:
    """
    Combine ligand and protein features for binding affinity prediction.
    
    This featurizer creates a unified feature vector by concatenating
    molecular fingerprints with protein features.
    """
    
    def __init__(self, 
                 ligand_config: Dict[str, Any],
                 protein_config: Dict[str, Any],
                 concatenation_method: str = 'concat'):
        """
        Initialize the combined featurizer.
        
        Args:
            ligand_config: Configuration for ligand featurizer
            protein_config: Configuration for protein featurizer
            concatenation_method: How to combine features ('concat', 'weighted')
        """
        self.ligand_featurizer = get_ligand_featurizer(ligand_config)
        self.protein_featurizer = get_protein_featurizer(protein_config)
        self.concatenation_method = concatenation_method
        
        self.ligand_dim = None
        self.protein_dim = None
        self.total_dim = None
        
        # For protein identifier featurizer, we need to fit it first
        self.protein_fitted = False
        
        self.name = f"combined_{self.ligand_featurizer.name}_{self.protein_featurizer.name}"
    
    def fit_protein_featurizer(self, protein_ids: List[str]) -> None:
        """
        Fit the protein featurizer on protein IDs.
        
        Args:
            protein_ids: List of protein identifiers
        """
        if hasattr(self.protein_featurizer, 'fit'):
            print("Fitting protein featurizer on protein IDs...")
            self.protein_featurizer.fit(protein_ids)
            self.protein_fitted = True
    
    def featurize(self,
                  smiles_list: List[str],
                  protein_ids: List[str] = None,
                  protein_sequences: List[str] = None,
                  show_progress: bool = True,
                  ligand_cache: Optional["FeatureCache"] = None,
                  protein_cache: Optional["FeatureCache"] = None) -> Tuple[np.ndarray, Dict[str, List[int]]]:
        """
        Generate combined features from ligands and proteins.
        
        Args:
            smiles_list: List of SMILES strings for ligands
            protein_ids: List of protein identifiers (for identifier featurizer)
            protein_sequences: List of protein sequences (for sequence featurizer)
            show_progress: Show progress bars
            
        Returns:
            Tuple of (combined feature matrix, dict of invalid indices)
        """
        if len(smiles_list) == 0:
            raise ValueError("Empty SMILES list")
        
        # Determine which protein input to use
        protein_feature_type = self.protein_featurizer.get_config()['type']
        
        if protein_feature_type == 'protein_identifier' and protein_ids is None:
            raise ValueError("protein_ids required for protein_identifier featurizer")
        
        if protein_feature_type == 'protein_sequence' and protein_sequences is None:
            raise ValueError("protein_sequences required for protein_sequence featurizer")
        
        # Generate ligand features
        print("Generating ligand features...")
        X_ligand, invalid_ligand = self.ligand_featurizer.featurize(
            smiles_list, show_progress=show_progress, cache=ligand_cache
        )

        # Generate protein features
        print("Generating protein features...")
        if protein_feature_type == 'protein_identifier':
            # Fit if not already fitted
            if not self.protein_fitted and hasattr(self.protein_featurizer, 'fit'):
                self.fit_protein_featurizer(protein_ids)

            # protein_identifier uses random embeddings — no caching needed
            X_protein, invalid_protein = self.protein_featurizer.transform(
                protein_ids, show_progress=show_progress
            )
        else:  # protein_sequence
            X_protein, invalid_protein = self.protein_featurizer.featurize(
                protein_sequences, show_progress=show_progress, cache=protein_cache
            )
        
        # Store dimensions
        self.ligand_dim = X_ligand.shape[1]
        self.protein_dim = X_protein.shape[1]
        
        # Combine features
        print(f"Combining features: ligand ({self.ligand_dim}D) + protein ({self.protein_dim}D)")
        
        if self.concatenation_method == 'concat':
            X_combined = np.concatenate([X_ligand, X_protein], axis=1)
        elif self.concatenation_method == 'weighted':
            # Weighted combination (can be tuned)
            ligand_weight = 0.7
            protein_weight = 0.3
            # Normalize to same scale
            X_ligand_norm = X_ligand / (np.linalg.norm(X_ligand, axis=1, keepdims=True) + 1e-8)
            X_protein_norm = X_protein / (np.linalg.norm(X_protein, axis=1, keepdims=True) + 1e-8)
            X_combined = np.concatenate([
                X_ligand_norm * ligand_weight,
                X_protein_norm * protein_weight
            ], axis=1)
        else:
            raise ValueError(f"Unknown concatenation method: {self.concatenation_method}")
        
        self.total_dim = X_combined.shape[1]
        
        print(f"Combined feature dimension: {self.total_dim}")
        
        # Combine invalid indices
        invalid_indices = {
            'ligand': invalid_ligand,
            'protein': invalid_protein,
            'any': list(set(invalid_ligand) | set(invalid_protein))
        }
        
        if invalid_indices['any']:
            print(f"Warning: {len(invalid_indices['any'])} samples have invalid features")
        
        return X_combined, invalid_indices
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the combined featurizer configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'type': 'combined',
            'ligand_config': self.ligand_featurizer.get_config(),
            'protein_config': self.protein_featurizer.get_config(),
            'concatenation_method': self.concatenation_method,
            'ligand_dim': self.ligand_dim,
            'protein_dim': self.protein_dim,
            'total_dim': self.total_dim
        }
    
    def save_protein_mapping(self, filepath: str) -> None:
        """
        Save protein featurizer mapping if applicable.
        
        Args:
            filepath: Path to save the mapping
        """
        if hasattr(self.protein_featurizer, 'save_mapping'):
            self.protein_featurizer.save_mapping(filepath)
            print(f"Protein mapping saved to {filepath}")
    
    def load_protein_mapping(self, filepath: str) -> None:
        """
        Load protein featurizer mapping if applicable.
        
        Args:
            filepath: Path to load the mapping from
        """
        if hasattr(self.protein_featurizer, 'load_mapping'):
            self.protein_featurizer.load_mapping(filepath)
            self.protein_fitted = True
            print(f"Protein mapping loaded from {filepath}")


def get_combined_featurizer(ligand_config: Dict[str, Any],
                            protein_config: Dict[str, Any],
                            concatenation_method: str = 'concat') -> CombinedFeaturizer:
    """
    Factory function to create a combined featurizer.
    
    Args:
        ligand_config: Configuration for ligand features
        protein_config: Configuration for protein features  
        concatenation_method: Method to combine features
        
    Returns:
        Initialized CombinedFeaturizer
    """
    return CombinedFeaturizer(
        ligand_config=ligand_config,
        protein_config=protein_config,
        concatenation_method=concatenation_method
    )
