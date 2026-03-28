"""
Protein Featurizer Module.

This module provides utilities for converting protein identifiers, sequences,
and structures into numerical features for machine learning.
"""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple
from collections import Counter
from tqdm import tqdm

if TYPE_CHECKING:
    from .feature_cache import FeatureCache


class ProteinSequenceFeaturizer:
    """
    Generate features from protein sequences.
    
    Computes amino acid composition, physicochemical properties,
    and other sequence-based descriptors.
    """
    
    # Standard amino acids
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Physicochemical property groups
    HYDROPHOBIC = set('AILMFWYV')
    POLAR = set('STNQ')
    POSITIVE = set('KRH')
    NEGATIVE = set('DE')
    AROMATIC = set('FYW')
    SMALL = set('AGST')
    
    def __init__(self, include_composition: bool = True,
                 include_properties: bool = True,
                 include_dipeptides: bool = False):
        """
        Initialize the protein sequence featurizer.
        
        Args:
            include_composition: Include amino acid composition features
            include_properties: Include physicochemical property features
            include_dipeptides: Include dipeptide composition (increases feature dim)
        """
        self.include_composition = include_composition
        self.include_properties = include_properties
        self.include_dipeptides = include_dipeptides
        self.name = "protein_sequence"
        
        # Calculate feature dimension
        self.n_features = 0
        if include_composition:
            self.n_features += 20  # AA composition
        if include_properties:
            self.n_features += 11  # Property features
        if include_dipeptides:
            self.n_features += 400  # Dipeptide composition (20x20)
    
    def sequence_to_features(self, sequence: str) -> Optional[np.ndarray]:
        """
        Convert a protein sequence to features.
        
        Args:
            sequence: Protein amino acid sequence
            
        Returns:
            Feature array or None if conversion fails
        """
        if not sequence or not isinstance(sequence, str):
            return None
        
        sequence = sequence.upper().strip()
        if len(sequence) == 0:
            return None
        
        features = []
        
        try:
            # Amino acid composition
            if self.include_composition:
                aa_comp = self._amino_acid_composition(sequence)
                features.extend(aa_comp)
            
            # Physicochemical properties
            if self.include_properties:
                props = self._physicochemical_properties(sequence)
                features.extend(props)
            
            # Dipeptide composition
            if self.include_dipeptides:
                dipep = self._dipeptide_composition(sequence)
                features.extend(dipep)
            
            return np.array(features)
        
        except Exception as e:
            print(f"Error computing sequence features: {sequence[:50]}... Error: {e}")
            return None
    
    def _amino_acid_composition(self, sequence: str) -> List[float]:
        """Compute amino acid composition (20 features)."""
        length = len(sequence)
        composition = []
        for aa in self.AMINO_ACIDS:
            count = sequence.count(aa)
            composition.append(count / length if length > 0 else 0)
        return composition
    
    def _physicochemical_properties(self, sequence: str) -> List[float]:
        """Compute physicochemical property features (11 features)."""
        length = len(sequence)
        if length == 0:
            return [0] * 11
        
        seq_set = set(sequence)
        
        properties = [
            len(sequence),  # Length
            sum(1 for aa in sequence if aa in self.HYDROPHOBIC) / length,  # Hydrophobic %
            sum(1 for aa in sequence if aa in self.POLAR) / length,  # Polar %
            sum(1 for aa in sequence if aa in self.POSITIVE) / length,  # Positive %
            sum(1 for aa in sequence if aa in self.NEGATIVE) / length,  # Negative %
            sum(1 for aa in sequence if aa in self.AROMATIC) / length,  # Aromatic %
            sum(1 for aa in sequence if aa in self.SMALL) / length,  # Small %
            (sum(1 for aa in sequence if aa in self.POSITIVE) - 
             sum(1 for aa in sequence if aa in self.NEGATIVE)) / length,  # Charge
            len(seq_set) / 20,  # Diversity (unique AA / 20)
            sequence.count('C') / length,  # Cysteine content (disulfide bonds)
            sequence.count('P') / length,  # Proline content (structure breaker)
        ]
        
        return properties
    
    def _dipeptide_composition(self, sequence: str) -> List[float]:
        """Compute dipeptide composition (400 features)."""
        length = len(sequence) - 1
        if length <= 0:
            return [0] * 400
        
        dipeptides = [sequence[i:i+2] for i in range(len(sequence) - 1)]
        dipep_count = Counter(dipeptides)
        
        composition = []
        for aa1 in self.AMINO_ACIDS:
            for aa2 in self.AMINO_ACIDS:
                dipep = aa1 + aa2
                composition.append(dipep_count.get(dipep, 0) / length)
        
        return composition
    
    def featurize(self, sequences: List[str], show_progress: bool = True,
                  cache: Optional["FeatureCache"] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Convert a list of protein sequences to features.

        Args:
            sequences: List of protein sequences
            show_progress: Show progress bar
            cache: Optional FeatureCache to load/store precomputed features

        Returns:
            Tuple of (feature matrix, invalid_indices)
        """
        if cache is not None:
            return cache.featurize_with_cache(
                sequences,
                lambda keys: self._compute_sequence_features(keys, show_progress),
                show_progress=show_progress,
            )
        return self._compute_sequence_features(sequences, show_progress)

    def _compute_sequence_features(self, sequences: List[str], show_progress: bool) -> Tuple[np.ndarray, List[int]]:
        invalid_indices = []
        # Pre-allocate float32 to avoid list→array memory doubling
        features = np.zeros((len(sequences), self.n_features), dtype=np.float32)

        iterator = tqdm(enumerate(sequences), total=len(sequences),
                       desc="Generating protein features") if show_progress else enumerate(sequences)

        for idx, seq in iterator:
            feat = self.sequence_to_features(seq)
            if feat is not None:
                features[idx] = feat
            else:
                invalid_indices.append(idx)

        if invalid_indices:
            print(f"Warning: {len(invalid_indices)} sequences could not be converted")

        return features, invalid_indices
    
    def get_config(self) -> Dict[str, Any]:
        """Get the featurizer configuration."""
        return {
            'type': 'protein_sequence',
            'include_composition': self.include_composition,
            'include_properties': self.include_properties,
            'include_dipeptides': self.include_dipeptides,
            'n_features': self.n_features
        }


class ProteinIdentifierFeaturizer:
    """
    Generate features from protein identifiers using learned embeddings.
    
    Creates a unique embedding vector for each protein, allowing the model
    to learn protein-specific patterns.
    """
    
    def __init__(self, embedding_dim: int = 32, 
                 use_onehot: bool = False):
        """
        Initialize the protein identifier featurizer.
        
        Args:
            embedding_dim: Dimension of protein embeddings
            use_onehot: Use one-hot encoding instead of learned embeddings
        """
        self.embedding_dim = embedding_dim
        self.use_onehot = use_onehot
        self.protein_to_idx = {}
        self.idx_to_protein = {}
        self.embeddings = None
        self.n_proteins = 0
        self.name = "protein_identifier"
    
    def fit(self, protein_ids: List[str]) -> None:
        """
        Fit the featurizer on a list of protein IDs.
        
        Args:
            protein_ids: List of protein identifiers
        """
        unique_proteins = sorted(set(protein_ids))
        self.n_proteins = len(unique_proteins)
        
        # Create mapping
        self.protein_to_idx = {prot: idx for idx, prot in enumerate(unique_proteins)}
        self.idx_to_protein = {idx: prot for prot, idx in self.protein_to_idx.items()}
        
        if self.use_onehot:
            self.embedding_dim = self.n_proteins
            # One-hot will be generated on-the-fly
            self.embeddings = None
        else:
            # Initialize random embeddings (can be updated during training in NN models)
            np.random.seed(42)
            self.embeddings = np.random.randn(self.n_proteins, self.embedding_dim).astype(np.float32)
            # Normalize
            self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        print(f"Fitted protein identifier featurizer with {self.n_proteins} unique proteins")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def transform(self, protein_ids: List[str], show_progress: bool = True) -> Tuple[np.ndarray, List[int]]:
        """
        Transform protein IDs to embeddings.
        
        Args:
            protein_ids: List of protein identifiers
            show_progress: Show progress bar
            
        Returns:
            Tuple of (embedding matrix, unknown_indices)
        """
        if not self.protein_to_idx:
            raise ValueError("Featurizer not fitted. Call fit() first.")
        
        unknown_indices = []
        # Pre-allocate float32 to avoid list→array memory doubling
        out_dim = self.n_proteins if self.use_onehot else self.embedding_dim
        embeddings = np.zeros((len(protein_ids), out_dim), dtype=np.float32)

        iterator = tqdm(enumerate(protein_ids), total=len(protein_ids),
                       desc="Encoding protein IDs") if show_progress else enumerate(protein_ids)

        for idx, prot_id in iterator:
            if prot_id in self.protein_to_idx:
                prot_idx = self.protein_to_idx[prot_id]
                if self.use_onehot:
                    embeddings[idx, prot_idx] = 1.0
                else:
                    embeddings[idx] = self.embeddings[prot_idx]
            else:
                unknown_indices.append(idx)
                # Row stays as zeros for unknown proteins

        if unknown_indices:
            print(f"Warning: {len(unknown_indices)} proteins not seen during fit")

        return embeddings, unknown_indices
    
    def fit_transform(self, protein_ids: List[str], show_progress: bool = True) -> Tuple[np.ndarray, List[int]]:
        """Fit and transform in one step."""
        self.fit(protein_ids)
        return self.transform(protein_ids, show_progress)
    
    def get_config(self) -> Dict[str, Any]:
        """Get the featurizer configuration."""
        return {
            'type': 'protein_identifier',
            'embedding_dim': self.embedding_dim,
            'use_onehot': self.use_onehot,
            'n_proteins': self.n_proteins,
            'proteins': list(self.protein_to_idx.keys())
        }
    
    def save_mapping(self, filepath: str) -> None:
        """Save the protein ID mapping."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'protein_to_idx': self.protein_to_idx,
                'embedding_dim': self.embedding_dim,
                'use_onehot': self.use_onehot
            }, f, indent=2)
    
    def load_mapping(self, filepath: str) -> None:
        """Load the protein ID mapping."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.protein_to_idx = data['protein_to_idx']
        self.idx_to_protein = {idx: prot for prot, idx in self.protein_to_idx.items()}
        self.embedding_dim = data['embedding_dim']
        self.use_onehot = data['use_onehot']
        self.n_proteins = len(self.protein_to_idx)
        
        if not self.use_onehot:
            # Reinitialize embeddings (in real use, these should be loaded from trained model)
            np.random.seed(42)
            self.embeddings = np.random.randn(self.n_proteins, self.embedding_dim).astype(np.float32)
            self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)


def get_protein_featurizer(config: Dict[str, Any]) -> Any:
    """
    Factory function to get the appropriate protein featurizer based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized protein featurizer object
    """
    feature_type = config.get('type', 'protein_identifier')
    
    if feature_type == 'protein_sequence':
        return ProteinSequenceFeaturizer(
            include_composition=config.get('include_composition', True),
            include_properties=config.get('include_properties', True),
            include_dipeptides=config.get('include_dipeptides', False)
        )
    elif feature_type == 'protein_identifier':
        return ProteinIdentifierFeaturizer(
            embedding_dim=config.get('embedding_dim', 32),
            use_onehot=config.get('use_onehot', False)
        )
    else:
        raise ValueError(f"Unknown protein feature type: {feature_type}")
