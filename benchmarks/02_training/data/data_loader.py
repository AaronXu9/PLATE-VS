"""
Data Loader Module.

This module handles loading and preprocessing of the training data
from the registry.csv file.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Load and preprocess training data from the registry.
    """
    
    def __init__(self, registry_path: str, root_dir: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            registry_path: Path to the registry.csv file
            root_dir: Root directory for resolving relative paths (optional)
        """
        self.registry_path = Path(registry_path)
        self.root_dir = Path(root_dir) if root_dir else self.registry_path.parent
        self.registry = None
        
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry file not found: {self.registry_path}")
    
    def load_registry(self) -> pd.DataFrame:
        """
        Load the registry CSV file.
        
        Returns:
            DataFrame containing the registry data
        """
        print(f"Loading registry from {self.registry_path}")
        self.registry = pd.read_csv(self.registry_path)
        print(f"Loaded {len(self.registry)} samples from registry")
        return self.registry
    
    def get_training_data(self,
                         similarity_threshold: str = '0p7',
                         include_decoys: bool = True,
                         split: Optional[str] = None,
                         protein_partition: Optional[str] = None) -> pd.DataFrame:
        """
        Get training data filtered by similarity threshold, ligand split, and protein partition.

        Args:
            similarity_threshold: Similarity threshold to use (e.g., '0p7', '0p5')
            include_decoys: Whether to include decoy compounds
            split: Ligand-based split ('train', 'test', or None for all)
            protein_partition: Protein cluster partition ('train', 'val', 'test', or None).
                Only used when the registry has a 'protein_partition' column
                (i.e. registry_2d_split.csv). When set, only proteins assigned to
                this partition are returned. Decoys are always included regardless
                of protein_partition (they are not protein-specific).

        Returns:
            Filtered DataFrame
        """
        if self.registry is None:
            self.load_registry()

        has_protein_partition = 'protein_partition' in self.registry.columns

        # Build filter conditions
        conditions = []

        if split:
            if include_decoys and split in ('train', 'test'):
                # Get the active compounds for this split + threshold
                actives_mask = (
                    (self.registry['split'] == split) &
                    (self.registry['similarity_threshold'] == similarity_threshold)
                )
                # Apply protein partition filter on actives (not decoys)
                if protein_partition and has_protein_partition:
                    actives_mask = actives_mask & (
                        self.registry['protein_partition'] == protein_partition
                    )
                if split == 'test':
                    # For test: include only protein-matched decoys (realistic per-target VS)
                    test_proteins = set(
                        self.registry.loc[actives_mask, 'uniprot_id'].unique()
                    )
                    decoys_mask = (
                        (self.registry['split'] == 'decoy') &
                        (self.registry['uniprot_id'].isin(test_proteins))
                    )
                else:
                    # For train: include all decoys (or protein-matched if partition given)
                    if protein_partition and has_protein_partition:
                        train_proteins = set(
                            self.registry.loc[actives_mask, 'uniprot_id'].unique()
                        )
                        decoys_mask = (
                            (self.registry['split'] == 'decoy') &
                            (self.registry['uniprot_id'].isin(train_proteins))
                        )
                    else:
                        decoys_mask = self.registry['split'] == 'decoy'
                conditions.append(actives_mask | decoys_mask)
            else:
                conditions.append(self.registry['split'] == split)
                if split != 'decoy':
                    conditions.append(self.registry['similarity_threshold'] == similarity_threshold)
                if protein_partition and has_protein_partition and split != 'decoy':
                    conditions.append(
                        self.registry['protein_partition'] == protein_partition
                    )
        else:
            # Get all data for the specified threshold
            if include_decoys:
                conditions.append(
                    (self.registry['similarity_threshold'] == similarity_threshold) |
                    (self.registry['split'] == 'decoy')
                )
            else:
                conditions.append(self.registry['similarity_threshold'] == similarity_threshold)
            if protein_partition and has_protein_partition:
                # Filter non-decoy rows by protein partition; keep all decoys
                conditions.append(
                    (self.registry['protein_partition'] == protein_partition) |
                    (self.registry['split'] == 'decoy')
                )

        # Apply filters
        mask = conditions[0]
        for cond in conditions[1:]:
            mask = mask & cond

        filtered_data = self.registry[mask].copy()
        print(f"Filtered data: {len(filtered_data)} samples for split='{split}', "
              f"threshold='{similarity_threshold}', include_decoys={include_decoys}"
              + (f", protein_partition='{protein_partition}'" if protein_partition else ""))

        return filtered_data
    
    def prepare_features_labels(self,
                                data: pd.DataFrame,
                                smiles_column: str = 'smiles',
                                label_column: str = 'is_active',
                                task: str = 'classification',
                                include_protein_info: bool = False,
                                protein_id_column: str = 'uniprot_id') -> Tuple:
        """
        Extract SMILES, labels, and optionally protein info from the data.

        Args:
            data: DataFrame from get_training_data()
            smiles_column: Column containing SMILES strings
            label_column: Column for binary labels (used only when task='classification')
            task: 'classification' returns binary int labels from label_column;
                  'regression' returns float pchembl values, rows with NaN pchembl excluded
            include_protein_info: Whether to also return protein identifier list
            protein_id_column: Column containing protein UniProt IDs

        Returns:
            Tuple of (SMILES list, labels array) or
            (SMILES list, labels array, protein_ids list) if include_protein_info=True
        """
        if smiles_column not in data.columns:
            raise ValueError(f"SMILES column '{smiles_column}' not found in data")

        if task == 'regression':
            if 'pchembl' not in data.columns:
                raise ValueError(
                    "Registry does not have a 'pchembl' column. "
                    "Run assign_protein_splits_soft.py to generate registry_soft_split.csv first."
                )
            data = data[data['pchembl'].notna()].copy()
            labels = data['pchembl'].values.astype(np.float32)
            print(f"Regression mode: {len(labels)} samples with valid pChEMBL values")
            print(f"  pChEMBL range: {labels.min():.2f} – {labels.max():.2f} "
                  f"(mean {labels.mean():.2f})")
        elif task == 'classification':
            if label_column not in data.columns:
                raise ValueError(f"Label column '{label_column}' not found in data")
            labels = data[label_column].astype(int).values
            print(f"Prepared {len(labels)} samples with labels")
            print(f"  Active compounds: {np.sum(labels == 1)}")
            print(f"  Inactive compounds: {np.sum(labels == 0)}")
            print(f"  Class balance: {np.mean(labels):.2%} active")
        else:
            raise ValueError(
                f"Unknown task='{task}'. Choose 'classification' or 'regression'."
            )

        smiles = data[smiles_column].tolist()

        if include_protein_info:
            if protein_id_column not in data.columns:
                raise ValueError(f"Protein ID column '{protein_id_column}' not found in data")
            protein_ids = data[protein_id_column].tolist()
            print(f"  Unique proteins: {len(set(protein_ids))}")
            return smiles, labels, protein_ids

        return smiles, labels
    
    def split_data(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   val_size: float = 0.2,
                   test_size: Optional[float] = None,
                   random_state: int = 42,
                   stratify: bool = True) -> Dict[str, np.ndarray]:
        """
        Split data into train/val/test sets.
        
        Args:
            X: Feature matrix
            y: Labels
            val_size: Validation set size (fraction)
            test_size: Test set size (fraction, optional)
            random_state: Random seed
            stratify: Whether to stratify the split
            
        Returns:
            Dictionary containing train/val/test splits
        """
        stratify_y = y if stratify else None
        
        if test_size is not None and test_size > 0:
            # Three-way split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
            )
            
            # Adjust val_size for the remaining data
            adjusted_val_size = val_size / (1 - test_size)
            stratify_temp = y_temp if stratify else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=adjusted_val_size, 
                random_state=random_state, stratify=stratify_temp
            )
            
            splits = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }
            
            print(f"Data split into:")
            print(f"  Train: {len(y_train)} samples ({np.mean(y_train):.2%} active)")
            print(f"  Val:   {len(y_val)} samples ({np.mean(y_val):.2%} active)")
            print(f"  Test:  {len(y_test)} samples ({np.mean(y_test):.2%} active)")
        else:
            # Two-way split (train/val only)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=random_state, stratify=stratify_y
            )
            
            splits = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val
            }
            
            print(f"Data split into:")
            print(f"  Train: {len(y_train)} samples ({np.mean(y_train):.2%} active)")
            print(f"  Val:   {len(y_val)} samples ({np.mean(y_val):.2%} active)")
        
        return splits
    
    def get_dataset_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_samples': len(data),
            'n_active': int(data['is_active'].sum()),
            'n_inactive': int((~data['is_active']).sum()),
            'active_ratio': float(data['is_active'].mean()),
        }
        
        if 'split' in data.columns:
            stats['split_distribution'] = data['split'].value_counts().to_dict()
        
        if 'similarity_threshold' in data.columns:
            stats['threshold_distribution'] = data['similarity_threshold'].value_counts().to_dict()
        
        if 'uniprot_id' in data.columns:
            stats['n_unique_proteins'] = int(data['uniprot_id'].nunique())
            stats['top_proteins'] = data['uniprot_id'].value_counts().head(10).to_dict()
        
        return stats
