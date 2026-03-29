"""
Example: Training with Protein Features

This example demonstrates how to train models that take both ligand
and protein features into account.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data.data_loader import DataLoader
from features.featurizer import MorganFingerprintFeaturizer
from features.protein_featurizer import ProteinIdentifierFeaturizer, ProteinSequenceFeaturizer
from features.combined_featurizer import CombinedFeaturizer
from models.rf_trainer import RandomForestTrainer


def example_protein_aware_training():
    """
    Example: Train a protein-aware binding affinity model.
    """
    print("\n" + "="*70)
    print("Example: Protein-Aware Binding Affinity Prediction")
    print("="*70 + "\n")
    
    # Load data with protein information
    loader = DataLoader('../../training_data_full/registry.csv')
    loader.load_registry()
    
    # Get training data
    train_data = loader.get_training_data(
        similarity_threshold='0p7',
        include_decoys=True,
        split='train'
    )
    
    # Extract SMILES, labels, AND protein IDs
    smiles, labels, protein_ids = loader.prepare_features_labels(
        train_data, 
        include_protein_info=True
    )
    
    # Use subset for demo
    n_samples = min(1000, len(smiles))
    smiles = smiles[:n_samples]
    labels = labels[:n_samples]
    protein_ids = protein_ids[:n_samples]
    
    print(f"\nDataset Info:")
    print(f"  Samples: {len(smiles)}")
    print(f"  Unique proteins: {len(set(protein_ids))}")
    print(f"  Active compounds: {sum(labels)}")
    
    # Create combined featurizer
    print("\n1. Setting up combined featurizer...")
    
    # Ligand featurizer
    ligand_featurizer = MorganFingerprintFeaturizer(radius=2, n_bits=2048)
    
    # Protein featurizer (using identifiers)
    protein_featurizer = ProteinIdentifierFeaturizer(embedding_dim=32)
    
    # Combine
    combined_featurizer = CombinedFeaturizer(
        ligand_config={'type': 'morgan_fingerprint', 'radius': 2, 'n_bits': 2048},
        protein_config={'type': 'protein_identifier', 'embedding_dim': 32},
        concatenation_method='concat'
    )
    
    # Fit protein featurizer on all protein IDs
    combined_featurizer.fit_protein_featurizer(protein_ids)
    
    # Generate combined features
    print("\n2. Generating combined ligand-protein features...")
    X, invalid = combined_featurizer.featurize(
        smiles_list=smiles,
        protein_ids=protein_ids,
        show_progress=True
    )
    
    print(f"\nFeature dimensions:")
    config = combined_featurizer.get_config()
    print(f"  Ligand features:  {config['ligand_dim']}D")
    print(f"  Protein features: {config['protein_dim']}D")
    print(f"  Combined:         {config['total_dim']}D")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train model
    print("\n3. Training Random Forest model...")
    config = {
        'hyperparameters': {
            'n_estimators': 50,
            'max_depth': 10,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    }
    
    trainer = RandomForestTrainer(config)
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    print("\n4. Model trained successfully!")
    print(f"   Val ROC-AUC: {history['val_metrics']['roc_auc']:.4f}")
    
    # Save model
    output_dir = './example_models/protein_aware_rf'
    trainer.save_model(output_dir)
    combined_featurizer.save_protein_mapping(f'{output_dir}/protein_mapping.json')
    
    print(f"\n✓ Model and protein mapping saved to {output_dir}")


def example_compare_with_without_protein():
    """
    Example: Compare models with and without protein features.
    """
    print("\n" + "="*70)
    print("Example: Comparing Ligand-Only vs Ligand+Protein Models")
    print("="*70 + "\n")
    
    # Load data
    loader = DataLoader('../../training_data_full/registry.csv')
    loader.load_registry()
    
    train_data = loader.get_training_data(
        similarity_threshold='0p7',
        split='train'
    )
    
    smiles, labels, protein_ids = loader.prepare_features_labels(
        train_data, include_protein_info=True
    )
    
    # Use subset
    n_samples = min(500, len(smiles))
    smiles = smiles[:n_samples]
    labels = labels[:n_samples]
    protein_ids = protein_ids[:n_samples]
    
    from sklearn.model_selection import train_test_split
    
    # Model 1: Ligand-only
    print("\n1. Training LIGAND-ONLY model...")
    ligand_featurizer = MorganFingerprintFeaturizer(radius=2, n_bits=2048)
    X_ligand, _ = ligand_featurizer.featurize(smiles, show_progress=False)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_ligand, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    config = {'hyperparameters': {'n_estimators': 50, 'random_state': 42}}
    trainer1 = RandomForestTrainer(config)
    history1 = trainer1.train(X_train, y_train, X_val, y_val)
    
    # Model 2: Ligand + Protein
    print("\n2. Training LIGAND+PROTEIN model...")
    combined_featurizer = CombinedFeaturizer(
        ligand_config={'type': 'morgan_fingerprint', 'radius': 2, 'n_bits': 2048},
        protein_config={'type': 'protein_identifier', 'embedding_dim': 32},
        concatenation_method='concat'
    )
    
    combined_featurizer.fit_protein_featurizer(protein_ids)
    X_combined, _ = combined_featurizer.featurize(
        smiles, protein_ids=protein_ids, show_progress=False
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    trainer2 = RandomForestTrainer(config)
    history2 = trainer2.train(X_train, y_train, X_val, y_val)
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print("\nLigand-Only Model:")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        val = history1['val_metrics'][metric]
        print(f"  {metric:15s}: {val:.4f}")
    
    print("\nLigand+Protein Model:")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        val = history2['val_metrics'][metric]
        print(f"  {metric:15s}: {val:.4f}")
    
    print("\nImprovement (Ligand+Protein vs Ligand-Only):")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        val1 = history1['val_metrics'][metric]
        val2 = history2['val_metrics'][metric]
        improvement = ((val2 - val1) / val1) * 100
        print(f"  {metric:15s}: {improvement:+.2f}%")
    
    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    print("\nProtein-Aware ML Training Examples")
    print("=" * 70)
    print("\nSelect an example:")
    print("  1. Basic protein-aware training")
    print("  2. Compare ligand-only vs ligand+protein models")
    
    # For non-interactive, run example 1
    try:
        example_protein_aware_training()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("  1. Dependencies installed: pip install -r requirements.txt")
        print("  2. Registry file exists: ../../training_data_full/registry.csv")
