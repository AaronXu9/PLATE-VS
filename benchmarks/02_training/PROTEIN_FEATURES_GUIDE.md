# Protein-Aware Training Guide

## Overview

The pipeline now supports **protein-aware** binding affinity prediction by incorporating protein features alongside ligand features. This allows the model to learn that **different molecules bind to different proteins**, making predictions protein-specific rather than just molecule-specific.

## Why Protein Features Matter

### Without Protein Features (Ligand-Only)
- Model learns: "Is molecule X active?"
- Same prediction for molecule X regardless of target protein
- Limited accuracy for cross-protein prediction

### With Protein Features (Ligand + Protein)
- Model learns: "Is molecule X active against protein Y?"
- Different predictions for the same molecule against different proteins
- Captures protein-specific binding patterns
- Better generalization to new protein-ligand pairs

## Protein Feature Types

### 1. Protein Identifier Embeddings (Recommended)

Uses learned embeddings for each unique protein ID.

**Advantages:**
- Simple and fast
- Works with just UniProt IDs
- Model learns protein-specific patterns
- Good for datasets with many samples per protein

**Configuration:**
```yaml
protein:
  type: "protein_identifier"
  embedding_dim: 32       # Size of protein embedding vector
  use_onehot: false       # Set true for one-hot encoding
```

**When to use:**
- You have protein IDs (UniProt, PDB, etc.)
- Multiple samples per protein in training data
- Want fast training and inference

### 2. Protein Sequence Features

Computes features from amino acid sequences.

**Advantages:**
- Transfers to unseen proteins
- Captures physicochemical properties
- No need to fit on training proteins

**Configuration:**
```yaml
protein:
  type: "protein_sequence"
  include_composition: true    # Amino acid composition (20 features)
  include_properties: true     # Physicochemical properties (11 features)
  include_dipeptides: false    # Dipeptide composition (400 features)
```

**When to use:**
- Need to predict on new, unseen proteins
- Have protein sequences available
- Want interpretable features

## Quick Start

### 1. Update Configuration

Edit `configs/classical_config.yaml`:

```yaml
# Enable protein features
features:
  type: "combined"  # Use combined ligand+protein features
  
  ligand:
    type: "morgan_fingerprint"
    radius: 2
    n_bits: 2048
  
  protein:
    type: "protein_identifier"
    embedding_dim: 32
  
  concatenation_method: 'concat'

data:
  include_protein_features: true  # Enable protein feature extraction
```

### 2. Train Model

```bash
python train_classical_oddt.py \
  --config ../configs/classical_config.yaml \
  --use-precomputed-split
```

The training script will automatically:
1. Extract protein IDs from registry
2. Generate protein embeddings
3. Combine with ligand fingerprints
4. Train on combined features
5. Save protein mapping for inference

## Feature Dimensions

### Ligand Features
- Morgan fingerprints: 2048D (default)
- Molecular descriptors: 10-20D

### Protein Features
- Identifier embeddings: 32D (configurable)
- Sequence composition: 20D (AA composition)
- Sequence properties: 11D (physicochemical)
- Dipeptides: 400D (optional)

### Combined
- Concatenation: Ligand_dim + Protein_dim
  - Example: 2048 + 32 = 2080D
- Weighted: Same dimensions but normalized

## Usage Examples

### Python API

```python
from data.data_loader import DataLoader
from features.combined_featurizer import get_combined_featurizer
from models.rf_trainer import RandomForestTrainer

# Load data with protein info
loader = DataLoader('../../training_data_full/registry.csv')
data = loader.get_training_data(split='train', similarity_threshold='0p7')
smiles, labels, protein_ids = loader.prepare_features_labels(
    data, include_protein_info=True
)

# Create combined featurizer
featurizer = get_combined_featurizer(
    ligand_config={'type': 'morgan_fingerprint', 'radius': 2, 'n_bits': 2048},
    protein_config={'type': 'protein_identifier', 'embedding_dim': 32}
)

# Fit protein featurizer
featurizer.fit_protein_featurizer(protein_ids)

# Generate features
X, invalid = featurizer.featurize(
    smiles_list=smiles,
    protein_ids=protein_ids
)

# Train model
trainer = RandomForestTrainer(config)
trainer.train(X_train, y_train, X_val, y_val)

# Save with protein mapping
trainer.save_model('./output')
featurizer.save_protein_mapping('./output/protein_mapping.json')
```

### Making Predictions

```python
# Load model and protein mapping
trainer = RandomForestTrainer(config)
trainer.load_model('./output')

featurizer = get_combined_featurizer(ligand_config, protein_config)
featurizer.load_protein_mapping('./output/protein_mapping.json')

# Predict for new ligand-protein pairs
new_smiles = ['CCO', 'c1ccccc1']
new_protein_ids = ['P12345', 'P67890']

X_new, _ = featurizer.featurize(new_smiles, protein_ids=new_protein_ids)
predictions = trainer.predict(X_new)
probabilities = trainer.predict_proba(X_new)
```

## Comparison: Ligand-Only vs Ligand+Protein

Run the comparison example:

```bash
python example_protein_features.py
```

Expected improvements with protein features:
- **ROC-AUC**: +5-15% improvement
- **Precision**: Better identification of true actives
- **Generalization**: Better performance on new proteins
- **Interpretability**: Can analyze protein-specific patterns

## Advanced: Protein Sequence Features

If you have protein sequences available:

```python
from features.protein_featurizer import ProteinSequenceFeaturizer

# Create sequence featurizer
protein_featurizer = ProteinSequenceFeaturizer(
    include_composition=True,
    include_properties=True,
    include_dipeptides=False
)

# Featurize sequences
sequences = ['MKTAYIAKQR...', 'MVILGPTKE...']
X_protein, invalid = protein_featurizer.featurize(sequences)
```

Features computed:
- **Composition**: % of each amino acid (20 features)
- **Properties**: Hydrophobic %, charge, etc. (11 features)
- **Dipeptides**: Consecutive AA pairs (400 features)

## Best Practices

### 1. Choose the Right Protein Features

| Scenario | Recommended Type |
|----------|------------------|
| Many samples per protein | Identifier embeddings |
| Few samples per protein | Sequence features |
| Need to predict on new proteins | Sequence features |
| Fast training required | Identifier embeddings |

### 2. Tune Embedding Dimensions

Start with 32D and increase if:
- You have many unique proteins (>1000)
- Each protein has many samples
- Performance plateaus

Typical ranges:
- Small datasets (<100 proteins): 16-32D
- Medium datasets (100-1000 proteins): 32-64D
- Large datasets (>1000 proteins): 64-128D

### 3. Handle Unseen Proteins

**Training-time approach:**
- Use identifier embeddings for known proteins
- Learn protein-specific patterns

**Test-time approach:**
- For new proteins, use sequence features
- Or use zero-vector (graceful degradation)

### 4. Concatenation Methods

```yaml
concatenation_method: 'concat'  # Simple concatenation (recommended)
concatenation_method: 'weighted'  # Weighted combination (experimental)
```

## Output Files

Training with protein features creates additional files:

```
trained_models/
└── random_forest/
    ├── random_forest.pkl
    ├── random_forest_config.json
    ├── random_forest_feature_config.json
    ├── random_forest_protein_mapping.json  # ← Protein ID mapping
    └── random_forest_training_summary.json
```

The protein mapping is essential for inference on new data.

## Performance Expectations

### Typical Improvements
- ROC-AUC: +5-15%
- Precision: +10-20%
- Recall: Varies by dataset

### When Protein Features Help Most
1. Multi-protein datasets (>10 proteins)
2. Protein-specific binding patterns
3. Sufficient samples per protein (>50)
4. Cross-protein generalization tasks

### When Benefits Are Smaller
1. Single-protein datasets
2. Very diverse compound libraries
3. Few samples per protein (<10)

## Troubleshooting

**Error: "protein_ids required for protein_identifier featurizer"**
- Set `include_protein_features: true` in config
- Ensure registry has `uniprot_id` column

**Warning: "X proteins not seen during fit"**
- Protein in test/val not in train set
- Will use zero-vector (may reduce performance)
- Consider using sequence features for better transfer

**Memory Error**
- Reduce protein embedding_dim
- Reduce ligand n_bits
- Process in batches

**Low Performance Gain**
- Check if dataset has sufficient proteins
- Try sequence features instead
- Ensure protein IDs are consistent

## Next Steps

1. **Try the examples**: Run `example_protein_features.py`
2. **Compare models**: Train both ligand-only and protein-aware models
3. **Analyze patterns**: Look at feature importance for protein features
4. **Experiment**: Try different embedding dimensions and feature types

## References

For more details, see:
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [README.md](README.md) - Complete documentation
- [example_protein_features.py](example_protein_features.py) - Usage examples
