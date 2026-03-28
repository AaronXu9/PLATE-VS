# Training Pipeline for Binding Affinity Prediction

This directory contains an extensible machine learning training pipeline for predicting protein-ligand binding affinity.

## ✨ NEW: Protein-Aware Models

The pipeline now supports **protein-aware** prediction! Models can learn protein-specific binding patterns by incorporating protein features alongside ligand features.

**Key Benefits:**
- 🎯 Protein-specific predictions (not just ligand-based)
- 📈 5-15% improvement in ROC-AUC
- 🔬 Better cross-protein generalization
- 🧬 Two feature types: protein IDs or sequences

**Quick Start with Protein Features:**
```bash
# Set features.type='combined' in config
python train_classical_oddt.py --use-precomputed-split
```

See [PROTEIN_FEATURES_GUIDE.md](PROTEIN_FEATURES_GUIDE.md) for details.

## Structure

```
02_training/
├── train_classical_oddt.py    # Main training script
├── data/
│   └── data_loader.py          # Data loading utilities
├── features/
│   ├── featurizer.py           # Molecular featurization
│   ├── protein_featurizer.py   # Protein featurization (NEW)
│   └── combined_featurizer.py  # Combined ligand+protein (NEW)
├── models/
│   ├── base_trainer.py         # Abstract base trainer class
│   └── rf_trainer.py           # Random Forest implementation
└── trained_models/             # Output directory for saved models
```

## Quick Start

### 0. Quick Test First (Recommended)

Before running full training, verify everything works:

```bash
# Quick test with 1000 samples
python train_classical_oddt.py --quick-test --use-precomputed-split

# Or run automated test suite
python test_training.py
```

This completes in seconds and verifies the entire pipeline.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Random Forest Model

**Option A: With Protein Features (Recommended)**
```bash
# Uses combined ligand+protein features
python train_classical_oddt.py --use-precomputed-split
```

**Option B: Ligand-Only (Legacy)**
```bash
# Edit config: set features.type='morgan_fingerprint' and 
# data.include_protein_features=false
python train_classical_oddt.py --use-precomputed-split
```

### 3. Custom Configuration

Modify `../configs/classical_config.yaml` to adjust hyperparameters:

```yaml
model_type: "random_forest"
hyperparameters:
  n_estimators: 200
  max_depth: 20
  class_weight: 'balanced'

# NEW: Protein-aware configuration
features:
  type: "combined"  # Use ligand + protein features
  ligand:
    type: "morgan_fingerprint"
    radius: 2
    n_bits: 2048
  protein:
    type: "protein_identifier"
    embedding_dim: 32

data:
  include_protein_features: true  # Enable protein features
```

## Command Line Options

```bash
python train_classical_oddt.py \
  --config ../configs/classical_config.yaml \
  --registry ../../training_data_full/registry.csv \
  --output ./trained_models \
  --use-precomputed-split

# With experiment tracking (WandB)
python train_classical_oddt.py \
    --config ../configs/classical_config.yaml \
    --use-precomputed-split \
    --wandb \
    --wandb-project binding-affinity
```

**Common Options:**
- `--config`: Path to YAML configuration file
- `--registry`: Path to registry.csv with training data
- `--output`: Directory to save trained models
- `--use-precomputed-split`: Use existing train/val/test splits from registry
- `--wandb`: Enable Weights & Biases experiment tracking
- `--wandb-project`: WandB project name
- `--quick-test`: Run quick test on small data subset (1000 samples)
- `--test-samples`: Number of samples for quick test mode

## 📊 Logging and Experiment Tracking

The pipeline includes comprehensive logging:

- **File Logging**: Timestamped logs saved to `trained_models/training_TIMESTAMP.log`
- **Console Logging**: Real-time progress updates
- **WandB Integration**: Track experiments, compare runs, visualize metrics

**Install WandB (optional):**
```bash
pip install wandb
wandb login
```

**Use WandB:**
```bash
python train_classical_oddt.py --wandb --wandb-project my-project
```

See [LOGGING_GUIDE.md](LOGGING_GUIDE.md) for complete documentation.

## Features

### Ligand Featurization

**Morgan Fingerprints (ECFP)**
- Circular fingerprints capturing structural features
- Configurable radius and bit length
- Default: radius=2, n_bits=2048

**Molecular Descriptors**
- Physicochemical properties (MW, LogP, etc.)
- RDKit descriptor calculations
- Configurable descriptor selection

### Protein Featurization (NEW)

**Protein Identifier Embeddings**
- Learned embeddings for each protein
- Fast and efficient
- Good for known proteins
- Default: 32-dimensional embeddings

**Protein Sequence Features**
- Amino acid composition
- Physicochemical properties
- Dipeptide composition (optional)
- Transfers to unseen proteins

**Combined Features**
- Concatenates ligand + protein features
- Total dimension: ligand_dim + protein_dim
- Example: 2048D (ligand) + 32D (protein) = 2080D

See [PROTEIN_FEATURES_GUIDE.md](PROTEIN_FEATURES_GUIDE.md) for detailed information.

### Supported Models

**Current:**
- ✅ Random Forest (RF)

**Planned:**
- 🔄 Gradient Boosting (GBM)
- 🔄 Support Vector Machines (SVM)
- 🔄 ODDT Scoring Functions
- 🔄 Neural Networks

## Adding New Models

The framework is designed to be extensible. To add a new model:

### 1. Create a new trainer class

```python
# models/your_model_trainer.py
from .base_trainer import BaseTrainer

class YourModelTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config, model_name='your_model')
    
    def build_model(self):
        # Initialize your model
        return your_model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Training logic
        self.model = self.build_model()
        self.model.fit(X_train, y_train)
        return self.training_history
```

### 2. Register in the factory function

```python
# train_classical_oddt.py
def get_trainer(model_type, config):
    if model_type == 'random_forest':
        return RandomForestTrainer(config)
    elif model_type == 'your_model':
        return YourModelTrainer(config)
    # ...
```

### 3. Create a config file

```yaml
# configs/your_model_config.yaml
model_type: "your_model"
hyperparameters:
  param1: value1
  param2: value2
```

## Output

After training, the following files are saved in the output directory:

- `{model_name}.pkl` - Trained model (joblib format)
- `{model_name}_config.json` - Model configuration
- `{model_name}_history.json` - Training history and metrics
- `{model_name}_feature_config.json` - Feature generation settings
- `{model_name}_training_summary.json` - Complete training summary

## Evaluation Metrics

The pipeline computes the following metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity / True positive rate
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Average Precision**: Area under precision-recall curve
- **MCC**: Matthews Correlation Coefficient

## Data Format

The pipeline expects a `registry.csv` file with the following columns:

- `smiles`: SMILES representation of molecules
- `is_active`: Binary label (True/False or 1/0)
- `similarity_threshold`: Threshold used for data filtering
- `split`: Data split ('train', 'val', 'test', 'decoy')
- `uniprot_id`: Protein identifier (optional)

## Example Usage in Python

```python
from data.data_loader import DataLoader
from features.featurizer import MorganFingerprintFeaturizer
from models.rf_trainer import RandomForestTrainer

# Load data
loader = DataLoader('../../training_data_full/registry.csv')
train_data = loader.get_training_data(split='train', similarity_threshold='0p7')
smiles, labels = loader.prepare_features_labels(train_data)

# Generate features
featurizer = MorganFingerprintFeaturizer(radius=2, n_bits=2048)
X, invalid = featurizer.featurize(smiles)

# Train model
config = {'hyperparameters': {'n_estimators': 100, 'random_state': 42}}
trainer = RandomForestTrainer(config)
history = trainer.train(X, labels)

# Save model
trainer.save_model('./my_model')
```

## Tips

1. **Quick Test First**: Always run `--quick-test` before full training to catch issues early
2. **Use WandB**: Track experiments with `--wandb` for better organization and comparison
3. **Class Imbalance**: Use `class_weight: 'balanced'` in the config for imbalanced datasets
4. **Feature Selection**: Start with Morgan fingerprints (radius=2, n_bits=2048)
5. **Hyperparameter Tuning**: Enable cross-validation to assess model stability
6. **Memory**: Reduce `n_bits` if running into memory issues
7. **Speed**: Adjust `n_jobs` based on available CPU cores
8. **Protein Features**: Use combined features for better cross-protein generalization

## 📚 Documentation

- **[LOGGING_GUIDE.md](LOGGING_GUIDE.md)** - Complete guide to logging, WandB, and quick testing
- **[PROTEIN_FEATURES_GUIDE.md](PROTEIN_FEATURES_GUIDE.md)** - Using protein structure features
- **[VISUAL_GUIDE.md](VISUAL_GUIDE.md)** - Visual diagrams of the architecture
- **[PROTEIN_INTEGRATION_SUMMARY.md](PROTEIN_INTEGRATION_SUMMARY.md)** - Technical integration details

## Troubleshooting

**Out of Memory**
- Reduce `n_bits` in feature configuration
- Process data in batches
- Use fewer `n_estimators`

**Poor Performance**
- Check class balance in the data
- Try different `class_weight` settings
- Increase model complexity (`n_estimators`, `max_depth`)
- Enable cross-validation to check for overfitting

**Invalid SMILES**
- Check the data loader output for invalid molecule warnings
- These are replaced with zero vectors automatically
