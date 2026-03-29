# Training Pipeline Architecture

## Overview

This is an extensible, modular machine learning pipeline for predicting protein-ligand binding affinity. The architecture is designed to make it easy to add new models, feature types, and evaluation methods.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │    Data      │─────▶│  Features    │─────▶│   Models     │  │
│  │   Loader     │      │  Generator   │      │   Trainer    │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│        │                      │                      │          │
│        │                      │                      │          │
│        ▼                      ▼                      ▼          │
│  registry.csv          Fingerprints           Trained Model    │
│  (SMILES + labels)     (2048-dim vectors)     (RF, GBM, etc.)  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Layer (`data/`)

**DataLoader** - Handles all data operations
- Loads registry.csv
- Filters by similarity threshold and split
- Extracts SMILES and labels
- Creates train/val/test splits
- Computes dataset statistics

```python
loader = DataLoader('registry.csv')
train_data = loader.get_training_data(split='train', threshold='0p7')
smiles, labels = loader.prepare_features_labels(train_data)
```

### 2. Feature Layer (`features/`)

**Featurizers** - Convert SMILES to numerical features

- **MorganFingerprintFeaturizer**: Circular fingerprints (ECFP)
  - Captures substructure patterns
  - Configurable radius and bit size
  - Most common for virtual screening

- **DescriptorFeaturizer**: Physicochemical properties
  - MW, LogP, H-bond donors/acceptors, etc.
  - Fast computation
  - Interpretable features

```python
featurizer = MorganFingerprintFeaturizer(radius=2, n_bits=2048)
X, invalid = featurizer.featurize(smiles)
```

### 3. Model Layer (`models/`)

**BaseTrainer** - Abstract base class
- Defines standard interface
- Common evaluation methods
- Model saving/loading
- Feature importance extraction

**Specific Trainers** - Model implementations
- RandomForestTrainer ✅
- GradientBoostingTrainer 🔄 (planned)
- SVMTrainer 🔄 (planned)
- NeuralNetTrainer 🔄 (planned)

```python
trainer = RandomForestTrainer(config)
history = trainer.train(X_train, y_train, X_val, y_val)
trainer.save_model('./output')
```

## Data Flow

```
Input Data (registry.csv)
    │
    ├─ sample_id, uniprot_id, pdb_id
    ├─ smiles                              ◀── Molecular structure
    ├─ is_active (True/False)              ◀── Label
    ├─ similarity_threshold (0p3, 0p5, 0p7, 0p9)
    └─ split (train, val, test, decoy)
    │
    ▼
DataLoader.get_training_data()
    │ Filters by threshold and split
    │ Includes/excludes decoys
    ▼
SMILES List + Labels Array
    │
    ▼
Featurizer.featurize()
    │ Converts SMILES → Fingerprints
    │ Handles invalid molecules
    ▼
Feature Matrix (N × 2048) + Labels (N,)
    │
    ├─ Train set (80%)
    ├─ Val set (20%)
    └─ Test set (optional)
    │
    ▼
Trainer.train()
    │ Fits model on train
    │ Evaluates on val/test
    ▼
Trained Model + Metrics
    │
    ├─ model.pkl
    ├─ config.json
    ├─ history.json
    └─ summary.json
```

## Extension Points

### Adding a New Model

1. **Create trainer class** inheriting from `BaseTrainer`:

```python
# models/gbm_trainer.py
from .base_trainer import BaseTrainer
from sklearn.ensemble import GradientBoostingClassifier

class GBMTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config, model_name='gradient_boosting')
    
    def build_model(self):
        return GradientBoostingClassifier(**self.hyperparameters)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model = self.build_model()
        self.model.fit(X_train, y_train)
        # ... evaluation logic
        return self.training_history
```

2. **Register in factory**:

```python
# train_classical_oddt.py
def get_trainer(model_type, config):
    if model_type == 'random_forest':
        return RandomForestTrainer(config)
    elif model_type == 'gradient_boosting':
        return GBMTrainer(config)
```

3. **Create config**:

```yaml
# configs/gbm_config.yaml
model_type: "gradient_boosting"
hyperparameters:
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 5
```

### Adding a New Feature Type

1. **Create featurizer class**:

```python
# features/featurizer.py
class CustomFeaturizer:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.name = "custom_features"
    
    def featurize(self, smiles_list):
        # Convert SMILES to features
        return feature_matrix, invalid_indices
    
    def get_config(self):
        return {'type': 'custom', 'param1': self.param1}
```

2. **Register in factory**:

```python
def get_featurizer(config):
    if config['type'] == 'morgan_fingerprint':
        return MorganFingerprintFeaturizer(...)
    elif config['type'] == 'custom':
        return CustomFeaturizer(...)
```

## Configuration System

The pipeline uses YAML configuration files for reproducibility:

```yaml
# Complete configuration structure
model_type: "random_forest"           # Which model to use

hyperparameters:                      # Model-specific parameters
  n_estimators: 100
  max_depth: null
  class_weight: 'balanced'
  random_state: 42

features:                             # Feature generation
  type: "morgan_fingerprint"
  radius: 2
  n_bits: 2048

data:                                 # Data loading
  similarity_threshold: '0p7'
  include_decoys: true
  val_split: 0.2

cross_validation: false               # Optional CV
cv_folds: 5
```

## Training Modes

### Mode 1: Precomputed Splits (Recommended)

Uses the existing train/val/test splits from registry:

```bash
python train_classical_oddt.py --use-precomputed-split
```

Advantages:
- Consistent with other experiments
- Proper test set isolation
- Reproducible across runs

### Mode 2: Custom Splits

Creates new train/val splits:

```bash
python train_classical_oddt.py
```

Advantages:
- Flexible split ratios
- Can combine different thresholds
- Useful for experimentation

## Output Structure

```
trained_models/
└── {model_name}/
    ├── {model_name}.pkl                 # Serialized model (joblib)
    ├── {model_name}_config.json         # Hyperparameters used
    ├── {model_name}_history.json        # Training metrics over time
    ├── {model_name}_feature_config.json # Feature generation settings
    └── {model_name}_training_summary.json # Complete run summary
```

## Metrics and Evaluation

All models report comprehensive metrics:

| Metric | Description | Range | Good Value |
|--------|-------------|-------|------------|
| Accuracy | Overall correctness | 0-1 | >0.7 |
| Precision | Positive predictive value | 0-1 | >0.7 |
| Recall | Sensitivity | 0-1 | >0.7 |
| F1 Score | Harmonic mean of P&R | 0-1 | >0.7 |
| ROC-AUC | Area under ROC curve | 0-1 | >0.8 |
| Avg Precision | Area under PR curve | 0-1 | >0.7 |
| MCC | Matthews correlation | -1 to 1 | >0.5 |

## Best Practices

1. **Start Simple**: Begin with Random Forest and Morgan fingerprints
2. **Use Cross-Validation**: Enable CV to check model stability
3. **Handle Class Imbalance**: Use `class_weight='balanced'`
4. **Monitor All Metrics**: Don't rely on accuracy alone
5. **Save Everything**: Keep configs and histories for reproducibility
6. **Test Incrementally**: Start with small datasets to verify pipeline

## Performance Considerations

- **Feature Generation**: Most time-consuming step (parallelizable)
- **Training**: RF is fast, scales well with cores (`n_jobs=-1`)
- **Memory**: ~2048 floats per molecule × N molecules
- **Disk Space**: Models are typically <100MB

## Future Enhancements

Planned additions:
- [ ] Gradient Boosting Models (XGBoost, LightGBM)
- [ ] Support Vector Machines
- [ ] Neural Networks (fully connected)
- [ ] Graph Neural Networks
- [ ] ODDT scoring functions
- [ ] Hyperparameter optimization (Optuna)
- [ ] Model ensembles
- [ ] Active learning
