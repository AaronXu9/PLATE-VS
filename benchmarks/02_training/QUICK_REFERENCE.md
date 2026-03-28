# Quick Reference Card

## 🚀 Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train with default settings
python train_classical_oddt.py --use-precomputed-split

# Train with custom config
python train_classical_oddt.py --config ../configs/classical_config.yaml

# Run examples
python example_usage.py
```

## 📁 File Guide

| File | Purpose |
|------|---------|
| `train_classical_oddt.py` | Main training script (CLI) |
| `requirements.txt` | Package dependencies |
| `data/data_loader.py` | Load registry.csv, filter data |
| `features/featurizer.py` | SMILES → fingerprints |
| `models/base_trainer.py` | Base class for all models |
| `models/rf_trainer.py` | Random Forest implementation |
| `models/template_trainer.py` | Template for new models |
| `example_usage.py` | Python API examples |

## 📚 Documentation Guide

| Document | What to Read |
|----------|--------------|
| **QUICKSTART.md** | First-time setup and basic usage |
| **README.md** | Complete feature documentation |
| **ARCHITECTURE.md** | System design and how to extend |
| **IMPLEMENTATION_SUMMARY.md** | What was built and why |
| **This file** | Quick reference for common tasks |

## 🔧 Common Tasks

### Load Data
```python
from data.data_loader import DataLoader

loader = DataLoader('../../training_data_full/registry.csv')
data = loader.get_training_data(split='train', similarity_threshold='0p7')
smiles, labels = loader.prepare_features_labels(data)
```

### Generate Features
```python
from features.featurizer import MorganFingerprintFeaturizer

featurizer = MorganFingerprintFeaturizer(radius=2, n_bits=2048)
X, invalid = featurizer.featurize(smiles)
```

### Train Model
```python
from models.rf_trainer import RandomForestTrainer

config = {'hyperparameters': {'n_estimators': 100, 'random_state': 42}}
trainer = RandomForestTrainer(config)
history = trainer.train(X_train, y_train, X_val, y_val)
```

### Save/Load Model
```python
# Save
trainer.save_model('./my_model')

# Load
trainer.load_model('./my_model')

# Predict
predictions = trainer.predict(X_test)
probabilities = trainer.predict_proba(X_test)
```

## ⚙️ Configuration Template

```yaml
model_type: "random_forest"

hyperparameters:
  n_estimators: 100
  max_depth: null
  class_weight: 'balanced'
  random_state: 42

features:
  type: "morgan_fingerprint"
  radius: 2
  n_bits: 2048

data:
  similarity_threshold: '0p7'
  include_decoys: true
```

## 📊 Evaluation Metrics

| Metric | What It Measures | Good Value |
|--------|------------------|------------|
| Accuracy | Overall correctness | >0.7 |
| Precision | Correct positives / predicted positives | >0.7 |
| Recall | Correct positives / actual positives | >0.7 |
| F1 Score | Balance of precision & recall | >0.7 |
| ROC-AUC | Ranking quality | >0.8 |
| Avg Precision | Area under PR curve | >0.7 |
| MCC | Correlation (-1 to 1) | >0.5 |

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError: sklearn | `pip install scikit-learn` |
| ModuleNotFoundError: rdkit | `conda install -c conda-forge rdkit` |
| FileNotFoundError: registry.csv | Check `--registry` path |
| MemoryError | Reduce `n_bits` or use fewer samples |
| Low accuracy | Try `class_weight='balanced'` |

## 🎯 Adding New Models (3 Steps)

1. **Create trainer**: Copy `models/template_trainer.py`
2. **Register**: Add to `get_trainer()` in `train_classical_oddt.py`
3. **Config**: Create YAML in `../configs/`

## 📦 Output Files

After training, find in `trained_models/`:
- `*.pkl` - Trained model
- `*_config.json` - Hyperparameters
- `*_history.json` - Training metrics
- `*_training_summary.json` - Complete summary

## 🔑 Key Classes

```python
# Data
DataLoader(registry_path)

# Features  
MorganFingerprintFeaturizer(radius=2, n_bits=2048)
DescriptorFeaturizer(descriptor_names=None)

# Models
RandomForestTrainer(config)
# Future: GBMTrainer, SVMTrainer, etc.
```

## 💡 Tips

- Start with `n_estimators=100`, `n_bits=2048`
- Use `--use-precomputed-split` for consistency
- Enable `cross_validation: true` to check stability
- Set `class_weight: 'balanced'` for imbalanced data
- Check `*_training_summary.json` for full results

## 🔗 Related Files

- Config: `../configs/classical_config.yaml`
- Data: `../../training_data_full/registry.csv`
- Output: `./trained_models/`

---

**Need more detail?** See README.md or ARCHITECTURE.md
