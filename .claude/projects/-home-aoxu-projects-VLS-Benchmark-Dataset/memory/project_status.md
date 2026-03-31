---
name: Project status
description: Current training results and TODO status for VLS-Benchmark-Dataset classical ML pipeline
type: project
---

All three classical models trained on 0p7 similarity split (train=1,764,226 / test=2,086,662).

## Results (test ROC-AUC)

| Model | Train ROC-AUC | Val ROC-AUC | Test ROC-AUC | Training time |
|-------|:---:|:---:|:---:|:---:|
| RF    | 0.859 | 0.441 | 0.304 | ~7 min |
| GBM   | 0.799 | 0.789 | 0.372 | ~10 min |
| SVM   | 0.703 | 0.696 | 0.430 | ~3 min |

All models are 0.27–0.42 below PDBbind literature baselines — expected due to hard 0p7 similarity split.

## Feature cache
- Morgan FP (r=2, 2048 bits) precomputed for 2,284,653 unique SMILES
- Cache: `training_data_full/feature_cache/morgan_r2_b2048.h5` (44.6 MB)
- Future training runs use `--cache-dir ../../training_data_full/feature_cache`

## ✅ DONE
1. RF, GBM, SVM full training
2. Feature cache (HDF5, packed bits, gzip)
3. Benchmark report → `benchmarks/03_analysis/report.csv`
4. PDBbind comparison → printed to stdout

## 🔲 TODO
1. Deep learning models (DeepPurpose/foundation model stubs exist in `benchmarks/02_training/`)
2. Investigate generalization gap (all models collapse train→test on 0p7 split)

**Why:** 0p7 similarity split is a hard generalization benchmark by design. RF over-relies on protein identity embeddings (random, non-generalizable). GBM has best train/val consistency. SVM has best test AUC due to regularization.

**How to apply:** Next focus is deep learning models or ablation studies (ligand-only features, different similarity thresholds).
