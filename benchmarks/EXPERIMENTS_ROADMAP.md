# Experiments Roadmap

End-to-end commands for every experiment in this repo, in the order you'd run them. Each workflow is self-contained and points to the methodology doc and config file that drive it.

For the high-level summary of results see `docs/presentation/results_summary.md`. For split construction details see `SOFT_SPLIT_REGISTRY.md` and `BENCHMARKING_GUIDE.md`.

---

## 0. Environments

| Conda env | Use for | Source |
|---|---|---|
| `rdkit_env` | Classical ML, GNINA, analysis, notebook execution | `benchmarks/envs/env_deep_learning.yml` |
| `binding_affinity` (CARC) | Dual-encoder training (06, 07) — needs torch 2.5 + PyG + torchmd-net | `benchmarks/envs/env_binding_affinity.yml` |
| `gems` | GEMS inference — needs the stack in `external/GEMS/README.md` §3 | `benchmarks/envs/env_gems.yml` |
| `boltzina_env` | Boltzina/Boltz-2 docking | `benchmarks/envs/env_boltzina.yml` |

`rdkit_env` on this machine needs `LD_LIBRARY_PATH=/home/aoxu/miniconda3/envs/rdkit_env/lib` for rdkit's `libstdc++` (GLIBCXX mismatch with system lib).

---

## 1. Classical ML — RF / GBM / SVM

**Methodology**: `benchmarks/02_training/README.md`, `benchmarks/02_training/PROTEIN_FEATURES_GUIDE.md`.
**Dataset**: PLATE-VS (both splits) and PDBbind CleanSplit.
**Features**: Morgan FP (ECFP4, 2048 bits) ⊕ 32-dim learned protein-ID embedding.

### 1.1 PLATE-VS hard 2D split (0p7 threshold)

```bash
conda run -n rdkit_env python3 benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/{classical,gbm,svm}_config.yaml \
    --registry training_data_full/registry_2d_split.csv \
    --output benchmarks/02_training/trained_models \
    --use-precomputed-split
```
Outputs: `benchmarks/02_training/trained_models/{random_forest,gradient_boosting,svm}_training_summary.json`.

### 1.2 PLATE-VS soft split (0p7 threshold)

```bash
conda run -n rdkit_env python3 benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/soft_split_config.yaml \
    --registry training_data_full/registry_soft_split.csv \
    --output trained_models/soft_split_classification \
    --use-2d-split
```
(Swap `model_type` inside the YAML for GBM/SVM or use the dedicated soft-split variants.) Outputs: `trained_models/soft_split_classification/*_training_summary.json`.

### 1.3 PLATE-VS soft-split regression (pChEMBL)

```bash
conda run -n rdkit_env python3 benchmarks/02_training/train_regression.py \
    --config benchmarks/configs/regression_{rf,gbm,svm}_config.yaml \
    --registry training_data_full/registry_soft_split_regression.csv \
    --use-2d-split
```
Outputs: `trained_models/regression/*_regressor_training_summary.json`.

### 1.4 PDBbind CleanSplit regression

```bash
conda run -n rdkit_env python3 benchmarks/05_pdbbind_comparison/train_classical_pdbbind.py \
    --config benchmarks/05_pdbbind_comparison/configs/classical_pdbbind_config.yaml
```
Two config variants: Morgan FP only (`classical/`) and FP + protein-embedding (`classical_with_prot_emb/`). Outputs under `benchmarks/05_pdbbind_comparison/results/<variant>/`.

---

## 2. GNINA — structure-based docking

**Methodology**: `benchmarks/BENCHMARKING_GUIDE.md` §6.
**Binary**: `/home/aoxu/projects/PoseBench/forks/GNINA/gnina` (GPU build).

### 2.1 PLATE-VS (15 diverse test-partition targets, hard split)

```bash
# Step-by-step (each idempotent)
conda run -n rdkit_env python3 benchmarks/04_docking/select_targets.py     --config benchmarks/04_docking/configs/gnina_config.yaml
conda run -n rdkit_env python3 benchmarks/04_docking/prepare_structures.py --config benchmarks/04_docking/configs/gnina_config.yaml
conda run -n rdkit_env python3 benchmarks/04_docking/prepare_ligands.py    --config benchmarks/04_docking/configs/gnina_config.yaml --max-decoys 500 --n-workers 16
conda run -n rdkit_env python3 benchmarks/04_docking/run_gnina_benchmark.py --config benchmarks/04_docking/configs/gnina_config.yaml
conda run -n rdkit_env python3 benchmarks/04_docking/collect_results.py    --config benchmarks/04_docking/configs/gnina_config.yaml
LD_LIBRARY_PATH=/home/aoxu/miniconda3/envs/rdkit_env/lib \
  /home/aoxu/miniconda3/envs/rdkit_env/bin/python3 benchmarks/04_docking/analyze_docking.py
```
Outputs: `benchmarks/04_docking/results/` — `gnina_training_summary.json` (pooled) + `docking_metrics/{uniprot}_docking_metrics.json` (per-target).

### 2.2 PDBbind CleanSplit CASF-2016

```bash
LD_LIBRARY_PATH=/home/aoxu/miniconda3/envs/rdkit_env/lib \
  /home/aoxu/miniconda3/envs/rdkit_env/bin/python benchmarks/05_pdbbind_comparison/run_gnina_pdbbind.py \
    --config benchmarks/05_pdbbind_comparison/configs/gnina_pdbbind_config.yaml
LD_LIBRARY_PATH=/home/aoxu/miniconda3/envs/rdkit_env/lib \
  /home/aoxu/miniconda3/envs/rdkit_env/bin/python benchmarks/05_pdbbind_comparison/collect_gnina_pdbbind.py \
    --config benchmarks/05_pdbbind_comparison/configs/gnina_pdbbind_config.yaml
```
**Subset config** for session-sized runs: `configs/gnina_pdbbind_subset20_config.yaml` (stratified 20-complex sample), backed by `data/pdbbind_cleansplit/labels/PDBbind_casf2016_subset20.json`. Use `python -c "..."` (see GNINA PDBbind §2 in `METHODS.md`) to regenerate the subset.
Outputs: `benchmarks/05_pdbbind_comparison/results/gnina_pdbbind_casf2016_training_summary.json`.

---

## 3. Dual-encoder (ours)

**Methodology**: `benchmarks/06_binding_affinity_model/METHODS.md`.

### 3.1 PDBbind CleanSplit — 5-fold CV (for each fold `f ∈ {0,1,2,3,4}`)

```bash
conda run -n binding_affinity python benchmarks/06_binding_affinity_model/train.py \
    --config benchmarks/06_binding_affinity_model/configs/et_hiqbind.yaml \
    --fold $f
```
Each fold writes `benchmarks/06_binding_affinity_model/results/dual_encoder_f${f}_casf2016_training_summary.json`.

After all 5 folds, aggregate ensemble metrics into `benchmarks/05_pdbbind_comparison/results/dual_encoder_ensemble_training_summary.json` by averaging per-complex predictions across folds (glue code lives in the PDBbind comparison analysis — see `06_binding_affinity_model/evaluate.py`).

### 3.2 PLATE-VS virtual screening (hard 0p7)

```bash
conda run -n binding_affinity python benchmarks/07_plate_vs_dl/train_vs.py \
    --config benchmarks/07_plate_vs_dl/configs/vs_default.yaml
```
Outputs: `benchmarks/07_plate_vs_dl/results/vs_0p7_training_summary.json` (once a full run completes). Current status: 2 epochs W&B-only, no persisted summary. Bottleneck: 11.5 h/epoch from on-the-fly RDKit conformer generation — next step is to precompute conformers.

---

## 4. GEMS inference (pretrained, PDBbind only)

**Methodology**: `benchmarks/05_pdbbind_comparison/GEMS_INTEGRATION.md`.

```bash
conda run -n gems python benchmarks/05_pdbbind_comparison/run_gems_inference.py \
    --config benchmarks/05_pdbbind_comparison/configs/gems_config.yaml \
    --test-set casf2016          # or casf2013
```
Outputs: `benchmarks/05_pdbbind_comparison/results/gems/gems_{test_set}_training_summary.json`.

PLATE-VS extension is not yet implemented — see §5 of `GEMS_INTEGRATION.md` for what it would entail.

---

## 5. Unified reporting

After any subset of experiments completes:

```bash
# Master classification + regression report (PLATE-VS + GNINA)
conda run -n rdkit_env python3 benchmarks/03_analysis/generate_benchmark_report.py \
    --results-dir benchmarks/02_training/trained_models \
    --extra-dirs trained_models/soft_split_classification trained_models/regression \
    --docking-dir benchmarks/04_docking/results \
    --output benchmarks/03_analysis/report.csv

# PDBbind-specific report
conda run -n rdkit_env python3 benchmarks/05_pdbbind_comparison/generate_pdbbind_report.py
#  → benchmarks/05_pdbbind_comparison/results/pdbbind_comparison_report.csv

# Presentation figures (reads all result JSONs above)
conda run -n rdkit_env jupyter nbconvert --to notebook --execute --inplace \
    docs/presentation/generate_presentation_figs.ipynb
```

---

## 6. Quick dependency map

```
split construction (01_preprocessing)
        ↓
   registry CSVs
        ├─ classical ML (02) ───┐
        ├─ docking (04) ────────┤
        ├─ dual-encoder (06/07) ├──► training_summary.json
        └─ PDBbind (05) ────────┘            │
                                             ▼
                               03_analysis + pdbbind report + notebook
                                             │
                                             ▼
                                    docs/presentation/
```

---

## 7. Current coverage (as of 2026-04-20)

| Method | PDBbind CASF-2016 | PLATE-VS hard 0p7 | PLATE-VS soft 0p7 |
|---|---|---|---|
| RF / GBM / SVM | ✓ | ✓ | ✓ |
| GNINA | ✓ (n=20 stratified) | ✓ (15 targets) | ✗ |
| GEMS (pretrained) | ✓ | ✗ (needs docking+dataprep) | ✗ |
| Dual-encoder | ✓ (5-fold ensemble) | △ (2 epochs, not persisted) | ✗ |

Next concrete runs that would complete the matrix — see `docs/presentation/results_summary.md` §7.
