# Dual-Encoder Binding-Affinity Model — Methods

This document is the single source of truth for the dual-encoder's architecture, training recipe, and 5-fold cross-validation protocol. For the code see `model/`, `data/`, and `train.py` in this directory; for configuration see `configs/`.

---

## 1. Task & target

Regression on `log_kd_ki` (pK) for protein–ligand complexes in PDBbind CleanSplit, held-out test set CASF-2016. All scalar targets are pre-scaled to `[0, 1]` by dividing by `pk_max = 16.0` during training and rescaled by the same factor at evaluation.

## 2. Architecture

Dual encoder + bidirectional cross-attention fusion + MLP head. Four modules, implemented in `model/`:

```
Input                                                   Output
-----                                                   ------

Protein side
  prot_padded   [B, R, 1088]      ──►  ProteinEncoder  ──►  [B, R, 256]
  prot_mask     [B, R]                   (2-layer MLP)
                                        freeze ESM2/ANKH

Ligand side
  z      [N_atoms]                ──►  LigandEncoder   ──►  per-atom [N, 256]
  pos    [N_atoms, 3]                  (TorchMD-NET ET       ─► scatter_to_padded
  batch  [N_atoms]                      or SchNet)            ─► [B, A, 256], mask

                 ProteinEncoder ─┐
                                 ├─►   CrossAttentionFusion    ─►  [B, 2·256]
                 LigandEncoder  ─┘       (3 bidirectional       (gated masked mean
                                          layers, 8 heads)       pooling)

                                                              ─►  PredictionHead
                                                                  (MLP 512→256→128→1)
                                                              ─►  pK (scaled [0,1])
```

### 2.1 Protein encoder (`model/protein_encoder.py`)

- Input: **precomputed** per-residue embeddings of dim 1088 = **ANKH-base (320) + ESM2-t6 (768)**, packed as `[B, max_res, 1088]` with a boolean mask. These are the same embeddings shipped with the GEMS "B6AEPL" preprocessed datasets (Zenodo).
- ESM2 and ANKH are **frozen** — no gradients flow into them.
- Projection: `Linear(1088, 256) → LayerNorm → SiLU → Linear(256, 256) → LayerNorm`.
- Pad length is capped at `max_pocket_residues = 80`.

### 2.2 Ligand encoder (`model/ligand_encoder.py`)

Two 3D equivariant backbones, selected by `model.ligand_backend` (`auto` prefers ET, falls back to SchNet):

| Backend | Package | Why |
|---|---|---|
| **TorchMD-NET ET** | `torchmdnet` 2.6.1 (custom-patched on CARC) | Equivariant transformer; sensitive to geometry quality |
| **SchNet** | `torch_geometric.nn.SchNet` | Always available; surprisingly robust with approximate conformers |

Inputs: atomic numbers `z`, 3D positions `pos`, and PyG batch indices. Per-atom hidden states are produced at the configured `hidden_channels = proj_dim = 256`, then linearly projected to 256 and scattered into padded `[B, max_atoms, 256]` with a mask. `max_ligand_atoms = 100`.

### 2.3 Cross-attention fusion (`model/cross_attention.py`)

3 stacked `CrossAttentionLayer`s, each with:
- **Protein-attends-to-ligand** multi-head attention (`d_model=256, heads=8, dropout=0.1`)
- **Ligand-attends-to-protein** multi-head attention
- Pre-norm residual + feed-forward block for each stream

After the stack, each stream is reduced via **gated masked mean pooling**: a sigmoid gate learned per token decides how much each position contributes to the pooled vector. Final fused vector is `[B, 512]` (protein-pool concatenated with ligand-pool).

### 2.4 Prediction head (`model/prediction_head.py`)

`Linear(512, 256) → SiLU → Dropout → Linear(256, 128) → SiLU → Dropout → Linear(128, 1)`. Output is a scalar `pK_scaled ∈ ℝ`. Dropout in the head is `2 × model.dropout`.

---

## 3. Loss function (`model/losses.py`)

Composite:

```
L_total = L_Huber(pred, target) + λ_rank · L_pair(pred, target)
```

with `λ_rank = 0.05` (was 0.1 originally — reduced after observing gradient spikes during warmup).

- **Huber / Smooth-L1** with `delta = 1.0` on the scaled pK. Robust to outliers in the training distribution.
- **Sampled pairwise margin ranking**: sample up to `rank_sample_size = 64` random ordered pairs `(i, j)` per batch; for each pair compute
  ```
  loss_ij = ReLU( m_ij − sign(target_i − target_j) · (pred_i − pred_j) )
  ```
  where the margin `m_ij = max(0.1, 0.5 · |target_i − target_j|)` is **adaptive** to the affinity gap. Only the mean over valid pairs is used. Directly improves ranking metrics (Spearman, Kendall).

The loss reports `{huber, rank, total}` so W&B can track each term separately.

---

## 4. Data pipeline

### 4.1 Dataset variants

Three PDBbind CleanSplit preprocessed `.pt` dataset directories under `data/pdbbind_cleansplit/`:

| Directory | 3D source | Config using it |
|---|---|---|
| `binding_affinity_dataset/` | RDKit ETKDGv3 conformers (fallback when crystal missing) | `default_config.yaml` |
| `binding_affinity_crystal_dataset/` | Raw crystal SDFs from RCSB model API | `et_crystal.yaml`, `schnet_crystal.yaml` |
| `binding_affinity_hiqbind_dataset/` | HiQBind **curated** crystal SDFs (correct bond orders) | `et_hiqbind.yaml` (best-performing) |

Every variant uses the **same** protein-side embeddings (GEMS B6AEPL), so the only thing changing across variants is the ligand 3D source. See `data/build_dataset.py` for construction.

### 4.2 Collate (`data/collate.py`)

Produces `AffinityBatch` with:
- `ligand_batch`: PyG `Batch` over stripped `Data(z, pos)` — for the ligand encoder's native batching
- `prot_padded`, `prot_mask`: padded residue embeddings
- `y`: `[B, 1]` scaled pK
- `pdb_ids`: list of complex identifiers (populated from either `Data.pdb_id` or `Data.uniprot_id` — PLATE-VS uses the latter, fixed in commit 55457d3b)

### 4.3 HiQBind integration

HiQBind is a curated reissue of PDBbind ligand SDFs with validated bond orders. Obtained from `https://huggingface.co/datasets/JZhang-lab/HiQBind` and landed at `/mnt/katritch_lab2/aoxu/data/hiqbind/` (local) / `/project2/katritch_223/aoxu/data/hiqbind/` (CARC). `data/build_dataset.py` prefers HiQBind SDF if present for a given PDB ID and falls back to the RDKit conformer for missing entries (**coverage: 88 % of CASF-2016, 57 % of train**). Empirical effect: **ET Pearson R 0.773 → 0.786 (+0.013)** on CASF-2016 at fold 0.

---

## 5. Training recipe

Canonical hyperparameters (`default_config.yaml`):

| Group | Parameter | Value | Notes |
|---|---|---|---|
| Data | `batch_size` | 16 | GPU-memory bound at proj_dim=256, ET=6L |
| Data | `num_workers` | 4 (PDBbind) / **0** (PLATE-VS) | PLATE-VS's AffinityBatch can't pickle |
| Optim | Optimizer | `AdamW(lr, weight_decay=1e-5)` | |
| Optim | `lr` | 1e-4 (default) / **5e-5** (et_hiqbind) / 3e-4 (old — unstable) | |
| Optim | LR schedule | Linear warmup (`warmup_steps=2000`) → cosine anneal to 0 | see `get_scheduler` |
| Optim | `grad_clip` | 0.5 | Was 1.0; tightened to prevent spikes |
| Train | `max_epochs` | 100 | Early-stopping usually fires ~40–60 |
| Train | `patience` | 20 | On validation Pearson R |
| Train | `seed` | 42 | Also used for `torch.manual_seed` + `np.random.seed` |

Hyperparameter history (summarised from W&B):
- **lr=3e-4 was unstable**: val R collapsed around epoch 9. Fixed by dropping to 1e-4, lengthening warmup to 2000 steps, and tightening grad_clip to 0.5.
- **proj_dim=512, 8L** (larger SchNet) overfit: R 0.766 → 0.729. Stuck with proj_dim=256.
- **ET cutoff=5 → 10 Å** with RDKit conformers: R 0.744 → 0.773. Wider cutoff compensates for approximate geometries.

---

## 6. 5-fold cross-validation protocol

Folds are the **published GEMS CleanSplit folds**: `data/pdbbind_cleansplit/labels/PDBbind_cleansplit_train_val_split_f{0..4}.json`. Each fold defines a `train` / `val` split over the CleanSplit training set of 16,908 complexes; the **CASF-2016 test set (285 complexes) is the same for every fold** — never mixed into training or validation.

For each fold `f ∈ {0,1,2,3,4}`:

```
conda run -n binding_affinity python benchmarks/06_binding_affinity_model/train.py \
    --config benchmarks/06_binding_affinity_model/configs/et_hiqbind.yaml \
    --fold $f
```

Outputs: `results/dual_encoder_f{f}_casf2016_training_summary.json` and the best checkpoint at `checkpoints/best_model_f{f}.pt` (selected on val Pearson R). Model selection is by val R, not train loss.

### Ensemble inference

The 5-fold ensemble prediction is the **mean of per-fold pK predictions on each CASF-2016 complex**:

```
pred_ensemble(c) = mean_{f=0..4}  pred_f(c)
```

Computed offline and written to `benchmarks/05_pdbbind_comparison/results/dual_encoder_ensemble_training_summary.json`. Ensembling adds ≈ **+0.036 Pearson R** over the best single fold (fold 0: 0.786 → ensemble: 0.804).

### Fold-level results (HiQBind ET, cutoff=10Å)

| Fold | Pearson R | R² | RMSE |
|---|---|---|---|
| f0 | 0.786 | 0.611 | 1.342 |
| f1 | 0.756 | — | — |
| f2 | 0.785 | — | — |
| f3 | 0.768 | — | — |
| f4 | 0.743 | — | — |
| **mean ± std** | **0.768 ± 0.016** | — | — |
| **Ensemble** | **0.804** | **0.629** | **1.311** |

Reference: GEMS published 5-fold ensemble **R = 0.815**.

---

## 7. PLATE-VS variant (`benchmarks/07_plate_vs_dl/`)

A binary-classification variant of the same dual-encoder:
- Head outputs a single logit; loss = `BCEWithLogitsLoss(pos_weight)` where `pos_weight` reflects the active/decoy imbalance.
- Evaluation adds **per-target ROC-AUC** (average over UniProt IDs) in addition to global ROC-AUC and AP.
- Dataset is `training_data_full/registry_2d_split.csv` (hard split, 0p7 threshold). PLATE-VS protein embeddings are ESM2-t6 only (320-dim), built by `data/build_protein_embeddings.py`.
- `evaluate()` lives in `benchmarks/07_plate_vs_dl/evaluation.py` (extracted to be test-importable without torchmd-net).

Current status: 2 epochs logged to W&B (global ROC-AUC ≈ 0.801). No persisted summary JSON yet; conformer pre-computation is the next step to reduce the 11.5 h/epoch cost.

---

## 8. Reproducibility checklist

1. `conda env create -f benchmarks/envs/env_binding_affinity.yml` (or use existing `binding_affinity` env on CARC).
2. Put preprocessed datasets at `data/pdbbind_cleansplit/binding_affinity_{,crystal_,hiqbind_}dataset/`.
3. `export WANDB_API_KEY=...` (or create `.env` at repo root).
4. Run `train.py --config <yaml> --fold <0..4>` per fold.
5. After all 5 folds, compute ensemble metrics via `benchmarks/05_pdbbind_comparison/` aggregation scripts.
