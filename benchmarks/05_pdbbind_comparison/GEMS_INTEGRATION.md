# GEMS Integration — Methodology & Inference Guide

How we run Graber et al. (2025)'s pretrained GEMS ensemble on PDBbind CleanSplit, what the `.pt` data format is, and what would be needed to extend this to PLATE-VS. The upstream code is `external/GEMS/`; our wrapper is `run_gems_inference.py`.

---

## 1. What GEMS is (for reference)

- **Paper**: Graber et al., *"Enhancing Generalizable Binding Affinity Prediction by Removing Data Leakage and Integrating Language Model Embeddings into Graph Neural Networks"*, bioRxiv 2024.12.09.627482.
- **Task**: Regression of protein-ligand binding affinity (pK).
- **Backbone**: GATv2-style message-passing GNN (`GEMS18d` / `GEMS18e` in `external/GEMS/model/GEMS18.py`).
- **Graph construction**: one graph per complex. Nodes are ligand atoms + pocket residues; edges encode covalent bonds and spatial contacts within 5 Å. Node features concatenate:
  - Atom / residue type one-hots
  - **Frozen language-model embeddings**: ChemBERTa-77M (ligand-side), optional **ANKH-base** + **ESM2-t6** (protein-side).
- **Output**: Scalar pK scaled to `[0, 1]` by dividing by 16 during training; multiply by 16 at inference to recover pK units.
- **Training**: 5-fold CV on PDBbind CleanSplit. Published ensemble achieves **Pearson R = 0.815** on CASF-2016.

**CleanSplit** (`data/pdbbind_cleansplit/labels/PDBbind_data_split_cleansplit.json`) is PDBbind v.2020 **filtered to remove train-test leakage into CASF-2013/2016**, contributed by the same paper. Splits: 16,908 train / 285 CASF-2016 / 195 CASF-2013.

---

## 2. GEMS variants and checkpoint selection

GEMS ships multiple models that differ only in which language-model embeddings are baked into the graph nodes. Our wrapper auto-detects the variant from the dataset it's pointed at:

| Flag tuple `(has_ankh, has_esm2, delete_protein)` | Dataset ID | Architecture |
|---|---|---|
| (F, F, F) | `00AEPL` | `GEMS18e` (no embeddings) |
| (T, F, F) | `B0AEPL` | `GEMS18d` (ANKH only) |
| (F, T, F) | `06AEPL` | `GEMS18d` (ESM2 only) |
| **(T, T, F)** | **`B6AEPL`** | **`GEMS18d` (best — ANKH + ESM2)** |
| (T, T, T)  | `B6AE0L` | Ablation: protein nodes deleted |

See `EMBEDDING_MAP` in `run_gems_inference.py:64`. Published checkpoints live at `external/GEMS/model/GEMS18{d,e}_<ID>_kikdic_d*_f{0..4}_best_stdict.pt` — five state-dicts per variant for the 5-fold ensemble.

Our CASF-2016 results use **B6AEPL (GEMS18d)** — the best published variant.

---

## 3. Data format — `.pt` dataset structure

Preprocessed datasets come from GEMS's [Zenodo record 15482796](https://doi.org/10.5281/zenodo.15482796) as PyTorch-pickle `.pt` files. One file per `(variant, split)`, e.g. `B6AEPL_casf2016.pt`. Load with:

```python
dataset = torch.load(path, map_location="cpu", weights_only=False)
```

The object is a GEMS `Dataset` (from `external/GEMS/Dataset.py`) — a list-like container of PyG `Data` objects with these attributes per complex:

| Field | Shape / type | Content |
|---|---|---|
| `x` | `[N_nodes, node_feat_dim]` | Concatenated one-hots + frozen LM embeddings |
| `edge_index` | `[2, N_edges]` | Graph connectivity |
| `edge_attr` | `[N_edges, edge_feat_dim]` | Bond type + distance bin features |
| `y` | scalar in `[0, 1]` | **Scaled pK** — multiply by 16 to recover pK |
| `id` | `str` | PDB ID |
| `lig_emb` | `[1, D_lig]` (optional) | ChemBERTa global descriptor |

Container-level attrs used by the variant detector:

| Attribute | Values | Meaning |
|---|---|---|
| `protein_embeddings` | `["ankh_base", "esm2_t6"]` or subset | Which protein LMs were used at preprocessing time |
| `ligand_embeddings` | `["chemberta_77M"]` | Ligand LM variants |
| `delete_protein` | bool | Ablation flag |

`node_feat_dim` and `edge_feat_dim` are read off `dataset[0].x.shape[1]` / `.edge_attr.shape[1]` and passed directly into `GEMS18d(in_channels=..., edge_dim=...)` — **do not hardcode these; GEMS variants have different feature dims**.

---

## 4. Running inference on PDBbind CleanSplit

**Prerequisites**
1. `external/GEMS/` submodule populated with checkpoints (`model/*.pt`).
2. `gems` conda env: `conda env create -f benchmarks/envs/env_gems.yml` (needs PyG + ESM2-compatible Transformers — see GEMS README §3 for the exact stack).
3. Preprocessed `.pt` datasets at `data/pdbbind_cleansplit/preprocessed/*.pt` (from Zenodo, ~34 GB for the full B6AEPL set).

**Command**
```bash
conda run -n gems python benchmarks/05_pdbbind_comparison/run_gems_inference.py \
    --config benchmarks/05_pdbbind_comparison/configs/gems_config.yaml \
    --test-set casf2016    # or casf2013
```

Config is `configs/gems_config.yaml`:
```yaml
model:
  repo_path: "../../../external/GEMS"
  n_folds: 5
data:
  preprocessed_dir: "../../../data/pdbbind_cleansplit/preprocessed"
  test_set: "casf2016"
inference:
  batch_size: 128
  num_workers: 4
  device: "auto"
```

**Inference flow** (`run_inference()` in `run_gems_inference.py`):
1. Load any `.pt` from `preprocessed_dir`, inspect its `protein_embeddings` / `ligand_embeddings` / `delete_protein` to pick `(arch_name, dataset_id)`.
2. Load the test dataset for that variant (e.g. `B6AEPL_casf2016.pt`).
3. Read `node_feat_dim` and `edge_feat_dim` from the first `Data` object.
4. Load all 5 checkpoints matching `{arch_name}_{dataset_id}_*_f{0..4}_best_stdict.pt`.
5. Run each complex through all 5 models, **mean** the per-model predictions, unscale by ×16, compute regression metrics.
6. Write `results/gems/gems_{test_set}_training_summary.json` with per-complex predictions and aggregate metrics.

**Performance on CASF-2016** (`benchmarks/05_pdbbind_comparison/results/gems/gems_casf2016_training_summary.json`):

| Metric | Value |
|---|---|
| Pearson R | 0.815 |
| Spearman ρ | 0.805 |
| R² | 0.654 |
| RMSE | 1.274 |
| n | 282 (3 complexes missing from `.pt` vs the 285 JSON split) |

---

## 5. Extending to PLATE-VS (not yet implemented)

Running GEMS on PLATE-VS would require generating a `.pt` dataset in GEMS's graph format for the PLATE-VS test set. Unlike PDBbind, PLATE-VS entries do **not** have co-crystal structures — only SMILES for actives and decoys, plus one reference PDB per UniProt. Pipeline:

1. **Dock each ligand** (active + decoy) against the reference receptor for its UniProt target. GNINA is already wired for this in `benchmarks/04_docking/` — outputs best poses per ligand.
2. **Run GEMS `GEMS_dataprep_workflow.py`** on each (receptor, docked-ligand) pair to construct the interaction graph:
   - Parse protein PDB, extract residues within 5 Å of ligand
   - Compute protein residue LM embeddings (ESM2-t6 + ANKH-base)
   - Compute ligand ChemBERTa embedding
   - Build graph with node + edge features
3. **Package** the graphs as a GEMS `Dataset` object, save as `.pt` with matching `B6AEPL_plate_vs.pt` convention.
4. **Run inference** via the same `run_gems_inference.py` — no wrapper changes needed provided the dataset is at `preprocessed_dir` and matches the variant-detection rules.

Estimated effort: 1–2 days of engineering on CARC. Biggest unknowns:
- Whether GEMS can score docked (non-native) poses with reasonable calibration. Its training set is crystal complexes; docked poses have different geometric noise.
- Disk — each PLATE-VS target has ~10K–100K ligands; full `.pt` for 15 targets could exceed 50 GB.

---

## 6. Code map

| File | Purpose |
|---|---|
| `benchmarks/05_pdbbind_comparison/run_gems_inference.py` | Our wrapper — variant detection, model loading, ensembling, metrics |
| `benchmarks/05_pdbbind_comparison/configs/gems_config.yaml` | Inference config |
| `benchmarks/utils/metrics.py` | `summarize_regression()` — shared metric helper |
| `external/GEMS/Dataset.py` | Upstream dataset class; read for field definitions |
| `external/GEMS/model/GEMS18.py` | `GEMS18d` / `GEMS18e` architecture |
| `external/GEMS/GEMS_dataprep_workflow.py` | Upstream dataprep entry point (needed for PLATE-VS extension) |
| `external/GEMS/docs/GEMS_variants_and_datasets.md` | Upstream: variant tables + checkpoint naming |
| `external/GEMS/docs/dataset_filtering.md` | CleanSplit filtering algorithm |

---

## 7. Known gotchas

- **pK scaling**: GEMS targets are `pk / 16` during training. Our wrapper multiplies predictions by 16 at inference. If you dig into raw outputs, remember they're in `[0, 1]`.
- **Feature dims vary per variant** — always read `node_feat_dim` and `edge_feat_dim` from the dataset, never hardcode them.
- **CASF-2016 coverage**: the shipped `B6AEPL_casf2016.pt` contains 282 complexes, not 285. 3 entries from the JSON split are missing — likely graph-construction failures upstream. Report `n` alongside metrics.
- **Auto variant detection** matches on attribute presence. If you preprocess your own `.pt` with a different embedding mix, update `EMBEDDING_MAP` or the detector will warn and default to `B6AEPL`.
