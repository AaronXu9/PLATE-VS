# Boltzina Benchmark Integration Design

**Date:** 2026-03-28
**Branch:** `feature/boltzina-benchmark` (worktree off `feature/ml-benchmarking`)
**Scope:** PoC — benchmark boltzina (AutoDock Vina + Boltz-2) on 10 soft-split test proteins

---

## Goal

Integrate [boltzina](https://github.com/ohuelab/boltzina) into the VLS-Benchmark-Dataset pipeline to benchmark a 3D docking + Boltz-2 neural scoring method on the soft-split test partition. Output a `training_summary.json` compatible with `generate_benchmark_report.py` so results appear alongside classical ML baselines.

---

## Repository Structure

```
benchmarks/05_boltzina/
├── configs/
│   └── poc_config.yaml            # n_proteins=10, threshold=0p7, n_decoys_ratio=50
├── 01_select_proteins.py          # pick top-10 test proteins → results/poc_proteins.json
├── 02_prep_boltz.py               # CIF → boltz predict per protein → results/work_dirs/
├── 03_prep_ligands.py             # SMILES → 3D PDB files → results/ligands/
├── 04_run_boltzina.py             # build config + run boltzina → results/raw_results/
├── 05_collect_results.py          # CSVs → metrics → results/boltzina_training_summary.json
└── results/                       # gitignored (large intermediate files)
    ├── poc_proteins.json
    ├── work_dirs/{uniprot_id}/    # boltz predict output per protein
    ├── ligands/{uniprot_id}/      # actives/ and decoys/ PDB files
    └── raw_results/{uniprot_id}/  # boltzina output CSVs

external/boltzina/                 # git submodule (pinned commit)
benchmarks/envs/boltzina_env.yml   # conda env spec
```

---

## Environment

**Two environments used:**

| Stage | Script | Env | Reason |
|-------|--------|-----|--------|
| Select proteins | 01 | `rdkit_env` | pandas + rdkit already available |
| Boltz predict | 02 | `boltzina_env` | Boltz-2 + boltzina deps |
| Ligand prep | 03 | `rdkit_env` | RDKit ETKDG conformer generation |
| Run boltzina | 04 | `boltzina_env` | boltzina + Meeko + Boltz-2 |
| Collect results | 05 | `rdkit_env` | pandas + scipy |

**Setup commands (one-time):**

```bash
# 1. Add submodule
git submodule add https://github.com/ohuelab/boltzina external/boltzina

# 2. Create env from boltzina's pyproject.toml
conda create -n boltzina_env python=3.10
conda run -n boltzina_env pip install -e external/boltzina

# 3. Download Boltz-2 checkpoints + ADFR suite (Meeko PDBQT prep)
conda run -n boltzina_env bash external/boltzina/setup.sh

# 4. Add CIF/conformer tools to boltzina_env
conda run -n boltzina_env pip install rdkit gemmi
```

**Note:** `rdkit_env` has `boltz 0.4.1` (Boltz-1). `boltzina_env` will install the correct Boltz-2 version from boltzina's `pyproject.toml` without touching `rdkit_env`.

---

## Data Sources

All data available locally — no downloads required.

| Data | Path |
|------|------|
| Registry + affinity | `training_data_full/registry_soft_split_regression.csv` |
| Protein CIF structures | `plate-vs/VLS_benchmark/zipped_uniprot_raw_cif/uniprot_{uid}/cif_files_raw/{pdb_id}.cif` |
| Active SDF files | `plate-vs/VLS_benchmark/chembl_affinity/uniprot_{uid}/sdf_filtered_by_ligand_similarity/` |
| Decoy SMILES | `plate-vs/VLS_benchmark/chembl_affinity/uniprot_{uid}/deepcoy_output/{uid}_generated_decoys.txt` |

---

## Stage Design

### Stage 01 — Select Proteins

**Input:** `registry_soft_split_regression.csv`
**Output:** `results/poc_proteins.json`

Filter: `similarity_threshold='0p7'`, `protein_partition='test'`. Per-protein stats: n_actives, pchembl coverage. Rank by:

1. `quality_score` DESC (crystal structure quality)
2. `n_actives` ≥ 50
3. `pchembl_coverage` ≥ 0.80 (fraction of actives with affinity label)

Take top 10. Each entry in `poc_proteins.json`:
```json
{
  "uniprot_id": "O00408",
  "pdb_id": "5TZW",
  "cif_path": "plate-vs/VLS_benchmark/...",
  "n_actives": 312,
  "n_decoys_to_sample": 2500,
  "pchembl_coverage": 0.91
}
```

`n_decoys_to_sample = min(n_actives * 50, 2500)` — caps total ligands per protein at ~2550.

---

### Stage 02 — Prep Boltz

**Input:** `poc_proteins.json`
**Output:** `results/work_dirs/{uniprot_id}/` (one per protein)

Per protein:
1. Parse CIF with gemmi → extract protein chain sequence + coordinates
2. Write boltz input YAML (structure prediction input using CIF coordinates as template)
3. Run `conda run -n boltzina_env boltz predict {input.yaml} --out_dir results/work_dirs/{uid}/`
4. Record: receptor PDB path (`work_dirs/{uid}/predictions/{uid}/{uid}_model_0_protein.pdb`)
5. Extract co-crystal ligand HETATM atoms from CIF → compute centroid → Vina box center (22×22×22 Å)
6. Write `results/work_dirs/{uid}/vina_config.txt`

Skips proteins where `work_dirs/{uid}/` already exists (resumable).

---

### Stage 03 — Prep Ligands

**Input:** `poc_proteins.json`, registry, decoy SMILES files
**Output:** `results/ligands/{uniprot_id}/actives/*.pdb`, `results/ligands/{uniprot_id}/decoys/*.pdb`

Per protein:
1. **Actives:** get SMILES from registry (`split='test'`, `protein_partition='test'`, `is_active=True`, `threshold='0p7'`). Convert each SMILES → 3D conformer via RDKit ETKDG → write PDB. Filename: `{compound_id}.pdb`.
2. **Decoys:** read `{uid}_generated_decoys.txt` (format: `{ref_smiles} {decoy_smiles}` per line). Sample `n_decoys_to_sample` rows randomly. Convert decoy SMILES → PDB via same ETKDG pipeline. Filename: `decoy_{idx}.pdb`.
3. Skip existing files (resumable per compound).

---

### Stage 04 — Run Boltzina

**Input:** `poc_proteins.json`, `results/work_dirs/`, `results/ligands/`
**Output:** `results/raw_results/{uniprot_id}/results.csv`

Per protein:
1. Collect all ligand PDB paths (actives + decoys)
2. Write `results/raw_results/{uid}/config.json`:
   ```json
   {
     "work_dir": "results/work_dirs/{uid}",
     "vina_config": "results/work_dirs/{uid}/vina_config.txt",
     "fname": "{uid}",
     "input_ligand_name": "UNL",
     "output_dir": "results/raw_results/{uid}",
     "receptor_pdb": "results/work_dirs/{uid}/predictions/...",
     "ligand_files": [...]
   }
   ```
3. Run `conda run -n boltzina_env python external/boltzina/run.py results/raw_results/{uid}/config.json`
4. Skip if `results/raw_results/{uid}/results.csv` already exists (resumable).

---

### Stage 05 — Collect Results

**Input:** `results/raw_results/*/results.csv`, `poc_proteins.json`, registry (for pchembl)
**Output:**
- `results/boltzina_training_summary.json` (report-compatible aggregate)
- `results/boltzina_per_protein_results.csv` (per-protein breakdown)

Per protein:
1. Load `results.csv` — columns include: ligand filename, vina_score, boltz_affinity
2. Derive `is_active` label from filename (actives/ vs decoys/ path)
3. Join actives with pchembl from registry via compound_id
4. Compute:
   - **ROC-AUC** using `boltz_affinity` to rank actives vs decoys (sklearn)
   - **EF1%** — fraction of actives in top 1% of ranked list / random expectation
   - **Spearman r** — `boltz_affinity` vs `pchembl` for actives with affinity labels

Macro-average across 10 proteins → aggregate metrics.

Output `boltzina_training_summary.json` schema (matches `generate_benchmark_report.py` expectations):
```json
{
  "model_type": "boltzina",
  "feature_type": "3d_docking_boltz2",
  "similarity_threshold": "0p7",
  "use_precomputed_split": true,
  "training_history": {
    "train_metrics": {},
    "val_metrics": {},
    "test_metrics": {
      "roc_auc": 0.0,
      "ef1pct": 0.0,
      "spearman_r": 0.0
    },
    "n_train_samples": 0,
    "n_val_samples": 0,
    "n_test_samples": 0,
    "training_time_s": 0.0
  }
}
```

---

## Git Workflow

```bash
# From existing feature/ml-benchmarking branch
git worktree add ../VLS-Benchmark-Dataset-boltzina -b feature/boltzina-benchmark
cd ../VLS-Benchmark-Dataset-boltzina
git submodule add https://github.com/ohuelab/boltzina external/boltzina
```

PR `feature/boltzina-benchmark` → `feature/ml-benchmarking` (so boltzina results appear alongside classical ML before the final merge to `main`).

---

## Success Criteria

- [ ] 10 proteins selected and written to `poc_proteins.json`
- [ ] `boltz predict` completes for all 10 proteins
- [ ] Ligand PDB files generated for all actives + sampled decoys
- [ ] Boltzina produces `results.csv` for all 10 proteins (no crashes)
- [ ] `boltzina_training_summary.json` contains non-null `test_roc_auc`
- [ ] `generate_benchmark_report.py --extra-dirs results/` includes boltzina row in output

---

## Out of Scope

- Scoring-only mode (no Vina docking)
- Full 97-protein test set (follow-on after PoC validation)
- DeepPurpose / GraphDTA integration (separate branch)
- Hyperparameter tuning of boltzina (use defaults)
