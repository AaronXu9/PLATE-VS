# PLATE-VS Dual-Encoder Results Summary

Last updated: 2026-05-09

## Split definitions (post-2026-05-07 rename)

A previous version of this benchmark called "hard" what was actually only a
ligand-novelty test. The labels were corrected:

| Tag | Filter | Test set | Generalization tested |
|-----|--------|----------|----------------------|
| **`hard`** | `split=='test' AND protein_partition=='test'` | 87 proteins, 20K actives | **Both** novel proteins AND novel ligands (true 2D) |
| **`soft`** | `protein_partition=='test'` | 97 proteins, 42K actives | Novel proteins only (ligand similarity unconstrained) |
| **`ligand_novel`** | `split=='test'` (legacy) | 678 proteins, 127K actives | Novel ligands only — proteins overlap with train |

Note: 91% of `ligand_novel` test proteins are also in train. That's why it was
the easiest split for protein-conditioned models like the dual-encoder.

Implementation: configs that need both axes use `split_column: "split"` +
`secondary_split_column: "protein_partition"` in `data:`. See
`benchmarks/07_plate_vs_dl/configs/vs_hard_*.yaml` for the canonical setup.

---

## Headline results

All numbers below are **per-target ROC-AUC** (mean ± std across test proteins).
Real-inactives = ChEMBL compounds with pChEMBL ≥ 6 vs < 5.

### Generalization gradient (best models)

| Generalization tested | DeepCoy (training test) | Real inactives | Notes |
|----------------------|-------------------------|-----------------|-------|
| Ligand novelty only | 0.840 (SchNet) | 0.528 ± 0.256 | Big DeepCoy↔real gap (overfits) |
| Protein novelty only | 0.621 (SchNet) | 0.581 ± 0.192 (ET v2) | Cleaner |
| **Both novel (true hard)** | **0.617** (SchNet) / **0.605** (ET) | **0.625 ± 0.201** (SchNet) / **0.630 ± 0.228** (ET) | **Best real-world transfer** |

The harder we make the train/test gap, the smaller the DeepCoy↔real-inactive
gap becomes. The TRUE 2D hard split's real-inactive AUC (0.625-0.630) is
**actually higher** than its DeepCoy AUC (0.605-0.617) — opposite of all
weaker splits. This is consistent with the model learning real binding
chemistry rather than DeepCoy-specific shortcuts.

### Full table — all classical + DL on real inactives

| Model | Backend | Split | DeepCoy AUC | Real-inactive AUC (per-target) |
|-------|---------|-------|-------------|--------------------------------|
| RF + ESM2 | classical | ligand_novel | 0.473 | 0.403 ± 0.185 |
| GB + ESM2 | classical | ligand_novel | 0.433 | 0.392 ± 0.247 |
| RF + ESM2 | classical | soft | 0.544 | 0.449 ± 0.216 |
| Dual-encoder (SchNet) | DL | ligand_novel | 0.873 | 0.528 ± 0.256 |
| Dual-encoder (SchNet) | DL | soft | 0.633 | 0.546 ± 0.212 |
| Dual-encoder (TorchMD-ET v2) | DL | soft | 0.633 | 0.581 ± 0.192 |
| **Dual-encoder (SchNet)** | DL | **hard (2D)** | 0.586 | **0.625 ± 0.201** |
| **Dual-encoder (TorchMD-ET v2)** | DL | **hard (2D)** | 0.619 | **0.630 ± 0.228** |

### Backend comparison (TRUE hard split)

| Metric | SchNet | TorchMD-ET (v2 HP) |
|--------|--------|--------------------|
| Params | 6,559,490 | 8,742,529 |
| DeepCoy global AUC | 0.586 | 0.619 |
| DeepCoy per-target | 0.617 ± 0.199 | 0.605 ± 0.219 |
| Real-inactive global | 0.596 | 0.584 |
| **Real-inactive per-target** | 0.625 ± 0.201 | **0.630 ± 0.228** |
| NaN-dropped at eval | 0 | 93 (0.6%) |
| Best epoch | 14 | 24 |
| Total training (V100/P100) | ~17h (16 epochs) | ~33h (25 epochs) |

ET very slightly edges SchNet on per-target real-inactive AUC, but with higher
variance and more numerical-stability headaches. **SchNet is the recommended
default.**

---

## Notable findings

1. **The split definition matters more than the architecture.** Switching from
   ligand_novel → soft → hard moves real-inactive AUC by ~0.10. Switching
   SchNet → ET on the same split moves it by < 0.01.

2. **DeepCoy AUC ≠ real-world performance.** The model with the highest
   DeepCoy AUC (SchNet ligand_novel: 0.873) has the *lowest* real-inactive
   transfer (0.528). The data pipeline bug (collapsed ionic salts) plus the
   easy ligand-only split together created the illusion of superhuman
   performance.

3. **ESM2 carries the protein generalization.** Replacing the random-init
   protein lookup in classical models with mean-pooled ESM2 embeddings
   improved them from anti-correlated (0.30 AUC) to near-random (0.47).
   The dual-encoder's ESM2 is doing the same job, more effectively.

4. **DL beats classical only modestly on the meaningful test.** On TRUE hard
   real inactives, best DL ≈ 0.630, best classical (RF+ESM2 soft) ≈ 0.449.
   That's +0.18 absolute — real but modest. The 0.999 we used to report was
   never real binding signal.

---

## Reproduction

### Files
- Configs: `benchmarks/07_plate_vs_dl/configs/vs_{hard,soft,ligand_novel}_{schnet,et}.yaml`
- Trained checkpoints (CARC): `benchmarks/07_plate_vs_dl/checkpoints_<tag>/best_vs_model.pt`
- Result JSONs: `benchmarks/07_plate_vs_dl/results/{hard,soft,ligand_novel}_*_real_inactives_eval.json`

### Submit a sweep on CARC
```bash
ssh aoxu@discovery.usc.edu
cd /project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset
git pull
bash slurm/submit_plate_vs_sweep.sh hard_schnet      # true hard, SchNet
bash slurm/submit_plate_vs_sweep.sh hard_et          # true hard, TorchMD-ET
```

### Re-eval an existing checkpoint
```bash
sbatch --job-name=eval_<tag> \
  --export=ALL,CKPT=benchmarks/07_plate_vs_dl/checkpoints_<tag>/best_vs_model.pt,\
CONFIG=benchmarks/07_plate_vs_dl/configs/vs_<tag>.yaml,RESULT_TAG=<tag> \
  slurm/run_eval_real_inactives.slurm
```

See `slurm/PLATE_VS_SWEEP_README.md` for full submission details.
