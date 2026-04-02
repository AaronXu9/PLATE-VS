# Virtual Screening Benchmark Results

Comparison of scoring methods on the PLATE-VS benchmark (hard split, 0.7 similarity threshold).

## Methods Compared

| Method | Docking | Scoring | Description |
|--------|---------|---------|-------------|
| **Vina** | AutoDock Vina | Vina energy | Physics-based scoring function |
| **GNINA CNN** | GNINA (Vina-based) | CNN score / CNN affinity | 3D CNN trained on protein-ligand grids |
| **Boltzina** | AutoDock Vina | Boltz-2 affinity | Transformer (Pairformer trunk + affinity head) |

## Results

### P09211 (GSTP1, 66 actives, ~485 decoys)

| Method | Score | ROC-AUC | EF1% | EF5% |
|--------|-------|---------|------|------|
| Boltzina | affinity_prob_binary | **0.795** | 1.39 | **2.68** |
| GNINA | cnn_affinity | 0.737 | 0.00 | 0.60 |
| Boltzina | docking_score (Vina) | 0.617 | **2.78** | 1.19 |
| Boltzina | affinity_pred_value | 0.397 | 0.00 | 0.00 |
| GNINA | vina_score | 0.472 | 0.00 | 0.00 |
| GNINA | cnn_score | 0.352 | 0.00 | 0.00 |

**Winner: Boltzina `affinity_prob_binary` (0.795 ROC-AUC)**

### Q13490 (BIRC5, 80 actives, ~520 decoys)

| Method | Score | ROC-AUC | EF1% | EF5% |
|--------|-------|---------|------|------|
| GNINA | cnn_score | **0.838** | **7.52** | **6.80** |
| GNINA | vina_score | 0.794 | 4.30 | 3.88 |
| GNINA | cnn_affinity | 0.776 | 3.22 | 3.64 |
| Boltzina | docking_score (Vina) | 0.771 | 1.23 | 2.45 |
| Boltzina | affinity_pred_value | 0.519 | 0.00 | 0.25 |
| Boltzina | affinity_prob_binary | 0.320 | 0.00 | 0.25 |

**Winner: GNINA `cnn_score` (0.838 ROC-AUC, 7.5x enrichment at top 1%)**

## Key Findings

1. **No single method dominates across targets.** Boltzina's `affinity_prob_binary` is best for P09211 (0.795), but GNINA's `cnn_score` is best for Q13490 (0.838).

2. **GNINA CNN scoring is more consistent.** GNINA's `cnn_affinity` scores 0.737 and 0.776 across both targets — solid performance without extremes. Boltzina's Boltz-2 scores range from 0.320 to 0.795 — highly target-dependent.

3. **Boltz-2 `affinity_pred_value` underperforms.** This continuous affinity prediction is near-random on both targets (0.397, 0.519). The binary `affinity_prob_binary` is the useful Boltz-2 output.

4. **Vina docking score is a strong baseline.** On Q13490, Vina alone (0.771-0.794) nearly matches the neural rescoring methods. On P09211, it's weaker (0.472-0.617) and neural rescoring adds real value.

5. **Enrichment factors tell a different story than ROC-AUC.** GNINA achieves 7.5x enrichment at top 1% on Q13490 — practically meaningful for lead discovery. Boltzina achieves 2.8x on P09211. Both are useful for prioritizing compounds.

## Predicted vs Experimental Structure (Boltzina only)

Using Boltz-predicted structure for complex CIF vs experimental crystal structure:

| Protein | Score | Predicted | Experimental | Better |
|---------|-------|-----------|-------------|--------|
| P09211 | affinity_prob_binary | **0.795** | 0.763 | Predicted |
| P09211 | affinity_pred_value | 0.397 | 0.343 | Predicted |
| Q13490 | affinity_prob_binary | 0.320 | 0.309 | ~Tie |
| Q13490 | affinity_pred_value | 0.519 | 0.529 | ~Tie |

Boltz-predicted structure gives slightly better Boltz-2 scores on P09211, no difference on Q13490. The affinity module may be trained on Boltz-predicted geometries.

## Methodology

- **Docking**: AutoDock Vina (CPU, 8 parallel workers) for Boltzina; GNINA (CPU, 16 workers) for GNINA
- **Receptor**: experimental crystal structure PDBQT (same for all methods)
- **Complex for Boltz-2**: docked ligand merged with receptor PDB → CIF via maxit
- **Ligands**: actives from ChEMBL (test split, 0.7 similarity), decoys from DeepCoy
- **Metrics**: ROC-AUC (ranking quality), EF1%/EF5% (early enrichment)
- **Score direction**: `vina_score`/`docking_score` negated (lower energy = better); `cnn_score`, `cnn_affinity`, `affinity_*` used as-is (higher = better)
