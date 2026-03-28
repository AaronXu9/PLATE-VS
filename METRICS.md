# Metrics Reference

This document explains every metric used in the benchmark, how it is computed,
and where in the codebase the computation lives.

---

## Table of Contents

1. [Standard Classification Metrics](#1-standard-classification-metrics)
2. [Virtual Screening Metrics](#2-virtual-screening-metrics)
   - [Enrichment Factor (EF)](#21-enrichment-factor-ef)
   - [BEDROC](#22-bedroc)
3. [Threshold Selection](#3-threshold-selection)
4. [Regression Metrics](#4-regression-metrics)
5. [Code Locations Summary](#5-code-locations-summary)

---

## 1. Standard Classification Metrics

All classification metrics are computed via scikit-learn. Unless otherwise noted,
binary predictions (`y_pred`) use a default threshold of 0.5 on the predicted
probability (`y_proba`).

### ROC-AUC

The area under the Receiver Operating Characteristic curve. Measures the
probability that a randomly chosen active is ranked higher than a randomly
chosen decoy. Threshold-free (uses the full score distribution).

- Range: [0, 1]. 0.5 = random, 1.0 = perfect.
- **Relevant for VS**: yes, but insensitive to early enrichment.

```python
roc_auc_score(y_true, y_proba)          # sklearn
```

### Average Precision (PR-AUC)

Area under the Precision-Recall curve. More sensitive than ROC-AUC under class
imbalance (typical in VS datasets: ~5–10 % actives).

- Range: [0, 1]. Baseline ≈ prevalence.

```python
average_precision_score(y_true, y_proba)   # sklearn
```

### Accuracy

Fraction of correctly classified samples.

- **Caveat**: misleading under class imbalance — a model predicting all-inactive
  achieves ~94 % accuracy on the 1D test split (6 % active rate).

```python
accuracy_score(y_true, y_pred)
```

### Precision

Of all samples predicted as active, what fraction are truly active.

```python
precision_score(y_true, y_pred, zero_division=0)
```

### Recall (Sensitivity / TPR)

Of all true actives, what fraction are correctly identified.

```python
recall_score(y_true, y_pred, zero_division=0)
```

### F1 Score

Harmonic mean of Precision and Recall.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

```python
f1_score(y_true, y_pred, zero_division=0)
```

### MCC (Matthews Correlation Coefficient)

A balanced measure that accounts for all four quadrants of the confusion matrix.
Preferred over F1/accuracy under heavy class imbalance.

- Range: [−1, 1]. 0 = no better than random, 1 = perfect.

```python
matthews_corrcoef(y_true, y_pred)
```

---

## 2. Virtual Screening Metrics

These metrics specifically measure **early enrichment** — how well a method
places actives at the top of a ranked list. They are the primary evaluation
criteria for VS benchmarking.

All VS metrics require only a **continuous ranking score** and **binary labels**
(1 = active, 0 = decoy). No probability threshold is needed.

Input format expected by RDKit:
```python
ranked = sorted(zip(y_score, y_true.astype(int)), reverse=True)
# list of (score, label) tuples, sorted descending by score
```

---

### 2.1 Enrichment Factor (EF)

**Definition:**

```
EF(χ) = (# actives in top χ fraction of ranked list) / (N_actives × χ)
```

where χ ∈ (0, 1] is the fraction of the dataset inspected.

- EF = 1 → same as random selection
- EF = 1/χ → perfect enrichment (all actives in the top χ fraction)
- EF < 1 → worse than random

**Interpretation by fraction:**

| Key | χ | Meaning |
|-----|---|---------|
| `ef_0.1pct` | 0.1 % | Top 1-in-1000 — most relevant for large-library screening |
| `ef_0.2pct` | 0.2 % | |
| `ef_0.5pct` | 0.5 % | |
| `ef_1pct`   | 1 %   | Standard VS benchmark fraction |
| `ef_2pct`   | 2 %   | |
| `ef_5pct`   | 5 %   | |
| `ef_10pct`  | 10 %  | |
| `ef_15pct`  | 15 %  | |
| `ef_20pct`  | 20 %  | |

**Limitation:** EF is sensitive to the precise cut-off fraction and ignores the
rank order *within* the top fraction. Two methods with the same EF@1% may place
actives very differently within that 1%.

**Implementation:**

```python
from rdkit.ML.Scoring import Scoring

ef_vals = Scoring.CalcEnrichment(ranked, col=1, fractions=ef_fractions)
```

RDKit's `CalcEnrichment` counts how many `col=1` entries (actives) appear in
the top `fraction × N` positions of `ranked`.

---

### 2.2 BEDROC

**Boltzmann-Enhanced Discrimination of Receiver Operating Characteristic**
(Truchon & Bayly, *J. Chem. Inf. Model.*, 2007).

BEDROC addresses EF's limitation by exponentially down-weighting actives found
late in the ranked list. It integrates over the entire ranking, giving
progressively less credit to actives found deeper in the list.

**Parameter α controls the emphasis window:**

| Key | α | Approximate emphasis region |
|-----|---|-----------------------------|
| `bedroc_a20`  | 20  | Top ~8 % of list |
| `bedroc_a80`  | 80  | Top ~2 % of list |
| `bedroc_a160` | 160 | Top ~1 % of list |

Higher α → tighter focus on the very top ranks. Use `bedroc_a160` to judge
performance in the most selective regime.

**Range:** [BEDROC_min, 1]
- The minimum (BEDROC_min) depends on prevalence and is slightly above 0 for
  typical VS datasets.
- Random ranking ≈ 0.5 × (1 + Ra) where Ra is the ratio of actives; for 6 %
  prevalence, random ≈ ~0.05–0.10 depending on α.
- 1.0 = perfect enrichment (all actives ranked first).

**Implementation:**

```python
from rdkit.ML.Scoring import Scoring

bedroc = Scoring.CalcBEDROC(ranked, col=1, alpha=alpha)
```

---

## 3. Threshold Selection

Binary metrics (accuracy, precision, recall, F1, MCC) require converting
`y_proba` into `y_pred`.

### Default threshold (ML training evaluation)

`y_pred = (y_proba >= 0.5).astype(int)`

Used in `BaseTrainer.evaluate()` and `SVMTrainer._evaluate_scaled()`.

### Youden's J threshold (docking analysis)

For GNINA docking result analysis (`collect_results.py`), the threshold is
chosen to maximise **Youden's J statistic** (= TPR − FPR), which is equivalent
to maximising the vertical distance from the ROC curve to the diagonal.

```python
fpr, tpr, thresholds = roc_curve(y_true, y_score)
best_idx   = np.argmax(tpr - fpr)
threshold  = thresholds[best_idx]
y_pred     = (y_score >= threshold).astype(int)
```

This is more appropriate than a fixed 0.5 threshold when class prevalence is low
(~5–30 % actives, as in the 10 docked targets).

---

## 4. Regression Metrics

Used by the DeepPurpose / affinity-prediction branch (`benchmarks/utils/metrics.py`).
Not computed in the main classification pipeline.

| Metric | Formula | Location |
|--------|---------|----------|
| MSE    | mean((ŷ − y)²) | `metrics.py:28` |
| RMSE   | √MSE | `metrics.py:35` |
| MAE    | mean(\|ŷ − y\|) | `metrics.py:40` |
| Pearson r | standard correlation | `metrics.py:47` |
| CI     | Concordance Index (Harrell) | `metrics.py:56` |

---

## 5. Code Locations Summary

### Primary computation

| Metric group | Function | File | Lines |
|---|---|---|---|
| All standard classification | `BaseTrainer.evaluate()` | `benchmarks/02_training/models/base_trainer.py` | 101–128 |
| All standard classification (SVM) | `SVMTrainer._evaluate_scaled()` | `benchmarks/02_training/models/svm_trainer.py` | 151–171 |
| EF + BEDROC (ML models) | `compute_vs_metrics()` | `benchmarks/02_training/models/base_trainer.py` | 161–210 |
| EF + BEDROC (re-evaluation) | calls `compute_vs_metrics()` | `benchmarks/02_training/evaluate_vs_metrics.py` | 150–154 |
| EF + BEDROC (per-target ML) | calls `compute_vs_metrics()` | `benchmarks/04_docking/compare_methods_per_target.py` | 97–99 |
| EF + BEDROC + ROC-AUC (GNINA) | `compute_metrics()` | `benchmarks/04_docking/analyze_docking.py` | 37–61 |
| Classification + Youden threshold (GNINA) | `compute_metrics()` | `benchmarks/04_docking/collect_results.py` | 74–98 |
| Regression metrics | `summarize_regression()` | `benchmarks/utils/metrics.py` | 87–94 |

### EF/BEDROC fractions used per context

| Context | EF fractions | BEDROC α values |
|---|---|---|
| ML training (all splits) | 0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%, 15%, 20% | 20, 80, 160 |
| Per-target ML analysis | 0.1%, 0.5%, 1%, 2%, 5%, 10%, 20% | 20, 80, 160 |
| GNINA docking analysis | 0.1%, 0.5%, 1%, 2%, 5%, 10%, 20% | 20, 80, 160 |

### Key dependency

All EF and BEDROC computations use `rdkit.ML.Scoring.Scoring`, which expects a
list of `(score, label)` tuples sorted **descending** by score. The `label`
column index is always `col=1`.

```python
from rdkit.ML.Scoring import Scoring
ranked = sorted(zip(y_score, y_true.astype(int)), reverse=True)
Scoring.CalcEnrichment(ranked, col=1, fractions=[0.01])
Scoring.CalcBEDROC(ranked, col=1, alpha=80.0)
```
