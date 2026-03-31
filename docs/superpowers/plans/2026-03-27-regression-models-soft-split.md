# Regression Models on Soft Split Registry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict pChEMBL binding affinity values (regression) using the soft-split registry, mirroring the existing RF/GBM/SVM classification pipeline.

**Architecture:** Two phases — (1) enrich `registry_soft_split.csv` with pChEMBL values from the local `data/chembl_activities_enriched.parquet` file (1.2M rows, median-aggregated per compound); (2) build `BaseRegressionTrainer` + three concrete regressors (RF, GBM, SVM), a `train_regression.py` script that mirrors `train_classical_oddt.py`, and three YAML configs.

**Tech Stack:** scikit-learn (RandomForestRegressor, HistGradientBoostingRegressor, LinearSVR), scipy.stats (Spearman), pandas + pyarrow, existing Morgan FP + protein embedding featurizers, rdkit_env conda environment.

> **Scope note:** pChEMBL enrichment and regression training are independent subsystems. This plan combines them because enrichment is a prerequisite for meaningful model evaluation. If either section grows, split into separate plans.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `benchmarks/01_preprocessing/enrich_pchembl.py` | Join registry with ChEMBL activities parquet, aggregate pChEMBL per compound |
| Create | `benchmarks/01_preprocessing/tests/test_enrich_pchembl.py` | Unit tests for enrichment |
| Modify | `benchmarks/utils/metrics.py` | Add `calculate_r2`, `calculate_spearman`, update `summarize_regression` |
| Create | `benchmarks/02_training/models/base_regression_trainer.py` | Abstract base with regression `evaluate()` using `summarize_regression` |
| Create | `benchmarks/02_training/models/rf_regressor.py` | `RandomForestRegressor` trainer |
| Create | `benchmarks/02_training/models/gbm_regressor.py` | `HistGradientBoostingRegressor` / XGBoost / LightGBM trainer |
| Create | `benchmarks/02_training/models/svm_regressor.py` | `LinearSVR` trainer with StandardScaler |
| Create | `benchmarks/02_training/tests/test_regression_trainers.py` | Tests for all three regressors |
| Create | `benchmarks/02_training/train_regression.py` | End-to-end regression training script |
| Create | `benchmarks/configs/regression_rf_config.yaml` | RF regression config |
| Create | `benchmarks/configs/regression_gbm_config.yaml` | GBM regression config |
| Create | `benchmarks/configs/regression_svm_config.yaml` | SVM regression config |

**All work is in the `feature/soft-partition-regression` worktree at `/home/aoxu/projects/VLS-soft-partition/`.**

---

## Task 1: Enrich pChEMBL from local ChEMBL parquet

**Files:**
- Create: `benchmarks/01_preprocessing/enrich_pchembl.py`
- Create: `benchmarks/01_preprocessing/tests/test_enrich_pchembl.py`

- [ ] **Step 1: Write the failing test**

```python
# benchmarks/01_preprocessing/tests/test_enrich_pchembl.py
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from enrich_pchembl import aggregate_pchembl, enrich_registry


def _make_activities(rows):
    return pd.DataFrame(rows, columns=[
        'molecule_chembl_id', 'pchembl_value', 'assay_type', 'standard_type'
    ])


def _make_registry(compound_ids):
    return pd.DataFrame({
        'compound_id': compound_ids,
        'pchembl': [np.nan] * len(compound_ids),
    })


class TestAggregatePChEMBL:
    def test_median_aggregation(self):
        acts = _make_activities([
            ('CHEMBL1', 7.0, 'B', 'Ki'),
            ('CHEMBL1', 8.0, 'B', 'Ki'),   # median of 7 and 8 → 7.5
            ('CHEMBL2', 6.0, 'B', 'IC50'),
        ])
        result = aggregate_pchembl(acts)
        assert pytest.approx(result['CHEMBL1'], abs=1e-6) == 7.5
        assert pytest.approx(result['CHEMBL2'], abs=1e-6) == 6.0

    def test_filters_non_binding_assays(self):
        acts = _make_activities([
            ('CHEMBL1', 7.0, 'F', 'Ki'),   # F = functional, excluded
            ('CHEMBL2', 6.0, 'B', 'Ki'),
        ])
        result = aggregate_pchembl(acts, assay_types={'B'})
        assert 'CHEMBL1' not in result
        assert 'CHEMBL2' in result

    def test_filters_out_of_range_pchembl(self):
        acts = _make_activities([
            ('CHEMBL1', 3.5, 'B', 'Ki'),   # below 4.0 — excluded
            ('CHEMBL2', 7.0, 'B', 'Ki'),
            ('CHEMBL3', 13.0, 'B', 'Ki'),  # above 12.0 — excluded
        ])
        result = aggregate_pchembl(acts, pchembl_min=4.0, pchembl_max=12.0)
        assert 'CHEMBL1' not in result
        assert 'CHEMBL2' in result
        assert 'CHEMBL3' not in result

    def test_filters_invalid_standard_types(self):
        acts = _make_activities([
            ('CHEMBL1', 7.0, 'B', 'MIC'),   # not in default valid types
            ('CHEMBL2', 7.0, 'B', 'IC50'),
        ])
        result = aggregate_pchembl(acts)
        assert 'CHEMBL1' not in result
        assert 'CHEMBL2' in result

    def test_empty_activities_returns_empty_series(self):
        acts = _make_activities([])
        result = aggregate_pchembl(acts)
        assert len(result) == 0


class TestEnrichRegistry:
    def test_pchembl_joined_by_compound_id(self):
        registry = _make_registry(['CHEMBL1', 'CHEMBL2', 'CHEMBL3'])
        agg = pd.Series({'CHEMBL1': 7.5, 'CHEMBL2': 6.0})
        result = enrich_registry(registry, agg)
        assert pytest.approx(result.loc[result['compound_id'] == 'CHEMBL1', 'pchembl'].iloc[0]) == 7.5
        assert pytest.approx(result.loc[result['compound_id'] == 'CHEMBL2', 'pchembl'].iloc[0]) == 6.0
        assert np.isnan(result.loc[result['compound_id'] == 'CHEMBL3', 'pchembl'].iloc[0])

    def test_original_non_null_pchembl_overwritten(self):
        # enrich_registry always replaces pchembl column with fresh join result
        registry = pd.DataFrame({
            'compound_id': ['CHEMBL1'],
            'pchembl': [5.0],   # stale value
        })
        agg = pd.Series({'CHEMBL1': 8.0})
        result = enrich_registry(registry, agg)
        assert pytest.approx(result['pchembl'].iloc[0]) == 8.0

    def test_row_count_unchanged(self):
        registry = _make_registry(['CHEMBL1'] * 3)   # same compound, 3 rows
        agg = pd.Series({'CHEMBL1': 7.0})
        result = enrich_registry(registry, agg)
        assert len(result) == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/aoxu/projects/VLS-soft-partition
conda run -n rdkit_env pytest benchmarks/01_preprocessing/tests/test_enrich_pchembl.py -v 2>&1 | tail -20
```
Expected: `ModuleNotFoundError: No module named 'enrich_pchembl'`

- [ ] **Step 3: Implement `enrich_pchembl.py`**

```python
# benchmarks/01_preprocessing/enrich_pchembl.py
"""
Enrich registry_soft_split.csv with pChEMBL values from chembl_activities_enriched.parquet.

Join key:  registry.compound_id  <->  activities.molecule_chembl_id
Strategy:  median pChEMBL per compound across binding assays with valid measurements.

Output:    new CSV at --output path (registry_soft_split.csv unchanged unless output
           path is the same).
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

VALID_ASSAY_TYPES: set = {'B'}
VALID_STANDARD_TYPES: set = {'IC50', 'Ki', 'Kd', 'EC50', 'Potency'}
PCHEMBL_MIN: float = 4.0
PCHEMBL_MAX: float = 12.0

_ACTIVITIES_COLS = ['molecule_chembl_id', 'pchembl_value', 'assay_type', 'standard_type']


def aggregate_pchembl(
    activities: pd.DataFrame,
    assay_types: set = VALID_ASSAY_TYPES,
    standard_types: set = VALID_STANDARD_TYPES,
    pchembl_min: float = PCHEMBL_MIN,
    pchembl_max: float = PCHEMBL_MAX,
) -> pd.Series:
    """
    Filter activities to binding assays with valid pChEMBL, then aggregate
    to one median pChEMBL per molecule_chembl_id.

    Returns a Series indexed by molecule_chembl_id.
    """
    if activities.empty:
        return pd.Series(dtype=float)

    mask = (
        activities['assay_type'].isin(assay_types)
        & activities['standard_type'].isin(standard_types)
        & activities['pchembl_value'].notna()
        & (activities['pchembl_value'] >= pchembl_min)
        & (activities['pchembl_value'] <= pchembl_max)
    )
    filtered = activities.loc[mask]
    if filtered.empty:
        return pd.Series(dtype=float)

    return filtered.groupby('molecule_chembl_id')['pchembl_value'].median()


def enrich_registry(registry: pd.DataFrame, agg: pd.Series) -> pd.DataFrame:
    """
    Map aggregated pChEMBL onto registry rows via compound_id.
    Always overwrites the pchembl column.
    Returns a new DataFrame (does not mutate registry).
    """
    out = registry.copy()
    out['pchembl'] = out['compound_id'].map(agg).astype(float)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description='Enrich soft-split registry with pChEMBL values')
    parser.add_argument('--registry', required=True, help='Path to registry_soft_split.csv')
    parser.add_argument('--activities', required=True, help='Path to chembl_activities_enriched.parquet')
    parser.add_argument('--output', required=True, help='Output path for enriched registry CSV')
    parser.add_argument('--assay-types', nargs='+', default=list(VALID_ASSAY_TYPES))
    parser.add_argument('--standard-types', nargs='+', default=list(VALID_STANDARD_TYPES))
    parser.add_argument('--pchembl-min', type=float, default=PCHEMBL_MIN)
    parser.add_argument('--pchembl-max', type=float, default=PCHEMBL_MAX)
    args = parser.parse_args()

    logging.info(f'Loading registry: {args.registry}')
    registry = pd.read_csv(args.registry, low_memory=False)
    logging.info(f'  {len(registry):,} rows, {registry["compound_id"].nunique():,} unique compounds')

    logging.info(f'Loading activities: {args.activities}')
    activities = pd.read_parquet(args.activities, columns=_ACTIVITIES_COLS)
    logging.info(f'  {len(activities):,} rows')

    logging.info('Aggregating pChEMBL...')
    agg = aggregate_pchembl(
        activities,
        assay_types=set(args.assay_types),
        standard_types=set(args.standard_types),
        pchembl_min=args.pchembl_min,
        pchembl_max=args.pchembl_max,
    )
    logging.info(f'  {len(agg):,} compounds with pChEMBL after filtering')

    enriched = enrich_registry(registry, agg)
    n_enriched = enriched['pchembl'].notna().sum()
    logging.info(f'  {n_enriched:,} rows enriched ({n_enriched/len(enriched)*100:.1f}%)')
    logging.info(f'  pChEMBL range: {enriched["pchembl"].min():.2f} – {enriched["pchembl"].max():.2f}')

    logging.info(f'Writing to {args.output}')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    logging.info('Done.')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n rdkit_env pytest benchmarks/01_preprocessing/tests/test_enrich_pchembl.py -v 2>&1 | tail -20
```
Expected: `8 passed`

- [ ] **Step 5: Run the enrichment on the full registry**

```bash
conda run -n rdkit_env python benchmarks/01_preprocessing/enrich_pchembl.py \
  --registry /home/aoxu/projects/VLS-Benchmark-Dataset/training_data_full/registry_soft_split.csv \
  --activities /home/aoxu/projects/VLS-Benchmark-Dataset/data/chembl_activities_enriched.parquet \
  --output /home/aoxu/projects/VLS-Benchmark-Dataset/training_data_full/registry_soft_split_regression.csv
```

Expected log lines:
```
INFO  Aggregating pChEMBL...
INFO    X,XXX compounds with pChEMBL after filtering
INFO    X,XXX rows enriched (X.X%)
```
Note the enrichment count — this is how many training samples the regression models will have.

- [ ] **Step 6: Commit**

```bash
cd /home/aoxu/projects/VLS-soft-partition
git add benchmarks/01_preprocessing/enrich_pchembl.py \
        benchmarks/01_preprocessing/tests/test_enrich_pchembl.py
git commit -m "feat: enrich registry pChEMBL from chembl_activities_enriched.parquet"
```

---

## Task 2: Add R² and Spearman to metrics.py

**Files:**
- Modify: `benchmarks/utils/metrics.py`

- [ ] **Step 1: Write the failing tests** (append to a test file or create one)

```python
# benchmarks/utils/tests/test_metrics.py
import math
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from benchmarks.utils.metrics import (
    calculate_r2, calculate_spearman, summarize_regression
)


class TestCalculateR2:
    def test_perfect_prediction(self):
        y = [1.0, 2.0, 3.0, 4.0]
        assert pytest.approx(calculate_r2(y, y), abs=1e-9) == 1.0

    def test_mean_prediction_is_zero(self):
        # predicting mean every time → R² = 0
        y = [1.0, 2.0, 3.0, 4.0]
        y_pred = [2.5, 2.5, 2.5, 2.5]
        assert pytest.approx(calculate_r2(y, y_pred), abs=1e-9) == 0.0

    def test_constant_true_returns_nan(self):
        assert math.isnan(calculate_r2([5.0, 5.0, 5.0], [5.0, 5.0, 5.1]))

    def test_known_value(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5,  0.0, 2.0, 8.0])
        # R² = 1 - SS_res/SS_tot
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        expected = 1 - ss_res / ss_tot
        assert pytest.approx(calculate_r2(y_true, y_pred), abs=1e-6) == expected


class TestCalculateSpearman:
    def test_perfect_rank_correlation(self):
        y = [1.0, 2.0, 3.0, 4.0]
        assert pytest.approx(calculate_spearman(y, y), abs=1e-9) == 1.0

    def test_inverse_rank_correlation(self):
        y = [1.0, 2.0, 3.0, 4.0]
        y_inv = [4.0, 3.0, 2.0, 1.0]
        assert pytest.approx(calculate_spearman(y, y_inv), abs=1e-9) == -1.0

    def test_fewer_than_two_samples_returns_nan(self):
        assert math.isnan(calculate_spearman([5.0], [5.0]))


class TestSummarizeRegressionKeys:
    def test_has_r2_and_spearman(self):
        y = [6.0, 7.0, 8.0, 7.5, 6.5]
        result = summarize_regression(y, y)
        assert 'r2' in result
        assert 'spearman' in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/aoxu/projects/VLS-soft-partition
conda run -n rdkit_env pytest benchmarks/utils/tests/test_metrics.py -v 2>&1 | tail -20
```
Expected: `ImportError: cannot import name 'calculate_r2'`

- [ ] **Step 3: Edit `benchmarks/utils/metrics.py`**

Add after the existing imports block (after `except ImportError: _lifelines_concordance_index = None`):

```python
try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
```

Add these two functions before `summarize_regression`:

```python
def calculate_r2(y_true, y_pred):
    true_arr, pred_arr = _prepare_arrays(y_true, y_pred)
    if len(true_arr) < 2:
        return float("nan")
    ss_tot = np.sum((true_arr - np.mean(true_arr)) ** 2)
    if ss_tot == 0.0:
        return float("nan")
    ss_res = np.sum((true_arr - pred_arr) ** 2)
    return float(1.0 - ss_res / ss_tot)


def calculate_spearman(y_true, y_pred):
    true_arr, pred_arr = _prepare_arrays(y_true, y_pred)
    if len(true_arr) < 2:
        return float("nan")
    if not _SCIPY_AVAILABLE:
        return float("nan")
    result = _scipy_stats.spearmanr(true_arr, pred_arr)
    return float(result.correlation)
```

Replace the existing `summarize_regression` body:

```python
def summarize_regression(y_true, y_pred):
    return {
        "mse":      calculate_mse(y_true, y_pred),
        "rmse":     calculate_rmse(y_true, y_pred),
        "mae":      calculate_mae(y_true, y_pred),
        "r2":       calculate_r2(y_true, y_pred),
        "pearson":  calculate_pearson(y_true, y_pred),
        "spearman": calculate_spearman(y_true, y_pred),
        "ci":       calculate_ci(y_true, y_pred),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n rdkit_env pytest benchmarks/utils/tests/test_metrics.py -v 2>&1 | tail -20
```
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add benchmarks/utils/metrics.py benchmarks/utils/tests/test_metrics.py
git commit -m "feat: add R² and Spearman to regression metrics"
```

---

## Task 3: BaseRegressionTrainer + three regression model trainers

**Files:**
- Create: `benchmarks/02_training/models/base_regression_trainer.py`
- Create: `benchmarks/02_training/models/rf_regressor.py`
- Create: `benchmarks/02_training/models/gbm_regressor.py`
- Create: `benchmarks/02_training/models/svm_regressor.py`
- Create: `benchmarks/02_training/tests/test_regression_trainers.py`

- [ ] **Step 1: Write the failing tests**

```python
# benchmarks/02_training/tests/test_regression_trainers.py
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.rf_regressor import RandomForestRegressorTrainer
from models.gbm_regressor import GBMRegressorTrainer
from models.svm_regressor import SVMRegressorTrainer


def _small_regression_data(n=200, n_features=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    # pChEMBL targets: 5-9 range
    y = (rng.standard_normal(n) + 7.0).astype(np.float32)
    split = int(n * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


_MINIMAL_CONFIG = {
    'hyperparameters': {'n_estimators': 10, 'random_state': 42, 'n_jobs': 1}
}


class TestRandomForestRegressor:
    def test_train_returns_history_with_metrics(self):
        X_tr, y_tr, X_v, y_v = _small_regression_data()
        t = RandomForestRegressorTrainer(_MINIMAL_CONFIG)
        hist = t.train(X_tr, y_tr, X_v, y_v)
        assert 'train_metrics' in hist
        assert 'val_metrics' in hist
        for key in ('rmse', 'mae', 'r2', 'pearson', 'spearman'):
            assert key in hist['train_metrics'], f"missing {key} in train_metrics"

    def test_predict_shape(self):
        X_tr, y_tr, X_v, y_v = _small_regression_data()
        t = RandomForestRegressorTrainer(_MINIMAL_CONFIG)
        t.train(X_tr, y_tr)
        preds = t.predict(X_v)
        assert preds.shape == (len(X_v),)

    def test_predict_before_train_raises(self):
        t = RandomForestRegressorTrainer(_MINIMAL_CONFIG)
        with pytest.raises(ValueError, match="not trained"):
            t.predict(np.zeros((5, 64)))

    def test_save_load_roundtrip(self, tmp_path):
        X_tr, y_tr, X_v, y_v = _small_regression_data()
        t = RandomForestRegressorTrainer(_MINIMAL_CONFIG)
        t.train(X_tr, y_tr)
        t.save_model(str(tmp_path))
        t2 = RandomForestRegressorTrainer(_MINIMAL_CONFIG)
        t2.load_model(str(tmp_path))
        np.testing.assert_array_almost_equal(t.predict(X_v), t2.predict(X_v))

    def test_evaluate_returns_all_regression_keys(self):
        X_tr, y_tr, X_v, y_v = _small_regression_data()
        t = RandomForestRegressorTrainer(_MINIMAL_CONFIG)
        t.train(X_tr, y_tr)
        metrics = t.evaluate(X_v, y_v)
        for key in ('mse', 'rmse', 'mae', 'r2', 'pearson', 'spearman', 'ci'):
            assert key in metrics, f"missing {key}"


class TestGBMRegressor:
    def test_train_and_evaluate(self):
        X_tr, y_tr, X_v, y_v = _small_regression_data()
        t = GBMRegressorTrainer({'hyperparameters': {'n_estimators': 10, 'random_state': 42}})
        hist = t.train(X_tr, y_tr, X_v, y_v)
        assert 'val_metrics' in hist
        assert 'rmse' in hist['val_metrics']


class TestSVMRegressor:
    def test_train_and_evaluate(self):
        X_tr, y_tr, X_v, y_v = _small_regression_data()
        t = SVMRegressorTrainer({'hyperparameters': {'C': 0.1, 'max_iter': 500}})
        hist = t.train(X_tr, y_tr, X_v, y_v)
        assert 'val_metrics' in hist
        assert 'rmse' in hist['val_metrics']
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n rdkit_env pytest benchmarks/02_training/tests/test_regression_trainers.py -v 2>&1 | tail -15
```
Expected: `ModuleNotFoundError: No module named 'models.rf_regressor'`

- [ ] **Step 3: Create `base_regression_trainer.py`**

```python
# benchmarks/02_training/models/base_regression_trainer.py
"""Abstract base class for regression model trainers."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import json

import joblib
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.metrics import summarize_regression


class BaseRegressionTrainer(ABC):
    def __init__(self, config: Dict[str, Any], model_name: str):
        self.config = config
        self.model_name = model_name
        self.model = None
        self.training_history: Dict[str, Any] = {}

    @abstractmethod
    def build_model(self) -> Any: ...

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]: ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X).astype(np.float32)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        return summarize_regression(y, y_pred)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def save_model(self, save_dir: str) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / f"{self.model_name}.pkl")
        with open(path / f"{self.model_name}_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        with open(path / f"{self.model_name}_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Model saved to {path}")

    def load_model(self, load_dir: str) -> None:
        path = Path(load_dir)
        model_path = path / f"{self.model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)
        print(f"Model loaded from {path}")
```

- [ ] **Step 4: Create `rf_regressor.py`**

```python
# benchmarks/02_training/models/rf_regressor.py
"""Random Forest regressor for pChEMBL prediction."""
import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base_regression_trainer import BaseRegressionTrainer


class RandomForestRegressorTrainer(BaseRegressionTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name='random_forest_regressor')
        self.hyperparameters = config.get('hyperparameters', {})

    def build_model(self) -> RandomForestRegressor:
        hp = self.hyperparameters
        model = RandomForestRegressor(
            n_estimators=hp.get('n_estimators', 100),
            max_depth=hp.get('max_depth', None),
            min_samples_split=hp.get('min_samples_split', 2),
            min_samples_leaf=hp.get('min_samples_leaf', 1),
            max_features=hp.get('max_features', 'sqrt'),
            bootstrap=hp.get('bootstrap', True),
            random_state=hp.get('random_state', 42),
            n_jobs=hp.get('n_jobs', -1),
        )
        print(f"RF Regressor: n_estimators={model.n_estimators}, "
              f"max_depth={model.max_depth}, max_features={model.max_features}")
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        print("\n" + "=" * 50)
        print("Training Random Forest Regressor")
        print("=" * 50)
        self.model = self.build_model()
        start = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"Training completed in {elapsed:.2f}s on {len(X_train)} samples")

        train_metrics = self.evaluate(X_train, y_train)
        print(f"Train  RMSE={train_metrics['rmse']:.3f}  R²={train_metrics['r2']:.3f}")

        self.training_history = {
            'train_metrics': train_metrics,
            'training_time': elapsed,
            'n_train_samples': len(X_train),
            'hyperparameters': self.hyperparameters,
        }
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            print(f"Val    RMSE={val_metrics['rmse']:.3f}  R²={val_metrics['r2']:.3f}")
            self.training_history['val_metrics'] = val_metrics
            self.training_history['n_val_samples'] = len(X_val)
        return self.training_history
```

- [ ] **Step 5: Create `gbm_regressor.py`**

```python
# benchmarks/02_training/models/gbm_regressor.py
"""Gradient Boosting regressor. Backend: hist (sklearn) / xgboost / lightgbm."""
import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import xgboost as xgb
    _XGBOOST = True
except ImportError:
    _XGBOOST = False

try:
    import lightgbm as lgb
    _LIGHTGBM = True
except ImportError:
    _LIGHTGBM = False

from .base_regression_trainer import BaseRegressionTrainer


class GBMRegressorTrainer(BaseRegressionTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name='gradient_boosting_regressor')
        self.hyperparameters = config.get('hyperparameters', {})
        self.backend = self._resolve_backend(config.get('backend', 'auto'))
        print(f"GBMRegressorTrainer using backend: {self.backend}")

    def _resolve_backend(self, requested: str) -> str:
        if requested == 'auto':
            if _XGBOOST:
                return 'xgboost'
            if _LIGHTGBM:
                return 'lightgbm'
            return 'hist'
        if requested == 'xgboost' and not _XGBOOST:
            print("Warning: xgboost not installed, falling back to hist")
            return 'hist'
        if requested == 'lightgbm' and not _LIGHTGBM:
            print("Warning: lightgbm not installed, falling back to hist")
            return 'hist'
        return requested

    def build_model(self):
        hp = self.hyperparameters
        if self.backend == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=hp.get('n_estimators', 300),
                max_depth=hp.get('max_depth', 6),
                learning_rate=hp.get('learning_rate', 0.05),
                subsample=hp.get('subsample', 0.8),
                colsample_bytree=hp.get('colsample_bytree', 0.8),
                random_state=hp.get('random_state', 42),
                n_jobs=hp.get('n_jobs', -1),
                verbosity=0,
            )
        if self.backend == 'lightgbm':
            return lgb.LGBMRegressor(
                n_estimators=hp.get('n_estimators', 300),
                learning_rate=hp.get('learning_rate', 0.05),
                subsample=hp.get('subsample', 0.8),
                colsample_bytree=hp.get('colsample_bytree', 0.8),
                random_state=hp.get('random_state', 42),
                n_jobs=hp.get('n_jobs', -1),
                verbose=-1,
            )
        # Default: sklearn HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=hp.get('n_estimators', 300),
            learning_rate=hp.get('learning_rate', 0.05),
            max_depth=hp.get('max_depth', None),
            max_leaf_nodes=hp.get('max_leaf_nodes', 63),
            min_samples_leaf=hp.get('min_samples_leaf', 20),
            l2_regularization=hp.get('l2_regularization', 0.1),
            random_state=hp.get('random_state', 42),
            early_stopping=False,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        print("\n" + "=" * 50)
        print(f"Training GBM Regressor [{self.backend}]")
        print("=" * 50)
        self.model = self.build_model()
        start = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"Training completed in {elapsed:.2f}s on {len(X_train)} samples")

        train_metrics = self.evaluate(X_train, y_train)
        print(f"Train  RMSE={train_metrics['rmse']:.3f}  R²={train_metrics['r2']:.3f}")

        self.training_history = {
            'train_metrics': train_metrics,
            'training_time': elapsed,
            'n_train_samples': len(X_train),
            'hyperparameters': self.hyperparameters,
            'backend': self.backend,
        }
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            print(f"Val    RMSE={val_metrics['rmse']:.3f}  R²={val_metrics['r2']:.3f}")
            self.training_history['val_metrics'] = val_metrics
            self.training_history['n_val_samples'] = len(X_val)
        return self.training_history
```

- [ ] **Step 6: Create `svm_regressor.py`**

```python
# benchmarks/02_training/models/svm_regressor.py
"""LinearSVR-based regressor with feature scaling. Sparse-safe."""
import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from .base_regression_trainer import BaseRegressionTrainer


class SVMRegressorTrainer(BaseRegressionTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name='svm_regressor')
        self.hyperparameters = config.get('hyperparameters', {})

    def build_model(self) -> Pipeline:
        hp = self.hyperparameters
        svr = LinearSVR(
            C=hp.get('C', 1.0),
            epsilon=hp.get('epsilon', 0.1),
            max_iter=hp.get('max_iter', 2000),
            random_state=hp.get('random_state', 42),
            dual='auto',
        )
        # StandardScaler with_mean=False is sparse-safe (Morgan FP are sparse)
        return Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('svr', svr),
        ])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X).astype(np.float32)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        print("\n" + "=" * 50)
        print("Training SVM Regressor (LinearSVR)")
        print("=" * 50)
        self.model = self.build_model()
        start = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"Training completed in {elapsed:.2f}s on {len(X_train)} samples")

        train_metrics = self.evaluate(X_train, y_train)
        print(f"Train  RMSE={train_metrics['rmse']:.3f}  R²={train_metrics['r2']:.3f}")

        self.training_history = {
            'train_metrics': train_metrics,
            'training_time': elapsed,
            'n_train_samples': len(X_train),
            'hyperparameters': self.hyperparameters,
        }
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            print(f"Val    RMSE={val_metrics['rmse']:.3f}  R²={val_metrics['r2']:.3f}")
            self.training_history['val_metrics'] = val_metrics
            self.training_history['n_val_samples'] = len(X_val)
        return self.training_history
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
conda run -n rdkit_env pytest benchmarks/02_training/tests/test_regression_trainers.py -v 2>&1 | tail -20
```
Expected: `7 passed`

- [ ] **Step 8: Commit**

```bash
git add benchmarks/02_training/models/base_regression_trainer.py \
        benchmarks/02_training/models/rf_regressor.py \
        benchmarks/02_training/models/gbm_regressor.py \
        benchmarks/02_training/models/svm_regressor.py \
        benchmarks/02_training/tests/test_regression_trainers.py
git commit -m "feat: add RF/GBM/SVM regression trainers with R²/RMSE/Spearman evaluation"
```

---

## Task 4: train_regression.py — end-to-end regression training script

**Files:**
- Create: `benchmarks/02_training/train_regression.py`
- Create: `benchmarks/configs/regression_rf_config.yaml`
- Create: `benchmarks/configs/regression_gbm_config.yaml`
- Create: `benchmarks/configs/regression_svm_config.yaml`

- [ ] **Step 1: Create the three YAML configs**

```yaml
# benchmarks/configs/regression_rf_config.yaml
model_type: "random_forest"
task: "regression"
backend: ~

hyperparameters:
  n_estimators: 100
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  max_features: "sqrt"
  bootstrap: true
  random_state: 42
  n_jobs: -1

features:
  type: "combined"
  ligand:
    type: "morgan_fingerprint"
    radius: 2
    n_bits: 2048
    use_features: false
  protein:
    type: "protein_identifier"
    embedding_dim: 32
    use_onehot: false
  concatenation_method: "concat"

data:
  similarity_threshold: "0p7"
  include_decoys: false          # regression uses actives only (pChEMBL defined)
  include_protein_features: true
  task: "regression"
  use_2d_split: true

metrics:
  - rmse
  - mae
  - r2
  - pearson
  - spearman
  - ci
```

```yaml
# benchmarks/configs/regression_gbm_config.yaml
model_type: "gradient_boosting"
task: "regression"
backend: "auto"

hyperparameters:
  n_estimators: 300
  learning_rate: 0.05
  max_depth: null
  max_leaf_nodes: 63
  min_samples_leaf: 20
  subsample: 0.8
  colsample_bytree: 0.8
  l2_regularization: 0.1
  random_state: 42
  n_jobs: -1

features:
  type: "combined"
  ligand:
    type: "morgan_fingerprint"
    radius: 2
    n_bits: 2048
    use_features: false
  protein:
    type: "protein_identifier"
    embedding_dim: 32
    use_onehot: false
  concatenation_method: "concat"

data:
  similarity_threshold: "0p7"
  include_decoys: false
  include_protein_features: true
  task: "regression"
  use_2d_split: true

metrics:
  - rmse
  - mae
  - r2
  - pearson
  - spearman
  - ci
```

```yaml
# benchmarks/configs/regression_svm_config.yaml
model_type: "svm"
task: "regression"
backend: ~

hyperparameters:
  C: 1.0
  epsilon: 0.1
  max_iter: 2000
  random_state: 42

features:
  type: "combined"
  ligand:
    type: "morgan_fingerprint"
    radius: 2
    n_bits: 2048
    use_features: false
  protein:
    type: "protein_identifier"
    embedding_dim: 32
    use_onehot: false
  concatenation_method: "concat"

data:
  similarity_threshold: "0p7"
  include_decoys: false
  include_protein_features: true
  task: "regression"
  use_2d_split: true

metrics:
  - rmse
  - mae
  - r2
  - pearson
  - spearman
  - ci
```

- [ ] **Step 2: Create `train_regression.py`**

```python
# benchmarks/02_training/train_regression.py
"""
Regression training script for pChEMBL binding affinity prediction.

Mirrors train_classical_oddt.py but:
  - targets pChEMBL (continuous) instead of is_active (binary)
  - excludes decoys (they have no pChEMBL)
  - uses regression trainers (RF/GBM/SVM Regressor)
  - reports RMSE/MAE/R²/Pearson/Spearman instead of ROC-AUC/F1

Usage:
  conda run -n rdkit_env python train_regression.py \
      --config ../configs/regression_rf_config.yaml \
      --registry ../../training_data_full/registry_soft_split_regression.csv \
      --use-2d-split
"""
import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent))

from data.data_loader import DataLoader
from features.featurizer import get_featurizer
from features.combined_featurizer import get_combined_featurizer
from models.rf_regressor import RandomForestRegressorTrainer
from models.gbm_regressor import GBMRegressorTrainer
from models.svm_regressor import SVMRegressorTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_regression_trainer(model_type: str, config: dict):
    if model_type == 'random_forest':
        return RandomForestRegressorTrainer(config)
    if model_type in ('gradient_boosting', 'gbm', 'xgboost', 'lightgbm'):
        return GBMRegressorTrainer(config)
    if model_type == 'svm':
        return SVMRegressorTrainer(config)
    raise ValueError(
        f"Unknown model_type: {model_type!r}. "
        "Supported: ['random_forest', 'gradient_boosting', 'svm']"
    )


def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('regression_training')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(ch)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(Path(output_dir) / f'regression_{ts}.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
    logger.addHandler(fh)
    return logger


def train_regression(
    config_path: str,
    registry_path: str,
    output_dir: str = './trained_models_regression',
    use_2d_split: bool = False,
    quick_test: bool = False,
    test_samples: int = 500,
    cache_dir: Optional[str] = None,
) -> None:
    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("Regression Training Pipeline — pChEMBL Prediction")
    logger.info("=" * 70)

    if quick_test:
        logger.warning(f"QUICK TEST MODE: {test_samples} samples")

    config = load_config(config_path)
    model_type = config['model_type']
    data_config = config.get('data', {})
    similarity_threshold = data_config.get('similarity_threshold', '0p7')
    include_protein_features = data_config.get('include_protein_features', True)

    if not use_2d_split:
        use_2d_split = data_config.get('use_2d_split', False)

    logger.info(f"Model: {model_type} | Threshold: {similarity_threshold} | 2D split: {use_2d_split}")

    # ------------------------------------------------------------------ data
    logger.info(f"\nLoading registry: {registry_path}")
    loader = DataLoader(registry_path)
    loader.load_registry()

    def _load(split_name, protein_partition=None):
        data = loader.get_training_data(
            similarity_threshold=similarity_threshold,
            include_decoys=False,   # no decoys for regression
            split=split_name,
            protein_partition=protein_partition,
        )
        if include_protein_features:
            smiles, y, pids = loader.prepare_features_labels(
                data, task='regression', include_protein_info=True
            )
        else:
            smiles, y = loader.prepare_features_labels(data, task='regression')
            pids = None
        return smiles, y, pids

    if use_2d_split:
        train_smiles, y_train, train_pids = _load('train', 'train')
        val_smiles,   y_val,   val_pids   = _load('test',  'val')
        test_smiles,  y_test,  test_pids  = _load('test',  'test')
    else:
        from sklearn.model_selection import train_test_split
        all_smiles, y_all, all_pids = _load('train')
        idx = np.arange(len(all_smiles))
        tr_idx, v_idx = train_test_split(idx, test_size=0.2, random_state=42)
        train_smiles = [all_smiles[i] for i in tr_idx]
        val_smiles   = [all_smiles[i] for i in v_idx]
        y_train, y_val = y_all[tr_idx], y_all[v_idx]
        train_pids = [all_pids[i] for i in tr_idx] if all_pids else None
        val_pids   = [all_pids[i] for i in v_idx]  if all_pids else None
        test_smiles, y_test, test_pids = _load('test')

    logger.info(f"  Train: {len(train_smiles)} | Val: {len(val_smiles)} | Test: {len(test_smiles)}")

    if quick_test:
        for name, smiles_list, y in [
            ('train', train_smiles, y_train),
            ('val',   val_smiles,   y_val),
            ('test',  test_smiles,  y_test),
        ]:
            cap = {'train': test_samples, 'val': test_samples // 5, 'test': test_samples // 2}[name]
            if len(smiles_list) > cap:
                idx = np.random.default_rng(42).choice(len(smiles_list), cap, replace=False)
                if name == 'train':
                    train_smiles = [smiles_list[i] for i in idx]; y_train = y[idx]
                elif name == 'val':
                    val_smiles   = [smiles_list[i] for i in idx]; y_val   = y[idx]
                else:
                    test_smiles  = [smiles_list[i] for i in idx]; y_test  = y[idx]

    # ------------------------------------------------------------------ features
    feature_config = config.get('features', {})
    if include_protein_features and feature_config.get('type') == 'combined':
        ligand_config = feature_config.get('ligand', {'type': 'morgan_fingerprint', 'radius': 2, 'n_bits': 2048})
        protein_config = feature_config.get('protein', {'type': 'protein_identifier', 'embedding_dim': 32})
        featurizer = get_combined_featurizer(
            ligand_config=ligand_config,
            protein_config=protein_config,
            concatenation_method=feature_config.get('concatenation_method', 'concat'),
        )
        all_pids = (train_pids or []) + (val_pids or []) + (test_pids or [])
        featurizer.fit_protein_featurizer(all_pids)
        X_train, _ = featurizer.featurize(train_smiles, protein_ids=train_pids, show_progress=True)
        X_val,   _ = featurizer.featurize(val_smiles,   protein_ids=val_pids,   show_progress=True)
        using_combined = True
    else:
        featurizer = get_featurizer(feature_config)
        X_train, _ = featurizer.featurize(train_smiles, show_progress=True)
        X_val,   _ = featurizer.featurize(val_smiles,   show_progress=True)
        using_combined = False

    logger.info(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")

    # ------------------------------------------------------------------ train
    trainer = get_regression_trainer(model_type, config)
    trainer.train(X_train, y_train, X_val, y_val)

    del X_train, X_val
    gc.collect()

    # ------------------------------------------------------------------ test
    if using_combined:
        X_test, _ = featurizer.featurize(test_smiles, protein_ids=test_pids, show_progress=True)
    else:
        X_test, _ = featurizer.featurize(test_smiles, show_progress=True)

    test_metrics = trainer.evaluate(X_test, y_test)
    logger.info("\nTest Set Metrics:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    trainer.training_history['test_metrics'] = test_metrics

    # ------------------------------------------------------------------ save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    if using_combined:
        featurizer.save_protein_mapping(str(out / f"{trainer.model_name}_protein_mapping.json"))
    with open(out / f"{trainer.model_name}_feature_config.json", 'w') as f:
        json.dump(featurizer.get_config(), f, indent=2)

    summary = {
        'model_type': model_type,
        'task': 'regression',
        'feature_type': featurizer.name,
        'similarity_threshold': similarity_threshold,
        'training_history': trainer.training_history,
        'data_config': data_config,
        'use_2d_split': use_2d_split,
    }
    with open(out / f"{trainer.model_name}_training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("Regression Training Complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train regression models for pChEMBL prediction')
    parser.add_argument('--config',   required=True, help='Path to regression YAML config')
    parser.add_argument('--registry', required=True, help='Path to registry_soft_split_regression.csv')
    parser.add_argument('--output',   default='./trained_models_regression')
    parser.add_argument('--use-2d-split', action='store_true')
    parser.add_argument('--quick-test',   action='store_true')
    parser.add_argument('--test-samples', type=int, default=500)
    parser.add_argument('--cache-dir',    default=None)
    args = parser.parse_args()
    train_regression(
        config_path=args.config,
        registry_path=args.registry,
        output_dir=args.output,
        use_2d_split=args.use_2d_split,
        quick_test=args.quick_test,
        test_samples=args.test_samples,
        cache_dir=args.cache_dir,
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Run a smoke test (quick_test mode)**

```bash
cd /home/aoxu/projects/VLS-soft-partition
conda run -n rdkit_env python benchmarks/02_training/train_regression.py \
    --config benchmarks/configs/regression_rf_config.yaml \
    --registry /home/aoxu/projects/VLS-Benchmark-Dataset/training_data_full/registry_soft_split_regression.csv \
    --use-2d-split \
    --quick-test \
    --output benchmarks/02_training/trained_models_regression_test \
    2>&1 | tail -30
```

Expected output includes:
```
Regression Training Complete!
Test Set Metrics:
  rmse: X.XXX
  mae:  X.XXX
  r2:   X.XXX
  pearson:  X.XXX
  spearman: X.XXX
  ci:       X.XXX
```

- [ ] **Step 4: Commit**

```bash
git add benchmarks/02_training/train_regression.py \
        benchmarks/configs/regression_rf_config.yaml \
        benchmarks/configs/regression_gbm_config.yaml \
        benchmarks/configs/regression_svm_config.yaml
git commit -m "feat: add regression training script and configs for RF/GBM/SVM"
```

---

## Task 5: Full training run + push

- [ ] **Step 1: Run full RF regression training**

```bash
conda run -n rdkit_env python benchmarks/02_training/train_regression.py \
    --config benchmarks/configs/regression_rf_config.yaml \
    --registry /home/aoxu/projects/VLS-Benchmark-Dataset/training_data_full/registry_soft_split_regression.csv \
    --use-2d-split \
    --output benchmarks/02_training/trained_models_regression \
    2>&1 | tee /tmp/rf_regression.log
```

This will take 5-20 minutes depending on the enriched sample count from Task 1.

- [ ] **Step 2: Verify outputs exist**

```bash
ls benchmarks/02_training/trained_models_regression/
# Expected:
# random_forest_regressor.pkl
# random_forest_regressor_config.json
# random_forest_regressor_history.json
# random_forest_regressor_training_summary.json
# random_forest_regressor_feature_config.json
# random_forest_regressor_protein_mapping.json
```

- [ ] **Step 3: Push the feature branch**

```bash
cd /home/aoxu/projects/VLS-soft-partition
git push https://AaronXu9:<TOKEN>@github.com/AaronXu9/PLATE-VS.git feature/soft-partition-regression
```

Replace `<TOKEN>` with the GitHub PAT.

---

## Self-Review

**Spec coverage:**
- ✅ pChEMBL enrichment from `chembl_activities_enriched.parquet` → Task 1
- ✅ RF regression model → Task 3 (rf_regressor.py) + Task 4
- ✅ GBM regression model → Task 3 (gbm_regressor.py) + Task 4
- ✅ SVM regression model → Task 3 (svm_regressor.py) + Task 4
- ✅ RMSE, MAE, R², Pearson, Spearman metrics → Task 2
- ✅ CI (concordance index) already in `metrics.py` — included in `summarize_regression`
- ✅ Same feature pipeline (Morgan FP + protein embeddings) → reused via `get_combined_featurizer`
- ✅ Soft split registry used throughout

**Placeholder scan:** None found — all code blocks are complete.

**Type consistency:**
- `BaseRegressionTrainer.predict()` → returns `np.float32` array — matches usage in `evaluate()`
- `summarize_regression(y, y_pred)` → called in `BaseRegressionTrainer.evaluate()` with matching signature
- `loader.prepare_features_labels(data, task='regression')` → returns `(smiles, y_float32)` or `(smiles, y_float32, pids)` — matches `_load()` in `train_regression.py`
- `GBMRegressorTrainer.model_name = 'gradient_boosting_regressor'` — consistent across save/load paths
