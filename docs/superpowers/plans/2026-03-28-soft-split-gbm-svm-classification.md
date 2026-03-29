# Soft-Split GBM/SVM Classification + Branch Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train GBM and SVM classifiers on the soft-split registry, regenerate a unified benchmark report covering all completed runs, and merge the branch to main.

**Architecture:** Re-use `train_classical_oddt.py` with existing `gbm_config.yaml` / `svm_config.yaml`, outputting to `trained_models/soft_split_classification/` alongside the existing RF. Add `--extra-dirs` CLI flag to `generate_benchmark_report.py` so one invocation covers hard-split, soft-split classification, and regression results.

**Tech Stack:** Python 3, scikit-learn (HistGradientBoosting, LinearSVC), conda env `rdkit_env`, existing CLI scripts.

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `benchmarks/03_analysis/generate_benchmark_report.py` | Modify | Add `--extra-dirs` CLI flag |
| `benchmarks/03_analysis/tests/test_generate_benchmark_report.py` | Create | Test new `--extra-dirs` flag |
| `trained_models/soft_split_classification/gradient_boosting_*.json` | Create (runtime) | GBM training outputs |
| `trained_models/soft_split_classification/svm_*.json` | Create (runtime) | SVM training outputs |
| `benchmarks/03_analysis/report.csv` | Overwrite (runtime) | Unified report with all 9 model runs |

---

## Task 1: Add `--extra-dirs` CLI flag to `generate_benchmark_report.py`

The `collect_summaries()` function already accepts `extra_dirs: list[Path]` but the CLI only exposes `--docking-dir` (single path). We need `--extra-dirs` that accepts multiple paths.

**Files:**
- Modify: `benchmarks/03_analysis/generate_benchmark_report.py`
- Create: `benchmarks/03_analysis/tests/test_generate_benchmark_report.py`

- [ ] **Step 1: Create tests directory and write the failing test**

```bash
mkdir -p benchmarks/03_analysis/tests
touch benchmarks/03_analysis/tests/__init__.py
```

Create `benchmarks/03_analysis/tests/test_generate_benchmark_report.py`:

```python
"""Tests for generate_benchmark_report CLI --extra-dirs flag."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "generate_benchmark_report.py"

MINIMAL_SUMMARY = {
    "model_type": "random_forest",
    "feature_type": "morgan",
    "similarity_threshold": "0p7",
    "use_precomputed_split": False,
    "training_history": {
        "n_train_samples": 100,
        "n_val_samples": 20,
        "n_test_samples": 20,
        "training_time": 1.0,
        "train_metrics": {"roc_auc": 0.9, "avg_precision": 0.5, "f1_score": 0.4,
                          "accuracy": 0.8, "precision": 0.4, "recall": 0.4, "mcc": 0.3},
        "val_metrics":   {"roc_auc": 0.7, "avg_precision": 0.3, "f1_score": 0.2,
                          "accuracy": 0.7, "precision": 0.2, "recall": 0.2, "mcc": 0.1},
        "test_metrics":  {"roc_auc": 0.6, "avg_precision": 0.2, "f1_score": 0.1,
                          "accuracy": 0.6, "precision": 0.1, "recall": 0.1, "mcc": 0.05},
    },
}


def _write_summary(directory: Path, model_name: str) -> Path:
    summary = dict(MINIMAL_SUMMARY, model_type=model_name)
    path = directory / f"{model_name}_training_summary.json"
    path.write_text(json.dumps(summary))
    return path


def test_extra_dirs_includes_summaries_from_additional_paths():
    """--extra-dirs should cause summaries from those dirs to appear in the CSV output."""
    with tempfile.TemporaryDirectory() as primary_str, \
         tempfile.TemporaryDirectory() as extra_str, \
         tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out_f:

        primary = Path(primary_str)
        extra = Path(extra_str)
        out_csv = Path(out_f.name)

        _write_summary(primary, "random_forest")
        _write_summary(extra, "gradient_boosting")

        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--results-dir", str(primary),
             "--extra-dirs", str(extra),
             "--output", str(out_csv)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

        content = out_csv.read_text()
        assert "random_forest" in content, "primary dir model missing from CSV"
        assert "gradient_boosting" in content, "extra dir model missing from CSV"


def test_extra_dirs_accepts_multiple_paths():
    """--extra-dirs should accept more than one path."""
    with tempfile.TemporaryDirectory() as primary_str, \
         tempfile.TemporaryDirectory() as extra1_str, \
         tempfile.TemporaryDirectory() as extra2_str, \
         tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out_f:

        primary = Path(primary_str)
        extra1 = Path(extra1_str)
        extra2 = Path(extra2_str)
        out_csv = Path(out_f.name)

        _write_summary(primary, "random_forest")
        _write_summary(extra1, "gradient_boosting")
        _write_summary(extra2, "svm")

        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--results-dir", str(primary),
             "--extra-dirs", str(extra1), str(extra2),
             "--output", str(out_csv)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

        content = out_csv.read_text()
        assert "random_forest" in content
        assert "gradient_boosting" in content
        assert "svm" in content
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/aoxu/projects/VLS-Benchmark-Dataset
conda run -n rdkit_env python3 -m pytest benchmarks/03_analysis/tests/test_generate_benchmark_report.py -v
```

Expected: FAIL — `error: unrecognized arguments: --extra-dirs`

- [ ] **Step 3: Add `--extra-dirs` to the CLI and `generate_report` function**

In `benchmarks/03_analysis/generate_benchmark_report.py`, make these two edits:

**Edit 1** — change `generate_report` signature (line ~145) to accept `extra_dirs_cli`:

```python
def generate_report(results_dir: str, output_csv: str = None,
                    split: str = 'all', verbose: bool = False,
                    docking_dir: str = None,
                    extra_dirs_cli: list = None) -> list[dict]:
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    extra_dirs = []
    if docking_dir:
        docking_path = Path(docking_dir)
        if docking_path.exists():
            extra_dirs.append(docking_path)
        else:
            print(f"Warning: docking dir not found: {docking_dir}", file=sys.stderr)
    if extra_dirs_cli:
        for p in extra_dirs_cli:
            ep = Path(p)
            if ep.exists():
                extra_dirs.append(ep)
            else:
                print(f"Warning: extra dir not found: {p}", file=sys.stderr)
```

**Edit 2** — add `--extra-dirs` argument and pass it through in `main()` (after the `--docking-dir` argument, before `args = parser.parse_args()`):

```python
    parser.add_argument(
        '--extra-dirs',
        type=str,
        nargs='+',
        default=None,
        help='Additional directories to scan for *_training_summary.json files'
    )
```

And update the `generate_report` call at the bottom of `main()`:

```python
    generate_report(args.results_dir, args.output, args.split, args.verbose,
                    args.docking_dir, args.extra_dirs)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/aoxu/projects/VLS-Benchmark-Dataset
conda run -n rdkit_env python3 -m pytest benchmarks/03_analysis/tests/test_generate_benchmark_report.py -v
```

Expected: PASS — 2 tests pass

- [ ] **Step 5: Commit**

```bash
git add benchmarks/03_analysis/generate_benchmark_report.py \
        benchmarks/03_analysis/tests/__init__.py \
        benchmarks/03_analysis/tests/test_generate_benchmark_report.py
git commit -m "feat: add --extra-dirs flag to generate_benchmark_report for multi-dir aggregation"
```

---

## Task 2: Quick-test gate — GBM and SVM

Validate that both configs load and run correctly against the soft-split registry with `--quick-test` (1000 samples) before committing to multi-hour full runs. The GBM `backend: auto` will try XGBoost, fall back to HistGBM since XGBoost is not installed in `rdkit_env`.

**Files:**
- (no file changes — just running existing scripts)

- [ ] **Step 1: Quick-test GBM**

```bash
cd /home/aoxu/projects/VLS-Benchmark-Dataset
conda run -n rdkit_env python3 benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/gbm_config.yaml \
    --registry training_data_full/registry_soft_split.csv \
    --use-2d-split \
    --output trained_models/soft_split_classification \
    --cache-dir training_data_full/feature_cache \
    --quick-test
```

Expected: completes in <3 min, prints test ROC-AUC, writes `trained_models/soft_split_classification/gradient_boosting_training_summary.json`

- [ ] **Step 2: Verify GBM quick-test output**

```bash
python3 -c "
import json
d = json.load(open('trained_models/soft_split_classification/gradient_boosting_training_summary.json'))
h = d['training_history']
print('n_train:', h['n_train_samples'])
print('test_roc_auc:', h['test_metrics']['roc_auc'])
print('model_type:', d['model_type'])
"
```

Expected: `model_type: gradient_boosting`, `n_train: 1000`, non-null `test_roc_auc`

- [ ] **Step 3: Quick-test SVM**

```bash
conda run -n rdkit_env python3 benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/svm_config.yaml \
    --registry training_data_full/registry_soft_split.csv \
    --use-2d-split \
    --output trained_models/soft_split_classification \
    --cache-dir training_data_full/feature_cache \
    --quick-test
```

Expected: completes in <2 min, writes `trained_models/soft_split_classification/svm_training_summary.json`

- [ ] **Step 4: Verify SVM quick-test output**

```bash
python3 -c "
import json
d = json.load(open('trained_models/soft_split_classification/svm_training_summary.json'))
h = d['training_history']
print('n_train:', h['n_train_samples'])
print('test_roc_auc:', h['test_metrics']['roc_auc'])
print('model_type:', d['model_type'])
"
```

Expected: `model_type: svm`, `n_train: 1000`, non-null `test_roc_auc`

> If either quick-test fails, inspect the log in `trained_models/soft_split_classification/training_*.log` before proceeding to full training.

---

## Task 3: Full GBM classification training on soft split

~1.73M train samples, HistGradientBoosting (sklearn fallback). Expect 15–30 min.

**Files:**
- Create (runtime): `trained_models/soft_split_classification/gradient_boosting_*.json`

- [ ] **Step 1: Run full GBM training**

```bash
cd /home/aoxu/projects/VLS-Benchmark-Dataset
conda run -n rdkit_env python3 benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/gbm_config.yaml \
    --registry training_data_full/registry_soft_split.csv \
    --use-2d-split \
    --output trained_models/soft_split_classification \
    --cache-dir training_data_full/feature_cache
```

Expected: completes without error. Final lines of output include `Training Complete!` and `Model saved to: trained_models/soft_split_classification`.

- [ ] **Step 2: Verify GBM full training results**

```bash
python3 -c "
import json
d = json.load(open('trained_models/soft_split_classification/gradient_boosting_training_summary.json'))
h = d['training_history']
print(f'n_train:       {h[\"n_train_samples\"]:,}')
print(f'n_val:         {h[\"n_val_samples\"]:,}')
print(f'n_test:        {h[\"n_test_samples\"]:,}')
print(f'train_roc_auc: {h[\"train_metrics\"][\"roc_auc\"]:.4f}')
print(f'val_roc_auc:   {h[\"val_metrics\"][\"roc_auc\"]:.4f}')
print(f'test_roc_auc:  {h[\"test_metrics\"][\"roc_auc\"]:.4f}')
"
```

Expected: n_train ~1,730,902, test_roc_auc is a real float (not null). Comparable to RF soft-split test ROC-AUC of 0.472.

- [ ] **Step 3: Commit GBM results**

```bash
git add trained_models/soft_split_classification/gradient_boosting_config.json \
        trained_models/soft_split_classification/gradient_boosting_feature_config.json \
        trained_models/soft_split_classification/gradient_boosting_history.json \
        trained_models/soft_split_classification/gradient_boosting_training_summary.json
git commit -m "feat: add GBM classification results on soft split"
```

---

## Task 4: Full SVM classification training on soft split

LinearSVC on ~1.73M samples. Expect 5–15 min.

**Files:**
- Create (runtime): `trained_models/soft_split_classification/svm_*.json`

- [ ] **Step 1: Run full SVM training**

```bash
cd /home/aoxu/projects/VLS-Benchmark-Dataset
conda run -n rdkit_env python3 benchmarks/02_training/train_classical_oddt.py \
    --config benchmarks/configs/svm_config.yaml \
    --registry training_data_full/registry_soft_split.csv \
    --use-2d-split \
    --output trained_models/soft_split_classification \
    --cache-dir training_data_full/feature_cache
```

Expected: completes without error, prints `Training Complete!`.

- [ ] **Step 2: Verify SVM full training results**

```bash
python3 -c "
import json
d = json.load(open('trained_models/soft_split_classification/svm_training_summary.json'))
h = d['training_history']
print(f'n_train:       {h[\"n_train_samples\"]:,}')
print(f'n_val:         {h[\"n_val_samples\"]:,}')
print(f'n_test:        {h[\"n_test_samples\"]:,}')
print(f'train_roc_auc: {h[\"train_metrics\"][\"roc_auc\"]:.4f}')
print(f'val_roc_auc:   {h[\"val_metrics\"][\"roc_auc\"]:.4f}')
print(f'test_roc_auc:  {h[\"test_metrics\"][\"roc_auc\"]:.4f}')
"
```

Expected: n_train ~1,730,902, non-null test_roc_auc.

- [ ] **Step 3: Commit SVM results**

```bash
git add trained_models/soft_split_classification/svm_config.json \
        trained_models/soft_split_classification/svm_feature_config.json \
        trained_models/soft_split_classification/svm_history.json \
        trained_models/soft_split_classification/svm_training_summary.json
git commit -m "feat: add SVM classification results on soft split"
```

---

## Task 5: Regenerate unified benchmark report

Produce a single `report.csv` covering all 9 completed model/task/split combinations.

**Files:**
- Modify: `benchmarks/03_analysis/report.csv`

- [ ] **Step 1: Run report generator with all three result directories**

```bash
cd /home/aoxu/projects/VLS-Benchmark-Dataset/benchmarks/03_analysis
conda run -n rdkit_env python3 generate_benchmark_report.py \
    --results-dir ../02_training/trained_models \
    --extra-dirs ../../trained_models/soft_split_classification \
                 ../../trained_models/regression \
    --output report.csv \
    --verbose
```

Expected: output lists 9+ training runs (RF/GBM/SVM hard-split classification, RF/GBM/SVM soft-split classification, RF/GBM/SVM regression). Some DL stub rows may appear with empty metrics — that is acceptable.

- [ ] **Step 2: Spot-check the CSV**

```bash
python3 -c "
import csv
rows = list(csv.DictReader(open('report.csv')))
print(f'Total rows: {len(rows)}')
for r in rows:
    print(f\"{r['model']:25s}  {r.get('source_file','')[-60:]}\")
    print(f\"  test_roc_auc={r.get('test_roc_auc','')}\")
"
```

Expected: rows for `gradient_boosting`, `svm`, `random_forest` appearing under both hard-split and soft-split source paths, plus all 3 regressor rows.

- [ ] **Step 3: Commit updated report**

```bash
cd /home/aoxu/projects/VLS-Benchmark-Dataset
git add benchmarks/03_analysis/report.csv
git commit -m "chore: regenerate benchmark report with soft-split classification results"
```

---

## Task 6: Create PR and merge to main

- [ ] **Step 1: Verify branch is clean**

```bash
git status
git log main..HEAD --oneline
```

Expected: no uncommitted changes. Log shows the new commits from Tasks 1–5 above the divergence point from main.

- [ ] **Step 2: Push branch to remote**

```bash
git push origin feature/ml-benchmarking
```

- [ ] **Step 3: Create PR**

```bash
gh pr create \
  --title "feat: ML benchmarking pipeline — classical models on hard/soft split + regression" \
  --body "$(cat <<'EOF'
## Summary

- Implements full classical ML benchmarking pipeline (RF, GBM, SVM) on PLATE-VS dataset
- Hard-split classification (protein cluster × ligand similarity): RF/GBM/SVM trained and evaluated
- Soft-split classification (intra-cluster protein partition): RF/GBM/SVM trained and evaluated
- Regression on pChEMBL targets (soft split): RF/GBM/SVM trained and evaluated
- Unified benchmark report in \`benchmarks/03_analysis/report.csv\`

## Key results

| Task | Split | Model | Test ROC-AUC |
|---|---|---|---|
| Classification | Hard (0p7) | RF | 0.304 |
| Classification | Hard (0p7) | GBM | 0.372 |
| Classification | Hard (0p7) | SVM | 0.431 |
| Classification | Soft (0p7) | RF | 0.472 |
| Classification | Soft (0p7) | GBM | (new) |
| Classification | Soft (0p7) | SVM | (new) |
| Regression | Soft (0p7) | RF | R²= -0.48 |
| Regression | Soft (0p7) | GBM | R²= -0.36 |
| Regression | Soft (0p7) | SVM | R²= -0.73 |

Poor generalization is expected: Morgan FP + protein identity features cannot generalize across a similarity-based split. This establishes the lower bound for the benchmark.

## Out of scope / follow-on

- Deep learning models (DeepPurpose, GraphDTA) — stubs exist, deferred to follow-on branch
- PDBbind comparison

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Merge PR after review**

After the PR is reviewed and approved:

```bash
gh pr merge --squash --delete-branch
```

Or merge via GitHub UI with squash-and-merge.
