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
