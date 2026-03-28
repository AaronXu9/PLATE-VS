"""
Comparative Analysis: PLATE-VS vs PDBbind baselines.

Loads results from generate_benchmark_report and compares them against
published benchmark numbers from PDBbind v.2020 virtual screening studies.

Literature baselines (ROC-AUC on held-out test sets):
  - Random Forest + ECFP4:   ~0.72–0.78  (Wallach et al. 2018; Riniker & Landrum 2013)
  - Gradient Boosting + ECFP: ~0.74–0.80
  - SVM + ECFP:              ~0.70–0.76
  - DeepDTA (CNN):           ~0.74–0.78  (Öztürk et al. 2018)
  - Graph-DTA:               ~0.76–0.80  (Nguyen et al. 2021)
  Note: exact values vary by dataset split and decoy scheme.

Usage:
    python compare_to_pdbbind.py --results-dir ../02_training/trained_models
    python compare_to_pdbbind.py --results-dir ../02_training/trained_models --metric roc_auc
"""

import argparse
import sys
from pathlib import Path

# Reuse the loader from generate_benchmark_report
sys.path.insert(0, str(Path(__file__).parent))
from generate_benchmark_report import collect_summaries, parse_row, REPORT_METRICS

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Published baselines
# ---------------------------------------------------------------------------

# Format: { display_name: {metric: (low, high)} }
PDBBIND_BASELINES = {
    'RF + ECFP4 (lit.)': {
        'roc_auc':       (0.72, 0.78),
        'avg_precision': (0.65, 0.72),
        'f1_score':      (0.60, 0.70),
    },
    'GBM + ECFP (lit.)': {
        'roc_auc':       (0.74, 0.80),
        'avg_precision': (0.67, 0.74),
        'f1_score':      (0.62, 0.72),
    },
    'SVM + ECFP (lit.)': {
        'roc_auc':       (0.70, 0.76),
        'avg_precision': (0.63, 0.70),
        'f1_score':      (0.58, 0.68),
    },
    'DeepDTA CNN (lit.)': {
        'roc_auc':       (0.74, 0.78),
        'avg_precision': (0.66, 0.72),
        'f1_score':      (0.61, 0.70),
    },
}


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def _delta_symbol(our_val: float, lit_low: float, lit_high: float) -> str:
    if our_val > lit_high:
        return f'+{our_val - lit_high:+.3f} (above range)'
    elif our_val < lit_low:
        return f'{our_val - lit_low:+.3f} (below range)'
    else:
        return '  within range'


def compare(rows: list[dict], metric: str = 'roc_auc', split: str = 'test') -> None:
    col = f'{split}_{metric}'

    print(f"\n{'='*72}")
    print(f"  Comparison: PLATE-VS vs PDBbind literature ({metric}, {split} split)")
    print('='*72)

    # Our results
    print(f"\n  Our results ({len(rows)} run(s)):")
    print(f"  {'Model':<30} {'Sim. threshold':<18} {metric.upper()}")
    print(f"  {'-'*60}")
    for r in rows:
        val = r.get(col)
        val_str = f'{val:.4f}' if val is not None else '   n/a'
        print(f"  {r['model']:<30} {r['similarity_threshold']:<18} {val_str}")

    # Literature baselines
    print(f"\n  Literature baselines (PDBbind v.2020 / similar datasets):")
    print(f"  {'Method':<30} {'Range'}")
    print(f"  {'-'*60}")
    for name, metrics in PDBBIND_BASELINES.items():
        if metric in metrics:
            lo, hi = metrics[metric]
            print(f"  {name:<30} {lo:.2f} – {hi:.2f}")

    # Delta comparison — match our models to closest literature entry
    print(f"\n  Delta analysis:")
    print(f"  {'Our model':<30} {'Our value':<12} {'Closest lit. ref':<30} {'Delta'}")
    print(f"  {'-'*80}")

    model_to_lit = {
        'random_forest':     'RF + ECFP4 (lit.)',
        'gradient_boosting': 'GBM + ECFP (lit.)',
        'svm':               'SVM + ECFP (lit.)',
    }

    for r in rows:
        val = r.get(col)
        if val is None:
            continue
        lit_key = model_to_lit.get(r['model'])
        if lit_key and metric in PDBBIND_BASELINES.get(lit_key, {}):
            lo, hi = PDBBIND_BASELINES[lit_key][metric]
            delta = _delta_symbol(val, lo, hi)
            label = f"{r['model']} ({r['similarity_threshold']})"
            print(f"  {label:<30} {val:<12.4f} {lit_key:<30} {delta}")
        else:
            label = f"{r['model']} ({r['similarity_threshold']})"
            print(f"  {label:<30} {val:<12.4f} {'(no direct baseline)':<30}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare_pdbbind(results_dir: str, metric: str = 'roc_auc', split: str = 'test') -> None:
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    summaries = collect_summaries(results_path)
    if not summaries:
        print(f"No *_training_summary.json files found in {results_dir}")
        print("Run train_classical_oddt.py first to generate results.")
        return

    rows = [parse_row(s) for s in summaries]
    compare(rows, metric=metric, split=split)


def main():
    parser = argparse.ArgumentParser(
        description='Compare PLATE-VS benchmark results against PDBbind literature baselines'
    )
    parser.add_argument(
        '--results-dir', '-d',
        type=str,
        default='../02_training/trained_models',
        help='Directory containing *_training_summary.json files'
    )
    parser.add_argument(
        '--metric', '-m',
        type=str,
        default='roc_auc',
        choices=REPORT_METRICS,
        help='Metric to compare (default: roc_auc)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Which split to compare (default: test)'
    )
    args = parser.parse_args()
    compare_pdbbind(args.results_dir, args.metric, args.split)


if __name__ == "__main__":
    main()
