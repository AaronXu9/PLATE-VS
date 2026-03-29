"""
Benchmark Report Generation Script.

Scans a directory for *_training_summary.json files produced by
train_classical_oddt.py and prints a comparison table across models,
similarity thresholds, and splits.

Usage:
    python generate_benchmark_report.py --results-dir ../02_training/trained_models
    python generate_benchmark_report.py --results-dir ../02_training/trained_models --output report.csv
"""

import argparse
import json
from pathlib import Path
import sys

# Try pandas for pretty tables; fall back to plain text
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# Metrics to include in the report (in display order)
REPORT_METRICS = ['roc_auc', 'avg_precision', 'f1_score', 'accuracy', 'precision', 'recall', 'mcc']
SPLIT_NAMES = ['train', 'val', 'test']


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_summaries(results_dir: Path, extra_dirs: list[Path] = None) -> list[dict]:
    """Recursively find all *_training_summary.json files in results_dir and any extra_dirs."""
    search_dirs = [results_dir]
    if extra_dirs:
        search_dirs.extend(extra_dirs)

    summaries = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for p in sorted(search_dir.rglob('*_training_summary.json')):
            try:
                s = load_summary(p)
                s['_source_file'] = str(p)
                summaries.append(s)
            except Exception as e:
                print(f"Warning: could not load {p}: {e}", file=sys.stderr)
    return summaries


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_row(summary: dict) -> dict:
    """Flatten a training summary into a single report row."""
    history = summary.get('training_history', {})
    row = {
        'model': summary.get('model_type', 'unknown'),
        'feature_type': summary.get('feature_type', 'unknown'),
        'similarity_threshold': summary.get('similarity_threshold', 'unknown'),
        'precomputed_split': summary.get('use_precomputed_split', False),
        'source_file': summary.get('_source_file', ''),
    }

    for split in SPLIT_NAMES:
        metrics = history.get(f'{split}_metrics') or {}
        for metric in REPORT_METRICS:
            val = metrics.get(metric)
            row[f'{split}_{metric}'] = round(val, 4) if val is not None else None

    row['n_train'] = history.get('n_train_samples')
    row['n_val'] = history.get('n_val_samples')
    row['n_test'] = summary.get('training_history', {}).get('n_test_samples')
    row['training_time_s'] = round(history.get('training_time', 0), 1)

    return row


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_val(v) -> str:
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.4f}'
    return str(v)


def print_plain_table(rows: list[dict], focus_split: str = 'test') -> None:
    """Print a plain-text comparison table for the given split."""
    if not rows:
        print("No results found.")
        return

    key_cols = ['model', 'similarity_threshold', 'n_train', 'n_test', 'training_time_s']
    metric_cols = [f'{focus_split}_{m}' for m in REPORT_METRICS]
    all_cols = key_cols + metric_cols

    # Header
    header = '  '.join(f'{c:<22}' for c in all_cols)
    print(header)
    print('-' * len(header))

    for row in rows:
        line = '  '.join(f'{_format_val(row.get(c)):<22}' for c in all_cols)
        print(line)


def print_pandas_table(rows: list[dict], focus_split: str = 'test') -> None:
    df = pd.DataFrame(rows)
    key_cols = ['model', 'similarity_threshold', 'n_train', 'n_test', 'training_time_s']
    metric_cols = [f'{focus_split}_{m}' for m in REPORT_METRICS if f'{focus_split}_{m}' in df.columns]
    display_cols = [c for c in key_cols + metric_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))


def print_all_splits(rows: list[dict]) -> None:
    """Print one section per split."""
    for split in SPLIT_NAMES:
        print(f"\n{'='*70}")
        print(f"  Split: {split.upper()}")
        print('='*70)
        if PANDAS_AVAILABLE:
            print_pandas_table(rows, focus_split=split)
        else:
            print_plain_table(rows, focus_split=split)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(results_dir: str, output_csv: str = None,
                    split: str = 'all', verbose: bool = False,
                    docking_dir: str = None,
                    extra_dirs: list = None) -> list[dict]:
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    extra_paths = []
    if docking_dir:
        docking_path = Path(docking_dir)
        if docking_path.exists():
            extra_paths.append(docking_path)
        else:
            print(f"Warning: docking dir not found: {docking_dir}", file=sys.stderr)

    if extra_dirs:
        for p in extra_dirs:
            ep = Path(p)
            if ep.exists():
                extra_paths.append(ep)
            else:
                print(f"Warning: extra dir not found: {p}", file=sys.stderr)

    summaries = collect_summaries(results_path, extra_paths)
    if not summaries:
        print(f"No *_training_summary.json files found in {results_dir}")
        print("Run train_classical_oddt.py first to generate results.")
        return []

    rows = [parse_row(s) for s in summaries]

    print(f"\nFound {len(rows)} training run(s) in {results_dir}\n")

    if split == 'all':
        print_all_splits(rows)
    else:
        print(f"\n{'='*70}")
        print(f"  Split: {split.upper()}")
        print('='*70)
        if PANDAS_AVAILABLE:
            print_pandas_table(rows, focus_split=split)
        else:
            print_plain_table(rows, focus_split=split)

    if verbose:
        print(f"\n{'='*70}")
        print("  Source files")
        print('='*70)
        for r in rows:
            print(f"  {r['model']:20s}  {r['source_file']}")

    if output_csv:
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(rows)
            df.to_csv(output_csv, index=False)
            print(f"\nReport saved to {output_csv}")
        else:
            import csv
            if rows:
                with open(output_csv, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"\nReport saved to {output_csv}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Generate benchmark report from training run summaries'
    )
    parser.add_argument(
        '--results-dir', '-d',
        type=str,
        default='../02_training/trained_models',
        help='Directory containing *_training_summary.json files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Optional CSV output path'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='all',
        choices=['train', 'val', 'test', 'all'],
        help='Which split to display (default: all)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show source file paths'
    )
    parser.add_argument(
        '--docking-dir',
        type=str,
        default=None,
        help='Optional additional directory to scan for docking *_training_summary.json files'
    )
    parser.add_argument(
        '--extra-dirs',
        type=str,
        nargs='+',
        default=None,
        help='Additional directories to scan for *_training_summary.json files'
    )
    args = parser.parse_args()
    generate_report(
        results_dir=args.results_dir,
        output_csv=args.output,
        split=args.split,
        verbose=args.verbose,
        docking_dir=args.docking_dir,
        extra_dirs=args.extra_dirs,
    )


if __name__ == "__main__":
    main()
