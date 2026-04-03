"""
Download PDBbind CleanSplit data for benchmarking.

Downloads:
  1. Label dictionary and split JSONs from GEMS GitHub repo
  2. (Optionally) Preprocessed PyTorch datasets from Zenodo (9.88 GB)
  3. Cross-validation fold files for reproducibility

Raw PDB+SDF structural files must be obtained separately from
http://www.pdbbind.org.cn/ (registration required).

Usage:
    # Download labels + splits only (fast)
    python download_pdbbind_cleansplit.py

    # Also download preprocessed .pt datasets from Zenodo
    python download_pdbbind_cleansplit.py --include-zenodo

    # Specify output directory
    python download_pdbbind_cleansplit.py --output-dir data/pdbbind_cleansplit
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/camlab-ethz/GEMS/main/PDBbind_data"
)

GITHUB_FILES = {
    "labels": [
        "PDBbind_data_dict.json",
    ],
    "splits": [
        "PDBbind_data_split_cleansplit.json",
        "PDBbind_data_split_pdbbind.json",
    ],
    "cv_folds": [
        f"PDBbind_cleansplit_train_val_split_f{i}.json" for i in range(5)
    ] + [
        f"PDBbind_original_train_val_split_f{i}.json" for i in range(5)
    ],
    "other": [
        "clusters_casf2016.json",
    ],
}

ZENODO_FILES = {
    "GEMS_pytorch_datasets.tar.gz": (
        "https://zenodo.org/api/records/15482796/files/"
        "GEMS_pytorch_datasets.tar.gz/content"
    ),
    "pairwise_similarity_matrices.tar.gz": (
        "https://zenodo.org/api/records/15482796/files/"
        "pairwise_similarity_matrices.tar.gz/content"
    ),
}


def download_file(url: str, dest: Path, description: str = "") -> bool:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True

    label = description or dest.name
    print(f"  Downloading {label} ...")

    try:
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(dest), url],
            check=True,
        )
        return True
    except FileNotFoundError:
        # wget not available, try curl
        try:
            subprocess.run(
                ["curl", "-L", "-o", str(dest), "--progress-bar", url],
                check=True,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"  [ERROR] Failed to download {label}: {exc}")
            if dest.exists():
                dest.unlink()
            return False
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] Failed to download {label}: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def download_github_files(output_dir: Path) -> int:
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for category, filenames in GITHUB_FILES.items():
        print(f"\n--- {category} ---")
        for fname in filenames:
            url = f"{GITHUB_RAW_BASE}/{fname}"
            dest = labels_dir / fname
            if download_file(url, dest, fname):
                downloaded += 1

    return downloaded


def download_zenodo_datasets(output_dir: Path, include_similarity: bool = False) -> int:
    preprocessed_dir = output_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for fname, url in ZENODO_FILES.items():
        if fname == "pairwise_similarity_matrices.tar.gz" and not include_similarity:
            continue

        dest = preprocessed_dir / fname
        if download_file(url, dest, f"{fname} (from Zenodo)"):
            downloaded += 1

            # Extract tarball
            if dest.suffix == ".gz" and dest.stem.endswith(".tar"):
                print(f"  Extracting {fname} ...")
                try:
                    subprocess.run(
                        ["tar", "-xzf", str(dest), "-C", str(preprocessed_dir)],
                        check=True,
                    )
                    print(f"  Extracted to {preprocessed_dir}/")
                except subprocess.CalledProcessError as exc:
                    print(f"  [ERROR] Extraction failed: {exc}")

    return downloaded


def validate_labels(output_dir: Path) -> bool:
    labels_dir = output_dir / "labels"

    dict_path = labels_dir / "PDBbind_data_dict.json"
    split_path = labels_dir / "PDBbind_data_split_cleansplit.json"

    if not dict_path.exists() or not split_path.exists():
        print("\n[WARN] Label files not found — cannot validate.")
        return False

    with open(dict_path) as f:
        data_dict = json.load(f)
    with open(split_path) as f:
        split_dict = json.load(f)

    print(f"\n--- Validation ---")
    print(f"  PDBbind_data_dict.json: {len(data_dict)} complexes")

    for split_name, pdb_ids in split_dict.items():
        n_with_labels = sum(1 for pid in pdb_ids if pid in data_dict)
        print(
            f"  {split_name}: {len(pdb_ids)} complexes "
            f"({n_with_labels} with labels)"
        )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download PDBbind CleanSplit data for benchmarking"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pdbbind_cleansplit",
        help="Output directory (default: data/pdbbind_cleansplit)",
    )
    parser.add_argument(
        "--include-zenodo",
        action="store_true",
        help="Also download preprocessed .pt datasets from Zenodo (~10 GB)",
    )
    parser.add_argument(
        "--include-similarity",
        action="store_true",
        help="Also download pairwise similarity matrices from Zenodo (~3 GB)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # 1. Download label + split files from GitHub
    print("=" * 60)
    print("Step 1: Downloading labels and split files from GitHub")
    print("=" * 60)
    n_github = download_github_files(output_dir)
    print(f"\n  Downloaded {n_github} files from GitHub.")

    # 2. Optionally download Zenodo datasets
    if args.include_zenodo:
        print("\n" + "=" * 60)
        print("Step 2: Downloading preprocessed datasets from Zenodo")
        print("=" * 60)
        n_zenodo = download_zenodo_datasets(
            output_dir, include_similarity=args.include_similarity
        )
        print(f"\n  Downloaded {n_zenodo} files from Zenodo.")

    # 3. Validate
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)
    validate_labels(output_dir)

    # 4. Remind about raw data
    print("\n" + "=" * 60)
    print("NOTE: Raw PDB+SDF structural files")
    print("=" * 60)
    print(
        "  Raw PDB+SDF files are required for GNINA redocking and\n"
        "  SMILES extraction for classical ML models.\n"
        "  Download from: http://www.pdbbind.org.cn/ (registration required)\n"
        f"  Place under: {output_dir / 'raw' / '<pdb_id>' / '<pdb_id>_protein.pdb'}\n"
        f"               {output_dir / 'raw' / '<pdb_id>' / '<pdb_id>_ligand.sdf'}"
    )


if __name__ == "__main__":
    main()
