"""
Extract fixed-length protein embeddings from GEMS preprocessed datasets.

Loads B6AEPL .pt datasets (which contain per-residue ESM2 + ANKH embeddings
concatenated into node features) and mean-pools protein residue embeddings
to produce one fixed-size vector per complex.

Node feature layout (B6AEPL, 1148 dims):
  [0:60]    = base atomic features
  [60:380]  = ANKH-base embeddings (320 dims)
  [380:1148] = ESM2-t6 embeddings (768 dims)

Output: .npz file with {pdb_id: embedding_vector} mapping.

Usage:
    conda run -p /path/to/diffdock/env python extract_protein_embeddings.py \
        --dataset-dir data/pdbbind_cleansplit/preprocessed/GEMS_pytorch_datasets \
        --output data/pdbbind_cleansplit/protein_embeddings.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add GEMS repo to path so torch.load can find the Dataset class
GEMS_REPO = Path(__file__).resolve().parent.parent.parent / "external" / "GEMS"
if GEMS_REPO.exists():
    sys.path.insert(0, str(GEMS_REPO))


BASE_FEAT_DIM = 60  # Base atomic features to skip


def extract_embeddings_from_dataset(dataset_path: str, embedding_start: int = BASE_FEAT_DIM) -> dict:
    """Extract mean-pooled protein embeddings from a GEMS .pt dataset.

    Returns:
        dict of {pdb_id: np.ndarray} where each array is the mean-pooled
        protein embedding vector.
    """
    print(f"  Loading {Path(dataset_path).name}...")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)

    embeddings = {}
    for data in dataset:
        # Parse PDB ID from complex ID (e.g., "3f3c" or "3f3c_L00001_graph")
        pdb_id = data.id.split("_")[0] if "_" in data.id else data.id
        pdb_id = pdb_id.lower()

        # Get node counts: [total, n_ligand, n_protein]
        n_total, n_lig, n_prot = data.n_nodes.tolist()

        if n_prot == 0:
            continue

        # Extract protein node features (skip base atomic features)
        protein_nodes = data.x[n_lig:n_lig + n_prot, embedding_start:]

        # Mean pool across residues → fixed-size vector
        protein_emb = protein_nodes.mean(dim=0).numpy()
        embeddings[pdb_id] = protein_emb

    print(f"    Extracted {len(embeddings)} embeddings, dim={protein_emb.shape[0]}")
    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Extract protein embeddings from GEMS datasets"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/pdbbind_cleansplit/preprocessed/GEMS_pytorch_datasets",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="B6AEPL",
        help="GEMS embedding variant (default: B6AEPL = ANKH + ESM2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/pdbbind_cleansplit/protein_embeddings.npz",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    variant = args.variant

    # Load train + all test sets
    all_embeddings = {}
    for split in ["train_cleansplit", "casf2016", "casf2013", "casf2016_indep", "casf2013_indep"]:
        dataset_path = dataset_dir / f"{variant}_{split}.pt"
        if not dataset_path.exists():
            print(f"  [skip] {dataset_path.name} not found")
            continue
        embs = extract_embeddings_from_dataset(str(dataset_path))
        all_embeddings.update(embs)

    print(f"\nTotal: {len(all_embeddings)} unique complex embeddings")

    # Save as npz
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **all_embeddings)
    print(f"Saved to {output_path}")

    # Also save a JSON index for quick lookup
    index_path = output_path.with_suffix(".json")
    index = {pid: emb.shape[0] for pid, emb in all_embeddings.items()}
    with open(index_path, "w") as f:
        json.dump({"n_complexes": len(index), "embedding_dim": list(index.values())[0] if index else 0}, f, indent=2)
    print(f"Index saved to {index_path}")


if __name__ == "__main__":
    main()
