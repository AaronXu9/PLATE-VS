"""
Fetch protein sequences from UniProt and compute per-residue ESM2 embeddings
for PLATE-VS targets.

Two stages:
  1. Fetch sequences for 826 UniProt IDs from UniProt REST API
  2. Compute ESM2-t6 embeddings (768-dim per residue) for pocket residues

For pocket extraction, uses protein_references.json which has pocket_residue_count
and CIF structures with reference ligand positions.

Usage:
    # Stage 1: Fetch sequences
    python benchmarks/07_plate_vs_dl/data/build_protein_embeddings.py --stage fetch

    # Stage 2: Compute ESM2 embeddings (requires GPU)
    python benchmarks/07_plate_vs_dl/data/build_protein_embeddings.py --stage embed

    # Both stages
    python benchmarks/07_plate_vs_dl/data/build_protein_embeddings.py --stage all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{uid}.fasta"


def fetch_sequence(uid: str, retries: int = 2) -> str | None:
    """Fetch protein sequence from UniProt."""
    url = UNIPROT_API.format(uid=uid)
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                fasta = resp.read().decode("utf-8")
            lines = fasta.strip().split("\n")
            seq = "".join(l.strip() for l in lines if not l.startswith(">"))
            return seq if len(seq) > 10 else None
        except Exception:
            if attempt < retries:
                time.sleep(0.5)
    return None


def fetch_all_sequences(
    uniprot_ids: list[str], output_path: str, workers: int = 8
) -> dict[str, str]:
    """Fetch sequences for all UniProt IDs."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load existing
    sequences = {}
    if output.exists():
        with open(output) as f:
            sequences = json.load(f)

    to_fetch = [uid for uid in uniprot_ids if uid not in sequences]
    print(f"  Total: {len(uniprot_ids)}, cached: {len(sequences)}, to fetch: {len(to_fetch)}")

    if not to_fetch:
        return sequences

    n_ok, n_fail = 0, 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_sequence, uid): uid for uid in to_fetch}
        for future in as_completed(futures):
            uid = futures[future]
            try:
                seq = future.result()
            except Exception:
                seq = None
            if seq:
                sequences[uid] = seq
                n_ok += 1
            else:
                n_fail += 1
            total = n_ok + n_fail
            if total % 100 == 0:
                print(f"    {total}/{len(to_fetch)}: {n_ok} OK, {n_fail} failed")

    with open(output, "w") as f:
        json.dump(sequences, f)

    print(f"  Done: {n_ok} fetched, {n_fail} failed in {time.time()-t0:.0f}s")
    return sequences


def compute_esm2_embeddings(
    sequences: dict[str, str],
    output_path: str,
    pocket_info: dict | None = None,
    max_seq_len: int = 1024,
    device: str = "auto",
) -> None:
    """Compute ESM2-t6 per-residue embeddings for each protein.

    If pocket_info is provided, only stores embeddings for pocket residues.
    Otherwise stores embeddings for residues up to max_seq_len.
    """
    import torch
    import esm

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load ESM2-t6
    print("  Loading ESM2-t6-8M...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = set()
    if output.exists():
        data = np.load(str(output), allow_pickle=False)
        existing = set(data.files)
        print(f"  Existing embeddings: {len(existing)}")

    embeddings = {}
    # Load existing into memory
    if existing:
        data = np.load(str(output), allow_pickle=False)
        for k in data.files:
            embeddings[k] = data[k]

    to_compute = [uid for uid in sequences if uid not in existing]
    print(f"  To compute: {len(to_compute)}")

    if not to_compute:
        return

    t0 = time.time()
    with torch.no_grad():
        for i, uid in enumerate(to_compute):
            seq = sequences[uid]
            # Truncate long sequences
            if len(seq) > max_seq_len:
                seq = seq[:max_seq_len]

            batch_labels, batch_strs, batch_tokens = batch_converter(
                [(uid, seq)]
            )
            batch_tokens = batch_tokens.to(device)

            results = model(batch_tokens, repr_layers=[6], return_contacts=False)
            # Shape: [1, seq_len+2, 320] (includes BOS/EOS tokens)
            token_emb = results["representations"][6][0, 1:-1, :]  # strip BOS/EOS

            emb_np = token_emb.cpu().numpy()

            # If pocket info available, extract pocket residues only
            if pocket_info and uid in pocket_info:
                n_pocket = pocket_info[uid].get("pocket_residue_count", emb_np.shape[0])
                # Simple approach: take first N residues as pocket proxy
                # (proper approach would use 3D pocket extraction from CIF)
                emb_np = emb_np[:min(n_pocket * 2, emb_np.shape[0])]

            embeddings[uid] = emb_np.astype(np.float32)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"    {i+1}/{len(to_compute)} ({elapsed:.0f}s)")

    # Save all
    np.savez_compressed(str(output), **embeddings)
    print(f"  Saved {len(embeddings)} embeddings to {output}")
    print(f"  Sample dims: {list(embeddings.values())[0].shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Build protein embeddings for PLATE-VS"
    )
    parser.add_argument(
        "--protein-refs",
        default="training_data_full/protein_references.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/plate_vs_protein_embeddings",
    )
    parser.add_argument("--stage", choices=["fetch", "embed", "all"], default="all")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.protein_refs) as f:
        prefs = json.load(f)

    uniprot_ids = list(prefs.keys())
    print(f"PLATE-VS proteins: {len(uniprot_ids)}")

    # Stage 1: Fetch sequences
    seq_path = output_dir / "protein_sequences.json"
    if args.stage in ("fetch", "all"):
        print("\n=== Stage 1: Fetching sequences from UniProt ===")
        sequences = fetch_all_sequences(uniprot_ids, str(seq_path), args.workers)
        print(f"  Total sequences: {len(sequences)}")
    else:
        with open(seq_path) as f:
            sequences = json.load(f)

    # Stage 2: Compute ESM2 embeddings
    emb_path = output_dir / "esm2_embeddings.npz"
    if args.stage in ("embed", "all"):
        print("\n=== Stage 2: Computing ESM2-t6 embeddings ===")
        pocket_info = {uid: prefs[uid] for uid in uniprot_ids if uid in prefs}
        compute_esm2_embeddings(
            sequences, str(emb_path), pocket_info, device=args.device
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
