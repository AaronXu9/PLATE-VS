"""
Fetch SMILES for PDBbind CleanSplit complexes from the RCSB PDB API.

Alternative to extract_pdbbind_smiles.py when raw PDBbind SDF files
are not available. Uses the RCSB Chemical Component Dictionary API
to look up canonical SMILES by HET code.

Output: CSV with columns (pdb_id, smiles, pK, split)

Usage:
    conda run -n rdkit_env python benchmarks/05_pdbbind_comparison/fetch_pdbbind_smiles_rcsb.py

    # Resume after interruption (skips already-fetched HET codes)
    conda run -n rdkit_env python benchmarks/05_pdbbind_comparison/fetch_pdbbind_smiles_rcsb.py --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


RCSB_CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"


def fetch_smiles_for_het(comp_id: str, retries: int = 2) -> str | None:
    """Fetch canonical SMILES for a PDB chemical component."""
    url = RCSB_CHEMCOMP_URL.format(comp_id=comp_id)
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            # Look for canonical SMILES (prefer OpenEye, then CACTVS)
            descriptors = data.get("pdbx_chem_comp_descriptor", [])
            smiles_candidates = []
            for d in descriptors:
                dtype = d.get("type", "")
                if "SMILES_CANONICAL" in dtype:
                    smiles_candidates.append((d.get("program", ""), d["descriptor"]))
                elif dtype == "SMILES":
                    smiles_candidates.append((d.get("program", ""), d["descriptor"]))

            if not smiles_candidates:
                return None

            # Prefer canonical from OpenEye or CACTVS
            for prog, smi in smiles_candidates:
                if "OpenEye" in prog and "CANONICAL" in str(smiles_candidates):
                    return smi
            return smiles_candidates[0][1]

        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < retries:
                time.sleep(1)
        except Exception:
            if attempt < retries:
                time.sleep(1)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Fetch SMILES from RCSB for PDBbind CleanSplit"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/pdbbind_cleansplit/labels/PDBbind_data_dict.json",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="data/pdbbind_cleansplit/labels/PDBbind_data_split_cleansplit.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/pdbbind_cleansplit/smiles/pdbbind_smiles.csv",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="data/pdbbind_cleansplit/smiles/het_smiles_cache.json",
        help="Cache file for HET code → SMILES mapping",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel API requests",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from cached HET codes",
    )
    args = parser.parse_args()

    with open(args.labels) as f:
        data_dict = json.load(f)
    with open(args.splits) as f:
        split_dict = json.load(f)

    # Build PDB ID → split lookup
    # Priority: casf2016 > casf2013 > casf2016_indep > casf2013_indep > train
    # (some IDs appear in multiple splits; assign the most specific test set)
    split_lookup = {}
    all_split_ids = set()
    split_priority = ["train", "casf2013_indep", "casf2016_indep", "casf2013", "casf2016"]
    for split_name in split_priority:
        for pid in split_dict.get(split_name, []):
            split_lookup[pid.lower()] = split_name
            all_split_ids.add(pid.lower())

    print(f"CleanSplit total PDB IDs: {len(all_split_ids)}")

    # Collect unique HET codes that need SMILES
    het_to_pdb_ids = {}  # het_code → [pdb_id, ...]
    skipped_multi = 0
    skipped_no_label = 0

    for pid in sorted(all_split_ids):
        entry = data_dict.get(pid) or data_dict.get(pid.upper())
        if not entry or entry.get("log_kd_ki") is None:
            skipped_no_label += 1
            continue

        lig_name = entry.get("ligand_name", "")
        if "&" in lig_name or not lig_name:
            skipped_multi += 1
            continue

        het_to_pdb_ids.setdefault(lig_name, []).append(pid)

    unique_hets = set(het_to_pdb_ids.keys())
    print(f"Unique HET codes to query: {len(unique_hets)}")
    print(f"Skipped (multi-ligand): {skipped_multi}")
    print(f"Skipped (no label): {skipped_no_label}")

    # Load cache
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    het_smiles_cache = {}
    if args.resume and cache_path.exists():
        with open(cache_path) as f:
            het_smiles_cache = json.load(f)
        print(f"Loaded {len(het_smiles_cache)} cached HET codes")

    # Determine which HET codes still need fetching
    to_fetch = [h for h in unique_hets if h not in het_smiles_cache]
    print(f"HET codes to fetch from RCSB: {len(to_fetch)}")

    if to_fetch:
        print(f"\nFetching SMILES from RCSB API ({args.workers} workers)...")
        n_fetched = 0
        n_failed = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(fetch_smiles_for_het, het): het
                for het in to_fetch
            }

            for future in as_completed(futures):
                het = futures[future]
                try:
                    smiles = future.result()
                except Exception:
                    smiles = None

                if smiles:
                    het_smiles_cache[het] = smiles
                    n_fetched += 1
                else:
                    het_smiles_cache[het] = None
                    n_failed += 1

                total = n_fetched + n_failed
                if total % 500 == 0:
                    elapsed = time.time() - t0
                    rate = total / elapsed if elapsed > 0 else 0
                    print(
                        f"  Progress: {total}/{len(to_fetch)} "
                        f"({n_fetched} OK, {n_failed} failed, "
                        f"{rate:.0f}/s)"
                    )

                # Save cache periodically
                if total % 2000 == 0:
                    with open(cache_path, "w") as f:
                        json.dump(het_smiles_cache, f)

        elapsed = time.time() - t0
        print(f"\n  Done: {n_fetched} fetched, {n_failed} failed in {elapsed:.0f}s")

        # Save final cache
        with open(cache_path, "w") as f:
            json.dump(het_smiles_cache, f)
        print(f"  Cache saved to {cache_path}")

    # Build output CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_no_smiles = 0

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["pdb_id", "smiles", "pK", "split"])

        for pid in sorted(all_split_ids):
            entry = data_dict.get(pid) or data_dict.get(pid.upper())
            if not entry:
                continue

            pk = entry.get("log_kd_ki")
            if pk is None:
                continue

            lig_name = entry.get("ligand_name", "")
            if "&" in lig_name or not lig_name:
                continue

            smiles = het_smiles_cache.get(lig_name)
            if not smiles:
                n_no_smiles += 1
                continue

            split_name = split_lookup.get(pid, "unknown")
            writer.writerow([pid, smiles, pk, split_name])
            n_written += 1

    print(f"\nOutput: {output_path}")
    print(f"  Written: {n_written} rows")
    print(f"  No SMILES found: {n_no_smiles}")

    # Summary by split
    split_counts = {}
    with open(output_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row["split"]
            split_counts[s] = split_counts.get(s, 0) + 1
    print(f"\n  Per-split counts:")
    for s, c in sorted(split_counts.items()):
        print(f"    {s}: {c}")


if __name__ == "__main__":
    main()
