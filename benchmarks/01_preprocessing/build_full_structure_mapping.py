#!/usr/bin/env python3
"""
Build a complete mapping CSV for all UniProts with available CIF files.
This prepares the input for batch structure selection.
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser(
        description="Build structure mapping CSV for all available UniProts"
    )
    ap.add_argument(
        "--cif_base_dir",
        required=True,
        help="Base directory containing uniprot_* subdirectories"
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV file path"
    )
    ap.add_argument(
        "--stats_file",
        help="Optional: Write statistics to this file"
    )
    args = ap.parse_args()
    
    base_dir = Path(args.cif_base_dir)
    if not base_dir.exists():
        raise ValueError(f"Base directory not found: {base_dir}")
    
    # Find all UniProt directories
    uniprot_dirs = sorted(base_dir.glob("uniprot_*"))
    print(f"Found {len(uniprot_dirs)} UniProt directories")
    
    mapping_rows = []
    stats = {
        "total_uniprots": 0,
        "uniprots_with_cifs": 0,
        "total_structures": 0,
        "structures_per_uniprot": [],
    }
    
    for uniprot_dir in tqdm(uniprot_dirs, desc="Scanning directories"):
        uniprot_id = uniprot_dir.name.replace("uniprot_", "")
        stats["total_uniprots"] += 1
        
        # Look for cif_files_raw subdirectory
        cif_dir = uniprot_dir / "cif_files_raw"
        if not cif_dir.exists():
            continue
        
        # Find all CIF files
        cif_files = list(cif_dir.glob("*.cif"))
        if not cif_files:
            continue
        
        stats["uniprots_with_cifs"] += 1
        stats["structures_per_uniprot"].append(len(cif_files))
        stats["total_structures"] += len(cif_files)
        
        # Add to mapping
        for cif_file in cif_files:
            mapping_rows.append({
                "uniprot_id": uniprot_id,
                "pdb_id": cif_file.stem,
                "cif_path": str(cif_file),
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(mapping_rows)
    df.to_csv(args.out_csv, index=False)
    
    print(f"\n✓ Wrote mapping CSV: {args.out_csv}")
    print(f"  Total UniProts scanned: {stats['total_uniprots']}")
    print(f"  UniProts with CIF files: {stats['uniprots_with_cifs']}")
    print(f"  Total structures: {stats['total_structures']}")
    
    if stats['structures_per_uniprot']:
        import numpy as np
        counts = np.array(stats['structures_per_uniprot'])
        print(f"\n  Structures per UniProt:")
        print(f"    Min: {counts.min()}")
        print(f"    Max: {counts.max()}")
        print(f"    Mean: {counts.mean():.1f}")
        print(f"    Median: {np.median(counts):.0f}")
    
    # Write detailed statistics if requested
    if args.stats_file:
        with open(args.stats_file, 'w') as f:
            f.write("# Structure Mapping Statistics\n\n")
            f.write(f"Total UniProts scanned: {stats['total_uniprots']}\n")
            f.write(f"UniProts with CIF files: {stats['uniprots_with_cifs']}\n")
            f.write(f"Total structures mapped: {stats['total_structures']}\n")
            
            if stats['structures_per_uniprot']:
                counts = np.array(stats['structures_per_uniprot'])
                f.write(f"\nStructures per UniProt:\n")
                f.write(f"  Min: {counts.min()}\n")
                f.write(f"  Max: {counts.max()}\n")
                f.write(f"  Mean: {counts.mean():.2f}\n")
                f.write(f"  Median: {np.median(counts):.0f}\n")
                f.write(f"  Std Dev: {counts.std():.2f}\n")
                f.write(f"\nPercentiles:\n")
                for p in [25, 50, 75, 90, 95, 99]:
                    f.write(f"  {p}th: {np.percentile(counts, p):.0f}\n")
        
        print(f"\n✓ Wrote statistics: {args.stats_file}")


if __name__ == "__main__":
    main()
