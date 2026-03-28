#!/usr/bin/env python3
"""
Fix parquet files that can't be read with pandas but work with DuckDB.
Rewrites them using DuckDB to ensure pandas compatibility.
"""

import os
import sys
import duckdb
from pathlib import Path
from tqdm import tqdm
import shutil

def fix_parquet_file(parquet_path, backup=True):
    """
    Rewrite a parquet file using DuckDB to fix compatibility issues.
    
    Args:
        parquet_path: Path to the parquet file to fix
        backup: Whether to create a backup of the original file
    
    Returns:
        True if successful, False otherwise
    """
    parquet_path = Path(parquet_path)
    
    if not parquet_path.exists():
        print(f"ERROR: File not found: {parquet_path}")
        return False
    
    # Create temp file path
    temp_path = parquet_path.with_suffix('.parquet.tmp')
    backup_path = parquet_path.with_suffix('.parquet.backup')
    
    try:
        # Read and rewrite using DuckDB
        duckdb.sql(f"""
            COPY (SELECT * FROM read_parquet('{parquet_path}'))
            TO '{temp_path}'
            (FORMAT PARQUET);
        """)
        
        # Create backup if requested
        if backup and not backup_path.exists():
            shutil.copy2(parquet_path, backup_path)
        
        # Replace original with fixed version
        shutil.move(str(temp_path), str(parquet_path))
        
        return True
        
    except Exception as e:
        print(f"ERROR processing {parquet_path}: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        return False

def main():
    # Base directory containing all the parquet files
    base_dir = Path("/mnt/katritch_lab2/aoxu/VLS_benchmark/chembl_affinity")
    
    if not base_dir.exists():
        print(f"ERROR: Base directory not found: {base_dir}")
        sys.exit(1)
    
    # Find all *_chembl_activities_filtered.parquet files
    parquet_files = list(base_dir.glob("uniprot_*/*_chembl_activities_filtered.parquet"))
    
    print(f"Found {len(parquet_files)} parquet files to process")
    print(f"Base directory: {base_dir}")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        sys.exit(0)
    
    # Process each file
    success_count = 0
    failed_files = []
    
    for parquet_file in tqdm(parquet_files, desc="Processing files"):
        if fix_parquet_file(parquet_file, backup=True):
            success_count += 1
        else:
            failed_files.append(parquet_file)
    
    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files: {len(parquet_files)}")
    print(f"Successfully fixed: {success_count}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    
    print("\nNote: Original files have been backed up with .backup extension")

if __name__ == "__main__":
    main()
