#!/usr/bin/env python3
"""
PLATE-VS Bulk Downloader
========================

A script to download the complete PLATE-VS similarity matrix dataset.
It iterates through all combinations of similarity thresholds and query coverage levels.

Usage:
    python download_all_data.py

"""

import time
from pathlib import Path
from platevs_client import PlateVSClient

def main():
    # Configuration
    OUTPUT_DIR = "./platevs_full_dataset"
    THRESHOLDS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    QCOV_LEVELS = [50, 70, 95, 100]
    
    print("=" * 60)
    print("PLATE-VS Bulk Downloader")
    print("=" * 60)
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Qcov Levels: {QCOV_LEVELS}")
    
    # Initialize client
    client = PlateVSClient(output_dir=OUTPUT_DIR)
    
    # Check connectivity first
    print("\nChecking service status...")
    status = client.check_service_status()
    if not status.get('main'):
        print("Error: Main website is not reachable. Aborting.")
        return
    
    start_time = time.time()
    
    # 1. Download CSVs (Matrix Data)
    print("\n" + "=" * 40)
    print("Phase 1: Downloading Similarity Matrix CSVs")
    print("=" * 40)
    
    csv_count = 0
    for qcov in QCOV_LEVELS:
        print(f"\nProcessing Query Coverage Level: {qcov}%")
        for threshold in THRESHOLDS:
            print(f"  Downloading CSV for Threshold {threshold}...", end=" ", flush=True)
            path = client.download_similarity_matrix_csv(
                similarity_threshold=threshold,
                qcov_level=qcov
            )
            
            if path:
                print("✓ Done")
                csv_count += 1
            else:
                print("✗ Failed")
            
            time.sleep(0.5) # Rate limiting
            
    # 2. Download SDFs (Structure Files)
    print("\n" + "=" * 40)
    print("Phase 2: Downloading SDF Archives")
    print("=" * 40)
    
    sdf_count = 0
    for threshold in THRESHOLDS:
        print(f"  Downloading SDF Archive for Threshold {threshold}...", end=" ", flush=True)
        path = client.download_similarity_sdf(similarity_threshold=threshold)
        
        if path:
            print("✓ Done")
            sdf_count += 1
        else:
            print("✗ Failed")
        
        time.sleep(1.0) # Rate limiting for larger files

    # 3. Download CIFs (Raw Structures)
    print("\n" + "=" * 40)
    print("Phase 3: Downloading CIF Archive")
    print("=" * 40)
    
    cif_path = client.download_similarity_cif()
    if cif_path:
        print("✓ Done")
    else:
        print("✗ Failed")
        
    # Summary
    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print("Download Complete")
    print("=" * 60)
    print(f"Time elapsed: {duration:.2f} seconds")
    print(f"CSVs downloaded: {csv_count}/{len(THRESHOLDS) * len(QCOV_LEVELS)}")
    print(f"SDFs downloaded: {sdf_count}/{len(THRESHOLDS)}")
    if cif_path:
        print("CIFs downloaded: 1/1")
    else:
        print("CIFs downloaded: 0/1")
    print(f"Data saved to: {Path(OUTPUT_DIR).absolute()}")

if __name__ == "__main__":
    main()
