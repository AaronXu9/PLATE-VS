#!/usr/bin/env python3
"""
Analyze the results of structure selection and generate detailed reports.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def generate_summary_stats(df, output_dir):
    """Generate summary statistics."""
    stats_file = output_dir / "summary_statistics.txt"
    
    with open(stats_file, 'w') as f:
        f.write("# Structure Selection Summary Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total UniProts: {len(df)}\n")
        valid = df[df['error'] == '']
        f.write(f"Valid structures: {len(valid)} ({len(valid)/len(df)*100:.1f}%)\n")
        f.write(f"Failed structures: {len(df) - len(valid)} ({(len(df)-len(valid))/len(df)*100:.1f}%)\n\n")
        
        if len(valid) > 0:
            # Method distribution
            f.write("## Experimental Methods\n")
            method_counts = valid['method'].value_counts()
            for method, count in method_counts.items():
                f.write(f"  {method}: {count} ({count/len(valid)*100:.1f}%)\n")
            f.write("\n")
            
            # Resolution statistics
            res_col = valid['resolution']
            res_valid = pd.to_numeric(res_col, errors='coerce').dropna()
            if len(res_valid) > 0:
                f.write("## Resolution Statistics (Å)\n")
                f.write(f"  Count: {len(res_valid)}\n")
                f.write(f"  Min: {res_valid.min():.2f}\n")
                f.write(f"  Max: {res_valid.max():.2f}\n")
                f.write(f"  Mean: {res_valid.mean():.2f}\n")
                f.write(f"  Median: {res_valid.median():.2f}\n")
                f.write(f"  Std Dev: {res_valid.std():.2f}\n")
                f.write("\n  Percentiles:\n")
                for p in [25, 50, 75, 90, 95]:
                    f.write(f"    {p}th: {res_valid.quantile(p/100):.2f}\n")
                f.write("\n")
            
            # Quality score statistics
            scores = valid['quality_score'].astype(float)
            f.write("## Quality Score Statistics\n")
            f.write(f"  Min: {scores.min():.2f}\n")
            f.write(f"  Max: {scores.max():.2f}\n")
            f.write(f"  Mean: {scores.mean():.2f}\n")
            f.write(f"  Median: {scores.median():.2f}\n")
            f.write(f"  Std Dev: {scores.std():.2f}\n\n")
            
            # Pocket statistics
            pocket_sizes = valid['pocket_residue_count'].astype(int)
            f.write("## Pocket Size Statistics\n")
            f.write(f"  Min: {pocket_sizes.min()}\n")
            f.write(f"  Max: {pocket_sizes.max()}\n")
            f.write(f"  Mean: {pocket_sizes.mean():.1f}\n")
            f.write(f"  Median: {pocket_sizes.median():.0f}\n\n")
            
            # Completeness statistics
            completeness = valid['pocket_completeness'].astype(float)
            f.write("## Pocket Completeness Statistics\n")
            f.write(f"  Min: {completeness.min():.3f}\n")
            f.write(f"  Max: {completeness.max():.3f}\n")
            f.write(f"  Mean: {completeness.mean():.3f}\n")
            f.write(f"  Structures with 100% completeness: {(completeness == 1.0).sum()} ({(completeness == 1.0).sum()/len(valid)*100:.1f}%)\n")
            f.write(f"  Structures with >95% completeness: {(completeness > 0.95).sum()} ({(completeness > 0.95).sum()/len(valid)*100:.1f}%)\n\n")
        
        # Error analysis if any
        if len(df) - len(valid) > 0:
            f.write("## Error Analysis\n")
            errors = df[df['error'] != '']['error'].value_counts()
            for err, count in errors.head(10).items():
                f.write(f"  {err[:60]}: {count}\n")
    
    print(f"✓ Wrote summary statistics: {stats_file}")
    return stats_file


def generate_visualizations(df, output_dir):
    """Generate visualization plots."""
    valid = df[df['error'] == ''].copy()
    
    if len(valid) == 0:
        print("No valid structures to visualize")
        return []
    
    # Convert numeric columns
    valid['resolution_num'] = pd.to_numeric(valid['resolution'], errors='coerce')
    valid['quality_score_num'] = valid['quality_score'].astype(float)
    valid['pocket_size'] = valid['pocket_residue_count'].astype(int)
    valid['completeness_num'] = valid['pocket_completeness'].astype(float)
    
    sns.set_style("whitegrid")
    created_files = []
    
    # 1. Quality score distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid['quality_score_num'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Quality Score', fontsize=12)
    ax.set_ylabel('Number of Structures', fontsize=12)
    ax.set_title('Distribution of Quality Scores', fontsize=14, fontweight='bold')
    ax.axvline(valid['quality_score_num'].median(), color='red', linestyle='--', 
               label=f'Median: {valid["quality_score_num"].median():.1f}')
    ax.legend()
    plt.tight_layout()
    fig_path = output_dir / "quality_score_distribution.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    created_files.append(fig_path)
    
    # 2. Resolution distribution
    res_valid = valid.dropna(subset=['resolution_num'])
    if len(res_valid) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(res_valid['resolution_num'], bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Resolution (Å)', fontsize=12)
        ax.set_ylabel('Number of Structures', fontsize=12)
        ax.set_title('Distribution of Resolution', fontsize=14, fontweight='bold')
        ax.axvline(res_valid['resolution_num'].median(), color='red', linestyle='--',
                   label=f'Median: {res_valid["resolution_num"].median():.2f} Å')
        ax.legend()
        plt.tight_layout()
        fig_path = output_dir / "resolution_distribution.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        created_files.append(fig_path)
    
    # 3. Method distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    method_counts = valid['method'].value_counts().head(10)
    method_counts.plot(kind='barh', ax=ax)
    ax.set_xlabel('Number of Structures', fontsize=12)
    ax.set_ylabel('Experimental Method', fontsize=12)
    ax.set_title('Top 10 Experimental Methods', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = output_dir / "method_distribution.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    created_files.append(fig_path)
    
    # 4. Resolution vs Quality Score scatter
    if len(res_valid) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(res_valid['resolution_num'], res_valid['quality_score_num'],
                           c=res_valid['completeness_num'], cmap='RdYlGn',
                           alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Resolution (Å)', fontsize=12)
        ax.set_ylabel('Quality Score', fontsize=12)
        ax.set_title('Resolution vs Quality Score\n(colored by pocket completeness)',
                    fontsize=14, fontweight='bold')
        ax.invert_xaxis()  # Better resolution = lower number
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Pocket Completeness', fontsize=11)
        plt.tight_layout()
        fig_path = output_dir / "resolution_vs_quality.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        created_files.append(fig_path)
    
    # 5. Pocket size distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid['pocket_size'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Pocket Size (number of residues)', fontsize=12)
    ax.set_ylabel('Number of Structures', fontsize=12)
    ax.set_title('Distribution of Pocket Sizes', fontsize=14, fontweight='bold')
    ax.axvline(valid['pocket_size'].median(), color='red', linestyle='--',
               label=f'Median: {valid["pocket_size"].median():.0f}')
    ax.legend()
    plt.tight_layout()
    fig_path = output_dir / "pocket_size_distribution.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    created_files.append(fig_path)
    
    # 6. Completeness distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid['completeness_num'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Pocket Completeness', fontsize=12)
    ax.set_ylabel('Number of Structures', fontsize=12)
    ax.set_title('Distribution of Pocket Completeness', fontsize=14, fontweight='bold')
    ax.axvline(valid['completeness_num'].median(), color='red', linestyle='--',
               label=f'Median: {valid["completeness_num"].median():.3f}')
    ax.legend()
    plt.tight_layout()
    fig_path = output_dir / "completeness_distribution.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    created_files.append(fig_path)
    
    print(f"✓ Created {len(created_files)} visualization plots")
    return created_files


def generate_top_structures_report(df, output_dir, n=50):
    """Generate a report of top N structures."""
    valid = df[df['error'] == ''].copy()
    
    if len(valid) == 0:
        return
    
    valid['quality_score_num'] = valid['quality_score'].astype(float)
    top_structures = valid.nlargest(n, 'quality_score_num')
    
    report_file = output_dir / f"top_{n}_structures.txt"
    with open(report_file, 'w') as f:
        f.write(f"# Top {n} Structures by Quality Score\n")
        f.write("=" * 80 + "\n\n")
        
        for i, (idx, row) in enumerate(top_structures.iterrows(), 1):
            f.write(f"{i}. {row['uniprot_id']} - {row['pdb_id']}\n")
            f.write(f"   Quality Score: {row['quality_score']}\n")
            f.write(f"   Method: {row['method']}\n")
            f.write(f"   Resolution: {row['resolution']} Å\n")
            f.write(f"   Ligand: {row['chosen_ligand']}\n")
            f.write(f"   Pocket: {row['pocket_residue_count']} residues, "
                   f"{float(row['pocket_completeness'])*100:.1f}% complete\n")
            f.write("\n")
    
    print(f"✓ Wrote top structures report: {report_file}")
    return report_file


def main():
    ap = argparse.ArgumentParser(
        description="Analyze structure selection results"
    )
    ap.add_argument(
        "--input_csv",
        required=True,
        help="CSV file with selected structures"
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Directory for analysis outputs"
    )
    ap.add_argument(
        "--top_n",
        type=int,
        default=50,
        help="Number of top structures to report (default: 50)"
    )
    args = ap.parse_args()
    
    # Load data
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} structures from {args.input_csv}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate reports
    print("\nGenerating analysis reports...")
    generate_summary_stats(df, output_dir)
    generate_visualizations(df, output_dir)
    generate_top_structures_report(df, output_dir, n=args.top_n)
    
    print(f"\n✓ All analysis outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
