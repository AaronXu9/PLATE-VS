# PLATE-VS API Client

A Python template script for accessing PLATE-VS web services at https://www.drugbench.org/

## Overview

This client provides programmatic access to the PLATE-VS database, which contains protein-ligand affinity data. The client supports:

1. **Search by UniProt ID** - Query affinity data for a specific protein
2. **Search by SMILES** - Query affinity data for a specific compound (exact or similarity search)
3. **Download Similarity Matrix Data** - Download CSV, SDF, and CIF files for given similarity thresholds

## Installation

### Option 1: Install via pip (Recommended)

You can install the client directly from GitHub:

```bash
pip install https://github.com/AaronXu9/plate-vs-client.git
```

### Option 2: Clone and Install

```bash
git clone https://github.com/AaronXu9/plate-vs-client.git
cd plate-vs-client
pip install -r requirements.txt
```

## Quick Start

```python
from platevs_client import PlateVSClient

# Initialize the client
client = PlateVSClient(output_dir="./downloads")

# Check service status
status = client.check_service_status()
print(status)

# Search by UniProt ID
df = client.get_protein_ligands("P00533")  # EGFR
print(df.head())

# Search by SMILES (Similarity)
df = client.search_by_smiles("CC(=O)Oc1ccccc1C(=O)O", exact_match=False)
print(df.head())

# Download similarity matrix data
csv_path = client.download_similarity_matrix_csv(0.9, qcov_level=100)
sdf_path = client.download_similarity_sdf(0.9)
cif_path = client.download_similarity_cif()
```

## API Reference

### PlateVSClient

#### Constructor

```python
PlateVSClient(timeout=30, output_dir="./platevs_data")
```

- `timeout`: Request timeout in seconds
- `output_dir`: Directory for saving downloaded files

#### Methods

| Method | Description |
|--------|-------------|
| `search_by_uniprot(uniprot_id, page, limit)` | Search affinity data by UniProt ID (JSON) |
| `get_protein_ligands(uniprot_id)` | Get ligands DataFrame for a protein (CSV download) |
| `search_by_smiles(smiles, exact_match)` | Search affinity data by SMILES |
| `download_affinity_data(query, query_type)` | Download affinity data to CSV |
| `download_similarity_matrix_csv(threshold, qcov_level)` | Download similarity CSV |
| `download_similarity_sdf(threshold)` | Download similarity SDF files (tar.gz) |
| `download_similarity_cif()` | Download raw CIF files (tar.gz) |
| `download_all_similarity_data(thresholds, qcov_level)` | Bulk download for multiple thresholds |
| `check_service_status()` | Check if services are accessible |

## Notes

- **Endpoints**: The client uses the `/api` endpoints of the PLATE-VS website.
- **Rate Limiting**: Be respectful of the server; the client includes 1-second delays between batch downloads.
- **SDF Downloads**: SDF files are downloaded as `.tar.gz` archives from S3 (via signed URLs).

## Files

- `platevs_client.py` - Main client implementation
- `platevs_example.ipynb` - Jupyter notebook with interactive examples
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## License

Part of the PLATE-VS project.
