# Benchmarking Workflow

## 5. Implementation Guide
This directory contains the implementation of the benchmarking workflow for the PLATE-VS dataset.

### 5.1 Data Preprocessing & Featurization
Located in `01_preprocessing/`

**Goal:** Standardize data representation before training.

- **Structure Preparation:** 
  - Ensure all PDB files are protonated correctly.
  - Consistent ligand atom types.
- **Graph Construction:** (For GNNs like GraphDTA, GEMS)
  - Convert PDB/SDF files into graph objects (PyTorch Geometric, DeepChem).
  - *Nodes:* Atom features (Symbol, Degree, Charge, Aromaticity).
  - *Edges:* Bond type (Single, Double) + Distance-based edges.
- **Sequence Tokenization:** (For DeepDTA/Transformers)
  - Tokenize protein sequences (k-mers or BPE).
  - Tokenize ligand SMILES.

### 5.2 Model Training Recommendations
Located in `02_training/`

- **Classical Models:** 
  - Scikit-learn or ODDT. 
  - Random Forest (n_estimators 100-1000).
- **Deep Learning:** 
  - DeepPurpose library for standard architectures (DeepDTA, CNN, MPNN).
- **Foundation Models:**
  - **Boltz-2:** NVIDIA NIM or open-source repo. 
    - *Note:* Fine-tuning structural trunk needs A100/H100. Use LoRA or freeze structure module if constrained.
  - **FlashAffinity:** Use pre-computed ESM-3 embeddings (check disk space).

### 5.3 Comparative Analysis Table
Located in `03_analysis/`

- Present results in a table.
- Compare dataset against a standard (e.g., PDBbind v.2020) on the same external test set.
