## Plan: ML Benchmarking Architecture & Implementation

### Proposed File Structure
We will populate the `benchmarks/` directory with a robust structure separating **Configs**, **Environments**, **Utils**, and the core pipeline stages.

```text
benchmarks/
├── configs/                          # Config files (e.g., classical_config.yaml, deep_purpose_config.yaml)
├── envs/                             # Environment specs (env_classical.yml vs env_deep_learning.yml)
├── utils/                            # Shared code
│   ├── data_loaders.py               # Dataset classes specifically for PLATE-VS data
│   ├── metrics.py                    # Standardized RMSE, PCC, CI metrics
│   └── atom_featurizers.py           # Custom featurizers if needed
├── 01_preprocessing/
│   ├── 01_prepare_structures.py      # PDB Cleaning & Protonation
│   ├── 02_build_graphs.py            # Graph construction (PyG/DeepChem)
│   ├── 03_tokenize_seqs.py           # Tokenization for Transformers
│   └── 04_create_splits.py           # Standardized Train/Val/Test splitting
├── 02_training/
│   ├── train_classical_oddt.py       # RF/Scikit-learn and ODDT baselines
│   ├── train_deeppurpose_wrapper.py  # Bridge to DeepPurpose library
│   └── train_foundation_model.py     # Custom loops for Boltz-2 / FlashAffinity
└── 03_analysis/
    ├── generate_benchmark_report.py  # Aggregates results
    └── compare_to_pdbbind.py         # Comparative analysis table
```

### Steps

1.  **Environment & Utils Setup**: Create `envs/` for dependency management (as Boltz-2 and legacy ODDT often conflict) and `utils/data_loaders.py` to robustly load your parquet/csv data.
2.  **Preprocessing Pipeline**: Implement `01_prepare_structures.py` (protonation) and `02_build_graphs.py` (PyG/DeepChem conversion) to create cached, ML-ready datasets.
3.  **Classical Baselines**: Implement `train_classical_oddt.py` to establish a performance floor using Random Forest and ODDT scoring.
4.  **Deep Learning Integration**: Build `train_deeppurpose_wrapper.py` to pipe your preprocessed graphs/sequences into DeepPurpose's models (DeepDTA, GraphDTA).
5.  **Foundation Models**: Set up `train_foundation_model.py` specifically for handling the heavier Boltz-2/FlashAffinity architectures (likely requiring separate envs).
6.  **Analysis**: Script the final comparison table in `03_analysis/` to match your PDBbind comparison requirement.

### Further Considerations
1.  **Boltz-2 Resources**: Do you have access to the A100/H100 GPUs mentioned for Boltz-2 fine-tuning, or should we strictly plan for the LoRA/Inference-only approach?
2.  **Environment Separation**: Strong recommendation to use separate conda environments for "Classical" (ODDT is often older) vs "Deep Learning" (PyTorch/Boltz) to avoid "dependency hell."
