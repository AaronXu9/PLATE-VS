# PLATE-VS Dual-Encoder Sweep on CARC

## Split definitions (post-2026-05-07 rename)

The split names were **clarified** after we discovered the original "hard"
config wasn't actually a hard generalization test — proteins overlapped
~91% between train and test. Current convention:

| Split | Filter | Test set | Generalization tested |
|-------|--------|----------|----------------------|
| **`hard`** (true 2D) | `split=='test' AND protein_partition=='test'` | 87 proteins, 20K actives | **Both** novel proteins AND novel ligands |
| **`soft`** | `protein_partition=='test'` | 97 proteins, 42K actives | Novel proteins (ligand similarity unconstrained) |
| **`ligand_novel`** (legacy) | `split=='test'` | 678 proteins, 127K actives | Novel ligands only — proteins overlap with train |

Both `hard` and `soft` use `registry_soft_split.csv` (which carries both
columns). `ligand_novel` configs use `registry_2d_split.csv` and represent
the *easy* baseline (kept for reference).

## Prerequisites on CARC

1. **Code synced**: `/project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset` is up to date with `main`
2. **Conformer cache**: `data/plate_vs_conformers/conformers_full.pkl` (1.84 GB) must exist on CARC
3. **Conda env**: `binding_affinity` with `torchmd-net`
4. **Registry files**: `registry_soft_split.csv` (training) and `registry_soft_split_regression.csv` (real-inactive eval) in `training_data_full/`

## Quick start

```bash
mkdir -p slurm/logs

# Headline jobs (true hard 2D split)
bash slurm/submit_plate_vs_sweep.sh hard_schnet
bash slurm/submit_plate_vs_sweep.sh hard_et

# Soft (protein-novel only)
bash slurm/submit_plate_vs_sweep.sh soft_schnet
bash slurm/submit_plate_vs_sweep.sh soft_et_v2

# Legacy / baseline (ligand-novel only)
bash slurm/submit_plate_vs_sweep.sh ligand_novel_schnet
```

## Available tags

| Tag | Split | Backend | Notes |
|-----|-------|---------|-------|
| `hard_schnet` | TRUE 2D | SchNet (4L) | Strictest, primary baseline |
| `hard_et` | TRUE 2D | TorchMD-ET (4L, v2 HP) | Strictest, ET version |
| `soft_schnet` | protein-novel | SchNet | Already trained — see `soft_schnet_*.json` |
| `soft_et` | protein-novel | TorchMD-ET (6L original) | NaN'd; use v2 |
| `soft_et_v2` | protein-novel | TorchMD-ET (4L, conservative HP) | Already trained |
| `soft_et_v3` | protein-novel | TorchMD-ET (6L, aggressive HP) | Untried |
| `ligand_novel_schnet` | legacy ligand-novel | SchNet | Already trained — see `hard_schnet_*.json` (renamed) |
| `ligand_novel_et_v2` | legacy ligand-novel | TorchMD-ET (4L) | Already trained |

## SLURM job spec

- Partition: `gpu`, account: `katritch_223`, 1 GPU, 8 CPUs, 64 GB RAM, 36h
- Trains via `python -u benchmarks/07_plate_vs_dl/train_vs.py --config <yaml>`
- Auto-evals on real ChEMBL inactives if training completes cleanly
- Logs: `slurm/logs/plate_vs_dl_pvs_<tag>_<jobid>.{out,err}`

## Outputs

After each job:
- Checkpoint: `benchmarks/07_plate_vs_dl/checkpoints_<tag>/best_vs_model.pt`
- Training summary: `benchmarks/07_plate_vs_dl/results/<tag>_training_summary.json`
- Real-inactives eval: `benchmarks/07_plate_vs_dl/results/<tag>_real_inactives_eval.json`
- W&B run under project `plate-vs-dl`

## Re-running just the eval (no retrain)

```bash
sbatch --job-name=eval_hard_et \
    --export=ALL,CKPT=benchmarks/07_plate_vs_dl/checkpoints_hard_et/best_vs_model.pt,\
CONFIG=benchmarks/07_plate_vs_dl/configs/vs_hard_et.yaml,RESULT_TAG=hard_et \
    slurm/run_eval_real_inactives.slurm
```

## Notes / gotchas

- Train+eval writes `vs_0p7_training_summary.json` (default name) into
  `benchmarks/07_plate_vs_dl/results/`. The SLURM wrapper copies it to
  `<tag>_training_summary.json` immediately, but two simultaneously-finishing
  jobs are a tiny race risk.
- TorchMD-NET on CARC is custom-patched. If `torchmd-net: NOT AVAILABLE`
  appears in the job log, fall back to SchNet variants.
- ET training can produce NaN logits on edge-case conformers; the dataset
  filter and `train_vs.py` NaN guard handle this. `eval_real_inactives.py`
  drops non-finite scores before sklearn metric calls.
- 36h SLURM limit: SchNet runs typically complete or hit early-stop within
  it (~2.5h/epoch on P100). ET takes longer (~2.5–3h/epoch); some jobs
  hit the limit mid-training.
