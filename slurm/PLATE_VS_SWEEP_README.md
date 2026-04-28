# PLATE-VS Dual-Encoder Sweep on CARC

4-job sweep covering `{hard, soft} × {torchmd_et, schnet}` with
auto-evaluation on real ChEMBL inactives.

## Prerequisites on CARC

1. **Code synced**: `/project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset` is up to date with `main`
2. **Conformer cache**: `data/plate_vs_conformers/conformers_full.pkl` (1.84 GB) must exist on CARC.
   If it doesn't, rsync from local:
   ```bash
   rsync -avP data/plate_vs_conformers/conformers_full.pkl \
       carc:/project2/katritch_223/aoxu/projects/VLS-Benchmark-Dataset/data/plate_vs_conformers/
   ```
3. **Conda env**: `binding_affinity` with `torchmd-net` (already set up per `benchmarks/envs/env_binding_affinity.yml`)
4. **Registry files**: Both `registry_2d_split.csv` and `registry_soft_split_regression.csv` present in `training_data_full/`

## Quick start

From the CARC project root:

```bash
# Create logs directory
mkdir -p slurm/logs

# Submit all 4 jobs at once
bash slurm/submit_plate_vs_sweep.sh

# Or submit a single variant
bash slurm/submit_plate_vs_sweep.sh hard_et      # hard split, TorchMD-NET ET
bash slurm/submit_plate_vs_sweep.sh hard_schnet  # hard split, SchNet
bash slurm/submit_plate_vs_sweep.sh soft_et      # soft split, TorchMD-NET ET
bash slurm/submit_plate_vs_sweep.sh soft_schnet  # soft split, SchNet
```

## Job configuration

Each job:
- Partition: `gpu`, account: `katritch_223`, 1 GPU, 8 CPUs, 64 GB RAM, 12h
- Logs: `slurm/logs/plate_vs_dl_pvs_<tag>_<jobid>.{out,err}`
- Trains via `python benchmarks/07_plate_vs_dl/train_vs.py --config <yaml>`
- Auto-evals on real ChEMBL inactives (pChEMBL ≥ 6 vs < 5)

| Tag | Split | Backend | Config |
|---|---|---|---|
| `hard_et` | hard (`registry_2d_split.csv`) | TorchMD-NET ET (6 layers) | `vs_hard_et.yaml` |
| `hard_schnet` | hard | SchNet (4 layers) | `vs_hard_schnet.yaml` |
| `soft_et` | soft (`protein_partition`) | TorchMD-NET ET (6 layers) | `vs_soft_et.yaml` |
| `soft_schnet` | soft | SchNet (4 layers) | `vs_soft_schnet.yaml` |

## Monitor

```bash
squeue --me                            # see queued/running jobs
tail -f slurm/logs/plate_vs_dl_pvs_hard_et_<JOBID>.out
```

## Outputs

After each job:
- Checkpoint: `benchmarks/07_plate_vs_dl/checkpoints_<tag>/best_vs_model.pt`
- Training summary: `benchmarks/07_plate_vs_dl/results/<tag>_training_summary.json`
- Real-inactives eval: `benchmarks/07_plate_vs_dl/results/<tag>_real_inactives_eval.json`
- W&B run logged under project `plate-vs-dl` with the tag in run name

## Re-running just the eval (no retrain)

```bash
python benchmarks/07_plate_vs_dl/eval_real_inactives.py \
    --checkpoint benchmarks/07_plate_vs_dl/checkpoints_hard_et/best_vs_model.pt \
    --config benchmarks/07_plate_vs_dl/configs/vs_hard_et.yaml \
    --output benchmarks/07_plate_vs_dl/results/hard_et_real_inactives_eval.json
```

## Notes / gotchas

- Train+eval writes `vs_0p7_training_summary.json` (the default name) into
  `benchmarks/07_plate_vs_dl/results/`. The SLURM wrapper immediately copies
  it to `<tag>_training_summary.json` so concurrent jobs don't lose data.
  But if two jobs finish *exactly* simultaneously, there's a tiny race.
  Stagger submissions or check `<tag>_training_summary.json` first.
- TorchMD-NET on CARC is custom-patched (per `METHODS.md`). If you see
  import errors, try the SchNet variants — they're guaranteed to work.
- 12h SLURM time limit is generous; previous local runs early-stopped at
  epoch 13 in ~10h on a 4090. CARC GPUs (V100/A100) should be similar or faster.
