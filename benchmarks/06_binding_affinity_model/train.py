"""
Training script for the dual-encoder binding affinity model.

Usage:
    # Train fold 0
    python benchmarks/06_binding_affinity_model/train.py \
        --config benchmarks/06_binding_affinity_model/configs/default_config.yaml \
        --fold 0

    # Override batch size
    python benchmarks/06_binding_affinity_model/train.py \
        --config benchmarks/06_binding_affinity_model/configs/default_config.yaml \
        --fold 0 --batch-size 8

    # Disable W&B
    python benchmarks/06_binding_affinity_model/train.py \
        --config benchmarks/06_binding_affinity_model/configs/default_config.yaml \
        --fold 0 --no-wandb
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.utils.metrics import summarize_regression  # noqa: E402

from model.binding_affinity_model import BindingAffinityModel  # noqa: E402
from model.losses import CompositeLoss  # noqa: E402
from data.collate import custom_collate  # noqa: E402
from data.build_dataset import load_dataset  # noqa: E402


def load_config(config_path: str, cli_overrides: dict) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if cli_overrides.get("fold") is not None:
        config["cv"]["fold"] = cli_overrides["fold"]
    if cli_overrides.get("batch_size") is not None:
        config["data"]["batch_size"] = cli_overrides["batch_size"]
    if cli_overrides.get("no_wandb"):
        config["wandb"]["enabled"] = False

    return config


def resolve_paths(config: dict, config_dir: Path) -> dict:
    """Resolve relative paths in config."""
    for section in ["data"]:
        for key in list(config[section].keys()):
            val = config[section][key]
            if isinstance(val, str) and ("/" in val or val.startswith(".")):
                config[section][key] = str((config_dir / val).resolve())
    return config


def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Cosine annealing with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_huber = 0.0
    total_rank = 0.0
    total_grad_norm = 0.0
    n_samples = 0
    n_steps = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        loss, loss_dict = criterion(pred, batch.y.view(-1, 1))

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        bs = batch.y.shape[0]
        total_loss += loss_dict["total"] * bs
        total_huber += loss_dict["huber"] * bs
        total_rank += loss_dict["rank"] * bs
        total_grad_norm += gn.item()
        n_samples += bs
        n_steps += 1

    return {
        "loss": total_loss / n_samples,
        "huber": total_huber / n_samples,
        "rank": total_rank / n_samples,
        "grad_norm": total_grad_norm / max(n_steps, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, pk_max=16.0):
    model.eval()
    total_loss = 0.0
    all_pred, all_target, all_pdb_ids = [], [], []
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss, loss_dict = criterion(pred, batch.y.view(-1, 1))

        bs = batch.y.shape[0]
        total_loss += loss_dict["total"] * bs
        n_samples += bs

        all_pred.append(pred.cpu())
        all_target.append(batch.y.cpu())
        if hasattr(batch, "pdb_id"):
            all_pdb_ids.extend(batch.pdb_id)

    all_pred = torch.cat(all_pred).numpy().flatten()
    all_target = torch.cat(all_target).numpy().flatten()

    # Unscale to pK
    pred_pk = all_pred * pk_max
    true_pk = all_target * pk_max

    # Compute metrics
    metrics = {
        "loss": total_loss / n_samples,
        "pearson": float(pearsonr(pred_pk, true_pk)[0]),
        "spearman": float(spearmanr(pred_pk, true_pk).correlation),
        "rmse": float(np.sqrt(np.mean((pred_pk - true_pk) ** 2))),
        "r2": float(1 - np.sum((pred_pk - true_pk) ** 2) / np.sum((true_pk - true_pk.mean()) ** 2)),
    }

    # Per-complex predictions
    per_complex = {}
    for i, pid in enumerate(all_pdb_ids):
        per_complex[pid] = {
            "y_true": round(float(true_pk[i]), 4),
            "y_pred": round(float(pred_pk[i]), 4),
        }

    return metrics, per_complex


def main():
    parser = argparse.ArgumentParser(description="Train binding affinity model")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--fold", type=int, default=None, help="CV fold (0-4)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    config_dir = Path(args.config).resolve().parent
    config = load_config(args.config, vars(args))
    config = resolve_paths(config, config_dir)

    fold = config["cv"]["fold"]
    seed = config["training"]["seed"]

    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # W&B
    wandb_run = None
    if config["wandb"]["enabled"]:
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
        except ImportError:
            pass
        import wandb
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb_run = wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"].get("entity"),
            name=f"fold{fold}_{config['model']['proj_dim']}d_{config['model']['et_layers']}L",
            config=config,
            tags=config["wandb"].get("tags", []),
        )

    # Load datasets
    dataset_dir = config["data"]["dataset_dir"]
    print(f"\nLoading datasets from {dataset_dir}...")

    train_path = Path(dataset_dir) / "processed" / f"train_f{fold}_data.pt"
    val_path = Path(dataset_dir) / "processed" / f"val_f{fold}_data.pt"
    test_path = Path(dataset_dir) / "processed" / "casf2016_data.pt"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            print(f"ERROR: Dataset not found: {p}")
            print("Run build_dataset.py first.")
            sys.exit(1)

    train_dataset = load_dataset(str(train_path))
    val_dataset = load_dataset(str(val_path))
    test_dataset = load_dataset(str(test_path))

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    bs = config["data"]["batch_size"]
    nw = config["data"]["num_workers"]
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=custom_collate)

    # Build model
    mc = config["model"]
    model = BindingAffinityModel(
        esm_dim=mc["esm_dim"],
        proj_dim=mc["proj_dim"],
        et_layers=mc["et_layers"],
        et_heads=mc["et_heads"],
        et_rbf=mc["et_rbf"],
        et_cutoff=mc["et_cutoff"],
        cross_attn_layers=mc["cross_attn_layers"],
        cross_attn_heads=mc["cross_attn_heads"],
        dropout=mc["dropout"],
        ligand_backend=mc.get("ligand_backend", "auto"),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    # Loss, optimizer, scheduler
    lc = config["loss"]
    criterion = CompositeLoss(
        lambda_rank=lc["lambda_rank"],
        huber_delta=lc["huber_delta"],
        rank_margin=lc["rank_margin"],
        rank_sample_size=lc["rank_sample_size"],
    )

    tc = config["training"]
    optimizer = AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total_steps = tc["max_epochs"] * len(train_loader)
    scheduler = get_scheduler(optimizer, tc["warmup_steps"], total_steps)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_dir = Path(config["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for up to {tc['max_epochs']} epochs (patience={tc['patience']})...\n")

    for epoch in range(1, tc["max_epochs"] + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, tc["grad_clip"]
        )
        val_metrics, _ = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_R={val_metrics['pearson']:.4f} | "
            f"val_rho={val_metrics['spearman']:.4f} | "
            f"gnorm={train_metrics['grad_norm']:.2f} | "
            f"lr={lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # W&B logging
        if wandb_run:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/huber": train_metrics["huber"],
                "train/rank": train_metrics["rank"],
                "train/grad_norm": train_metrics["grad_norm"],
                "val/loss": val_metrics["loss"],
                "val/pearson": val_metrics["pearson"],
                "val/spearman": val_metrics["spearman"],
                "val/rmse": val_metrics["rmse"],
                "val/r2": val_metrics["r2"],
                "lr": lr,
            })

        # Early stopping + checkpointing
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / f"best_model_f{fold}.pt")
        else:
            patience_counter += 1
            if patience_counter >= tc["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Load best model and evaluate on CASF-2016
    print("\nEvaluating best model on CASF-2016...")
    model.load_state_dict(torch.load(ckpt_dir / f"best_model_f{fold}.pt", weights_only=True))
    test_metrics, test_predictions = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*50}")
    print(f"  CASF-2016 Results (fold {fold})")
    print(f"{'='*50}")
    for k, v in test_metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    # Save results JSON (compatible with existing report format)
    summary = {
        "task_type": "regression",
        "model_type": "dual_encoder_et_crossattn",
        "model_architecture": f"ET{mc['et_layers']}L_CrossAttn{mc['cross_attn_layers']}L_proj{mc['proj_dim']}",
        "feature_type": "3d_ligand_et_esm2_ankh_crossattn",
        "dataset": "pdbbind_cleansplit",
        "test_set": "casf2016",
        "similarity_threshold": "cleansplit",
        "use_precomputed_split": True,
        "training_history": {
            "train_metrics": None,
            "val_metrics": {k: round(v, 6) for k, v in val_metrics.items()},
            "test_metrics": {k: round(v, 6) for k, v in test_metrics.items()},
            "n_train_samples": len(train_dataset),
            "n_val_samples": len(val_dataset),
            "n_test_samples": len(test_dataset),
            "training_time": None,
            "n_params": n_params,
        },
        "cv_fold": fold,
        "per_complex_predictions": test_predictions,
        "config": config,
    }

    out_path = results_dir / f"dual_encoder_f{fold}_casf2016_training_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if wandb_run:
        import wandb
        wandb.log({
            "test/pearson": test_metrics["pearson"],
            "test/spearman": test_metrics["spearman"],
            "test/rmse": test_metrics["rmse"],
            "test/r2": test_metrics["r2"],
        })
        wandb.finish()


if __name__ == "__main__":
    # Resolve config path before chdir (user may pass relative path from project root)
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--config" and i + 1 < len(sys.argv):
                sys.argv[i + 1] = str(Path(sys.argv[i + 1]).resolve())
    # Change to script directory for relative imports
    os.chdir(SCRIPT_DIR)
    main()
