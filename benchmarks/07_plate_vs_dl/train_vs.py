"""
Training script for PLATE-VS virtual screening model.

Usage:
    python benchmarks/07_plate_vs_dl/train_vs.py \
        --config benchmarks/07_plate_vs_dl/configs/vs_default.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Add paths for imports
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks" / "06_binding_affinity_model"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "GEMS"))

from data.plate_vs_dataset import PlateVSDataset
from data.collate import custom_collate  # reuse from 06_binding_affinity_model
from model.vs_model import VirtualScreeningModel


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_paths(config: dict, config_dir: Path) -> dict:
    for key in ["registry_path", "protein_emb_path"]:
        if key in config["data"]:
            config["data"][key] = str((config_dir / config["data"][key]).resolve())
    return config


def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, criterion, device, grad_clip=0.5):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)
        loss = criterion(logits.squeeze(-1), batch.y.squeeze(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        bs = batch.y.shape[0]
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_scores, all_labels, all_uids = [], [], []
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits.squeeze(-1), batch.y.squeeze(-1))

        total_loss += loss.item() * batch.y.shape[0]
        n_samples += batch.y.shape[0]

        scores = torch.sigmoid(logits).cpu().numpy().flatten()
        labels = batch.y.cpu().numpy().flatten()
        all_scores.extend(scores)
        all_labels.extend(labels)
        if hasattr(batch, "uniprot_ids"):
            all_uids.extend(batch.uniprot_ids)
        elif hasattr(batch, "uniprot_id"):
            all_uids.extend(batch.uniprot_id)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Global metrics
    metrics = {"loss": total_loss / n_samples}
    try:
        metrics["roc_auc"] = float(roc_auc_score(all_labels, all_scores))
    except ValueError:
        metrics["roc_auc"] = 0.0
    try:
        metrics["avg_precision"] = float(average_precision_score(all_labels, all_scores))
    except ValueError:
        metrics["avg_precision"] = 0.0

    # Per-target ROC-AUC
    if all_uids:
        target_scores = defaultdict(lambda: {"scores": [], "labels": []})
        for score, label, uid in zip(all_scores, all_labels, all_uids):
            target_scores[uid]["scores"].append(score)
            target_scores[uid]["labels"].append(label)

        per_target_auc = []
        for uid, data in target_scores.items():
            labels = np.array(data["labels"])
            if len(np.unique(labels)) < 2:
                continue
            try:
                auc = roc_auc_score(labels, data["scores"])
                per_target_auc.append(auc)
            except ValueError:
                continue

        if per_target_auc:
            metrics["per_target_roc_auc_mean"] = float(np.mean(per_target_auc))
            metrics["per_target_roc_auc_std"] = float(np.std(per_target_auc))
            metrics["n_targets_evaluated"] = len(per_target_auc)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train PLATE-VS virtual screening model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config_dir = Path(args.config).resolve().parent
    config = load_config(args.config)
    config = resolve_paths(config, config_dir)

    if args.no_wandb:
        config["wandb"]["enabled"] = False

    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
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
            name=f"vs_{config['data']['similarity_threshold']}_{config['model']['ligand_backend']}",
            config=config,
            tags=config["wandb"].get("tags", []),
        )

    # Build datasets
    dc = config["data"]
    print("\nLoading train set...")
    train_dataset = PlateVSDataset(
        registry_path=dc["registry_path"],
        protein_emb_path=dc["protein_emb_path"],
        split="train",
        similarity_threshold=dc["similarity_threshold"],
        include_decoys=dc["include_decoys"],
        max_decoys_per_target=dc.get("max_decoys_per_target"),
        max_ligand_atoms=dc["max_ligand_atoms"],
        max_pocket_res=dc["max_pocket_residues"],
    )

    print("Loading test set...")
    test_dataset = PlateVSDataset(
        registry_path=dc["registry_path"],
        protein_emb_path=dc["protein_emb_path"],
        split="test",
        similarity_threshold=dc["similarity_threshold"],
        include_decoys=dc["include_decoys"],
        max_decoys_per_target=dc.get("max_decoys_per_target"),
        max_ligand_atoms=dc["max_ligand_atoms"],
        max_pocket_res=dc["max_pocket_residues"],
    )

    bs = dc["batch_size"]
    nw = dc["num_workers"]
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=custom_collate)

    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Model
    mc = config["model"]
    model = VirtualScreeningModel(
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

    # Loss + optimizer
    lc = config["loss"]
    pos_weight = torch.tensor([lc.get("pos_weight", 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tc = config["training"]
    optimizer = AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total_steps = tc["max_epochs"] * len(train_loader)
    scheduler = get_scheduler(optimizer, tc["warmup_steps"], total_steps)

    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    ckpt_dir = Path(config["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for up to {tc['max_epochs']} epochs...\n")

    for epoch in range(1, tc["max_epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, tc["grad_clip"])
        test_metrics = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        auc = test_metrics.get("roc_auc", 0)
        pt_auc = test_metrics.get("per_target_roc_auc_mean", 0)
        ap = test_metrics.get("avg_precision", 0)

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"test_AUC={auc:.4f} | "
            f"pt_AUC={pt_auc:.4f} | "
            f"AP={ap:.4f} | "
            f"lr={lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        if wandb_run:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "test/roc_auc": auc,
                "test/per_target_auc": pt_auc,
                "test/avg_precision": ap,
                "test/loss": test_metrics["loss"],
                "lr": lr,
            })

        # Checkpoint on best per-target AUC
        if pt_auc > best_val_auc:
            best_val_auc = pt_auc
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_vs_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= tc["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final evaluation with best model
    print("\nEvaluating best model...")
    model.load_state_dict(torch.load(ckpt_dir / "best_vs_model.pt", weights_only=True))
    final_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*50}")
    print(f"  PLATE-VS Test Results ({dc['similarity_threshold']})")
    print(f"{'='*50}")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    # Save results
    summary = {
        "task_type": "classification",
        "model_type": "dual_encoder_vs",
        "dataset": "plate_vs",
        "similarity_threshold": dc["similarity_threshold"],
        "test_metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in final_metrics.items()},
        "n_train": len(train_dataset),
        "n_test": len(test_dataset),
        "n_params": n_params,
        "config": config,
    }

    out_path = results_dir / f"vs_{dc['similarity_threshold']}_training_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if wandb_run:
        import wandb
        wandb.log({k: v for k, v in final_metrics.items() if isinstance(v, float)})
        wandb.finish()


if __name__ == "__main__":
    os.chdir(SCRIPT_DIR)
    main()
