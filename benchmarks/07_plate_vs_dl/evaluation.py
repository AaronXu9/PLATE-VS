"""Evaluation helpers for PLATE-VS virtual screening.

Kept separate from train_vs.py so tests can import without loading the full
training stack (torchmd-net, rdkit, W&B, etc.).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


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
        if hasattr(batch, "pdb_ids"):
            all_uids.extend(batch.pdb_ids)
        elif hasattr(batch, "pdb_id"):
            all_uids.extend(batch.pdb_id)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    metrics = {"loss": total_loss / n_samples}
    try:
        metrics["roc_auc"] = float(roc_auc_score(all_labels, all_scores))
    except ValueError:
        metrics["roc_auc"] = 0.0
    try:
        metrics["avg_precision"] = float(average_precision_score(all_labels, all_scores))
    except ValueError:
        metrics["avg_precision"] = 0.0

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
