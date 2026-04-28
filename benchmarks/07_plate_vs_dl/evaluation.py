"""Evaluation helpers for PLATE-VS virtual screening.

Kept separate from train_vs.py so tests can import without loading the full
training stack (torchmd-net, rdkit, W&B, etc.).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


# Default percentage cutoffs to match existing deepcoy training summaries
DEFAULT_EF_PERCENTAGES = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0)
DEFAULT_BEDROC_ALPHAS = (20.0, 80.0, 160.0)


def enrichment_factor(
    labels: Sequence[float],
    scores: Sequence[float],
    percentages: Iterable[float] = DEFAULT_EF_PERCENTAGES,
) -> dict[str, float]:
    """Enrichment factor at given top-k percentages.

    EF@k% = (n_actives_in_top_k / n_top_k) / (n_actives_total / n_total)
    Returns dict keyed `ef_<pct>pct` (matching existing summary schema).
    Returns empty dict if labels are degenerate (single class or empty).
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n = len(labels)
    n_active = int(labels.sum())
    if n == 0 or n_active == 0 or n_active == n:
        return {}

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    base_rate = n_active / n

    out: dict[str, float] = {}
    for pct in percentages:
        k = max(1, int(round(n * pct / 100.0)))
        hits = int(sorted_labels[:k].sum())
        out[_ef_key(pct)] = round((hits / k) / base_rate, 4)
    return out


def _ef_key(pct: float) -> str:
    if pct == int(pct):
        return f"ef_{int(pct)}pct"
    return f"ef_{pct}pct"


def bedroc(
    labels: Sequence[float],
    scores: Sequence[float],
    alphas: Iterable[float] = DEFAULT_BEDROC_ALPHAS,
) -> dict[str, float]:
    """BEDROC at given alphas (Truchon & Bayly, 2007).

    Higher alpha = more weight on top-ranked. alpha=20 is a common default.
    Returns dict keyed `bedroc_a<alpha_int>`.
    Returns empty dict if labels are degenerate.
    """
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(labels)
    n_active = int(labels.sum())
    if n == 0 or n_active == 0 or n_active == n:
        return {}

    order = np.argsort(-scores)
    sorted_labels = labels[order]

    out: dict[str, float] = {}
    for alpha in alphas:
        out[f"bedroc_a{int(alpha)}"] = round(_bedroc_truchon(sorted_labels, alpha), 4)
    return out


def _bedroc_truchon(sorted_labels: np.ndarray, alpha: float) -> float:
    """BEDROC normalized to [0, 1] (Truchon & Bayly, 2007).

    BEDROC = (RIE - RIE_min) / (RIE_max - RIE_min)
    Matches rdkit.ML.Scoring.Scoring.CalcBEDROC.
    """
    n = len(sorted_labels)
    n_active = int(sorted_labels.sum())
    if n_active == 0 or n_active == n:
        return 0.0
    ratio = n_active / n
    ranks = np.where(sorted_labels > 0)[0] + 1  # 1-based
    sum_exp = float(np.exp(-alpha * ranks / n).sum())
    denom = (1.0 / n) * ((1.0 - np.exp(-alpha)) / (np.exp(alpha / n) - 1.0))
    rie = sum_exp / (n_active * denom)
    rie_max = (1.0 - np.exp(-alpha * ratio)) / (ratio * (1.0 - np.exp(-alpha)))
    rie_min = (1.0 - np.exp(alpha * ratio)) / (ratio * (1.0 - np.exp(alpha)))
    if rie_max == rie_min:
        return 1.0
    return float((rie - rie_min) / (rie_max - rie_min))


def per_target_ef(
    uids: Sequence,
    labels: Sequence[float],
    scores: Sequence[float],
    percentages: Iterable[float] = (1.0, 5.0),
) -> dict[str, float]:
    """Mean per-target EF at given percentages (skips targets with <2 classes)."""
    uids = np.asarray(uids)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    pct_to_vals: dict[str, list[float]] = {_ef_key(p): [] for p in percentages}
    for uid in np.unique(uids):
        mask = uids == uid
        y = labels[mask]
        if len(np.unique(y)) < 2:
            continue
        ef = enrichment_factor(y, scores[mask], percentages)
        for k, v in ef.items():
            pct_to_vals[k].append(v)
    return {
        f"per_target_{k}_mean": round(float(np.mean(v)), 4)
        for k, v in pct_to_vals.items()
        if v
    }


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

    metrics.update(enrichment_factor(all_labels, all_scores))
    metrics.update(bedroc(all_labels, all_scores))

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

        metrics.update(per_target_ef(all_uids, all_labels, all_scores))

    return metrics
