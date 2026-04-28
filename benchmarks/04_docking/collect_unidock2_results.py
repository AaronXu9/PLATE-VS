"""Collect UniDock2 docking results, compute per-target + aggregate VS metrics.

Mirrors collect_results.py but reads `vina_binding_free_energy` (UniDock2 SDF
property) instead of `CNNscore` (gnina). Lower (more negative) Vina energy =
better binder, so we negate before treating as a discrimination score.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


_ENERGY_RE = re.compile(r"<vina_binding_free_energy>[^\n]*\n([-\d.eE+]+)")
_IS_ACTIVE_RE = re.compile(r"<is_active>[^\n]*\n(\d+)")


def parse_target_sdf(sdf_path: Path, index: dict) -> list[dict]:
    """One row per molecule, keeping the best (most negative) Vina energy.

    UniDock2 preserves the input SDF's `is_active` tag on every output pose,
    so we read it directly from the output rather than re-joining via index —
    that way we are insensitive to title-line collisions or chunk reordering.
    The `index` arg is kept as a fallback for ligands missing the tag.
    """
    blocks = sdf_path.read_text().split("$$$$")
    by_name: dict[str, tuple[float, int]] = {}
    for blk in blocks:
        if not blk.strip():
            continue
        first_line = blk.lstrip("\n").splitlines()[0].strip()
        if not first_line:
            continue
        em = _ENERGY_RE.search(blk)
        if em is None:
            continue
        e = float(em.group(1))
        am = _IS_ACTIVE_RE.search(blk)
        if am is not None:
            is_active = int(am.group(1))
        else:
            is_active = int(index.get(first_line, {}).get("is_active", 0))
        prev = by_name.get(first_line)
        if prev is None or e < prev[0]:
            by_name[first_line] = (e, is_active)
    return [
        {"name": name, "score": e, "is_active": ia} for name, (e, ia) in by_name.items()
    ]


def compute_target_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"n_actives": 0, "n_decoys": 0, "roc_auc": float("nan")}
    y = np.array([r["is_active"] for r in rows])
    s = -np.array([r["score"] for r in rows])  # higher = better-binder
    n_act = int(y.sum())
    n_dec = int((1 - y).sum())
    metrics: dict = {"n_actives": n_act, "n_decoys": n_dec}
    if n_act > 0 and n_dec > 0:
        metrics["roc_auc"] = float(roc_auc_score(y, s))
        metrics["ap"] = float(average_precision_score(y, s))
        # Youden-J threshold for the discrete metrics
        order = np.argsort(-s)
        ys = y[order]
        tpr = np.cumsum(ys) / max(n_act, 1)
        fpr = np.cumsum(1 - ys) / max(n_dec, 1)
        j = tpr - fpr
        thr_idx = int(np.argmax(j))
        thr = s[order][thr_idx]
        pred = (s >= thr).astype(int)
        metrics.update(
            {
                "f1": float(f1_score(y, pred, zero_division=0)),
                "accuracy": float(accuracy_score(y, pred)),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y, pred)) if len(set(pred)) > 1 else 0.0,
            }
        )
        # Enrichment factor at top 1%
        k = max(1, int(0.01 * len(rows)))
        ef1 = (ys[:k].sum() / k) / (n_act / len(rows))
        metrics["ef1pct"] = float(ef1)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    paths = {k: Path(v) for k, v in cfg["paths"].items()}

    targets = json.loads((paths["output_dir"] / "selected_targets.json").read_text())
    prep = json.loads((paths["output_dir"] / "unidock2_targets.json").read_text())
    runs = json.loads((paths["output_dir"] / "unidock2_results.json").read_text())

    per_target = {}
    pooled_y, pooled_s = [], []
    for tgt in targets:
        u = tgt["uniprot_id"]
        run_status = runs.get(u, {}).get("status")
        if run_status not in ("ok", "ok_partial"):
            per_target[u] = {"status": run_status or "missing"}
            continue
        sdf = paths["docking_dir"] / f"{u}_docked.sdf"
        index = json.loads((paths["unidock2_inputs_dir"] / u / "index.json").read_text())
        rows = parse_target_sdf(sdf, index)
        m = compute_target_metrics(rows)
        m["status"] = run_status
        m["elapsed_s"] = runs[u].get("elapsed_s", 0)
        m["n_chunks_failed"] = runs[u].get("n_chunks_failed", 0)
        per_target[u] = m
        pooled_y.extend([r["is_active"] for r in rows])
        pooled_s.extend([-r["score"] for r in rows])

    weights = [m.get("n_actives", 0) for m in per_target.values()]
    aucs = [m.get("roc_auc", 0.0) for m in per_target.values()]
    aps = [m.get("ap", 0.0) for m in per_target.values()]
    total_w = sum(weights) or 1
    weighted = {
        "weighted_roc_auc": float(sum(a * w for a, w in zip(aucs, weights)) / total_w),
        "weighted_ap": float(sum(a * w for a, w in zip(aps, weights)) / total_w),
        "n_targets": len(per_target),
        "total_actives": sum(weights),
    }
    if pooled_y:
        py = np.array(pooled_y)
        ps = np.array(pooled_s)
        if py.sum() > 0 and (1 - py).sum() > 0:
            weighted["pooled_roc_auc"] = float(roc_auc_score(py, ps))
            weighted["pooled_ap"] = float(average_precision_score(py, ps))

    summary = {"per_target": per_target, "aggregate": weighted, "config": cfg}
    out = paths["output_dir"] / "unidock2_training_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    print(
        f"Weighted ROC-AUC = {weighted['weighted_roc_auc']:.3f}, "
        f"AP = {weighted['weighted_ap']:.3f}"
    )


if __name__ == "__main__":
    main()
