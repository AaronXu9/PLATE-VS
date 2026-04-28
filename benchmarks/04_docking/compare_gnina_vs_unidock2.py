"""Side-by-side comparison: GNINA vs UniDock2 on the same 15 targets.

Reads `gnina_training_summary.json` and `unidock2_training_summary.json` from
benchmarks/04_docking/results/ for per-target ROC-AUC + AP, then re-parses
both pipelines' raw docked SDFs to compute Enrichment Factor at top 1% and
top 5% with a consistent definition. Emits CSV + Markdown.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


_VINA_RE = re.compile(r"<vina_binding_free_energy>[^\n]*\n([-\d.eE+]+)")
_IS_ACTIVE_RE = re.compile(r"<is_active>[^\n]*\n(\d+)")


def _ef(scores: np.ndarray, labels: np.ndarray, fraction: float) -> float | None:
    """Enrichment factor at top `fraction` of scores. Higher score = better.

    EF = (actives in top f%) / (f% × total) ÷ (actives / total).
    Returns None if not enough samples or no actives.
    """
    n = len(scores)
    n_act = int(labels.sum())
    if n == 0 or n_act == 0:
        return None
    k = max(1, int(fraction * n))
    order = np.argsort(-scores)  # descending
    top_actives = int(labels[order][:k].sum())
    return float((top_actives / k) / (n_act / n))


def _parse_unidock2_sdf(sdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Best (most-negative) Vina energy per molecule + is_active. Returns
    (-vina, label) so higher = better."""
    text = sdf_path.read_text()
    by_name: dict[str, tuple[float, int]] = {}
    for blk in text.split("$$$$"):
        if not blk.strip():
            continue
        title = blk.lstrip("\n").splitlines()[0].strip()
        if not title:
            continue
        em = _VINA_RE.search(blk)
        am = _IS_ACTIVE_RE.search(blk)
        if em is None or am is None:
            continue
        e = float(em.group(1))
        ia = int(am.group(1))
        prev = by_name.get(title)
        if prev is None or e < prev[0]:
            by_name[title] = (e, ia)
    if not by_name:
        return np.array([]), np.array([])
    energies = np.array([v[0] for v in by_name.values()])
    labels = np.array([v[1] for v in by_name.values()])
    return -energies, labels


def _parse_gnina_sdf(sdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Best CNNscore per molecule + is_active. Returns (CNNscore, label).
    GNINA's own collector keeps the first (best) pose per molecule so we
    mirror that here."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    by_name: dict[str, tuple[float, int]] = {}
    for mol in suppl:
        if mol is None:
            continue
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        if not name or name in by_name:
            continue
        try:
            score = float(mol.GetProp("CNNscore"))
        except (KeyError, ValueError):
            try:
                score = -float(mol.GetProp("minimizedAffinity"))
            except (KeyError, ValueError):
                continue
        try:
            ia = int(mol.GetProp("is_active"))
        except (KeyError, ValueError):
            continue
        by_name[name] = (score, ia)
    if not by_name:
        return np.array([]), np.array([])
    return (
        np.array([v[0] for v in by_name.values()]),
        np.array([v[1] for v in by_name.values()]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gnina-summary",
        default="/home/aoxu/projects/VLS-Benchmark-Dataset/benchmarks/04_docking/results/gnina_training_summary.json",
    )
    ap.add_argument(
        "--unidock2-summary",
        default="/home/aoxu/projects/VLS-Benchmark-Dataset/benchmarks/04_docking/results/unidock2_training_summary.json",
    )
    ap.add_argument(
        "--out-csv",
        default="/home/aoxu/projects/VLS-Benchmark-Dataset/benchmarks/04_docking/results/comparison_gnina_vs_unidock2.csv",
    )
    ap.add_argument(
        "--out-md",
        default="/home/aoxu/projects/VLS-Benchmark-Dataset/benchmarks/04_docking/results/comparison_gnina_vs_unidock2.md",
    )
    ap.add_argument(
        "--gnina-docked-dir",
        default="/home/aoxu/projects/VLS-Benchmark-Dataset/benchmarks/04_docking/results/docking",
    )
    ap.add_argument(
        "--unidock2-docked-dir",
        default="/home/aoxu/projects/VLS-Benchmark-Dataset/benchmarks/04_docking/results/docking_unidock2",
    )
    args = ap.parse_args()

    g = json.loads(Path(args.gnina_summary).read_text())
    u = json.loads(Path(args.unidock2_summary).read_text())
    # gnina ships per-target under "per_target_metrics" with keys
    # avg_precision/f1_score; unidock2 collector uses per_target with ap/f1.
    g_per = g.get("per_target_metrics", g.get("per_target", {}))
    u_per = u.get("per_target", u)

    gnina_dir = Path(args.gnina_docked_dir)
    unidock2_dir = Path(args.unidock2_docked_dir)

    rows = []
    pooled: dict[str, list] = {"g_score": [], "g_y": [], "u_score": [], "u_y": []}
    for uniprot in sorted(set(g_per) | set(u_per)):
        gm = g_per.get(uniprot, {}) if isinstance(g_per.get(uniprot), dict) else {}
        um = u_per.get(uniprot, {}) if isinstance(u_per.get(uniprot), dict) else {}

        # Recompute EF directly from each pipeline's docked SDF.
        g_sdf = gnina_dir / f"{uniprot}_docked.sdf"
        u_sdf = unidock2_dir / f"{uniprot}_docked.sdf"
        g_ef1 = g_ef5 = None
        u_ef1 = u_ef5 = None
        if g_sdf.exists():
            g_score, g_y = _parse_gnina_sdf(g_sdf)
            if len(g_score):
                g_ef1 = _ef(g_score, g_y, 0.01)
                g_ef5 = _ef(g_score, g_y, 0.05)
                pooled["g_score"].extend(g_score.tolist())
                pooled["g_y"].extend(g_y.tolist())
        if u_sdf.exists():
            u_score, u_y = _parse_unidock2_sdf(u_sdf)
            if len(u_score):
                u_ef1 = _ef(u_score, u_y, 0.01)
                u_ef5 = _ef(u_score, u_y, 0.05)
                pooled["u_score"].extend(u_score.tolist())
                pooled["u_y"].extend(u_y.tolist())

        rows.append(
            {
                "uniprot": uniprot,
                "n_actives": gm.get("n_actives") or um.get("n_actives"),
                "n_decoys": gm.get("n_decoys") or um.get("n_decoys"),
                "gnina_roc_auc": gm.get("roc_auc"),
                "unidock2_roc_auc": um.get("roc_auc"),
                "gnina_ap": gm.get("avg_precision"),
                "unidock2_ap": um.get("ap"),
                "gnina_ef1pct": g_ef1,
                "unidock2_ef1pct": u_ef1,
                "gnina_ef5pct": g_ef5,
                "unidock2_ef5pct": u_ef5,
                "unidock2_status": um.get("status"),
                "unidock2_dropped": um.get("n_dropped", um.get("n_chunks_failed", 0)),
                "unidock2_elapsed_s": um.get("elapsed_s"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    g_pool_score = np.array(pooled["g_score"])
    g_pool_y = np.array(pooled["g_y"])
    u_pool_score = np.array(pooled["u_score"])
    u_pool_y = np.array(pooled["u_y"])
    aggregate = {
        "gnina": {
            "test_metrics": g.get("training_history", {}).get("test_metrics", {}),
            "global_pooled_metrics": g.get("global_pooled_metrics", {}),
            "pooled_ef1pct": _ef(g_pool_score, g_pool_y, 0.01) if len(g_pool_score) else None,
            "pooled_ef5pct": _ef(g_pool_score, g_pool_y, 0.05) if len(g_pool_score) else None,
        },
        "unidock2": {
            **u.get("aggregate", {}),
            "pooled_ef1pct": _ef(u_pool_score, u_pool_y, 0.01) if len(u_pool_score) else None,
            "pooled_ef5pct": _ef(u_pool_score, u_pool_y, 0.05) if len(u_pool_score) else None,
        },
    }

    cols = list(df.columns)

    def _fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body_rows = ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in df.to_dict("records")]
    table_md = "\n".join([header, sep] + body_rows)

    Path(args.out_md).write_text(
        "# GNINA vs UniDock2 — 15-target VS comparison\n\n"
        + table_md
        + "\n\n## Aggregate\n\n```json\n"
        + json.dumps(aggregate, indent=2)
        + "\n```\n"
    )
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
