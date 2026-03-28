"""
Analyze GNINA docking results: compute per-target EF and BEDROC.

For each target with a docked SDF:
  1. Parse all poses — best pose per ligand (max CNN_VS)
  2. Compute EF@[0.1%, 0.5%, 1%, 2%, 5%] and BEDROC(α=20,80,160)
     using CNN_VS score (higher = more active) and minimizedAffinity
     (negated so higher = better binder)
  3. Save per-target JSON + combined summary CSV

Usage (from project root):
  conda run -n rdkit_env python3 benchmarks/04_docking/analyze_docking.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.ML.Scoring import Scoring

DOCKING_DIR = Path(__file__).parent / "results/docking"
OUT_DIR     = Path(__file__).parent / "results/docking_metrics"

EF_FRACTIONS  = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
BEDROC_ALPHAS = [20.0, 80.0, 160.0]


def ef_key(frac):
    if frac >= 0.01:
        return f"ef_{int(frac*100)}pct"
    return f"ef_{frac*100:.1f}pct"


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Compute EF and BEDROC from continuous scores (higher = more active)."""
    n_active = int(labels.sum())
    n_total  = len(labels)
    if n_active == 0 or n_active == n_total:
        return {}

    ranked = sorted(zip(scores.tolist(), labels.astype(int).tolist()), reverse=True)

    metrics = {"n_total": n_total, "n_active": n_active,
               "prevalence": round(n_active / n_total, 4)}

    ef_vals = Scoring.CalcEnrichment(ranked, col=1, fractions=EF_FRACTIONS)
    for frac, val in zip(EF_FRACTIONS, ef_vals):
        metrics[ef_key(frac)] = round(float(val), 4)

    for alpha in BEDROC_ALPHAS:
        val = Scoring.CalcBEDROC(ranked, col=1, alpha=alpha)
        metrics[f"bedroc_a{int(alpha)}"] = round(float(val), 4)

    # ROC-AUC
    from sklearn.metrics import roc_auc_score
    metrics["roc_auc"] = round(float(roc_auc_score(labels, scores)), 4)

    return metrics


def parse_docked_sdf(sdf_path: Path) -> pd.DataFrame:
    """
    Parse a GNINA docked SDF. Returns one row per ligand (best pose = max CNN_VS).
    """
    # sanitize=False: GNINA docked poses can have valence issues from explicit Hs
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    rows = []
    for mol in suppl:
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        name  = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        rows.append({
            "name":              name,
            "is_active":         int(props.get("is_active", 0)),
            "CNN_VS":            float(props.get("CNN_VS", props.get("CNNscore", 0))),
            "CNNscore":          float(props.get("CNNscore", 0)),
            "CNNaffinity":       float(props.get("CNNaffinity", 0)),
            "minimizedAffinity": float(props.get("minimizedAffinity", 0)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Best pose per ligand = max CNN_VS
    best = (df.sort_values("CNN_VS", ascending=False)
              .groupby("name", sort=False)
              .first()
              .reset_index())
    return best


def analyze_target(uniprot: str) -> dict:
    sdf_path = DOCKING_DIR / f"{uniprot}_docked.sdf"
    if not sdf_path.exists():
        return {"status": "missing"}

    df = parse_docked_sdf(sdf_path)
    if df.empty or "is_active" not in df.columns:
        return {"status": "empty"}

    labels = df["is_active"].values

    result = {
        "status":   "ok",
        "uniprot":  uniprot,
        "n_ligands": len(df),
    }

    # CNN_VS score (higher = better)
    cnn_metrics = compute_metrics(df["CNN_VS"].values, labels)
    result["CNN_VS"] = cnn_metrics

    # minimizedAffinity (negate: more negative affinity = better binder)
    aff_metrics = compute_metrics(-df["minimizedAffinity"].values, labels)
    result["minimizedAffinity"] = aff_metrics

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sdf_files = sorted(DOCKING_DIR.glob("*_docked.sdf"))
    targets   = [f.stem.replace("_docked", "") for f in sdf_files]
    print(f"Found {len(targets)} docked targets: {targets}\n")

    all_results = {}
    summary_rows = []

    for uniprot in targets:
        print(f"[{uniprot}] analysing...", end=" ", flush=True)
        res = analyze_target(uniprot)
        all_results[uniprot] = res

        # Save per-target JSON
        with open(OUT_DIR / f"{uniprot}_docking_metrics.json", "w") as f:
            json.dump(res, f, indent=2)

        if res["status"] == "ok":
            cnn  = res["CNN_VS"]
            aff  = res["minimizedAffinity"]
            print(f"n={res['n_ligands']} active={cnn.get('n_active','?')}  "
                  f"CNN_VS EF@1%={cnn.get('ef_1pct','?')}  "
                  f"BEDROC-80={cnn.get('bedroc_a80','?')}")
            row = {"uniprot": uniprot,
                   "n_ligands": res["n_ligands"],
                   "prevalence": cnn.get("prevalence")}
            for k, v in cnn.items():
                if k not in ("n_total","n_active","prevalence"):
                    row[f"cnn_{k}"] = v
            for k, v in aff.items():
                if k not in ("n_total","n_active","prevalence"):
                    row[f"aff_{k}"] = v
            summary_rows.append(row)
        else:
            print(res["status"])

    # Save combined JSON
    combined_path = OUT_DIR / "all_targets_docking_metrics.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined JSON → {combined_path}")

    # Save summary CSV
    if summary_rows:
        csv_path = OUT_DIR / "docking_metrics_summary.csv"
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        print(f"Saved summary CSV  → {csv_path}")

        # Print table
        df = pd.DataFrame(summary_rows)
        cols = ["uniprot","n_ligands","prevalence",
                "cnn_roc_auc","cnn_ef_1pct","cnn_ef_5pct","cnn_bedroc_a80",
                "aff_roc_auc","aff_ef_1pct","aff_bedroc_a80"]
        cols = [c for c in cols if c in df.columns]
        print("\n" + df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
