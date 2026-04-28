"""Side-by-side comparison: GNINA vs UniDock2 on the same 15 targets.

Reads `gnina_training_summary.json` and `unidock2_training_summary.json` from
benchmarks/04_docking/results/ and emits CSV + Markdown tables of per-target
ROC-AUC, AP, EF1%, and wall time, plus a final aggregate row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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
    args = ap.parse_args()

    g = json.loads(Path(args.gnina_summary).read_text())
    u = json.loads(Path(args.unidock2_summary).read_text())
    # gnina ships per-target under "per_target_metrics" with keys
    # avg_precision/f1_score; unidock2 collector uses per_target with ap/f1.
    g_per = g.get("per_target_metrics", g.get("per_target", {}))
    u_per = u.get("per_target", u)

    rows = []
    for uniprot in sorted(set(g_per) | set(u_per)):
        gm = g_per.get(uniprot, {}) if isinstance(g_per.get(uniprot), dict) else {}
        um = u_per.get(uniprot, {}) if isinstance(u_per.get(uniprot), dict) else {}
        rows.append(
            {
                "uniprot": uniprot,
                "n_actives": gm.get("n_actives") or um.get("n_actives"),
                "n_decoys": gm.get("n_decoys") or um.get("n_decoys"),
                "gnina_roc_auc": gm.get("roc_auc"),
                "unidock2_roc_auc": um.get("roc_auc"),
                "gnina_ap": gm.get("avg_precision"),
                "unidock2_ap": um.get("ap"),
                "unidock2_ef1pct": um.get("ef1pct"),
                "unidock2_status": um.get("status"),
                "unidock2_dropped": um.get("n_dropped", um.get("n_chunks_failed", 0)),
                "unidock2_elapsed_s": um.get("elapsed_s"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    aggregate = {
        "gnina": {
            "test_metrics": g.get("training_history", {}).get("test_metrics", {}),
            "global_pooled_metrics": g.get("global_pooled_metrics", {}),
        },
        "unidock2": u.get("aggregate", {}),
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
