"""Regression test: evaluate() must compute per-target AUC on PLATE-VS batches.

Reproduces the bug where AffinityBatch exposes `pdb_ids` (populated from the
Data.uniprot_id attribute via custom_collate), but evaluate() was checking for
`batch.uniprot_ids` — so the per-target branch never ran.

Run:
    /home/aoxu/miniconda3/envs/SurfDock/bin/python \
        benchmarks/07_plate_vs_dl/tests/test_per_target_auc.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[3]
BA_DIR = REPO_ROOT / "benchmarks" / "06_binding_affinity_model"
VS_DIR = REPO_ROOT / "benchmarks" / "07_plate_vs_dl"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_collate = _load_module("ba_collate", BA_DIR / "data" / "collate.py")
_evaluation = _load_module("vs_evaluation", VS_DIR / "evaluation.py")

AffinityBatch = _collate.AffinityBatch
custom_collate = _collate.custom_collate
evaluate = _evaluation.evaluate


class _StubModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(self, batch):
        return self._logits


def _make_data(uniprot_id: str, label: int) -> Data:
    return Data(
        z=torch.tensor([6], dtype=torch.long),
        pos=torch.zeros(1, 3, dtype=torch.float32),
        prot_emb=torch.zeros(1, 8, dtype=torch.float32),
        num_lig_atoms=torch.tensor(1, dtype=torch.long),
        num_pocket_res=torch.tensor(1, dtype=torch.long),
        y=torch.tensor([float(label)], dtype=torch.float32),
        uniprot_id=uniprot_id,
    )


def test_collate_preserves_uniprot_ids_as_pdb_ids():
    """custom_collate must populate AffinityBatch.pdb_ids from Data.uniprot_id."""
    data_list = [_make_data("P1", 1), _make_data("P1", 0),
                 _make_data("P2", 1), _make_data("P2", 0)]
    batch = custom_collate(data_list)
    assert batch.pdb_ids == ["P1", "P1", "P2", "P2"], f"got {batch.pdb_ids}"


def test_evaluate_populates_per_target_auc_from_uniprot_id_batches():
    """evaluate() must compute per-target AUC when Data carries uniprot_id (PLATE-VS)."""
    data_list = [_make_data("P1", 1), _make_data("P1", 0),
                 _make_data("P2", 1), _make_data("P2", 0)]
    batch = custom_collate(data_list)

    # Logits chosen so each target has perfect separation → per-target AUC = 1.0.
    logits = torch.tensor([[2.0], [-2.0], [2.0], [-2.0]])
    model = _StubModel(logits)
    criterion = nn.BCEWithLogitsLoss()

    metrics = evaluate(model, [batch], criterion, device=torch.device("cpu"))

    assert "per_target_roc_auc_mean" in metrics, \
        f"per-target AUC missing from metrics: {sorted(metrics)}"
    assert metrics["n_targets_evaluated"] == 2
    assert metrics["per_target_roc_auc_mean"] == 1.0


if __name__ == "__main__":
    test_collate_preserves_uniprot_ids_as_pdb_ids()
    print("OK: batch.pdb_ids populated from Data.uniprot_id")
    test_evaluate_populates_per_target_auc_from_uniprot_id_batches()
    print("OK: evaluate() returns per_target_roc_auc_mean")
