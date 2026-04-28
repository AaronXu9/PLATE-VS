from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collect_unidock2_results import compute_target_metrics, parse_target_sdf  # noqa: E402


def _write_synthetic_poses_sdf(p: Path):
    """3 molecules x 2 poses each. Best (lowest) energy listed first per molecule."""
    blocks = []
    for name, e1, e2 in [("active_0", -9.5, -9.0), ("decoy_0", -7.0, -6.5), ("decoy_1", -6.0, -5.5)]:
        for e in (e1, e2):
            blocks.append(
                f"{name}\n"
                f"     RDKit          3D\n"
                f"\n"
                f"  1  0  0  0  0  0  0  0  0  0999 V2000\n"
                f"    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
                f"M  END\n"
                f">  <vina_binding_free_energy>  (1) \n{e}\n\n"
                f"$$$$\n"
            )
    p.write_text("".join(blocks))


def test_parse_keeps_best_pose_per_molecule(tmp_path):
    p = tmp_path / "U_docked.sdf"
    _write_synthetic_poses_sdf(p)
    index = {
        "active_0": {"is_active": 1},
        "decoy_0": {"is_active": 0},
        "decoy_1": {"is_active": 0},
    }
    rows = parse_target_sdf(p, index)
    assert len(rows) == 3
    by_name = {r["name"]: r for r in rows}
    assert by_name["active_0"]["score"] == -9.5
    assert by_name["active_0"]["is_active"] == 1
    assert by_name["decoy_0"]["score"] == -7.0
    assert by_name["decoy_1"]["score"] == -6.0


def test_metrics_perfect_separation():
    rows = [
        {"name": "a1", "score": -10.0, "is_active": 1},
        {"name": "d1", "score": -6.0, "is_active": 0},
        {"name": "d2", "score": -5.0, "is_active": 0},
    ]
    m = compute_target_metrics(rows)
    assert m["roc_auc"] == 1.0
    assert m["n_actives"] == 1
    assert m["n_decoys"] == 2
    # EF1% with 3 ligands: top-1 is the active out of 1/3 prevalence -> EF = 3.0
    assert m["ef1pct"] == 3.0
