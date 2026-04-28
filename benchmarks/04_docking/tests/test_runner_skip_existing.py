from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_unidock2_benchmark import _skip_target  # noqa: E402


def test_skip_when_sdf_exists_and_status_ok(tmp_path):
    sdf = tmp_path / "U_docked.sdf"
    sdf.write_text("$$$$\n")
    prior = {"U": {"status": "ok", "elapsed_s": 100, "out_sdf": str(sdf)}}
    keep = _skip_target("U", sdf, prior)
    assert keep == prior["U"]


def test_no_skip_when_sdf_missing(tmp_path):
    sdf = tmp_path / "U_docked.sdf"
    prior = {"U": {"status": "ok", "out_sdf": str(sdf)}}
    assert _skip_target("U", sdf, prior) is None


def test_no_skip_when_status_error(tmp_path):
    sdf = tmp_path / "U_docked.sdf"
    sdf.write_text("$$$$\n")
    prior = {"U": {"status": "error", "out_sdf": str(sdf)}}
    assert _skip_target("U", sdf, prior) is None


def test_skip_accepts_ok_partial(tmp_path):
    sdf = tmp_path / "U_docked.sdf"
    sdf.write_text("$$$$\n")
    prior = {"U": {"status": "ok_partial", "out_sdf": str(sdf), "n_dropped": 25}}
    assert _skip_target("U", sdf, prior) == prior["U"]
