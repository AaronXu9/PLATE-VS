from pathlib import Path
import json
import sys

# Allow `from build_unidock2_batches import ...` when pytest is run from
# benchmarks/04_docking/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdkit import Chem
from rdkit.Chem import AllChem

from build_unidock2_batches import build_target_inputs, derive_box  # noqa: E402


def _write_three_ligand_sdf(p: Path) -> None:
    w = Chem.SDWriter(str(p))
    for i, smi in enumerate(["CCO", "CCN", "CCC"]):
        m = Chem.MolFromSmiles(smi)
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m, randomSeed=42)
        m.SetProp("_Name", f"mol_{i}")
        m.SetProp("is_active", "1" if i == 0 else "0")
        w.write(m)
    w.close()


def _write_ref_ligand(p: Path) -> None:
    m = Chem.MolFromSmiles("c1ccccc1")
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=42)
    Chem.SDWriter(str(p)).write(m)


def test_split_writes_per_ligand_sdf_and_batch_txt(tmp_path):
    src = tmp_path / "X_all_ligands.sdf"
    _write_three_ligand_sdf(src)
    ref = tmp_path / "X_ref_ligand.sdf"
    _write_ref_ligand(ref)
    out = tmp_path / "unidock2_inputs" / "X"
    info = build_target_inputs(
        uniprot="X",
        all_ligands_sdf=src,
        ref_ligand_sdf=ref,
        out_dir=out,
        padding=4.0,
        size_min=22.5,
    )
    assert info["n_ligands"] == 3
    assert (out / "batch.txt").exists()
    lines = (out / "batch.txt").read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        assert Path(line).exists() and line.endswith(".sdf")
    idx = json.loads((out / "index.json").read_text())
    assert idx["mol_0"]["is_active"] == 1
    assert idx["mol_1"]["is_active"] == 0


def test_box_derived_from_ref_ligand(tmp_path):
    ref = tmp_path / "ref.sdf"
    _write_ref_ligand(ref)
    cx, cy, cz, sx, sy, sz = derive_box(ref, padding=4.0, size_min=22.5)
    # benzene bbox is small → size floored at 22.5
    assert sx == sy == sz == 22.5
