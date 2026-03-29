# benchmarks/05_boltzina/tests/test_boltz_prep.py
import json
import pytest
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
from boltz_prep import (
    get_cif_path, write_vina_config, write_boltz_yaml,
    get_receptor_pdb, extract_ligand_centroid,
)

REPO_ROOT = Path('/home/aoxu/projects/VLS-Benchmark-Dataset')
SAMPLE_CIF = (REPO_ROOT / 'plate-vs/VLS_benchmark/zipped_uniprot_raw_cif'
              / 'uniprot_O00408/cif_files_raw/5TZW.cif')


def test_get_cif_path():
    path = get_cif_path('O00408', '5TZW', str(REPO_ROOT))
    assert path.endswith('5TZW.cif')
    assert 'uniprot_O00408' in path
    assert 'zipped_uniprot_raw_cif' in path


def test_write_vina_config(tmp_path):
    out = tmp_path / 'vina.txt'
    write_vina_config((1.5, -2.3, 7.0), (22.0, 22.0, 22.0), str(out))
    content = out.read_text()
    assert 'center_x = 1.500' in content
    assert 'center_y = -2.300' in content
    assert 'center_z = 7.000' in content
    assert 'size_x = 22.0' in content


def test_write_boltz_yaml(tmp_path):
    out = tmp_path / 'input.yaml'
    write_boltz_yaml('CC(=O)O', 'TESTSEQ', str(out))
    with open(out) as f:
        doc = yaml.safe_load(f)
    assert doc['version'] == 1
    seqs = doc['sequences']
    protein_entry = next(s for s in seqs if 'protein' in s)
    ligand_entry = next(s for s in seqs if 'ligand' in s)
    assert protein_entry['protein']['sequence'] == 'TESTSEQ'
    assert ligand_entry['ligand']['smiles'] == 'CC(=O)O'
    assert doc['properties'][0]['affinity']['binder'] == 'B'


def test_get_receptor_pdb_canonical(tmp_path):
    uid = 'O00408'
    pdb_file = (tmp_path / f'boltz_results_{uid}' / 'predictions'
                / uid / f'{uid}_model_0_protein.pdb')
    pdb_file.parent.mkdir(parents=True)
    pdb_file.write_text('ATOM  ...')
    result = get_receptor_pdb(str(tmp_path), uid)
    assert result == str(pdb_file)


def test_get_receptor_pdb_fallback(tmp_path):
    uid = 'O00408'
    # Fallback: different model number (not canonical _model_0_)
    pdb_file = (tmp_path / f'boltz_results_{uid}' / 'predictions'
                / uid / f'{uid}_model_1_protein.pdb')
    pdb_file.parent.mkdir(parents=True)
    pdb_file.write_text('ATOM  ...')
    result = get_receptor_pdb(str(tmp_path), uid)
    assert result.endswith('_protein.pdb')


@pytest.mark.skipif(not SAMPLE_CIF.exists(), reason='CIF not available locally')
def test_extract_ligand_centroid():
    cx, cy, cz = extract_ligand_centroid(str(SAMPLE_CIF))
    assert isinstance(cx, float)
    assert isinstance(cy, float)
    assert isinstance(cz, float)
    # Reasonable coordinate range for a protein crystal structure
    for coord in (cx, cy, cz):
        assert -300 < coord < 300
