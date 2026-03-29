# benchmarks/05_boltzina/tests/test_ligands.py
import sys
import random
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
from ligands import smiles_to_pdb, sample_decoys


def test_smiles_to_pdb_valid(tmp_path):
    out = tmp_path / 'benzene.pdb'
    result = smiles_to_pdb('c1ccccc1', str(out))
    assert result is True
    assert out.exists()
    content = out.read_text()
    assert 'HETATM' in content
    assert 'UNL' in content


def test_smiles_to_pdb_invalid_smiles(tmp_path):
    out = tmp_path / 'bad.pdb'
    result = smiles_to_pdb('not_a_smiles!!!', str(out))
    assert result is False
    assert not out.exists()


def test_smiles_to_pdb_unique_atom_names(tmp_path):
    out = tmp_path / 'aspirin.pdb'
    smiles_to_pdb('CC(=O)Oc1ccccc1C(=O)O', str(out))
    lines = [l for l in out.read_text().splitlines() if l.startswith('HETATM')]
    atom_names = [l[12:16].strip() for l in lines]
    assert len(atom_names) == len(set(atom_names)), 'Atom names must be unique'


def test_sample_decoys_returns_correct_count(tmp_path):
    decoy_file = tmp_path / 'decoys.txt'
    # actual format: "{active_smiles} {decoy_smiles}"
    lines = [f'c1ccccc1 CC{i}C' for i in range(100)]
    decoy_file.write_text('\n'.join(lines))
    result = sample_decoys(str(decoy_file), n=20)
    assert len(result) == 20


def test_sample_decoys_returns_all_when_n_exceeds_available(tmp_path):
    decoy_file = tmp_path / 'decoys.txt'
    lines = [f'c1ccccc1 CC{i}C' for i in range(10)]
    decoy_file.write_text('\n'.join(lines))
    result = sample_decoys(str(decoy_file), n=100)
    assert len(result) == 10


def test_sample_decoys_are_second_token(tmp_path):
    decoy_file = tmp_path / 'decoys.txt'
    decoy_file.write_text('c1ccccc1 CCO\nc1ccccc1 CCC\n')
    result = sample_decoys(str(decoy_file), n=2)
    assert 'CCO' in result
    assert 'CCC' in result
