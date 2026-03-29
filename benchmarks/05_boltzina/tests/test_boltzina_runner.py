# benchmarks/05_boltzina/tests/test_boltzina_runner.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
from boltzina_runner import write_boltzina_config


def test_write_boltzina_config_creates_file(tmp_path):
    ligand_files = [str(tmp_path / f'lig{i}.pdb') for i in range(3)]
    config_path = write_boltzina_config(
        uid='O00408',
        work_dir=str(tmp_path / 'work_dirs/O00408'),
        vina_config=str(tmp_path / 'work_dirs/O00408/vina_config.txt'),
        receptor_pdb=str(tmp_path / 'work_dirs/O00408/predictions/O00408/O00408_model_0_protein.pdb'),
        ligand_files=ligand_files,
        output_dir=str(tmp_path / 'raw_results/O00408'),
    )
    assert Path(config_path).exists()


def test_write_boltzina_config_required_keys(tmp_path):
    ligand_files = [str(tmp_path / 'lig.pdb')]
    config_path = write_boltzina_config(
        uid='O00408',
        work_dir=str(tmp_path / 'work_dir'),
        vina_config=str(tmp_path / 'vina.txt'),
        receptor_pdb=str(tmp_path / 'receptor.pdb'),
        ligand_files=ligand_files,
        output_dir=str(tmp_path / 'out'),
    )
    with open(config_path) as f:
        config = json.load(f)

    required_keys = {'work_dir', 'vina_config', 'fname', 'input_ligand_name',
                     'output_dir', 'receptor_pdb', 'ligand_files'}
    assert required_keys.issubset(config.keys())


def test_write_boltzina_config_ligand_files(tmp_path):
    ligand_files = [str(tmp_path / f'lig{i}.pdb') for i in range(5)]
    config_path = write_boltzina_config(
        uid='O00408',
        work_dir=str(tmp_path / 'work_dir'),
        vina_config=str(tmp_path / 'vina.txt'),
        receptor_pdb=str(tmp_path / 'receptor.pdb'),
        ligand_files=ligand_files,
        output_dir=str(tmp_path / 'out'),
    )
    with open(config_path) as f:
        config = json.load(f)
    assert len(config['ligand_files']) == 5
    assert config['fname'] == 'O00408'
    assert config['input_ligand_name'] == 'UNL'
