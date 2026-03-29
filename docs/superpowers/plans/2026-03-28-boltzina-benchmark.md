# Boltzina Benchmark Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 5-stage benchmarking pipeline integrating boltzina (AutoDock Vina + Boltz-2) on 10 soft-split test proteins, producing a `training_summary.json` that slots into `generate_benchmark_report.py`.

**Architecture:** Staged CLI scripts (01–05) in `benchmarks/05_boltzina/`. Core logic lives in `benchmarks/05_boltzina/lib/` as importable modules; numbered scripts are thin argparse wrappers. Boltzina is a git submodule at `external/boltzina/`. Data stages use `rdkit_env`; model stages use `boltzina_env`.

**Tech Stack:** Python 3.9, RDKit (ETKDG conformer gen), gemmi (CIF parsing), boltz-2 (protein trunk + structure prediction), AutoDock Vina (docking), Meeko (PDBQT prep), pandas, scipy, sklearn

---

## File Map

**New files:**
- `benchmarks/05_boltzina/lib/__init__.py`
- `benchmarks/05_boltzina/lib/select.py` — `select_proteins()`
- `benchmarks/05_boltzina/lib/boltz_prep.py` — CIF parsing, boltz YAML, Vina config, receptor PDB lookup
- `benchmarks/05_boltzina/lib/ligands.py` — `smiles_to_pdb()`, `sample_decoys()`, `prep_protein_ligands()`
- `benchmarks/05_boltzina/lib/boltzina_runner.py` — `write_boltzina_config()`, `run_boltzina()`
- `benchmarks/05_boltzina/lib/metrics.py` — `compute_vs_metrics()`, `parse_boltzina_csv()`, `aggregate_results()`, `write_training_summary()`
- `benchmarks/05_boltzina/01_select_proteins.py`
- `benchmarks/05_boltzina/02_prep_boltz.py`
- `benchmarks/05_boltzina/03_prep_ligands.py`
- `benchmarks/05_boltzina/04_run_boltzina.py`
- `benchmarks/05_boltzina/05_collect_results.py`
- `benchmarks/05_boltzina/configs/poc_config.yaml`
- `benchmarks/05_boltzina/tests/conftest.py`
- `benchmarks/05_boltzina/tests/test_select.py`
- `benchmarks/05_boltzina/tests/test_boltz_prep.py`
- `benchmarks/05_boltzina/tests/test_ligands.py`
- `benchmarks/05_boltzina/tests/test_boltzina_runner.py`
- `benchmarks/05_boltzina/tests/test_metrics.py`
- `benchmarks/envs/boltzina_env.yml`

**Modified:**
- `.gitignore` — add `benchmarks/05_boltzina/results/`
- `.gitmodules` — by `git submodule add`

---

## Task 1: Git Worktree + Submodule + Scaffold

**Files:** directory structure, `.gitignore`, `external/boltzina/`

- [ ] **Step 1: Create worktree on new branch**

```bash
cd /home/aoxu/projects/VLS-Benchmark-Dataset
git worktree add ../VLS-Benchmark-Dataset-boltzina -b feature/boltzina-benchmark
cd ../VLS-Benchmark-Dataset-boltzina
```

- [ ] **Step 2: Add boltzina as a git submodule**

```bash
git submodule add https://github.com/ohuelab/boltzina external/boltzina
git submodule update --init
```

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p benchmarks/05_boltzina/lib
mkdir -p benchmarks/05_boltzina/configs
mkdir -p benchmarks/05_boltzina/tests
mkdir -p benchmarks/05_boltzina/results
touch benchmarks/05_boltzina/lib/__init__.py
touch benchmarks/05_boltzina/tests/__init__.py
```

- [ ] **Step 4: Add results/ to .gitignore**

Add this line to `.gitignore`:
```
benchmarks/05_boltzina/results/
```

- [ ] **Step 5: Write poc_config.yaml**

```yaml
# benchmarks/05_boltzina/configs/poc_config.yaml
registry: training_data_full/registry_soft_split_regression.csv
base_dir: /home/aoxu/projects/VLS-Benchmark-Dataset
results_dir: benchmarks/05_boltzina/results
similarity_threshold: "0p7"
n_proteins: 10
min_actives: 50
min_pchembl_coverage: 0.80
n_decoys_ratio: 50
max_decoys: 2500
vina_box_size: [22.0, 22.0, 22.0]
boltzina_env: boltzina_env
```

- [ ] **Step 6: Commit scaffold**

```bash
git add benchmarks/05_boltzina/ external/boltzina .gitmodules .gitignore
git commit -m "feat: scaffold boltzina benchmark pipeline + submodule"
```

Expected: commit succeeds, `external/boltzina/` contains cloned repo at pinned commit.

---

## Task 2: Conda Environment Spec

**Files:** `benchmarks/envs/boltzina_env.yml`

- [ ] **Step 1: Inspect boltzina's boltz version requirement**

```bash
cat external/boltzina/pyproject.toml | grep -i boltz
```

Note the exact boltz version specifier; use it in the yml.

- [ ] **Step 2: Write boltzina_env.yml**

```yaml
# benchmarks/envs/boltzina_env.yml
name: boltzina_env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
    - -e ../../external/boltzina   # installs boltzina + its boltz-2 dep
    - rdkit
    - gemmi
```

- [ ] **Step 3: Install gemmi in rdkit_env (needed for test_boltz_prep.py)**

```bash
conda run -n rdkit_env pip install gemmi
```

Expected: `conda run -n rdkit_env python -c "import gemmi; print(gemmi.__version__)"` prints a version.

- [ ] **Step 4: Create the boltzina_env**

```bash
conda env create -f benchmarks/envs/boltzina_env.yml
```

- [ ] **Step 5: Run boltzina setup (downloads Boltz-2 checkpoints + ADFR suite)**

```bash
conda run -n boltzina_env bash external/boltzina/setup.sh
```

Expected: downloads complete, no errors. This is a one-time ~2–5 GB download.

- [ ] **Step 6: Verify boltzina_env**

```bash
conda run -n boltzina_env python -c "import boltzina; print('boltzina ok')"
conda run -n boltzina_env boltz --help | head -3
```

Expected: both commands succeed.

- [ ] **Step 7: Commit env spec**

```bash
git add benchmarks/envs/boltzina_env.yml
git commit -m "feat: add boltzina_env conda spec"
```

---

## Task 3: Stage 01 — select_proteins (TDD)

**Files:** `lib/select.py`, `01_select_proteins.py`, `tests/test_select.py`

- [ ] **Step 1: Write failing tests**

```python
# benchmarks/05_boltzina/tests/test_select.py
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
from select import select_proteins


def _make_registry(specs):
    """specs: list of (uid, pdb_id, n_actives, pchembl_frac, quality_score)"""
    rows = []
    for uid, pdb_id, n, cov, qscore in specs:
        for i in range(n):
            rows.append({
                'uniprot_id': uid, 'pdb_id': pdb_id,
                'cif_path': f'plate-vs/.../cif_files_raw/{pdb_id}.cif',
                'similarity_threshold': '0p7',
                'protein_partition': 'test',
                'is_active': True,
                'pchembl': 6.5 if i < int(n * cov) else None,
                'quality_score': float(qscore),
            })
    return pd.DataFrame(rows)


def test_filters_min_actives():
    df = _make_registry([
        ('P00001', '1ABC', 60, 0.9, 300),  # passes: 60 actives
        ('P00002', '2DEF', 30, 0.9, 400),  # fails: only 30 actives
    ])
    result = select_proteins(df, n=10, min_actives=50, min_pchembl_coverage=0.5)
    uids = [p['uniprot_id'] for p in result]
    assert 'P00001' in uids
    assert 'P00002' not in uids


def test_filters_pchembl_coverage():
    df = _make_registry([
        ('P00001', '1ABC', 60, 0.9, 300),  # passes: 90% coverage
        ('P00002', '2DEF', 60, 0.3, 400),  # fails: 30% coverage
    ])
    result = select_proteins(df, n=10, min_actives=50, min_pchembl_coverage=0.80)
    uids = [p['uniprot_id'] for p in result]
    assert 'P00001' in uids
    assert 'P00002' not in uids


def test_sorts_by_quality_score():
    df = _make_registry([
        ('P00001', '1ABC', 60, 0.9, 100),
        ('P00002', '2DEF', 60, 0.9, 300),
        ('P00003', '3GHI', 60, 0.9, 200),
    ])
    result = select_proteins(df, n=3)
    assert result[0]['uniprot_id'] == 'P00002'
    assert result[1]['uniprot_id'] == 'P00003'
    assert result[2]['uniprot_id'] == 'P00001'


def test_n_decoys_capped():
    df = _make_registry([('P00001', '1ABC', 100, 0.9, 300)])
    result = select_proteins(df, n=1)
    assert result[0]['n_decoys_to_sample'] == 2500  # 100*50=5000 → capped


def test_n_decoys_not_capped_when_small():
    df = _make_registry([('P00001', '1ABC', 50, 0.9, 300)])
    result = select_proteins(df, n=1)
    assert result[0]['n_decoys_to_sample'] == 50 * 50  # 2500


def test_output_keys():
    df = _make_registry([('P00001', '1ABC', 60, 0.9, 300)])
    result = select_proteins(df, n=1)
    assert len(result) == 1
    expected_keys = {'uniprot_id', 'pdb_id', 'cif_path', 'n_actives',
                     'n_decoys_to_sample', 'pchembl_coverage', 'quality_score'}
    assert expected_keys.issubset(result[0].keys())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/aoxu/projects/VLS-Benchmark-Dataset-boltzina
conda run -n rdkit_env pytest benchmarks/05_boltzina/tests/test_select.py -v 2>&1 | head -20
```

Expected: `ImportError: No module named 'select'`

- [ ] **Step 3: Implement lib/select.py**

```python
# benchmarks/05_boltzina/lib/select.py
import pandas as pd


def select_proteins(df, n=10, min_actives=50, min_pchembl_coverage=0.80):
    """Select top-n test proteins from registry DataFrame.

    Args:
        df: Full registry DataFrame (all rows, all columns).
        n: Number of proteins to return.
        min_actives: Minimum active compounds required per protein.
        min_pchembl_coverage: Minimum fraction of actives with pchembl value.

    Returns:
        List of dicts, sorted by quality_score DESC, each with keys:
        uniprot_id, pdb_id, cif_path, n_actives, n_decoys_to_sample,
        pchembl_coverage, quality_score.
    """
    test = df[
        (df['similarity_threshold'] == '0p7') &
        (df['protein_partition'] == 'test') &
        (df['is_active'] == True)
    ].copy()

    stats = test.groupby('uniprot_id').agg(
        pdb_id=('pdb_id', 'first'),
        cif_path=('cif_path', 'first'),
        quality_score=('quality_score', 'first'),
        n_actives=('is_active', 'count'),
        pchembl_coverage=('pchembl', lambda x: x.notna().mean()),
    ).reset_index()

    filtered = stats[
        (stats['n_actives'] >= min_actives) &
        (stats['pchembl_coverage'] >= min_pchembl_coverage)
    ]

    top = filtered.sort_values('quality_score', ascending=False).head(n)

    result = []
    for _, row in top.iterrows():
        n_actives = int(row['n_actives'])
        result.append({
            'uniprot_id': row['uniprot_id'],
            'pdb_id': row['pdb_id'],
            'cif_path': str(row['cif_path']),
            'n_actives': n_actives,
            'n_decoys_to_sample': min(n_actives * 50, 2500),
            'pchembl_coverage': round(float(row['pchembl_coverage']), 3),
            'quality_score': float(row['quality_score']),
        })
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n rdkit_env pytest benchmarks/05_boltzina/tests/test_select.py -v
```

Expected: 6 tests PASSED.

- [ ] **Step 5: Write CLI wrapper 01_select_proteins.py**

```python
# benchmarks/05_boltzina/01_select_proteins.py
"""Stage 01: Select top-N test proteins from regression registry."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from select import select_proteins


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', required=True,
                   help='Path to registry_soft_split_regression.csv')
    p.add_argument('--output', required=True,
                   help='Path to write poc_proteins.json')
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--min-actives', type=int, default=50)
    p.add_argument('--min-pchembl-coverage', type=float, default=0.80)
    args = p.parse_args()

    print(f"Loading registry from {args.registry}...")
    df = pd.read_csv(args.registry)
    proteins = select_proteins(df, args.n, args.min_actives, args.min_pchembl_coverage)
    print(f"Selected {len(proteins)} proteins:")
    for prot in proteins:
        print(f"  {prot['uniprot_id']} ({prot['pdb_id']}): "
              f"{prot['n_actives']} actives, "
              f"quality={prot['quality_score']:.0f}, "
              f"pchembl_coverage={prot['pchembl_coverage']:.2f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(proteins, f, indent=2)
    print(f"Written to {args.output}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 6: Smoke-test Stage 01 on real data**

```bash
conda run -n rdkit_env python benchmarks/05_boltzina/01_select_proteins.py \
    --registry training_data_full/registry_soft_split_regression.csv \
    --output benchmarks/05_boltzina/results/poc_proteins.json \
    --n 10
```

Expected: 10 proteins printed, `poc_proteins.json` written with 10 entries. Inspect:
```bash
python -c "import json; d=json.load(open('benchmarks/05_boltzina/results/poc_proteins.json')); [print(p['uniprot_id'], p['n_actives'], p['quality_score']) for p in d]"
```

- [ ] **Step 7: Commit**

```bash
git add benchmarks/05_boltzina/lib/select.py benchmarks/05_boltzina/01_select_proteins.py \
        benchmarks/05_boltzina/tests/test_select.py benchmarks/05_boltzina/configs/poc_config.yaml
git commit -m "feat: stage 01 select_proteins with tests"
```

---

## Task 4: Stage 02 — CIF Helpers (TDD)

**Files:** `lib/boltz_prep.py`, `tests/test_boltz_prep.py`

- [ ] **Step 1: Write failing tests**

```python
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

REPO_ROOT = Path(__file__).parents[4]
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
    pdb_file = tmp_path / 'predictions' / uid / f'{uid}_model_0_protein.pdb'
    pdb_file.parent.mkdir(parents=True)
    pdb_file.write_text('ATOM  ...')
    result = get_receptor_pdb(str(tmp_path), uid)
    assert result == str(pdb_file)


def test_get_receptor_pdb_fallback(tmp_path):
    uid = 'O00408'
    # Fallback: file with different naming
    pdb_file = tmp_path / 'predictions' / uid / f'{uid}_model_1_protein.pdb'
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n rdkit_env pytest benchmarks/05_boltzina/tests/test_boltz_prep.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'boltz_prep'`

- [ ] **Step 3: Implement lib/boltz_prep.py**

```python
# benchmarks/05_boltzina/lib/boltz_prep.py
"""CIF parsing helpers, boltz predict YAML generation, Vina config, and runner."""
import subprocess
from pathlib import Path

import gemmi
import yaml

COMMON_SOLVENTS = {
    'HOH', 'WAT', 'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'MES', 'ACT',
    'MPD', 'BME', 'DTT', 'NA', 'MG', 'ZN', 'CA', 'CL', 'BR', 'MN',
    'FE', 'CU', 'K', 'IOD', 'NO3', 'ACE', 'NH4', 'FMT',
}

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def get_cif_path(uniprot_id, pdb_id, base_dir):
    """Return absolute CIF path from repo base_dir."""
    return str(
        Path(base_dir) / 'plate-vs' / 'VLS_benchmark'
        / 'zipped_uniprot_raw_cif'
        / f'uniprot_{uniprot_id}' / 'cif_files_raw' / f'{pdb_id}.cif'
    )


def _extract_sequence_from_cif(cif_path):
    """Return one-letter sequence of the longest polymer chain in CIF."""
    structure = gemmi.read_structure(cif_path)
    best_seq = ''
    for model in structure:
        for chain in model:
            seq = ''.join(
                THREE_TO_ONE.get(res.name, 'X')
                for res in chain
                if res.entity_type == gemmi.EntityType.Polymer
            )
            if len(seq) > len(best_seq):
                best_seq = seq
    return best_seq


def extract_ligand_centroid(cif_path):
    """Return (cx, cy, cz) centroid of the largest non-solvent HETATM group."""
    structure = gemmi.read_structure(cif_path)
    best_residue = None
    best_count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.entity_type != gemmi.EntityType.NonPolymer:
                    continue
                if residue.name in COMMON_SOLVENTS:
                    continue
                heavy = [a for a in residue if a.element != gemmi.Element('H')]
                if len(heavy) > best_count:
                    best_count = len(heavy)
                    best_residue = residue

    if best_residue is None:
        raise ValueError(f'No suitable ligand found in {cif_path}')

    heavy = [a for a in best_residue if a.element != gemmi.Element('H')]
    cx = sum(a.pos.x for a in heavy) / len(heavy)
    cy = sum(a.pos.y for a in heavy) / len(heavy)
    cz = sum(a.pos.z for a in heavy) / len(heavy)
    return (cx, cy, cz)


def write_vina_config(center, box_size, output_path):
    """Write Vina box config (no receptor path; boltzina handles PDBQT prep).

    Args:
        center: (x, y, z) tuple of floats
        box_size: (sx, sy, sz) tuple/list of floats (Angstroms)
        output_path: where to write the config file
    """
    cx, cy, cz = center
    sx, sy, sz = box_size
    content = (
        f'center_x = {cx:.3f}\n'
        f'center_y = {cy:.3f}\n'
        f'center_z = {cz:.3f}\n'
        f'size_x = {sx:.1f}\n'
        f'size_y = {sy:.1f}\n'
        f'size_z = {sz:.1f}\n'
        f'num_modes = 1\n'
        f'seed = 1\n'
        f'cpu = 1\n'
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(content)


def write_boltz_yaml(reference_smiles, sequence, output_path):
    """Write boltz predict input YAML (protein + reference ligand + affinity).

    The reference ligand is used to establish the binding context for Boltz-2.
    The YAML filename (without .yaml) determines the predictions/ subdirectory name.

    Args:
        reference_smiles: SMILES string of a representative active ligand
        sequence: one-letter protein sequence string
        output_path: path to write the YAML (name it {uid}.yaml)
    """
    doc = {
        'version': 1,
        'sequences': [
            {'protein': {'id': ['A'], 'sequence': sequence}},
            {'ligand': {'id': 'B', 'smiles': reference_smiles}},
        ],
        'properties': [
            {'affinity': {'binder': 'B'}}
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(doc, f, default_flow_style=False, allow_unicode=True)


def get_receptor_pdb(work_dir, uid):
    """Return path to predicted receptor PDB from boltz predict output.

    Tries canonical path first, then searches for any *_protein.pdb.
    """
    canonical = Path(work_dir) / 'predictions' / uid / f'{uid}_model_0_protein.pdb'
    if canonical.exists():
        return str(canonical)
    matches = list((Path(work_dir) / 'predictions').rglob('*_protein.pdb'))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f'No receptor PDB in {work_dir}/predictions/')


def run_boltz_predict(yaml_path, work_dir, boltzina_env='boltzina_env'):
    """Run boltz predict via conda run.

    Args:
        yaml_path: path to the boltz input YAML (filename stem = uid)
        work_dir: output directory for boltz predict
        boltzina_env: conda environment name
    """
    result = subprocess.run(
        ['conda', 'run', '-n', boltzina_env,
         'boltz', 'predict', str(yaml_path),
         '--out_dir', str(work_dir),
         '--accelerator', 'gpu'],
        check=True,
        capture_output=True,
        text=True,
    )
    return result


def get_reference_smiles(registry_df, uniprot_id):
    """Return first active SMILES for uniprot_id (used as boltz reference ligand)."""
    rows = registry_df[
        (registry_df['uniprot_id'] == uniprot_id) &
        (registry_df['is_active'] == True) &
        (registry_df['smiles'].notna())
    ]
    if rows.empty:
        raise ValueError(f'No active SMILES found for {uniprot_id}')
    smiles = rows.iloc[0]['smiles']
    # Registry DeepCoy SMILES have format "{active} {decoy}" — take first token
    return smiles.split()[0]


def prep_protein(protein, registry_df, results_dir, base_dir, boltzina_env):
    """Run the full Stage 02 pipeline for one protein.

    Skips if predictions/ subdirectory already exists (resumable).
    """
    uid = protein['uniprot_id']
    work_dir = Path(results_dir) / 'work_dirs' / uid

    if (work_dir / 'predictions' / uid).exists():
        print(f'  [skip] {uid}: boltz predict already done')
        return

    cif_path = get_cif_path(uid, protein['pdb_id'], base_dir)
    sequence = _extract_sequence_from_cif(cif_path)
    reference_smiles = get_reference_smiles(registry_df, uid)
    centroid = extract_ligand_centroid(cif_path)

    # Write boltz YAML — filename stem becomes the prediction subdirectory name
    yaml_path = work_dir / f'{uid}.yaml'
    write_boltz_yaml(reference_smiles, sequence, str(yaml_path))

    # Vina config (box around co-crystal ligand centroid)
    vina_config_path = work_dir / 'vina_config.txt'
    write_vina_config(centroid, [22.0, 22.0, 22.0], str(vina_config_path))

    print(f'  Running boltz predict for {uid} (sequence length={len(sequence)})...')
    run_boltz_predict(yaml_path, work_dir, boltzina_env)
    print(f'  ✓ {uid}: boltz predict complete')
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n rdkit_env pytest benchmarks/05_boltzina/tests/test_boltz_prep.py -v
```

Expected: all tests PASSED (including the `extract_ligand_centroid` test if CIF exists).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/05_boltzina/lib/boltz_prep.py benchmarks/05_boltzina/tests/test_boltz_prep.py
git commit -m "feat: stage 02 CIF helpers with tests"
```

---

## Task 5: Stage 02 — Boltz Predict Runner + CLI

**Files:** `02_prep_boltz.py`

- [ ] **Step 1: Write CLI wrapper**

```python
# benchmarks/05_boltzina/02_prep_boltz.py
"""Stage 02: Run boltz predict per protein → results/work_dirs/{uid}/"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from boltz_prep import prep_protein


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poc-proteins', required=True,
                   help='Path to poc_proteins.json from Stage 01')
    p.add_argument('--registry', required=True,
                   help='Path to registry_soft_split_regression.csv')
    p.add_argument('--results-dir', required=True,
                   help='Root results directory (e.g. benchmarks/05_boltzina/results)')
    p.add_argument('--base-dir', required=True,
                   help='Repo root (for resolving CIF paths)')
    p.add_argument('--boltzina-env', default='boltzina_env')
    args = p.parse_args()

    with open(args.poc_proteins) as f:
        proteins = json.load(f)

    print(f'Loading registry...')
    registry_df = pd.read_csv(
        args.registry,
        usecols=['uniprot_id', 'is_active', 'smiles'],
    )

    print(f'Running boltz predict for {len(proteins)} proteins...')
    for i, protein in enumerate(proteins, 1):
        print(f'[{i}/{len(proteins)}] {protein["uniprot_id"]}')
        prep_protein(protein, registry_df, args.results_dir,
                     args.base_dir, args.boltzina_env)

    print('Stage 02 complete.')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run Stage 02 on the first protein only (smoke test)**

```bash
# Edit poc_proteins.json temporarily to keep only first entry, or use --n 1 in Stage 01
conda run -n rdkit_env python -c "
import json
with open('benchmarks/05_boltzina/results/poc_proteins.json') as f:
    proteins = json.load(f)
with open('benchmarks/05_boltzina/results/poc_proteins_1.json', 'w') as f:
    json.dump(proteins[:1], f, indent=2)
print('Written 1-protein subset')
"

conda run -n boltzina_env python benchmarks/05_boltzina/02_prep_boltz.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins_1.json \
    --registry training_data_full/registry_soft_split_regression.csv \
    --results-dir benchmarks/05_boltzina/results \
    --base-dir /home/aoxu/projects/VLS-Benchmark-Dataset
```

Expected: boltz predict runs (~10–30 min on GPU), output written to `results/work_dirs/{uid}/predictions/{uid}/`.

- [ ] **Step 3: Verify output structure**

```bash
UID=$(python -c "import json; print(json.load(open('benchmarks/05_boltzina/results/poc_proteins_1.json'))[0]['uniprot_id'])")
find benchmarks/05_boltzina/results/work_dirs/${UID}/predictions/ -name "*.pdb" | head -3
```

Expected: at least one `*_protein.pdb` file present.

- [ ] **Step 4: Note actual boltz predict output structure**

After running, inspect the output directory and update `get_receptor_pdb()` if the path pattern differs from the assumed `{uid}_model_0_protein.pdb`. This is a critical calibration step.

```bash
ls benchmarks/05_boltzina/results/work_dirs/${UID}/predictions/${UID}/
```

- [ ] **Step 5: Commit**

```bash
git add benchmarks/05_boltzina/02_prep_boltz.py
git commit -m "feat: stage 02 boltz predict runner CLI"
```

---

## Task 6: Stage 03 — Ligand Preparation (TDD)

**Files:** `lib/ligands.py`, `03_prep_ligands.py`, `tests/test_ligands.py`

- [ ] **Step 1: Write failing tests**

```python
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
    lines = [f'CC active_{i} CC(C) decoy_{i}' for i in range(100)]
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n rdkit_env pytest benchmarks/05_boltzina/tests/test_ligands.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'ligands'`

- [ ] **Step 3: Implement lib/ligands.py**

```python
# benchmarks/05_boltzina/lib/ligands.py
"""SMILES → 3D PDB conversion and decoy sampling."""
import random
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_pdb(smiles, output_path, ligand_name='UNL'):
    """Convert SMILES to 3D PDB file with unique per-element atom names.

    Uses RDKit ETKDG v3 for conformer generation + MMFF optimization.
    Atom names follow the convention required by boltzina: C1, C2, N1, etc.

    Args:
        smiles: SMILES string (for DeepCoy paired format, pass the decoy SMILES only)
        output_path: path to write the PDB file
        ligand_name: residue name in the PDB (default 'UNL' as boltzina expects)

    Returns:
        True on success, False if SMILES is invalid or conformer generation fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return False
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)

    # Assign unique atom names: C1, C2, N1, N2, ...
    element_counters = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        element_counters[sym] = element_counters.get(sym, 0) + 1
        atom_name = f'{sym}{element_counters[sym]}'
        mi = Chem.AtomPDBResidueInfo()
        mi.SetName(atom_name.ljust(4))
        mi.SetResidueName(ligand_name)
        mi.SetResidueNumber(1)
        mi.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(mi)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.PDBWriter(str(output_path))
    writer.write(mol)
    writer.close()
    return True


def sample_decoys(decoy_file, n, seed=42):
    """Sample n decoy SMILES from deepcoy output file.

    File format: one line per decoy, each line is '{active_smiles} {decoy_smiles}'.
    Returns list of decoy SMILES strings (the second token on each line).
    If n >= available, returns all decoys.
    """
    with open(decoy_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    decoy_smiles = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            decoy_smiles.append(parts[-1])

    if n >= len(decoy_smiles):
        return decoy_smiles

    rng = random.Random(seed)
    return rng.sample(decoy_smiles, n)


def prep_protein_ligands(protein, registry_df, results_dir, base_dir):
    """Prepare all active + decoy PDB files for one protein.

    Active PDB files → results/ligands/{uid}/actives/{compound_id}.pdb
    Decoy PDB files  → results/ligands/{uid}/decoys/decoy_{idx:05d}.pdb

    Skips files that already exist (resumable).
    Returns (n_actives_written, n_decoys_written, n_failed).
    """
    uid = protein['uniprot_id']
    actives_dir = Path(results_dir) / 'ligands' / uid / 'actives'
    decoys_dir = Path(results_dir) / 'ligands' / uid / 'decoys'
    actives_dir.mkdir(parents=True, exist_ok=True)
    decoys_dir.mkdir(parents=True, exist_ok=True)

    # --- Actives ---
    actives = registry_df[
        (registry_df['uniprot_id'] == uid) &
        (registry_df['similarity_threshold'] == '0p7') &
        (registry_df['protein_partition'] == 'test') &
        (registry_df['split'] == 'test') &
        (registry_df['is_active'] == True)
    ][['compound_id', 'smiles']].drop_duplicates('compound_id')

    n_actives_written = n_failed = 0
    for _, row in actives.iterrows():
        cid = str(row['compound_id'])
        out_path = actives_dir / f'{cid}.pdb'
        if out_path.exists():
            continue
        smiles = str(row['smiles']).split()[0]  # handle paired format
        if smiles_to_pdb(smiles, str(out_path)):
            n_actives_written += 1
        else:
            n_failed += 1

    # --- Decoys ---
    decoy_file = (
        Path(base_dir) / 'plate-vs' / 'VLS_benchmark'
        / 'chembl_affinity' / f'uniprot_{uid}'
        / 'deepcoy_output' / f'{uid}_generated_decoys.txt'
    )
    decoy_smiles = sample_decoys(str(decoy_file), protein['n_decoys_to_sample'])
    n_decoys_written = 0
    for idx, smi in enumerate(decoy_smiles):
        out_path = decoys_dir / f'decoy_{idx:05d}.pdb'
        if out_path.exists():
            continue
        if smiles_to_pdb(smi, str(out_path)):
            n_decoys_written += 1
        else:
            n_failed += 1

    return n_actives_written, n_decoys_written, n_failed
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n rdkit_env pytest benchmarks/05_boltzina/tests/test_ligands.py -v
```

Expected: all tests PASSED.

- [ ] **Step 5: Write CLI wrapper 03_prep_ligands.py**

```python
# benchmarks/05_boltzina/03_prep_ligands.py
"""Stage 03: Convert SMILES to 3D PDB files for actives and sampled decoys."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from ligands import prep_protein_ligands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poc-proteins', required=True)
    p.add_argument('--registry', required=True)
    p.add_argument('--results-dir', required=True)
    p.add_argument('--base-dir', required=True)
    args = p.parse_args()

    with open(args.poc_proteins) as f:
        proteins = json.load(f)

    print('Loading registry...')
    registry_df = pd.read_csv(
        args.registry,
        usecols=['uniprot_id', 'compound_id', 'smiles', 'similarity_threshold',
                 'protein_partition', 'split', 'is_active'],
    )

    for i, protein in enumerate(proteins, 1):
        uid = protein['uniprot_id']
        print(f'[{i}/{len(proteins)}] {uid}: preparing ligands...')
        n_act, n_dec, n_fail = prep_protein_ligands(
            protein, registry_df, args.results_dir, args.base_dir)
        print(f'  actives={n_act}, decoys={n_dec}, failed={n_fail}')

    print('Stage 03 complete.')


if __name__ == '__main__':
    main()
```

- [ ] **Step 6: Smoke-test Stage 03 on 1 protein**

```bash
conda run -n rdkit_env python benchmarks/05_boltzina/03_prep_ligands.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins_1.json \
    --registry training_data_full/registry_soft_split_regression.csv \
    --results-dir benchmarks/05_boltzina/results \
    --base-dir /home/aoxu/projects/VLS-Benchmark-Dataset
```

Verify:
```bash
UID=$(python -c "import json; print(json.load(open('benchmarks/05_boltzina/results/poc_proteins_1.json'))[0]['uniprot_id'])")
ls benchmarks/05_boltzina/results/ligands/${UID}/actives/ | wc -l
ls benchmarks/05_boltzina/results/ligands/${UID}/decoys/ | wc -l
```

Expected: directories exist with PDB files.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/05_boltzina/lib/ligands.py benchmarks/05_boltzina/03_prep_ligands.py \
        benchmarks/05_boltzina/tests/test_ligands.py
git commit -m "feat: stage 03 ligand SMILES→PDB prep with tests"
```

---

## Task 7: Stage 04 — Boltzina Config + Runner (TDD)

**Files:** `lib/boltzina_runner.py`, `04_run_boltzina.py`, `tests/test_boltzina_runner.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n rdkit_env pytest benchmarks/05_boltzina/tests/test_boltzina_runner.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'boltzina_runner'`

- [ ] **Step 3: Implement lib/boltzina_runner.py**

```python
# benchmarks/05_boltzina/lib/boltzina_runner.py
"""Boltzina config generation and subprocess runner."""
import json
import subprocess
from pathlib import Path


def write_boltzina_config(uid, work_dir, vina_config, receptor_pdb,
                           ligand_files, output_dir):
    """Write boltzina config.json for one protein.

    Args:
        uid: UniProt ID (used as fname, must match boltz predict YAML stem)
        work_dir: path to boltz predict output directory
        vina_config: path to Vina config file
        receptor_pdb: path to predicted receptor PDB
        ligand_files: list of ligand PDB file paths (actives + decoys)
        output_dir: where boltzina writes its results CSV

    Returns:
        path to written config.json (str)
    """
    config = {
        'work_dir': str(work_dir),
        'vina_config': str(vina_config),
        'fname': uid,
        'input_ligand_name': 'UNL',
        'output_dir': str(output_dir),
        'receptor_pdb': str(receptor_pdb),
        'ligand_files': [str(f) for f in ligand_files],
    }
    config_path = Path(output_dir) / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return str(config_path)


def run_boltzina(config_path, boltzina_submodule_path, boltzina_env='boltzina_env'):
    """Run boltzina via conda run.

    Args:
        config_path: path to boltzina config.json
        boltzina_submodule_path: path to external/boltzina/ (where run.py lives)
        boltzina_env: conda environment name
    """
    run_py = Path(boltzina_submodule_path) / 'run.py'
    result = subprocess.run(
        ['conda', 'run', '-n', boltzina_env,
         'python', str(run_py), str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result


def collect_ligand_paths(ligands_dir, uid):
    """Return sorted list of all PDB paths for actives and decoys.

    Args:
        ligands_dir: root ligands directory (contains {uid}/actives/ and {uid}/decoys/)
        uid: UniProt ID

    Returns:
        list of absolute path strings, actives first then decoys.
    """
    actives_dir = Path(ligands_dir) / uid / 'actives'
    decoys_dir = Path(ligands_dir) / uid / 'decoys'
    actives = sorted(actives_dir.glob('*.pdb'))
    decoys = sorted(decoys_dir.glob('*.pdb'))
    return [str(p) for p in actives + decoys]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n rdkit_env pytest benchmarks/05_boltzina/tests/test_boltzina_runner.py -v
```

Expected: 3 tests PASSED.

- [ ] **Step 5: Write CLI wrapper 04_run_boltzina.py**

```python
# benchmarks/05_boltzina/04_run_boltzina.py
"""Stage 04: Run boltzina docking + Boltz-2 scoring per protein."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from boltz_prep import get_receptor_pdb
from boltzina_runner import collect_ligand_paths, write_boltzina_config, run_boltzina


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poc-proteins', required=True)
    p.add_argument('--results-dir', required=True)
    p.add_argument('--boltzina-dir', default='external/boltzina',
                   help='Path to cloned boltzina submodule')
    p.add_argument('--boltzina-env', default='boltzina_env')
    args = p.parse_args()

    with open(args.poc_proteins) as f:
        proteins = json.load(f)

    results_dir = Path(args.results_dir)

    for i, protein in enumerate(proteins, 1):
        uid = protein['uniprot_id']
        raw_dir = results_dir / 'raw_results' / uid

        if (raw_dir / 'results.csv').exists():
            print(f'[{i}/{len(proteins)}] {uid}: skip (results.csv exists)')
            continue

        work_dir = results_dir / 'work_dirs' / uid
        vina_config = work_dir / 'vina_config.txt'
        receptor_pdb = get_receptor_pdb(str(work_dir), uid)
        ligand_files = collect_ligand_paths(str(results_dir / 'ligands'), uid)

        print(f'[{i}/{len(proteins)}] {uid}: {len(ligand_files)} ligands...')
        config_path = write_boltzina_config(
            uid=uid,
            work_dir=str(work_dir),
            vina_config=str(vina_config),
            receptor_pdb=receptor_pdb,
            ligand_files=ligand_files,
            output_dir=str(raw_dir),
        )
        run_boltzina(config_path, args.boltzina_dir, args.boltzina_env)
        print(f'  ✓ {uid}: boltzina complete')

    print('Stage 04 complete.')


if __name__ == '__main__':
    main()
```

- [ ] **Step 6: Smoke-test Stage 04 on 1 protein**

```bash
conda run -n boltzina_env python benchmarks/05_boltzina/04_run_boltzina.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins_1.json \
    --results-dir benchmarks/05_boltzina/results
```

Expected: `results/raw_results/{uid}/results.csv` appears. Inspect:
```bash
head -3 benchmarks/05_boltzina/results/raw_results/${UID}/results.csv
```

**Important:** Record the exact CSV column names shown in the header. You will need these for `parse_boltzina_csv()` in the next task.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/05_boltzina/lib/boltzina_runner.py benchmarks/05_boltzina/04_run_boltzina.py \
        benchmarks/05_boltzina/tests/test_boltzina_runner.py
git commit -m "feat: stage 04 boltzina config + runner with tests"
```

---

## Task 8: Stage 05 — Metrics + Output (TDD)

**Files:** `lib/metrics.py`, `05_collect_results.py`, `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# benchmarks/05_boltzina/tests/test_metrics.py
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
from metrics import compute_vs_metrics, write_training_summary


def test_roc_auc_perfect():
    # Actives score higher → perfect separation
    scores = [1.0, 0.9, 0.8, 0.1, 0.05, 0.02]
    labels = [1,   1,   1,   0,   0,    0  ]
    m = compute_vs_metrics(scores, labels)
    assert m['roc_auc'] == pytest.approx(1.0)


def test_roc_auc_random():
    rng = np.random.default_rng(42)
    scores = rng.random(200).tolist()
    labels = ([1] * 10 + [0] * 190)
    rng.shuffle(labels)
    m = compute_vs_metrics(scores, labels)
    assert 0.3 < m['roc_auc'] < 0.7  # roughly random


def test_ef1pct_perfect():
    # 10 actives in 1000 total; top 1% = 10 slots → all actives ranked first
    n = 1000
    n_actives = 10
    scores = [2.0] * n_actives + [1.0] * (n - n_actives)
    labels = [1] * n_actives + [0] * (n - n_actives)
    m = compute_vs_metrics(scores, labels)
    # Expected EF1% = (10/10) / (10/1000) = 100
    assert m['ef1pct'] == pytest.approx(100.0, rel=0.01)


def test_spearman_perfect_correlation():
    scores =  [1.0, 2.0, 3.0, 4.0, 5.0]
    labels =  [1,   1,   1,   1,   1  ]
    pchembl = [4.0, 5.0, 6.0, 7.0, 8.0]  # monotone with scores
    m = compute_vs_metrics(scores, labels, pchembl_values=pchembl)
    assert m['spearman_r'] == pytest.approx(1.0, abs=0.01)


def test_spearman_none_when_too_few_actives():
    scores =  [1.0, 2.0, 0.5]
    labels =  [1,   1,   0  ]
    pchembl = [5.0, 6.0, None]
    m = compute_vs_metrics(scores, labels, pchembl_values=pchembl)
    assert m['spearman_r'] is None  # < 5 actives with pchembl


def test_write_training_summary_schema(tmp_path):
    metrics = {'roc_auc': 0.75, 'ef1pct': 5.2, 'spearman_r': 0.31}
    out = tmp_path / 'summary.json'
    write_training_summary(metrics, str(out), n_test=1234, elapsed_s=3600.0)

    with open(out) as f:
        doc = json.load(f)

    assert doc['model_type'] == 'boltzina'
    assert doc['feature_type'] == '3d_docking_boltz2'
    assert doc['similarity_threshold'] == '0p7'
    assert doc['use_precomputed_split'] is True
    hist = doc['training_history']
    assert hist['test_metrics']['roc_auc'] == pytest.approx(0.75)
    assert hist['test_metrics']['ef1pct'] == pytest.approx(5.2)
    assert hist['test_metrics']['spearman_r'] == pytest.approx(0.31)
    assert hist['n_test_samples'] == 1234
    assert hist['training_time'] == pytest.approx(3600.0)
    # Required by generate_benchmark_report.py
    assert 'train_metrics' in hist
    assert 'val_metrics' in hist
    assert 'n_train_samples' in hist
    assert 'n_val_samples' in hist
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n rdkit_env pytest benchmarks/05_boltzina/tests/test_metrics.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'metrics'`

- [ ] **Step 3: Implement lib/metrics.py**

```python
# benchmarks/05_boltzina/lib/metrics.py
"""VS benchmark metrics and training_summary.json output."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def compute_vs_metrics(scores, labels, pchembl_values=None):
    """Compute ROC-AUC, EF1%, and Spearman r for one protein benchmark.

    Args:
        scores: list/array of boltz_affinity scores (higher = more active)
        labels: list/array of binary labels (1=active, 0=decoy)
        pchembl_values: list/array of pchembl values (None/NaN for decoys or missing)

    Returns:
        dict with keys: roc_auc, ef1pct, spearman_r (each float or None)
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    # ROC-AUC
    n_pos = labels.sum()
    if n_pos == 0 or n_pos == len(labels):
        roc_auc = None
    else:
        roc_auc = round(float(roc_auc_score(labels, scores)), 4)

    # EF1% — enrichment factor at top 1%
    n_total = len(scores)
    n_top = max(1, int(np.ceil(n_total * 0.01)))
    top_idx = np.argsort(scores)[::-1][:n_top]
    hits = int(labels[top_idx].sum())
    random_rate = n_pos / n_total if n_total > 0 else 0
    ef1pct = round((hits / n_top) / random_rate, 4) if random_rate > 0 else None

    # Spearman r — affinity correlation for actives with pchembl labels
    spearman_r = None
    if pchembl_values is not None:
        pchembl_arr = np.asarray(
            [float(v) if v is not None else np.nan for v in pchembl_values],
            dtype=float,
        )
        mask = (labels == 1) & ~np.isnan(pchembl_arr)
        if mask.sum() >= 5:
            r, _ = spearmanr(scores[mask], pchembl_arr[mask])
            spearman_r = round(float(r), 4)

    return {'roc_auc': roc_auc, 'ef1pct': ef1pct, 'spearman_r': spearman_r}


def parse_boltzina_csv(csv_path, affinity_col=None):
    """Parse boltzina results CSV into a DataFrame with is_active labels.

    Args:
        csv_path: path to boltzina output CSV
        affinity_col: name of the Boltz-2 affinity column (auto-detected if None)

    Returns:
        DataFrame with columns: ligand_file, boltz_affinity, is_active
    """
    df = pd.read_csv(csv_path)

    # Auto-detect affinity column (first column containing 'affinity' in name)
    if affinity_col is None:
        candidates = [c for c in df.columns if 'affinity' in c.lower()]
        if not candidates:
            raise ValueError(
                f'No affinity column found in {csv_path}. '
                f'Columns: {df.columns.tolist()}. '
                f'Pass affinity_col= explicitly.'
            )
        affinity_col = candidates[0]

    # Identify ligand file column
    file_col = next(
        (c for c in df.columns if 'ligand' in c.lower() or 'file' in c.lower()),
        df.columns[0]
    )

    # Derive is_active from path: actives/ → 1, decoys/ → 0
    df['is_active'] = df[file_col].str.contains('/actives/').astype(int)
    df['boltz_affinity'] = df[affinity_col].astype(float)
    df['ligand_file'] = df[file_col]

    return df[['ligand_file', 'boltz_affinity', 'is_active']]


def aggregate_results(per_protein_metrics):
    """Macro-average metrics across proteins, ignoring None values.

    Args:
        per_protein_metrics: list of dicts from compute_vs_metrics()

    Returns:
        dict with macro-averaged roc_auc, ef1pct, spearman_r
    """
    aggregated = {}
    for key in ('roc_auc', 'ef1pct', 'spearman_r'):
        values = [m[key] for m in per_protein_metrics if m.get(key) is not None]
        aggregated[key] = round(float(np.mean(values)), 4) if values else None
    return aggregated


def write_training_summary(metrics, output_path, n_test, elapsed_s):
    """Write boltzina_training_summary.json in generate_benchmark_report.py format.

    Args:
        metrics: dict from aggregate_results() with roc_auc, ef1pct, spearman_r
        output_path: where to write the JSON
        n_test: total number of compounds evaluated (actives + decoys)
        elapsed_s: total wall-clock time in seconds
    """
    doc = {
        'model_type': 'boltzina',
        'feature_type': '3d_docking_boltz2',
        'similarity_threshold': '0p7',
        'use_precomputed_split': True,
        'training_history': {
            'train_metrics': {},
            'val_metrics': {},
            'test_metrics': {
                'roc_auc': metrics.get('roc_auc'),
                'ef1pct': metrics.get('ef1pct'),
                'spearman_r': metrics.get('spearman_r'),
            },
            'n_train_samples': 0,
            'n_val_samples': 0,
            'n_test_samples': int(n_test),
            'training_time': round(float(elapsed_s), 1),
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(doc, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n rdkit_env pytest benchmarks/05_boltzina/tests/test_metrics.py -v
```

Expected: all tests PASSED.

- [ ] **Step 5: Write CLI wrapper 05_collect_results.py**

```python
# benchmarks/05_boltzina/05_collect_results.py
"""Stage 05: Parse boltzina CSVs → compute metrics → write training_summary.json."""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from metrics import parse_boltzina_csv, compute_vs_metrics, aggregate_results, write_training_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poc-proteins', required=True)
    p.add_argument('--registry', required=True)
    p.add_argument('--results-dir', required=True)
    p.add_argument('--affinity-col', default=None,
                   help='Name of boltz affinity column in results CSV (auto-detected if omitted)')
    args = p.parse_args()

    with open(args.poc_proteins) as f:
        proteins = json.load(f)

    # Load pchembl values for joining
    registry_df = pd.read_csv(
        args.registry,
        usecols=['uniprot_id', 'compound_id', 'pchembl', 'similarity_threshold',
                 'protein_partition', 'split', 'is_active'],
    )

    results_dir = Path(args.results_dir)
    per_protein = []
    per_protein_rows = []
    t0 = time.time()
    total_n = 0

    for protein in proteins:
        uid = protein['uniprot_id']
        csv_path = results_dir / 'raw_results' / uid / 'results.csv'
        if not csv_path.exists():
            print(f'  [missing] {uid}: no results.csv, skipping')
            continue

        df = parse_boltzina_csv(str(csv_path), affinity_col=args.affinity_col)
        total_n += len(df)

        # Build pchembl array aligned with actives
        active_df = df[df['is_active'] == 1].copy()
        active_df['compound_id'] = (
            active_df['ligand_file']
            .apply(lambda x: Path(x).stem)
        )
        pchembl_map = (
            registry_df[
                (registry_df['uniprot_id'] == uid) &
                (registry_df['is_active'] == True) &
                (registry_df['similarity_threshold'] == '0p7')
            ]
            .drop_duplicates('compound_id')
            .set_index('compound_id')['pchembl']
            .to_dict()
        )
        pchembl_vals = [pchembl_map.get(cid) for cid in active_df['compound_id']]

        # Combine for metrics
        scores = df['boltz_affinity'].tolist()
        labels = df['is_active'].tolist()
        pchembl_full = [None] * len(df)
        for j, (idx, row) in enumerate(df.iterrows()):
            if row['is_active'] == 1:
                cid = Path(row['ligand_file']).stem
                pchembl_full[j] = pchembl_map.get(cid)

        metrics = compute_vs_metrics(scores, labels, pchembl_values=pchembl_full)
        per_protein.append(metrics)
        per_protein_rows.append({
            'uniprot_id': uid,
            'n_actives': int(df['is_active'].sum()),
            'n_decoys': int((df['is_active'] == 0).sum()),
            **metrics,
        })
        print(f'  {uid}: roc_auc={metrics["roc_auc"]}, '
              f'ef1pct={metrics["ef1pct"]}, spearman_r={metrics["spearman_r"]}')

    elapsed = time.time() - t0
    agg = aggregate_results(per_protein)
    print(f'\nAggregate ({len(per_protein)} proteins):')
    print(f'  ROC-AUC={agg["roc_auc"]}, EF1%={agg["ef1pct"]}, Spearman r={agg["spearman_r"]}')

    summary_path = results_dir / 'boltzina_training_summary.json'
    write_training_summary(agg, str(summary_path), n_test=total_n, elapsed_s=elapsed)
    print(f'Written: {summary_path}')

    per_protein_csv = results_dir / 'boltzina_per_protein_results.csv'
    pd.DataFrame(per_protein_rows).to_csv(per_protein_csv, index=False)
    print(f'Written: {per_protein_csv}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 6: Commit**

```bash
git add benchmarks/05_boltzina/lib/metrics.py benchmarks/05_boltzina/05_collect_results.py \
        benchmarks/05_boltzina/tests/test_metrics.py
git commit -m "feat: stage 05 metrics + training_summary output with tests"
```

---

## Task 9: End-to-End Smoke Test (1 Protein)

- [ ] **Step 1: Run full test suite to confirm all unit tests pass**

```bash
conda run -n rdkit_env pytest benchmarks/05_boltzina/tests/ -v
```

Expected: all tests PASSED (the `extract_ligand_centroid` test requires the CIF file and the boltz predict integration tests were already run manually).

- [ ] **Step 2: Run Stage 05 on the 1-protein result from Task 5/6**

```bash
# Assuming poc_proteins_1.json and raw_results/{uid}/results.csv exist
conda run -n rdkit_env python benchmarks/05_boltzina/05_collect_results.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins_1.json \
    --registry training_data_full/registry_soft_split_regression.csv \
    --results-dir benchmarks/05_boltzina/results
```

Expected: `boltzina_training_summary.json` written. Verify:
```bash
python -c "
import json
d = json.load(open('benchmarks/05_boltzina/results/boltzina_training_summary.json'))
print('model_type:', d['model_type'])
print('test_metrics:', d['training_history']['test_metrics'])
print('n_test:', d['training_history']['n_test_samples'])
"
```

Expected: `roc_auc` is a float, not None.

- [ ] **Step 3: Verify report integration**

```bash
conda run -n rdkit_env python benchmarks/03_analysis/generate_benchmark_report.py \
    --results-dir benchmarks/02_training/trained_models \
    --extra-dirs benchmarks/05_boltzina/results \
    --output /tmp/report_with_boltzina.csv
```

Expected: output includes a row with `model=boltzina`. Check:
```bash
python -c "import pandas as pd; df=pd.read_csv('/tmp/report_with_boltzina.csv'); print(df[df['model']=='boltzina'][['model','similarity_threshold','n_test','test_roc_auc']].to_string())"
```

- [ ] **Step 4: Commit smoke test results**

```bash
git add benchmarks/05_boltzina/results/poc_proteins_1.json  # small JSON, ok to track
git commit -m "test: end-to-end smoke test passes for 1 protein"
```

---

## Task 10: Scale to 10 Proteins + Final Commit

- [ ] **Step 1: Run Stage 02 for all 10 proteins (GPU run, ~2–5 hours)**

```bash
conda run -n boltzina_env python benchmarks/05_boltzina/02_prep_boltz.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins.json \
    --registry training_data_full/registry_soft_split_regression.csv \
    --results-dir benchmarks/05_boltzina/results \
    --base-dir /home/aoxu/projects/VLS-Benchmark-Dataset
```

Each protein is skipped if already done. Re-runnable.

- [ ] **Step 2: Run Stage 03 for all 10 proteins**

```bash
conda run -n rdkit_env python benchmarks/05_boltzina/03_prep_ligands.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins.json \
    --registry training_data_full/registry_soft_split_regression.csv \
    --results-dir benchmarks/05_boltzina/results \
    --base-dir /home/aoxu/projects/VLS-Benchmark-Dataset
```

- [ ] **Step 3: Run Stage 04 for all 10 proteins**

```bash
conda run -n boltzina_env python benchmarks/05_boltzina/04_run_boltzina.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins.json \
    --results-dir benchmarks/05_boltzina/results
```

- [ ] **Step 4: Run Stage 05 for all 10 proteins**

```bash
conda run -n rdkit_env python benchmarks/05_boltzina/05_collect_results.py \
    --poc-proteins benchmarks/05_boltzina/results/poc_proteins.json \
    --registry training_data_full/registry_soft_split_regression.csv \
    --results-dir benchmarks/05_boltzina/results
```

- [ ] **Step 5: Regenerate unified benchmark report**

```bash
cd benchmarks/03_analysis
conda run -n rdkit_env python generate_benchmark_report.py \
    --results-dir ../02_training/trained_models \
    --extra-dirs ../../benchmarks/05_boltzina/results \
    --output report.csv
```

Verify boltzina row present and ROC-AUC non-null.

- [ ] **Step 6: Final commit**

```bash
git add benchmarks/05_boltzina/results/poc_proteins.json \
        benchmarks/05_boltzina/results/boltzina_training_summary.json \
        benchmarks/05_boltzina/results/boltzina_per_protein_results.csv \
        benchmarks/03_analysis/report.csv
git commit -m "feat: boltzina PoC benchmark complete — 10 proteins, ROC-AUC in report"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ 5 staged scripts (01–05) with CLI wrappers
- ✅ git submodule at `external/boltzina/`
- ✅ boltzina_env spec
- ✅ Select 10 proteins by quality_score + n_actives + pchembl_coverage filters
- ✅ CIF → boltz predict → work_dir (resumable, skips existing)
- ✅ SMILES → 3D PDB via RDKit ETKDG (actives + sampled decoys, unique atom names)
- ✅ Decoys sampled from deepcoy_output text files at 50:1 ratio, capped at 2500
- ✅ Boltzina config.json generated per protein with all required keys
- ✅ ROC-AUC, EF1%, Spearman r computed and macro-averaged
- ✅ `training_summary.json` schema matches `generate_benchmark_report.py parse_row()`
- ✅ Per-protein breakdown CSV also written
- ✅ All stages resumable (skip-if-exists logic)
- ✅ End-to-end smoke test task

**Type consistency:** All function signatures defined in lib/ match their usage in CLI wrappers and tests.

**No placeholders:** All code blocks are complete and runnable.
