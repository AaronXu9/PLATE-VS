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
    # Registry SMILES have format "{active} {decoy}" in some rows — take first token
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
