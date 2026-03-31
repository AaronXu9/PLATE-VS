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
            {'protein': {'id': ['A'], 'sequence': sequence, 'msa': 'empty'}},
            {'ligand': {'id': 'B', 'smiles': reference_smiles}},
        ],
        'properties': [
            {'affinity': {'binder': 'B'}}
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(doc, f, default_flow_style=False, allow_unicode=True)


def get_boltz_results_dir(work_dir, uid):
    """Return the boltz_results_{uid}/ subdirectory created by boltz predict."""
    return Path(work_dir) / f'boltz_results_{uid}'


def _cif_to_pdb(cif_path, pdb_path):
    """Extract protein chain A from a boltz CIF and write as PDB."""
    st = gemmi.read_structure(str(cif_path))
    # Keep only chain A (protein; chain B is the ligand in boltz output)
    for model in st:
        chains_to_remove = [
            chain.name for chain in model if chain.name != 'A'
        ]
        for name in chains_to_remove:
            model.remove_chain(name)
    st.write_pdb(str(pdb_path))
    return str(pdb_path)


def _exp_cif_to_pdb(cif_path, pdb_path, chain_a_only=False):
    """Extract polymer chains from experimental CIF, keep only standard amino acids.

    Strips non-standard residues (modified AAs, ligands classified as polymer,
    crystallographic additives) that cause Meeko/RDKit crashes during PDBQT prep.
    If chain_a_only=True, discards all chains except the first (usually chain A).
    """
    st = gemmi.read_structure(str(cif_path))
    st.remove_waters()
    st.remove_hydrogens()
    for model in st:
        # Optionally keep only the first chain (chain A)
        if chain_a_only and len(model) > 0:
            first_chain = list(model)[0].name
            chains_to_drop = [ch.name for ch in model if ch.name != first_chain]
            for name in chains_to_drop:
                try:
                    model.remove_chain(name)
                except Exception:
                    pass
        # Remove chains that contain no standard amino acids
        chains_to_remove = []
        for chain in model:
            has_std_aa = any(r.name in THREE_TO_ONE for r in chain)
            if not has_std_aa:
                chains_to_remove.append(chain.name)
        for name in chains_to_remove:
            try:
                model.remove_chain(name)
            except Exception:
                pass
        # Strip non-standard residues within remaining chains (iterate reversed to
        # preserve indices)
        for chain in model:
            idxs = [i for i, r in enumerate(chain) if r.name not in THREE_TO_ONE]
            for i in reversed(idxs):
                del chain[i]
    Path(pdb_path).parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(pdb_path))
    return str(pdb_path)


def prep_receptor_pdbqt(exp_cif_path, output_pdbqt, boltzina_env='boltzina_env'):
    """Pre-generate receptor.pdbqt from experimental CIF for a protein.

    Uses the experimental crystal structure (more reliable than boltz-2 predicted
    structures which can have clashing atoms that Meeko rejects).

    Skips if output_pdbqt already exists.
    Returns path to receptor.pdbqt.
    """
    pdbqt = Path(output_pdbqt)
    if pdbqt.exists():
        return str(pdbqt)
    pdbqt.parent.mkdir(parents=True, exist_ok=True)

    def _run_mk_prepare(tmp_pdb):
        return subprocess.run(
            ['conda', 'run', '-n', boltzina_env,
             'mk_prepare_receptor.py', '-i', str(tmp_pdb),
             '-o', str(pdbqt.parent / pdbqt.stem), '-p',
             '--default_altloc', 'A', '-a'],
            capture_output=True, text=True,
        )

    # Convert experimental CIF to PDB (temp file); try all chains, then chain A only
    tmp_pdb = pdbqt.parent / '_exp_receptor_tmp.pdb'
    _exp_cif_to_pdb(exp_cif_path, str(tmp_pdb))
    result = _run_mk_prepare(tmp_pdb)

    if result.returncode != 0 or not pdbqt.exists():
        # Fallback: retry with only the first chain (avoids inter-chain bond issues)
        _exp_cif_to_pdb(exp_cif_path, str(tmp_pdb), chain_a_only=True)
        result = _run_mk_prepare(tmp_pdb)

    try:
        tmp_pdb.unlink()
    except Exception:
        pass
    if result.returncode != 0 or not pdbqt.exists():
        raise RuntimeError(
            f'mk_prepare_receptor.py failed for {exp_cif_path}:\n{result.stderr}'
        )
    return str(pdbqt)


def get_receptor_pdb(work_dir, uid):
    """Return path to predicted receptor PDB from boltz predict output.

    Boltz-2 outputs {uid}_model_0.cif; boltz-1 outputs *_protein.pdb.
    Converts CIF → PDB (protein chain only) if needed.
    """
    pred_dir = get_boltz_results_dir(work_dir, uid) / 'predictions' / uid
    # Boltz-2: CIF output
    cif_path = pred_dir / f'{uid}_model_0.cif'
    pdb_path = pred_dir / f'{uid}_model_0_protein.pdb'
    if cif_path.exists() and not pdb_path.exists():
        _cif_to_pdb(cif_path, pdb_path)
    if pdb_path.exists():
        return str(pdb_path)
    # Legacy fallback: recursive search for any *_protein.pdb
    matches = list(Path(work_dir).rglob('*_protein.pdb'))
    if matches:
        return str(sorted(matches)[0])
    raise FileNotFoundError(f'No receptor PDB found under {work_dir}')


def generate_boltz_artifacts(yaml_path, output_dir, boltzina_env='boltzina_env'):
    """Generate manifest.json, constraints/, mols/ from boltz YAML without prediction.

    Uses boltz's own process_input() to parse the YAML and extract metadata,
    ligand constraints, and molecule objects — skipping the expensive neural
    network forward pass that boltz predict would run.

    Args:
        yaml_path: path to the boltz input YAML
        output_dir: base directory for processed/ artifacts
        boltzina_env: conda environment name (needs boltz installed)
    """
    output_dir = Path(output_dir)
    processed_dir = output_dir / 'processed'

    script = f'''
import json, pickle
from pathlib import Path
from boltz.main import process_input

yaml_path = Path("{yaml_path}")
out = Path("{processed_dir}")
for d in ["msa", "constraints", "templates", "mols", "structures", "records"]:
    (out / d).mkdir(parents=True, exist_ok=True)

ccd_path = Path.home() / ".boltz" / "ccd.pkl"
ccd = {{}}
if ccd_path.exists():
    with open(ccd_path, "rb") as f:
        ccd = pickle.load(f)

process_input(
    path=yaml_path, ccd=ccd, msa_dir=out / "msa", mol_dir=out / "mols",
    boltz2=True, use_msa_server=False, msa_server_url="",
    msa_pairing_strategy="greedy", msa_server_username=None,
    msa_server_password=None, api_key_header=None, api_key_value=None,
    max_msa_seqs=4096, processed_msa_dir=out / "msa",
    processed_constraints_dir=out / "constraints",
    processed_templates_dir=out / "templates",
    processed_mols_dir=out / "mols",
    structure_dir=out / "structures", records_dir=out / "records",
)

# Build manifest.json from records
records = []
for f in sorted((out / "records").glob("*.json")):
    records.append(json.load(open(f)))
with open(out / "manifest.json", "w") as f:
    json.dump({{"records": records}}, f, indent=4)
'''
    result = subprocess.run(
        ['conda', 'run', '-n', boltzina_env, 'python3', '-c', script],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'generate_boltz_artifacts failed:\n{result.stderr[-1000:]}'
        )
    manifest = processed_dir / 'manifest.json'
    if not manifest.exists():
        raise RuntimeError(f'manifest.json not generated at {manifest}')
    return str(processed_dir)


def run_boltz_predict(yaml_path, work_dir, boltzina_env='boltzina_env'):
    """Run boltz predict via conda run (legacy, prefer generate_boltz_artifacts).

    Args:
        yaml_path: path to the boltz input YAML (filename stem = uid)
        work_dir: output directory for boltz predict
        boltzina_env: conda environment name
    """
    result = subprocess.run(
        ['conda', 'run', '-n', boltzina_env,
         'boltz', 'predict', str(yaml_path),
         '--out_dir', str(work_dir),
         '--accelerator', 'gpu',
         '--model', 'boltz2'],
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

    Generates boltz artifacts (manifest, constraints, mols) directly from the
    input YAML without running the full boltz predict neural network. The
    experimental crystal structure is used for docking, not a predicted one.

    Skips if processed/manifest.json already exists (resumable).
    """
    uid = protein['uniprot_id']
    work_dir = Path(results_dir) / 'work_dirs' / uid

    boltz_results = get_boltz_results_dir(work_dir, uid)
    if (boltz_results / 'processed' / 'manifest.json').exists():
        print(f'  [skip] {uid}: artifacts already generated')
        return

    cif_path = get_cif_path(uid, protein['pdb_id'], base_dir)
    sequence = _extract_sequence_from_cif(cif_path)
    reference_smiles = get_reference_smiles(registry_df, uid)
    centroid = extract_ligand_centroid(cif_path)

    # Write boltz YAML — filename stem determines artifact naming
    yaml_path = work_dir / f'{uid}.yaml'
    write_boltz_yaml(reference_smiles, sequence, str(yaml_path))

    # Vina config (box around co-crystal ligand centroid)
    vina_config_path = work_dir / 'vina_config.txt'
    write_vina_config(centroid, [22.0, 22.0, 22.0], str(vina_config_path))

    print(f'  Generating boltz artifacts for {uid} (sequence length={len(sequence)})...')
    generate_boltz_artifacts(yaml_path, str(boltz_results), boltzina_env)
    print(f'  ✓ {uid}: artifacts generated')
