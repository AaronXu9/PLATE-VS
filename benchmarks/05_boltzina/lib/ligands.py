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
    try:
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
    except Exception:
        return False

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
    # Deduplicate by SMILES (compound_id is null in some registry versions)
    actives = registry_df[
        (registry_df['uniprot_id'] == uid) &
        (registry_df['similarity_threshold'] == '0p7') &
        (registry_df['protein_partition'] == 'test') &
        (registry_df['split'] == 'test') &
        (registry_df['is_active'] == True)
    ][['smiles']].drop_duplicates('smiles')

    n_actives_written = n_failed = 0
    for i, (_, row) in enumerate(actives.iterrows()):
        out_path = actives_dir / f'active_{i:05d}.pdb'
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
