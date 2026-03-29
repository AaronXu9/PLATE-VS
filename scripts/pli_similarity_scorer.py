"""
Protein-Ligand Interaction (PLI) Similarity Scorer

This module computes similarity between protein-ligand systems following the methodology
from Škrinjar et al. 2025: "Have protein-ligand cofolding methods moved beyond memorisation?"

Main Similarity Metric:
    SuCOS-pocket = SuCOS × (pocket_qcov / 100)
    
    Where:
    - SuCOS: Shape/pharmacophore overlap after ligand-to-ligand alignment (using RDKit's rdShapeAlign)
    - pocket_qcov: Pocket query coverage from Foldseek (% of query pocket residues aligned to target)

Additional Metrics:
1. Tanimoto ECFP4: 2D fingerprint similarity using Morgan fingerprints (radius=2) - paper standard
2. Tanimoto RDKit: 2D fingerprint similarity using RDKit topological fingerprints
3. SuCOS Ligand-Aligned: 3D overlap after optimal ligand shape alignment
4. SuCOS Protein-Aligned: 3D overlap after protein superposition (legacy, less reliable per paper)
5. Shape Similarity: Shape Tanimoto after ligand alignment
6. Pocket qcov: Percentage of aligned pocket residues from Foldseek

The key insight from the paper is that protein-aligned approaches are "too sensitive to the 
efficacy of rigid protein superposition", so the recommended approach is:
1. Use Foldseek to identify protein alignments and get pocket_qcov
2. Align ligands independently using RDKit shape alignment
3. Compute SuCOS on ligand-aligned poses
4. Multiply by pocket_qcov to get the final SuCOS-pocket score

Reference:
    Škrinjar et al. (2025). Have protein-ligand cofolding methods moved beyond memorisation?
    bioRxiv 2025.03.20.644243. https://doi.org/10.1101/2025.03.20.644243
"""

import os
import subprocess
import tempfile
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, rdMolAlign, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps

# Configure logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


# ============================================================================
# Feature Map Configuration for SuCOS Scoring
# ============================================================================

FDEF = AllChem.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)

FEAT_MAP_PARAMS = {k: FeatMaps.FeatMapParams() for k in FDEF.GetFeatureFamilies()}

PHARMACOPHORE_FEATURES = (
    "Donor",
    "Acceptor",
    "NegIonizable",
    "PosIonizable",
    "ZnBinder",
    "Aromatic",
    "Hydrophobe",
    "LumpedHydrophobe",
)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SystemInfo:
    """Information about a protein-ligand system."""
    system_id: str
    pdb_id: str
    ligand_chain: str
    protein_chains: List[str]
    ligand_smiles: Optional[str] = None
    sdf_path: Optional[Path] = None
    receptor_path: Optional[Path] = None


@dataclass
class FoldseekAlignment:
    """Result of Foldseek structural alignment."""
    query_id: str
    target_id: str
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3D translation vector
    lddt: float  # Local distance difference test score
    rmsd: float = 0.0
    aligned_length: int = 0
    qcov: float = 0.0  # Query coverage (0-100)


@dataclass
class SimilarityScore:
    """Container for all similarity scores between two systems.
    
    The main similarity score used is sucos_pocket (SuCOS × pocket_qcov),
    following the methodology from Škrinjar et al. 2025:
    "Have protein-ligand cofolding methods moved beyond memorisation?"
    
    SuCOS-pocket measures the overlap of ligand poses within the same pocket,
    where ligands are aligned using RDKit's shape alignment (not protein superposition).
    """
    query_system: str
    target_system: str
    # 2D Fingerprint Similarities
    tanimoto_ecfp4: float = np.nan  # ECFP4 (Morgan radius=2) - paper standard
    tanimoto_rdkit: float = np.nan  # RDKit topological fingerprint
    # 3D Ligand-Aligned Scores (using RDKit shape alignment)
    sucos_ligand_aligned: float = np.nan  # SuCOS after ligand shape alignment
    shape_similarity: float = np.nan  # Shape Tanimoto after alignment
    alignment_rmsd: float = np.nan  # RMSD of ligand alignment
    # Pocket Metrics (from Foldseek)
    pocket_qcov: float = np.nan  # Pocket query coverage (0-100)
    foldseek_lddt: float = np.nan  # Local distance difference test score
    # Combined Score (main metric from paper)
    sucos_pocket: float = np.nan  # SuCOS × (pocket_qcov / 100)
    # Legacy: Protein-aligned SuCOS (less reliable per paper)
    sucos_protein_aligned: float = np.nan


# ============================================================================
# Core Scoring Functions
# ============================================================================

def get_feature_map_score(
    mol_1: Chem.Mol,
    mol_2: Chem.Mol,
    score_mode: FeatMaps.FeatMapScoreMode = FeatMaps.FeatMapScoreMode.All,
) -> float:
    """
    Calculate pharmacophore feature overlap score between two molecules.
    
    Args:
        mol_1: Reference molecule
        mol_2: Probe molecule
        score_mode: Scoring mode for feature maps
        
    Returns:
        Normalized feature overlap score (0-1)
    """
    feat_lists = []
    for molecule in [mol_1, mol_2]:
        raw_feats = FDEF.GetFeaturesForMol(molecule)
        feat_lists.append([
            f for f in raw_feats if f.GetFamily() in PHARMACOPHORE_FEATURES
        ])

    if len(feat_lists[0]) == 0 or len(feat_lists[1]) == 0:
        return 0.0

    feat_maps = [
        FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=FEAT_MAP_PARAMS)
        for x in feat_lists
    ]
    feat_maps[0].scoreMode = score_mode

    score = feat_maps[0].ScoreFeats(feat_lists[1])
    return score / min(feat_maps[0].GetNumFeatures(), len(feat_lists[1]))


def get_sucos_score(
    mol_1: Chem.Mol,
    mol_2: Chem.Mol,
    score_mode: FeatMaps.FeatMapScoreMode = FeatMaps.FeatMapScoreMode.All,
) -> float:
    """
    Calculate SuCOS (Shape and Color Overlap Score) between two molecules.
    
    SuCOS = 0.5 * FeatureScore + 0.5 * (1 - Protrusion)
    
    Args:
        mol_1: Reference molecule (stays fixed)
        mol_2: Probe molecule (may be transformed)
        score_mode: Scoring mode for feature maps
        
    Returns:
        SuCOS score (0-1)
    """
    fm_score = get_feature_map_score(mol_1, mol_2, score_mode)
    fm_score = np.clip(fm_score, 0, 1)

    protrude_dist = rdShapeHelpers.ShapeProtrudeDist(
        mol_1, mol_2, allowReordering=False
    )
    protrude_dist = np.clip(protrude_dist, 0, 1)

    return 0.5 * fm_score + 0.5 * (1 - protrude_dist)


def align_molecules_crippen(mol_ref: Chem.Mol, mol_probe: Chem.Mol, iterations: int = 100) -> None:
    """
    Align probe molecule to reference using Crippen O3A algorithm.
    Modifies mol_probe in place.
    """
    crippenO3A = Chem.rdMolAlign.GetCrippenO3A(mol_probe, mol_ref, maxIters=iterations)
    crippenO3A.Align()


def align_molecules(
    reference: Chem.Mol,
    mobile: Chem.Mol,
    max_iters: int = 100,
) -> Tuple[float, float]:
    """
    Align mobile molecule to reference using Crippen O3A alignment.
    
    Args:
        reference: Reference molecule (fixed)
        mobile: Mobile molecule (will be transformed in place)
        max_iters: Max iterations for alignment
        
    Returns:
        Tuple of (shape_tanimoto, rmsd)
        - shape_tanimoto: Shape Tanimoto similarity after alignment
        - rmsd: RMSD of the alignment
    """
    # Use Crippen O3A for alignment
    try:
        o3a = rdMolAlign.GetCrippenO3A(mobile, reference, maxIters=max_iters)
        rmsd = o3a.Align()
        score = o3a.Score()
        
        # Calculate shape Tanimoto after alignment
        shape_tanimoto = rdShapeHelpers.ShapeTanimotoDist(reference, mobile)
        # Convert distance to similarity (1 - distance)
        shape_similarity = 1.0 - shape_tanimoto
        
        return shape_similarity, rmsd
    except Exception as e:
        LOG.warning(f"Alignment failed: {e}")
        return np.nan, np.nan


def calculate_tanimoto_ecfp4(smiles_1: str, smiles_2: str) -> float:
    """
    Calculate Tanimoto similarity using ECFP4 (Morgan) fingerprints.
    
    ECFP4 = Morgan fingerprint with radius 2, which is the standard
    used in the PLI similarity literature.
    
    Args:
        smiles_1: SMILES string of first molecule
        smiles_2: SMILES string of second molecule
        
    Returns:
        Tanimoto similarity (0-1)
    """
    try:
        mol_1 = Chem.MolFromSmiles(smiles_1)
        mol_2 = Chem.MolFromSmiles(smiles_2)
        
        if mol_1 is None or mol_2 is None:
            return np.nan
        
        # ECFP4 = Morgan fingerprint with radius 2
        fp_1 = AllChem.GetMorganFingerprintAsBitVect(mol_1, radius=2, nBits=2048)
        fp_2 = AllChem.GetMorganFingerprintAsBitVect(mol_2, radius=2, nBits=2048)
        
        return DataStructs.TanimotoSimilarity(fp_1, fp_2)
    except Exception as e:
        LOG.warning(f"Failed to calculate ECFP4 Tanimoto similarity: {e}")
        return np.nan


def calculate_tanimoto_rdkit(smiles_1: str, smiles_2: str) -> float:
    """
    Calculate Tanimoto similarity using RDKit topological fingerprints.
    
    Args:
        smiles_1: SMILES string of first molecule
        smiles_2: SMILES string of second molecule
        
    Returns:
        Tanimoto similarity (0-1)
    """
    try:
        mol_1 = Chem.MolFromSmiles(smiles_1)
        mol_2 = Chem.MolFromSmiles(smiles_2)
        
        if mol_1 is None or mol_2 is None:
            return np.nan
            
        fpgen = AllChem.GetRDKitFPGenerator()
        fp_1 = fpgen.GetFingerprint(mol_1)
        fp_2 = fpgen.GetFingerprint(mol_2)
        
        return DataStructs.TanimotoSimilarity(fp_1, fp_2)
    except Exception as e:
        LOG.warning(f"Failed to calculate RDKit Tanimoto similarity: {e}")
        return np.nan


def calculate_tanimoto_similarity(smiles_1: str, smiles_2: str) -> float:
    """Deprecated: Use calculate_tanimoto_ecfp4 instead."""
    return calculate_tanimoto_rdkit(smiles_1, smiles_2)


def apply_transformation_to_molecule(
    mol: Chem.Mol,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Chem.Mol:
    """
    Apply rotation and translation to molecule coordinates.
    
    Args:
        mol: RDKit molecule with 3D conformer
        rotation: 3x3 rotation matrix
        translation: 3D translation vector
        
    Returns:
        New molecule with transformed coordinates
    """
    mol_copy = Chem.Mol(mol)
    conf = mol_copy.GetConformer()
    
    coords = np.array([
        list(conf.GetAtomPosition(i))
        for i in range(mol_copy.GetNumAtoms())
    ])
    
    # Apply transformation: rotated_coords = coords @ R^T + t
    rotated_coords = coords @ rotation.T + translation
    
    for i in range(mol_copy.GetNumAtoms()):
        conf.SetAtomPosition(i, rotated_coords[i].tolist())
    
    return mol_copy


# ============================================================================
# Foldseek Integration
# ============================================================================

class FoldseekRunner:
    """Wrapper for running Foldseek structural alignments with caching."""
    
    def __init__(
        self,
        foldseek_path: str = "foldseek",
        tmp_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.foldseek_path = foldseek_path
        self.tmp_dir = tmp_dir or Path(tempfile.gettempdir())
        self.cache_dir = cache_dir
        
        # In-memory cache of loaded alignments
        self._alignment_cache: Dict[str, pd.DataFrame] = {}
    
    def save_alignments_parquet(
        self,
        alignments: List[FoldseekAlignment],
        output_path: Path,
        pdb_id: str,
    ) -> None:
        """
        Save Foldseek alignments to a parquet file in a format compatible with similarity_scoring.py.
        
        The format includes columns: query_pdb_id, target_pdb_id, query_chain, target_chain, u, t, lddt, rmsd, alnlen, qcov
        
        Args:
            alignments: List of FoldseekAlignment objects
            output_path: Path to save the parquet file
            pdb_id: PDB ID for grouping
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        records = []
        for aln in alignments:
            # Convert rotation matrix to comma-separated string (like original format)
            u_str = ",".join(map(str, aln.rotation.flatten()))
            t_str = ",".join(map(str, aln.translation))
            
            # Parse query/target to extract chain info
            query_parts = aln.query_id.replace(".pdb", "").split("_")
            target_parts = aln.target_id.replace(".pdb", "").split("_")
            
            records.append({
                "query_pdb_id": pdb_id,
                "target_pdb_id": target_parts[0] if target_parts else aln.target_id,
                "query_chain": query_parts[-1] if len(query_parts) > 1 else "A",
                "target_chain": target_parts[-1] if len(target_parts) > 1 else "A",
                "u": u_str,
                "t": t_str,
                "lddt": aln.lddt,
                "rmsd": aln.rmsd,
                "alnlen": aln.aligned_length,
                "qcov": aln.qcov,
            })
        
        if records:
            df = pd.DataFrame(records)
            df.to_parquet(output_path, index=False)
            LOG.info(f"Saved {len(records)} alignments to {output_path}")
    
    def load_alignments_parquet(self, parquet_path: Path) -> pd.DataFrame:
        """
        Load cached Foldseek alignments from a parquet file.
        
        Args:
            parquet_path: Path to the parquet file
            
        Returns:
            DataFrame with alignment data
        """
        cache_key = str(parquet_path)
        if cache_key in self._alignment_cache:
            return self._alignment_cache[cache_key]
        
        if not parquet_path.exists():
            return pd.DataFrame()
        
        df = pd.read_parquet(parquet_path)
        self._alignment_cache[cache_key] = df
        return df
    
    def get_cached_alignment(
        self,
        query_pdb_id: str,
        target_pdb_id: str,
        query_chains: Optional[List[str]] = None,
        target_chains: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
    ) -> Optional[FoldseekAlignment]:
        """
        Try to load alignment from cached parquet file.
        
        Args:
            query_pdb_id: Query PDB ID (e.g., "4grb")
            target_pdb_id: Target PDB ID (e.g., "6gra")
            query_chains: Optional list of query protein chains to filter
            target_chains: Optional list of target protein chains to filter
            cache_dir: Directory containing cached alignment parquet files
            
        Returns:
            FoldseekAlignment if found, None otherwise
        """
        cache_dir = cache_dir or self.cache_dir
        if cache_dir is None:
            return None
        
        parquet_path = cache_dir / f"{query_pdb_id}.parquet"
        df = self.load_alignments_parquet(parquet_path)
        
        if df.empty:
            return None
        
        # Filter by target PDB ID
        mask = df["target_pdb_id"] == target_pdb_id
        
        # Optionally filter by chains
        if query_chains and "query_chain" in df.columns:
            mask &= df["query_chain"].isin(query_chains)
        if target_chains and "target_chain" in df.columns:
            mask &= df["target_chain"].isin(target_chains)
        
        filtered = df[mask]
        if filtered.empty:
            return None
        
        # Get best alignment by LDDT
        best = filtered.sort_values("lddt", ascending=False).iloc[0]
        
        # Parse rotation and translation
        rotation = np.array([float(x) for x in best["u"].split(",")]).reshape(3, 3)
        translation = np.array([float(x) for x in best["t"].split(",")])
        
        return FoldseekAlignment(
            query_id=f"{query_pdb_id}_{best.get('query_chain', 'A')}",
            target_id=f"{target_pdb_id}_{best.get('target_chain', 'A')}",
            rotation=rotation,
            translation=translation,
            lddt=best["lddt"],
            rmsd=best.get("rmsd", 0.0),
            aligned_length=best.get("alnlen", 0),
            qcov=best.get("qcov", 0.0),
        )
        
    def run_easy_search(
        self,
        query_pdb: Path,
        target_pdb: Path,
        output_dir: Optional[Path] = None,
        save_to_cache: bool = True,
    ) -> Optional[FoldseekAlignment]:
        """
        Run Foldseek easy-search between two PDB structures.
        
        Args:
            query_pdb: Path to query PDB file
            target_pdb: Path to target PDB file
            output_dir: Directory for output files
            save_to_cache: Whether to save results to cache directory
            
        Returns:
            FoldseekAlignment with rotation/translation matrices, or None if failed
        """
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(dir=self.tmp_dir))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "alignment.tsv"
        tmp_dir = output_dir / "tmp"
        
        # Format string to get rotation (u) and translation (t) vectors
        # qcov = query coverage, percentage of query residues that are aligned
        format_str = "query,target,u,t,lddt,rmsd,alnlen,qcov"
        
        cmd = [
            self.foldseek_path,
            "easy-search",
            str(query_pdb),
            str(target_pdb),
            str(output_file),
            str(tmp_dir),
            "--format-output", format_str,
            "-e", "inf",  # No E-value cutoff
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            
            if result.returncode != 0:
                LOG.warning(f"Foldseek failed: {result.stderr}")
                return None
            
            alignment = self._parse_alignment_result(output_file)
            
            # Save to cache if requested
            if save_to_cache and alignment is not None and self.cache_dir is not None:
                # Extract PDB ID from the query path (e.g., "receptor.pdb" -> use parent dir name)
                query_pdb_id = query_pdb.parent.name[:4]  # First 4 chars of system_id
                self.save_alignments_parquet(
                    [alignment],
                    self.cache_dir / f"{query_pdb_id}.parquet",
                    query_pdb_id,
                )
            
            return alignment
            
        except subprocess.TimeoutExpired:
            LOG.warning("Foldseek timed out")
            return None
        except Exception as e:
            LOG.warning(f"Foldseek error: {e}")
            return None
    
    def _parse_alignment_result(self, output_file: Path) -> Optional[FoldseekAlignment]:
        """Parse Foldseek output file to extract alignment information."""
        if not output_file.exists():
            return None
            
        try:
            with open(output_file, 'r') as f:
                line = f.readline().strip()
                
            if not line:
                return None
                
            parts = line.split('\t')
            if len(parts) < 7:
                return None
            
            query_id = parts[0]
            target_id = parts[1]
            u_str = parts[2]  # Rotation matrix as comma-separated values
            t_str = parts[3]  # Translation vector as comma-separated values
            lddt = float(parts[4])
            rmsd = float(parts[5])
            alnlen = int(parts[6])
            qcov = float(parts[7]) if len(parts) > 7 else 0.0  # Query coverage
            
            # Parse rotation matrix (9 values -> 3x3)
            rotation = np.array([float(x) for x in u_str.split(',')]).reshape(3, 3)
            
            # Parse translation vector (3 values)
            translation = np.array([float(x) for x in t_str.split(',')])
            
            return FoldseekAlignment(
                query_id=query_id,
                target_id=target_id,
                rotation=rotation,
                translation=translation,
                lddt=lddt,
                rmsd=rmsd,
                aligned_length=alnlen,
                qcov=qcov,
            )
            
        except Exception as e:
            LOG.warning(f"Failed to parse Foldseek output: {e}")
            return None


# ============================================================================
# Main Similarity Scorer Class
# ============================================================================

class PLISimilarityScorer:
    """
    Protein-Ligand Interaction Similarity Scorer.
    
    Computes similarity between protein-ligand systems using:
    1. Tanimoto (2D fingerprint similarity)
    2. SuCOS Protein-Aligned (3D overlap after protein superposition)
    3. SuCOS Ligand-Aligned (3D overlap after optimal ligand alignment)
    """
    
    def __init__(
        self,
        systems_dir: Path,
        annotation_table: Optional[pd.DataFrame] = None,
        foldseek_path: str = "foldseek",
        output_dir: Optional[Path] = None,
        foldseek_cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the scorer.
        
        Args:
            systems_dir: Directory containing system folders (e.g., /mnt/.../systems/)
            annotation_table: DataFrame with system annotations (matched_annotation_table)
            foldseek_path: Path to foldseek executable
            output_dir: Directory for output files and cached alignments
            foldseek_cache_dir: Directory for cached Foldseek alignments (parquet files)
        """
        self.systems_dir = Path(systems_dir)
        self.annotation_table = annotation_table
        self.output_dir = Path(output_dir) if output_dir else Path("similarity_scores")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Foldseek cache directory
        self.foldseek_cache_dir = Path(foldseek_cache_dir) if foldseek_cache_dir else self.output_dir / "foldseek_cache"
        self.foldseek_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.foldseek = FoldseekRunner(
            foldseek_path=foldseek_path,
            cache_dir=self.foldseek_cache_dir,
        )
        self.fpgen = AllChem.GetRDKitFPGenerator()
        
        # Cache for loaded molecules and fingerprints
        self._mol_cache: Dict[str, Chem.Mol] = {}
        self._fp_cache: Dict[str, Any] = {}
        self._system_info_cache: Dict[str, SystemInfo] = {}
        
    def get_system_info(self, system_id: str) -> Optional[SystemInfo]:
        """
        Get system information from annotation table or directory structure.
        
        Args:
            system_id: System ID (e.g., "4grb__1__1.A__1.C")
            
        Returns:
            SystemInfo object or None if not found
        """
        if system_id in self._system_info_cache:
            return self._system_info_cache[system_id]
        
        # Parse system_id: pdbid__biounit__protein_chains__ligand_chain
        parts = system_id.split("__")
        if len(parts) < 4:
            LOG.warning(f"Invalid system_id format: {system_id}")
            return None
            
        pdb_id = parts[0]
        # Handle multiple protein chains separated by underscore
        protein_chain_part = parts[2]
        ligand_chain = parts[3]
        
        # Protein chains can be like "1.A_1.B" or just "1.A"
        protein_chains = protein_chain_part.split("_")
        
        # Construct paths
        system_dir = self.systems_dir / system_id
        ligand_files_dir = system_dir / "ligand_files"
        receptor_path = system_dir / "receptor.pdb"
        
        # Handle multi-ligand systems (e.g., "1.B_1.C" -> ["1.B.sdf", "1.C.sdf"])
        # First try the combined name, then individual files
        sdf_path = ligand_files_dir / f"{ligand_chain}.sdf"
        if not sdf_path.exists():
            # Try to find individual ligand files
            ligand_parts = ligand_chain.split("_")
            individual_sdfs = [ligand_files_dir / f"{part}.sdf" for part in ligand_parts]
            if all(p.exists() for p in individual_sdfs):
                # Store the first one as sdf_path, we'll handle multi-ligand in load_ligand_mol
                sdf_path = individual_sdfs[0]
            else:
                sdf_path = None
        
        # Get SMILES from annotation table if available
        ligand_smiles = None
        if self.annotation_table is not None and len(self.annotation_table) > 0:
            if "system_id" in self.annotation_table.columns:
                mask = self.annotation_table["system_id"] == system_id
                if mask.any():
                    row = self.annotation_table[mask].iloc[0]
                    ligand_smiles = row.get("ligand_rdkit_canonical_smiles")
        
        info = SystemInfo(
            system_id=system_id,
            pdb_id=pdb_id,
            ligand_chain=ligand_chain,
            protein_chains=protein_chains,
            ligand_smiles=ligand_smiles,
            sdf_path=sdf_path if sdf_path is not None and sdf_path.exists() else None,
            receptor_path=receptor_path if receptor_path.exists() else None,
        )
        
        self._system_info_cache[system_id] = info
        return info
    
    def load_ligand_mol(self, system_id: str) -> Optional[Chem.Mol]:
        """Load ligand molecule from SDF file(s).
        
        For multi-ligand systems (e.g., ligand_chain='1.B_1.C'), this will
        load all individual SDF files and combine them into a single molecule.
        """
        cache_key = f"mol_{system_id}"
        if cache_key in self._mol_cache:
            return self._mol_cache[cache_key]
        
        info = self.get_system_info(system_id)
        if info is None:
            return None
        
        try:
            system_dir = self.systems_dir / system_id
            ligand_files_dir = system_dir / "ligand_files"
            ligand_chain = info.ligand_chain
            
            # Check if it's a multi-ligand system
            if "_" in ligand_chain:
                # Load and combine multiple ligand files
                ligand_parts = ligand_chain.split("_")
                mols = []
                for part in ligand_parts:
                    sdf_path = ligand_files_dir / f"{part}.sdf"
                    if sdf_path.exists():
                        mol = Chem.MolFromMolFile(str(sdf_path))
                        if mol is not None:
                            mols.append(mol)
                
                if not mols:
                    return None
                
                # Combine molecules into one
                if len(mols) == 1:
                    combined = mols[0]
                else:
                    combined = mols[0]
                    for m in mols[1:]:
                        combined = Chem.CombineMols(combined, m)
                
                # Initialize ring info for combined molecule (needed for alignment)
                Chem.FastFindRings(combined)
                
                self._mol_cache[cache_key] = combined
                return combined
            else:
                # Single ligand file
                if info.sdf_path is None:
                    return None
                mol = Chem.MolFromMolFile(str(info.sdf_path))
                if mol is not None:
                    self._mol_cache[cache_key] = mol
                return mol
                
        except Exception as e:
            LOG.warning(f"Failed to load ligand for {system_id}: {e}")
            return None
    
    def get_fingerprint(self, system_id: str) -> Optional[Any]:
        """Get RDKit fingerprint for a ligand."""
        if system_id in self._fp_cache:
            return self._fp_cache[system_id]
        
        info = self.get_system_info(system_id)
        if info is None:
            return None
        
        try:
            if info.ligand_smiles:
                mol = Chem.MolFromSmiles(info.ligand_smiles)
            else:
                mol = self.load_ligand_mol(system_id)
            
            if mol is None:
                return None
            
            fp = self.fpgen.GetFingerprint(mol)
            self._fp_cache[system_id] = fp
            return fp
        except Exception as e:
            LOG.warning(f"Failed to get fingerprint for {system_id}: {e}")
            return None
    
    def get_ecfp4_fingerprint(self, system_id: str) -> Optional[Any]:
        """Get ECFP4 (Morgan radius=2) fingerprint for a ligand - paper standard."""
        cache_key = f"ecfp4_{system_id}"
        if cache_key in self._fp_cache:
            return self._fp_cache[cache_key]
        
        info = self.get_system_info(system_id)
        if info is None:
            return None
        
        try:
            if info.ligand_smiles:
                mol = Chem.MolFromSmiles(info.ligand_smiles)
            else:
                mol = self.load_ligand_mol(system_id)
            
            if mol is None:
                return None
            
            # ECFP4 = Morgan fingerprint with radius 2
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            self._fp_cache[cache_key] = fp
            return fp
        except Exception as e:
            LOG.warning(f"Failed to get ECFP4 fingerprint for {system_id}: {e}")
            return None
    
    def compute_tanimoto_ecfp4(self, system_1: str, system_2: str) -> float:
        """Compute ECFP4 Tanimoto similarity between two systems' ligands (paper standard)."""
        fp_1 = self.get_ecfp4_fingerprint(system_1)
        fp_2 = self.get_ecfp4_fingerprint(system_2)
        
        if fp_1 is None or fp_2 is None:
            return np.nan
        
        return DataStructs.TanimotoSimilarity(fp_1, fp_2)
    
    def compute_tanimoto_rdkit(self, system_1: str, system_2: str) -> float:
        """Compute RDKit Tanimoto similarity between two systems' ligands."""
        fp_1 = self.get_fingerprint(system_1)
        fp_2 = self.get_fingerprint(system_2)
        
        if fp_1 is None or fp_2 is None:
            return np.nan
        
        return DataStructs.TanimotoSimilarity(fp_1, fp_2)
    
    def compute_tanimoto(self, system_1: str, system_2: str) -> float:
        """Deprecated: Use compute_tanimoto_ecfp4 instead."""
        return self.compute_tanimoto_rdkit(system_1, system_2)
    
    def compute_sucos_protein_aligned(
        self,
        system_1: str,
        system_2: str,
        alignment: Optional[FoldseekAlignment] = None,
        use_cache: bool = True,
    ) -> Tuple[float, Optional[FoldseekAlignment]]:
        """
        Compute SuCOS score after aligning proteins via Foldseek.
        
        This answers: "Do the ligands overlap when the proteins are superimposed?"
        
        Args:
            system_1: Query system ID (reference)
            system_2: Target system ID (will be transformed)
            alignment: Pre-computed Foldseek alignment, or None to compute
            use_cache: Whether to try loading from/saving to cache
            
        Returns:
            Tuple of (sucos_score, alignment_used)
        """
        info_1 = self.get_system_info(system_1)
        info_2 = self.get_system_info(system_2)
        
        if info_1 is None or info_2 is None:
            return np.nan, None
        
        if info_1.receptor_path is None or info_2.receptor_path is None:
            return np.nan, None
        
        # Try to load alignment from cache first
        if alignment is None and use_cache:
            alignment = self.foldseek.get_cached_alignment(
                query_pdb_id=info_1.pdb_id,
                target_pdb_id=info_2.pdb_id,
                query_chains=info_1.protein_chains,
                target_chains=info_2.protein_chains,
            )
        
        # Run Foldseek if alignment not found in cache
        if alignment is None:
            alignment = self.foldseek.run_easy_search(
                info_1.receptor_path,
                info_2.receptor_path,
                output_dir=self.output_dir / "foldseek_runs" / f"{system_1}_vs_{system_2}",
                save_to_cache=use_cache,
            )
        
        if alignment is None:
            return np.nan, None
        
        # Load ligands
        mol_1 = self.load_ligand_mol(system_1)
        mol_2 = self.load_ligand_mol(system_2)
        
        if mol_1 is None or mol_2 is None:
            return np.nan, alignment
        
        # Apply protein alignment transformation to target ligand
        try:
            mol_2_transformed = apply_transformation_to_molecule(
                mol_2,
                alignment.rotation,
                alignment.translation,
            )
            
            sucos = get_sucos_score(mol_1, mol_2_transformed)
            return sucos, alignment
        except Exception as e:
            LOG.warning(f"Failed to compute protein-aligned SuCOS: {e}")
            return np.nan, alignment
    
    def compute_sucos_ligand_aligned(
        self,
        system_1: str,
        system_2: str,
    ) -> Tuple[float, float, float]:
        """
        Compute SuCOS score after optimal ligand alignment.
        
        This answers: "Do the ligands look alike in 3D when optimally overlaid?"
        
        Args:
            system_1: Query system ID (reference)
            system_2: Target system ID (will be aligned)
            
        Returns:
            Tuple of (sucos_score, shape_similarity, alignment_rmsd)
        """
        mol_1 = self.load_ligand_mol(system_1)
        mol_2 = self.load_ligand_mol(system_2)
        
        if mol_1 is None or mol_2 is None:
            return np.nan, np.nan, np.nan
        
        try:
            # Make copies since alignment modifies the molecule
            mol_1_copy = Chem.Mol(mol_1)
            mol_2_copy = Chem.Mol(mol_2)
            
            # Align mol_2 to mol_1
            shape_sim, rmsd = align_molecules(mol_1_copy, mol_2_copy)
            
            # Compute SuCOS after alignment
            sucos = get_sucos_score(mol_1_copy, mol_2_copy)
            
            return sucos, shape_sim, rmsd
        except Exception as e:
            LOG.warning(f"Failed to compute ligand-aligned SuCOS: {e}")
            return np.nan, np.nan, np.nan
    
    def score_pair(
        self,
        system_1: str,
        system_2: str,
        compute_protein_aligned: bool = True,
    ) -> SimilarityScore:
        """
        Compute all similarity scores between two systems.
        
        The main metric is sucos_pocket = SuCOS × (pocket_qcov / 100),
        following Škrinjar et al. 2025 "Have protein-ligand cofolding methods
        moved beyond memorisation?"
        
        Args:
            system_1: Query system ID
            system_2: Target system ID
            compute_protein_aligned: Whether to run Foldseek for protein alignment
            
        Returns:
            SimilarityScore with all computed metrics
        """
        score = SimilarityScore(
            query_system=system_1,
            target_system=system_2,
        )
        
        alignment = None
        
        # 1. Tanimoto (2D) - Both ECFP4 (paper standard) and RDKit
        score.tanimoto_ecfp4 = self.compute_tanimoto_ecfp4(system_1, system_2)
        score.tanimoto_rdkit = self.compute_tanimoto_rdkit(system_1, system_2)
        
        # 2. SuCOS Protein-Aligned (3D) - Legacy, less reliable per paper
        if compute_protein_aligned:
            sucos_protein, alignment = self.compute_sucos_protein_aligned(system_1, system_2)
            score.sucos_protein_aligned = sucos_protein
            if alignment is not None:
                score.foldseek_lddt = alignment.lddt
                score.pocket_qcov = alignment.qcov  # Pocket query coverage
        
        # 3. SuCOS Ligand-Aligned (3D) - Main approach per paper
        sucos_ligand, shape_sim, rmsd = self.compute_sucos_ligand_aligned(system_1, system_2)
        score.sucos_ligand_aligned = sucos_ligand
        score.shape_similarity = shape_sim
        score.alignment_rmsd = rmsd
        
        # 4. Combined Score: SuCOS-pocket = SuCOS × (pocket_qcov / 100)
        # This is the MAIN similarity metric from the paper
        if not np.isnan(sucos_ligand) and not np.isnan(score.pocket_qcov):
            score.sucos_pocket = sucos_ligand * (score.pocket_qcov / 100.0)
        
        return score
    
    def score_batch(
        self,
        system_pairs: List[Tuple[str, str]],
        compute_protein_aligned: bool = True,
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        Score multiple system pairs.
        
        Args:
            system_pairs: List of (query_system, target_system) tuples
            compute_protein_aligned: Whether to compute protein-aligned scores
            progress: Whether to show progress bar
            
        Returns:
            DataFrame with similarity scores
        """
        results = []
        
        iterator = system_pairs
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(system_pairs, desc="Scoring pairs")
            except ImportError:
                pass
        
        for system_1, system_2 in iterator:
            score = self.score_pair(system_1, system_2, compute_protein_aligned)
            results.append({
                "query_system": score.query_system,
                "target_system": score.target_system,
                "tanimoto_ecfp4": score.tanimoto_ecfp4,
                "tanimoto_rdkit": score.tanimoto_rdkit,
                "sucos_ligand_aligned": score.sucos_ligand_aligned,
                "shape_similarity": score.shape_similarity,
                "alignment_rmsd": score.alignment_rmsd,
                "pocket_qcov": score.pocket_qcov,
                "foldseek_lddt": score.foldseek_lddt,
                "sucos_pocket": score.sucos_pocket,  # Main metric from paper
                "sucos_protein_aligned": score.sucos_protein_aligned,  # Legacy
            })
        
        return pd.DataFrame(results)


# ============================================================================
# Utility Functions
# ============================================================================

def load_annotation_table(path: Path) -> pd.DataFrame:
    """Load the matched annotation table."""
    return pd.read_parquet(path)


def get_systems_by_pdb(annotation_table: pd.DataFrame, pdb_id: str) -> List[str]:
    """Get all system IDs for a given PDB ID."""
    mask = annotation_table["system_id"].str.startswith(pdb_id)
    return annotation_table[mask]["system_id"].tolist()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Example usage of the PLISimilarityScorer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute PLI similarity scores")
    parser.add_argument("--systems-dir", type=Path, required=True,
                       help="Directory containing system folders")
    parser.add_argument("--annotation-table", type=Path, required=True,
                       help="Path to matched_annotation_table.parquet")
    parser.add_argument("--query-system", type=str, required=True,
                       help="Query system ID")
    parser.add_argument("--target-system", type=str, required=True,
                       help="Target system ID")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output file for scores")
    
    args = parser.parse_args()
    
    # Load annotation table
    annotation_table = load_annotation_table(args.annotation_table)
    
    # Create scorer
    scorer = PLISimilarityScorer(
        systems_dir=args.systems_dir,
        annotation_table=annotation_table,
    )
    
    # Score the pair
    score = scorer.score_pair(args.query_system, args.target_system)
    
    print(f"\nSimilarity Scores: {args.query_system} vs {args.target_system}")
    print(f"\n=== Main Metric (from paper) ===")
    print(f"  SuCOS-pocket:            {score.sucos_pocket:.4f}  (SuCOS × pocket_qcov/100)")
    print(f"\n=== 2D Fingerprint Similarities ===")
    print(f"  Tanimoto (ECFP4):        {score.tanimoto_ecfp4:.4f}  (paper standard)")
    print(f"  Tanimoto (RDKit):        {score.tanimoto_rdkit:.4f}")
    print(f"\n=== 3D Ligand-Aligned Scores ===")
    print(f"  SuCOS Ligand-Aligned:    {score.sucos_ligand_aligned:.4f}")
    print(f"  Shape Similarity:        {score.shape_similarity:.4f}")
    print(f"  Alignment RMSD:          {score.alignment_rmsd:.4f}")
    print(f"\n=== Pocket Metrics (Foldseek) ===")
    print(f"  Pocket qcov:             {score.pocket_qcov:.4f}")
    print(f"  Foldseek LDDT:           {score.foldseek_lddt:.4f}")
    print(f"\n=== Legacy (protein-aligned) ===")
    print(f"  SuCOS Protein-Aligned:   {score.sucos_protein_aligned:.4f}")
    
    if args.output:
        df = pd.DataFrame([{
            "query_system": score.query_system,
            "target_system": score.target_system,
            "tanimoto_ecfp4": score.tanimoto_ecfp4,
            "tanimoto_rdkit": score.tanimoto_rdkit,
            "sucos_ligand_aligned": score.sucos_ligand_aligned,
            "shape_similarity": score.shape_similarity,
            "alignment_rmsd": score.alignment_rmsd,
            "pocket_qcov": score.pocket_qcov,
            "foldseek_lddt": score.foldseek_lddt,
            "sucos_pocket": score.sucos_pocket,
            "sucos_protein_aligned": score.sucos_protein_aligned,
        }])
        df.to_parquet(args.output, index=False)
        print(f"\nScores saved to {args.output}")


if __name__ == "__main__":
    main()
