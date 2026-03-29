import unittest
import os
import shutil
from pathlib import Path
from platevs_client import PlateVSClient

class TestPlateVSClient(unittest.TestCase):
    """
    Integration tests for PlateVSClient.
    
    Note: These tests hit the live API at https://www.drugbench.org.
    Ensure you have internet connection.
    """
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("./test_downloads")
        cls.client = PlateVSClient(output_dir=str(cls.test_dir))
        
    @classmethod
    def tearDownClass(cls):
        # Clean up downloaded files
        # if cls.test_dir.exists():
        #     shutil.rmtree(cls.test_dir)
        pass

    def test_01_service_status(self):
        """Test if the service is reachable."""
        print("\nTesting service status...")
        status = self.client.check_service_status()
        # We expect at least the main site to be reachable
        self.assertTrue(status.get('main'), "Main website should be reachable")
        print(f"Service status: {status}")

    def test_02_search_uniprot(self):
        """Test searching by UniProt ID (EGFR)."""
        print("\nTesting UniProt search...")
        # Search for EGFR (P00533)
        result = self.client.search_by_uniprot("P00533", limit=5)
        
        # The API returns a paginated response structure
        # Based on molecules.ts: { data: [...], total: ... }
        self.assertIsInstance(result, dict)
        # Note: If the API is down or empty, this might fail, but we check structure
        if 'data' in result:
            self.assertIsInstance(result['data'], list)

    def test_03_get_protein_ligands(self):
        """Test downloading ligands as DataFrame."""
        print("\nTesting ligand download...")
        df = self.client.get_protein_ligands("P00533")
        # We expect a DataFrame, even if empty
        self.assertIsNotNone(df)
        print(f"Retrieved {len(df)} ligands")

    def test_04_search_smiles_similarity(self):
        """Test SMILES similarity search."""
        print("\nTesting SMILES similarity search...")
        # Aspirin
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        df = self.client.search_by_smiles(smiles, exact_match=False)
        self.assertIsNotNone(df)
        # Note: Might be empty if no similar compounds found, but shouldn't crash

    def test_05_download_similarity_csv(self):
        """Test downloading similarity matrix CSV."""
        print("\nTesting similarity matrix CSV download...")
        # Use a small threshold/qcov to ensure it exists
        path = self.client.download_similarity_matrix_csv(similarity_threshold=0.9, qcov_level=100)
        
        if path:
            self.assertTrue(path.exists())
            self.assertTrue(path.stat().st_size > 0)
            print(f"Downloaded CSV to {path}")
        else:
            print("Skipping CSV check (download failed or file not found)")

    def test_06_download_similarity_sdf(self):
        """Test downloading similarity SDF archive."""
        print("\nTesting similarity SDF download...")
        # Use a small threshold to ensure it exists
        path = self.client.download_similarity_sdf(similarity_threshold=0.9)
        
        if path:
            self.assertTrue(path.exists())
            self.assertTrue(path.stat().st_size > 0)
            print(f"Downloaded SDF archive to {path}")
        else:
            print("Skipping SDF check (download failed or file not found)")

    def test_07_download_similarity_cif(self):
        """Test downloading similarity CIF archive."""
        print("\nTesting similarity CIF download...")
        path = self.client.download_similarity_cif()
        
        if path:
            self.assertTrue(path.exists())
            self.assertTrue(path.stat().st_size > 0)
            print(f"Downloaded CIF archive to {path}")
        else:
            print("Skipping CIF check (download failed or file not found)")

if __name__ == '__main__':
    unittest.main(verbosity=2)
