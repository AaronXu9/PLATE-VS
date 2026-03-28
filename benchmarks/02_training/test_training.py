"""
Quick Test Script for Training Pipeline

This script runs a quick test to verify the training pipeline works correctly
by training on a small subset of data.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from train_classical_oddt import train_classical


def test_basic_training():
    """Test basic ligand-only training."""
    print("\n" + "="*70)
    print("TEST 1: Basic Ligand-Only Training")
    print("="*70)
    
    trainer = train_classical(
        config_path='../configs/classical_config.yaml',
        registry_path='../../training_data_full/registry.csv',
        output_dir='./test_models/ligand_only',
        use_precomputed_split=True,
        quick_test=True,
        test_samples=500
    )
    
    print("\n✓ Test 1 PASSED: Ligand-only training works")
    return trainer


def test_protein_aware_training():
    """Test protein-aware training."""
    print("\n" + "="*70)
    print("TEST 2: Protein-Aware Training")
    print("="*70)
    
    # First, ensure config has protein features enabled
    import yaml
    
    config_path = '../configs/classical_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Temporarily modify config for protein features
    config['features']['type'] = 'combined'
    config['data']['include_protein_features'] = True
    
    # Save temporary config
    temp_config_path = './test_protein_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    trainer = train_classical(
        config_path=temp_config_path,
        registry_path='../../training_data_full/registry.csv',
        output_dir='./test_models/protein_aware',
        use_precomputed_split=True,
        quick_test=True,
        test_samples=500
    )
    
    # Cleanup
    Path(temp_config_path).unlink()
    
    print("\n✓ Test 2 PASSED: Protein-aware training works")
    return trainer


def test_custom_split():
    """Test custom data split."""
    print("\n" + "="*70)
    print("TEST 3: Custom Data Split")
    print("="*70)
    
    trainer = train_classical(
        config_path='../configs/classical_config.yaml',
        registry_path='../../training_data_full/registry.csv',
        output_dir='./test_models/custom_split',
        use_precomputed_split=False,  # Custom split
        quick_test=True,
        test_samples=500
    )
    
    print("\n✓ Test 3 PASSED: Custom split training works")
    return trainer


def test_model_save_load():
    """Test model saving and loading."""
    print("\n" + "="*70)
    print("TEST 4: Model Save/Load")
    print("="*70)
    
    from models.rf_trainer import RandomForestTrainer
    from features.featurizer import MorganFingerprintFeaturizer
    import numpy as np
    
    # Train a simple model
    np.random.seed(42)
    X = np.random.randn(100, 2048)
    y = np.random.randint(0, 2, 100)
    
    config = {'hyperparameters': {'n_estimators': 10, 'random_state': 42}}
    trainer = RandomForestTrainer(config)
    trainer.train(X, y)
    
    # Save
    save_dir = './test_models/save_load'
    trainer.save_model(save_dir)
    
    # Load
    trainer2 = RandomForestTrainer(config)
    trainer2.load_model(save_dir)
    
    # Test predictions match
    pred1 = trainer.predict(X[:10])
    pred2 = trainer2.predict(X[:10])
    
    assert np.array_equal(pred1, pred2), "Predictions don't match after load!"
    
    print("\n✓ Test 4 PASSED: Model save/load works")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("Running Training Pipeline Tests")
    print("="*70)
    
    tests = [
        ("Basic Training", test_basic_training),
        ("Protein-Aware Training", test_protein_aware_training),
        ("Custom Split", test_custom_split),
        ("Save/Load", test_model_save_load),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test training pipeline')
    parser.add_argument('--test', type=str, choices=['basic', 'protein', 'split', 'save', 'all'],
                       default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    if args.test == 'basic':
        test_basic_training()
    elif args.test == 'protein':
        test_protein_aware_training()
    elif args.test == 'split':
        test_custom_split()
    elif args.test == 'save':
        test_model_save_load()
    else:
        run_all_tests()
