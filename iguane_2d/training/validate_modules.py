#!/usr/bin/env python3
"""
Quick validation script for modular training implementation
Tests that all modules import correctly and basic functionality works
"""

import sys
from pathlib import Path

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("VALIDATING MODULAR TRAINING IMPLEMENTATION")
print("=" * 80)


def test_imports():
    """Test that all modules import successfully"""
    print("\n[1/8] Testing imports...")
    
    try:
        import config
        print("  ✓ config.py imports successfully")
    except Exception as e:
        print(f"  ✗ config.py failed: {e}")
        return False
    
    try:
        import gpu_utils
        print("  ✓ gpu_utils.py imports successfully")
    except Exception as e:
        print(f"  ✗ gpu_utils.py failed: {e}")
        return False
    
    try:
        import data_loader
        print("  ✓ data_loader.py imports successfully")
    except Exception as e:
        print(f"  ✗ data_loader.py failed: {e}")
        return False
    
    try:
        import models
        print("  ✓ models.py imports successfully")
    except Exception as e:
        print(f"  ✗ models.py failed: {e}")
        return False
    
    try:
        import losses
        print("  ✓ losses.py imports successfully")
    except Exception as e:
        print(f"  ✗ losses.py failed: {e}")
        return False
    
    try:
        import cyclegan
        print("  ✓ cyclegan.py imports successfully")
    except Exception as e:
        print(f"  ✗ cyclegan.py failed: {e}")
        return False
    
    try:
        import evaluation
        print("  ✓ evaluation.py imports successfully")
    except Exception as e:
        print(f"  ✗ evaluation.py failed: {e}")
        return False
    
    try:
        import train
        print("  ✓ train.py imports successfully")
    except Exception as e:
        print(f"  ✗ train.py failed: {e}")
        return False
    
    return True


def test_config():
    """Test configuration module"""
    print("\n[2/8] Testing config module...")
    
    try:
        from config import TrainingConfig
        
        # Create default config
        config = TrainingConfig()
        
        # Check key attributes
        assert hasattr(config, 'epochs'), "Missing epochs attribute"
        assert hasattr(config, 'batch_size'), "Missing batch_size attribute"
        assert hasattr(config, 'lr_gen'), "Missing lr_gen attribute"
        assert config.epochs == 200, f"Default epochs should be 200, got {config.epochs}"
        
        print("  ✓ TrainingConfig works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False


def test_models():
    """Test model building"""
    print("\n[3/8] Testing models module...")
    
    try:
        import tensorflow as tf
        from models import build_2d_generator, build_2d_discriminator, SpectralNormalization
        
        # Test generator
        gen = build_2d_generator(img_shape=(138, 176, 1), ga_embedding_dim=32)
        assert gen is not None, "Generator is None"
        print("  ✓ Generator builds successfully")
        
        # Test discriminator
        disc = build_2d_discriminator(img_shape=(138, 176, 1))
        assert disc is not None, "Discriminator is None"
        print("  ✓ Discriminator builds successfully")
        
        # Test spectral normalization
        layer = SpectralNormalization(tf.keras.layers.Dense(64))
        assert layer is not None, "SpectralNormalization is None"
        print("  ✓ SpectralNormalization layer works")
        
        return True
    except Exception as e:
        print(f"  ✗ Models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_losses():
    """Test loss functions"""
    print("\n[4/8] Testing losses module...")
    
    try:
        import tensorflow as tf
        from losses import (
            cycle_consistency_loss,
            identity_loss,
            discriminator_loss_smooth,
            generator_loss,
            check_for_nan_loss
        )
        
        # Create dummy tensors
        real = tf.random.normal((4, 138, 176, 1))
        fake = tf.random.normal((4, 138, 176, 1))
        disc_real = tf.random.normal((4, 17, 22, 1))
        disc_fake = tf.random.normal((4, 17, 22, 1))
        
        # Test cycle loss
        cyc_loss = cycle_consistency_loss(real, fake)
        assert cyc_loss is not None, "Cycle loss is None"
        print("  ✓ Cycle consistency loss works")
        
        # Test identity loss
        id_loss = identity_loss(real, fake)
        assert id_loss is not None, "Identity loss is None"
        print("  ✓ Identity loss works")
        
        # Test discriminator loss
        disc_loss = discriminator_loss_smooth(disc_real, disc_fake)
        assert disc_loss is not None, "Discriminator loss is None"
        print("  ✓ Discriminator loss works")
        
        # Test generator loss
        gen_loss = generator_loss(disc_fake)
        assert gen_loss is not None, "Generator loss is None"
        print("  ✓ Generator loss works")
        
        # Test NaN check
        loss_dict = {'gen': 1.0, 'disc': 0.5}
        has_nan = check_for_nan_loss(loss_dict)
        assert has_nan == False, "False NaN detection"
        print("  ✓ NaN detection works")
        
        return True
    except Exception as e:
        print(f"  ✗ Losses test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cyclegan():
    """Test CycleGAN model"""
    print("\n[5/8] Testing cyclegan module...")
    
    try:
        from cyclegan import CycleGAN2D_MultiSite
        
        # Create model
        cyclegan = CycleGAN2D_MultiSite(
            img_shape=(138, 176, 1),
            ga_embedding_dim=32,
            target_sites=['TMC', 'VGH'],
            use_multi_gpu=False
        )
        
        assert cyclegan is not None, "CycleGAN is None"
        assert hasattr(cyclegan, 'gen_site2BCH'), "Missing forward generator"
        assert hasattr(cyclegan, 'gen_BCH2site'), "Missing backward generators"
        assert len(cyclegan.gen_BCH2site) == 2, "Wrong number of backward generators"
        
        print("  ✓ CycleGAN2D_MultiSite instantiates correctly")
        
        # Test compilation
        cyclegan.compile(lr_gen=2e-4, lr_disc=2e-4)
        print("  ✓ CycleGAN compiles successfully")
        
        return True
    except Exception as e:
        print(f"  ✗ CycleGAN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loader():
    """Test data loader (basic structure only)"""
    print("\n[6/8] Testing data_loader module...")
    
    try:
        from data_loader import DataAugmenter
        
        # Test augmenter
        augmenter = DataAugmenter()
        assert augmenter is not None, "DataAugmenter is None"
        print("  ✓ DataAugmenter instantiates correctly")
        
        # Note: Can't test actual loading without data file
        print("  ℹ Skipping data loading test (requires data file)")
        
        return True
    except Exception as e:
        print(f"  ✗ Data loader test failed: {e}")
        return False


def test_evaluation():
    """Test evaluation module (basic structure only)"""
    print("\n[7/8] Testing evaluation module...")
    
    try:
        from evaluation import detect_collapse, generate_quality_report
        
        # Test collapse detection
        import numpy as np
        mean_diff = np.array([0.1, 0.2, 0.05])
        std_diff = np.array([0.05, 0.08, 0.03])
        collapse = detect_collapse(mean_diff, std_diff)
        
        assert isinstance(collapse, (bool, np.bool_)), "Collapse detection returned wrong type"
        print("  ✓ Collapse detection works")
        
        # Test quality report (with dummy history)
        history = {
            'gen_loss': [3.0, 2.5, 2.0],
            'disc_BCH_loss': [0.8, 0.7, 0.6],
            'cycle_loss': [0.5, 0.4, 0.3],
            'identity_loss': [0.1, 0.08, 0.06]
        }
        generate_quality_report(history)
        print("  ✓ Quality report generation works")
        
        return True
    except Exception as e:
        print(f"  ✗ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_utils():
    """Test GPU utilities"""
    print("\n[8/8] Testing gpu_utils module...")
    
    try:
        from gpu_utils import enable_xla_optimization
        
        # Test XLA enablement (doesn't fail even if no GPU)
        enable_xla_optimization()
        print("  ✓ XLA optimization enabled")
        
        # Note: Can't fully test GPU config without GPU
        print("  ℹ Skipping GPU configuration test (requires GPU)")
        
        return True
    except Exception as e:
        print(f"  ✗ GPU utils test failed: {e}")
        return False


def main():
    """Run all tests"""
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Models", test_models),
        ("Losses", test_losses),
        ("CycleGAN", test_cyclegan),
        ("Data Loader", test_data_loader),
        ("Evaluation", test_evaluation),
        ("GPU Utils", test_gpu_utils),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} | {name}")
    
    print("=" * 80)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Modular implementation is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == '__main__':
    exit(main())
