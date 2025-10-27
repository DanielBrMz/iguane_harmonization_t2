#!/usr/bin/env python3
"""
CORRECTED Diagnostic Test - Using Universal Forward Generator (IGUANe Style)
Uses gen_site2BCH which harmonizes ANY site → BCH
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import exposure
from scipy.ndimage import sobel

print("=" * 80)
print("CORRECTED DIAGNOSTIC - UNIVERSAL GENERATOR (IGUANe)")
print("=" * 80)


def find_test_data():
    """Find test data file"""
    search_paths = [
        'processed_data_4slice_fixed/test_4slice_data.pkl',
        'processed_data_4slice/test_4slice_data.pkl',
        'test_4slice_data.pkl',
    ]
    
    for path in search_paths:
        if Path(path).exists():
            return path
    return None


def diagnose_output(img, name="Image"):
    """Diagnose image statistics"""
    
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[:, :, 0]
    
    # Check if completely black or white
    if np.all(img == 0):
        print(f"\n{name}:")
        print(f"  ❌ COMPLETELY BLACK (all zeros)")
        print(f"  This means generator failed or weights didn't load")
        return {'failed': True}
    
    if np.all(img == 1):
        print(f"\n{name}:")
        print(f"  ❌ COMPLETELY WHITE (all ones)")
        return {'failed': True}
    
    mean_val = np.mean(img)
    std_val = np.std(img)
    min_val = np.min(img)
    max_val = np.max(img)
    
    # Edge detection
    sx = sobel(img, axis=0)
    sy = sobel(img, axis=1)
    edges = np.sqrt(sx**2 + sy**2)
    edge_strength = np.mean(edges)
    
    # Compression indicators
    is_compressed = std_val < 0.05
    is_gray = (mean_val > 0.4) and (mean_val < 0.6)
    
    print(f"\n{name}:")
    print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print(f"  Range: [{min_val:.4f}, {max_val:.4f}] (span: {max_val-min_val:.4f})")
    print(f"  Edge Strength: {edge_strength:.4f}")
    
    if std_val < 0.001:
        if mean_val < 0.1:
            print(f"  ❌ NEARLY BLACK - Generator not working")
        elif mean_val > 0.9:
            print(f"  ❌ NEARLY WHITE - Generator not working")
        else:
            print(f"  ⚠️  UNIFORM GRAY - Severe compression")
    elif is_compressed and is_gray:
        if edge_strength > 0.01:
            print(f"  ⚠️  CONTRAST COMPRESSED (structure present)")
        else:
            print(f"  ❌ MODE COLLAPSE")
    else:
        print(f"  ✓ Normal output")
    
    return {
        'mean': mean_val,
        'std': std_val,
        'range': max_val - min_val,
        'edge_strength': edge_strength,
        'is_compressed': is_compressed,
        'failed': False
    }


def main():
    print("\n1. FINDING TEST DATA...")
    test_data_path = find_test_data()
    
    if test_data_path is None:
        print("  ❌ Could not find test data!")
        return
    
    print(f"  ✓ Found: {test_data_path}")
    
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    
    test_images = test_data['images'].astype(float) / 255.0
    test_ga = test_data['gestational_age']
    test_sites = test_data['site']
    
    unique_sites = np.unique(test_sites)
    print(f"  ✓ Loaded {len(test_images)} slices from {len(unique_sites)} sites")
    print(f"  Sites: {', '.join(unique_sites)}")
    
    # Find a non-BCH sample
    non_bch_sites = [s for s in unique_sites if 'BCH' not in s]
    if not non_bch_sites:
        print("  ❌ No non-BCH sites in test data")
        return
    
    test_site = non_bch_sites[0]
    site_idx = np.where(test_sites == test_site)[0]
    sample_idx = site_idx[0]
    
    print(f"\n  Using sample from: {test_site}")
    print(f"  GA: {test_ga[sample_idx]:.1f} weeks")
    
    print("\n2. FINDING UNIVERSAL FORWARD GENERATOR...")
    
    weight_dir = Path('weights/cyclegan_2d')
    
    # Look for the UNIVERSAL forward generator
    forward_gen_weight = weight_dir / 'gen_site2BCH_final.weights.h5'
    
    if not forward_gen_weight.exists():
        # Try alternative names
        alternatives = [
            'gen_A2B_final.weights.h5',
            'gen_forward_final.weights.h5',
            'generator_final.weights.h5',
        ]
        
        for alt in alternatives:
            if (weight_dir / alt).exists():
                forward_gen_weight = weight_dir / alt
                break
    
    if not forward_gen_weight.exists():
        print(f"  ❌ Universal forward generator not found!")
        print(f"     Expected: {forward_gen_weight}")
        print(f"\n  Available weights:")
        for f in sorted(weight_dir.glob('*.h5')):
            print(f"    - {f.name}")
        print(f"\n  🔍 DIAGNOSIS:")
        print(f"     The forward generator (gen_site2BCH) should work for ALL sites")
        print(f"     Backward generators (gen_BCH2{site}) are only for training")
        return
    
    print(f"  ✓ Found universal generator: {forward_gen_weight.name}")
    print(f"  This generator should work for ANY input site → BCH")
    
    print("\n3. LOADING GENERATOR...")
    try:
        import tensorflow as tf
        from train_fetal_2d_cyclegan import build_2d_generator
        
        # Build the UNIVERSAL forward generator
        gen_forward = build_2d_generator((138, 176, 1), 16, name='gen_site2BCH')
        gen_forward.load_weights(str(forward_gen_weight))
        print(f"  ✓ Loaded universal forward generator")
        print(f"  This generator harmonizes: ANY_SITE → BCH_CHD")
        
    except Exception as e:
        print(f"  ❌ Error loading generator: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n4. TESTING GENERATOR...")
    original = test_images[sample_idx]
    ga_value = test_ga[sample_idx].reshape(1, 1)
    
    # Check input is valid
    print(f"\n  Input Statistics:")
    print(f"    Mean: {original.mean():.4f}")
    print(f"    Std: {original.std():.4f}")
    print(f"    Range: [{original.min():.4f}, {original.max():.4f}]")
    
    if original.std() < 0.01:
        print(f"    ⚠️  WARNING: Input image has very low contrast!")
        print(f"    This might be a bad sample or preprocessing issue")
    
    # Generate harmonized output
    print(f"\n  Generating harmonized output...")
    img_batch = np.expand_dims(original, axis=0)
    
    try:
        harmonized = gen_forward([img_batch, ga_value], training=False).numpy()[0]
        print(f"    ✓ Generation successful")
    except Exception as e:
        print(f"    ❌ Generation failed: {e}")
        return
    
    print("\n" + "="*80)
    print("DIAGNOSTIC RESULTS")
    print("="*80)
    
    orig_stats = diagnose_output(original, "Original")
    harm_stats = diagnose_output(harmonized, "Harmonized (Universal Generator)")
    
    if harm_stats.get('failed', False):
        print("\n" + "="*80)
        print("🚨 CRITICAL ISSUE DETECTED")
        print("="*80)
        print("\nThe generator is outputting zeros/ones. Possible causes:")
        print("\n1. WEIGHTS NOT LOADED:")
        print("   - Check if gen_site2BCH_final.weights.h5 actually exists")
        print("   - Verify file is not corrupted (should be ~50-200 MB)")
        print("   - Try loading from earlier epoch checkpoint")
        
        print("\n2. ARCHITECTURE MISMATCH:")
        print("   - Generator architecture changed after training")
        print("   - Weight shapes don't match layer shapes")
        print("   - Solution: Use exact same model definition as training")
        
        print("\n3. INPUT PREPROCESSING ERROR:")
        print("   - Check if input is properly normalized to [0, 1]")
        print("   - Verify input is not all black/white")
        
        print("\n4. ACTIVATION FUNCTION ISSUE:")
        print("   - If using tanh, outputs can saturate to -1 or +1")
        print("   - After rescaling: -1 → 0 (black), +1 → 1 (white)")
        print("   - Solution: Switch to sigmoid activation")
        
        print("\n📋 RECOMMENDED ACTIONS:")
        print("   1. Check weight file size:")
        print(f"      ls -lh {forward_gen_weight}")
        print("   2. Try earlier checkpoint (e.g., epoch 150 instead of final)")
        print("   3. Print model summary to verify architecture")
        print("   4. Test with BCH sample (should be identity mapping)")
        
        return
    
    # If generation worked, continue with analysis
    harm_2d = harmonized[:, :, 0] if harmonized.ndim == 3 else harmonized
    orig_2d = original[:, :, 0] if original.ndim == 3 else original
    
    # Apply CLAHE
    if harm_stats['std'] < 0.05:
        print(f"\n  Applying CLAHE enhancement...")
        harm_clahe = exposure.equalize_adapthist(harm_2d, clip_limit=0.03)
        clahe_stats = diagnose_output(harm_clahe, "CLAHE Enhanced")
    
    # Create visualization
    output_dir = Path('diagnostics_universal_gen')
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Images
    axes[0, 0].imshow(orig_2d, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Original ({test_site})\nStd: {orig_stats["std"]:.4f}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(harm_2d, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Harmonized (Universal Gen)\nStd: {harm_stats["std"]:.4f}')
    axes[0, 1].axis('off')
    
    if harm_stats['std'] < 0.05:
        axes[0, 2].imshow(harm_clahe, cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title(f'CLAHE Enhanced\nStd: {clahe_stats["std"]:.4f}')
    else:
        axes[0, 2].imshow(np.abs(orig_2d - harm_2d), cmap='hot', vmin=0, vmax=0.3)
        axes[0, 2].set_title(f'Difference Map')
    axes[0, 2].axis('off')
    
    # Row 2: Analysis
    axes[1, 0].hist(orig_2d.flatten(), bins=50, alpha=0.7, color='blue', density=True)
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlabel('Intensity')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].hist(harm_2d.flatten(), bins=50, alpha=0.7, color='red', density=True)
    axes[1, 1].axvline(harm_stats['mean'], color='k', linestyle='--', linewidth=2)
    axes[1, 1].set_title(f'Harmonized Histogram')
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].grid(alpha=0.3)
    
    # Profile
    center = harm_2d.shape[0] // 2
    axes[1, 2].plot(orig_2d[center, :], label='Original', linewidth=2, alpha=0.7)
    axes[1, 2].plot(harm_2d[center, :], label='Harmonized', linewidth=2, alpha=0.7)
    axes[1, 2].set_title('Central Profile')
    axes[1, 2].set_xlabel('Position')
    axes[1, 2].set_ylabel('Intensity')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.suptitle('Universal Generator Test (IGUANe Style)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'universal_generator_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualization: {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    mae = np.mean(np.abs(orig_2d - harm_2d))
    print(f"\nMAE: {mae:.4f}")
    
    if mae < 0.01:
        print(f"⚠️  Generator is copying input (identity mapping)")
        print(f"   This is expected if input is already from BCH")
        print(f"   For other sites, this suggests generator didn't learn")
    elif mae > 0.5:
        print(f"⚠️  Very large changes - check if this is reasonable")
    else:
        print(f"✓ Reasonable harmonization magnitude")
    
    if harm_stats['std'] < 0.01:
        print(f"\n❌ CRITICAL: Output has zero/near-zero variation")
        print(f"   Generator is broken or weights didn't load")
    elif harm_stats['std'] < 0.05:
        print(f"\n⚠️  Contrast compression detected")
        print(f"   See corrected diagnostic report for fixes")
    else:
        print(f"\n✓ Output has reasonable contrast")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()