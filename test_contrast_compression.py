#!/usr/bin/env python3
"""
Quick Diagnostic Test for Contrast Compression
Tests if harmonized outputs suffer from contrast compression and whether
CLAHE/contrast stretching can reveal hidden structure.
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
print("CONTRAST COMPRESSION DIAGNOSTIC TEST")
print("=" * 80)


def diagnose_output(harmonized_img, name="Harmonized"):
    """Diagnose a single harmonized output"""
    
    if harmonized_img.shape[-1] == 1:
        img_2d = harmonized_img[:, :, 0]
    else:
        img_2d = harmonized_img
    
    # Statistics
    mean_val = np.mean(img_2d)
    std_val = np.std(img_2d)
    min_val = np.min(img_2d)
    max_val = np.max(img_2d)
    
    # Edge detection
    sx = sobel(img_2d, axis=0)
    sy = sobel(img_2d, axis=1)
    edges = np.sqrt(sx**2 + sy**2)
    edge_strength = np.mean(edges)
    edge_pixels_pct = (edges > 0.01).sum() / edges.size * 100
    
    # Compression indicators
    is_compressed = std_val < 0.05
    is_gray = (mean_val > 0.45) and (mean_val < 0.55)
    range_narrow = (max_val - min_val) < 0.2
    
    print(f"\n{name} Diagnostics:")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Std:  {std_val:.4f}")
    print(f"  Min:  {min_val:.4f}")
    print(f"  Max:  {max_val:.4f}")
    print(f"  Range: {max_val - min_val:.4f}")
    print(f"  Edge Strength: {edge_strength:.4f}")
    print(f"  Pixels with edges: {edge_pixels_pct:.2f}%")
    print(f"\n  Compression Indicators:")
    print(f"    Low std (< 0.05): {'⚠️  YES' if is_compressed else '✓ NO'}")
    print(f"    Centered at gray: {'⚠️  YES' if is_gray else '✓ NO'}")
    print(f"    Narrow range: {'⚠️  YES' if range_narrow else '✓ NO'}")
    
    if is_compressed or (is_gray and range_narrow):
        if edge_strength > 0.01:
            print(f"\n  ⚠️  DIAGNOSIS: Structure detected but SEVERELY COMPRESSED")
            print(f"      Recommendation: Apply CLAHE or retrain with sigmoid activation")
        else:
            print(f"\n  ❌ DIAGNOSIS: Possible mode collapse or complete compression")
            print(f"      Recommendation: Retrain with sigmoid + adjusted losses")
    else:
        print(f"\n  ✓ DIAGNOSIS: Output appears normal")
    
    return {
        'mean': mean_val,
        'std': std_val,
        'range': max_val - min_val,
        'edge_strength': edge_strength,
        'is_compressed': is_compressed
    }


def test_enhancement_methods(harmonized_img, original_img):
    """Test different enhancement methods"""
    
    print(f"\n{'='*80}")
    print("TESTING ENHANCEMENT METHODS")
    print(f"{'='*80}")
    
    if harmonized_img.shape[-1] == 1:
        harm_2d = harmonized_img[:, :, 0]
    else:
        harm_2d = harmonized_img
    
    if original_img.shape[-1] == 1:
        orig_2d = original_img[:, :, 0]
    else:
        orig_2d = original_img
    
    # Method 1: CLAHE
    harm_clahe = exposure.equalize_adapthist(harm_2d, clip_limit=0.03)
    diagnose_output(harm_clahe[:, :, np.newaxis], "CLAHE Enhanced")
    mae_clahe = np.mean(np.abs(harm_clahe - orig_2d))
    print(f"    MAE vs Original: {mae_clahe:.4f}")
    
    # Method 2: Contrast Stretching
    mask = (harm_2d > 0.05) & (harm_2d < 0.95)
    if mask.sum() > 100:
        p_low = np.percentile(harm_2d[mask], 2)
        p_high = np.percentile(harm_2d[mask], 98)
        if p_high > p_low:
            harm_stretched = np.clip((harm_2d - p_low) / (p_high - p_low), 0, 1)
        else:
            harm_stretched = harm_2d
    else:
        harm_stretched = harm_2d
    
    diagnose_output(harm_stretched[:, :, np.newaxis], "Contrast Stretched")
    mae_stretched = np.mean(np.abs(harm_stretched - orig_2d))
    print(f"    MAE vs Original: {mae_stretched:.4f}")
    
    return harm_clahe, harm_stretched


def create_comparison_figure(original, harmonized_raw, harmonized_clahe, 
                            harmonized_stretched, output_path):
    """Create comprehensive comparison figure"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Squeeze channel dimension
    if original.shape[-1] == 1:
        original = original[:, :, 0]
    if harmonized_raw.shape[-1] == 1:
        harmonized_raw = harmonized_raw[:, :, 0]
    
    # Row 1: Images
    axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Original\nStd: {np.std(original):.4f}', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(harmonized_raw, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Raw Harmonized\nStd: {np.std(harmonized_raw):.4f}', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(harmonized_clahe, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'CLAHE Enhanced\nStd: {np.std(harmonized_clahe):.4f}', fontsize=12)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(harmonized_stretched, cmap='gray', vmin=0, vmax=1)
    axes[0, 3].set_title(f'Contrast Stretched\nStd: {np.std(harmonized_stretched):.4f}', fontsize=12)
    axes[0, 3].axis('off')
    
    # Row 2: Histograms and profiles
    axes[1, 0].hist(original.flatten(), bins=50, alpha=0.7, color='blue', density=True)
    axes[1, 0].set_title('Original Histogram', fontsize=10)
    axes[1, 0].set_xlabel('Intensity')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].hist(harmonized_raw.flatten(), bins=50, alpha=0.7, color='red', density=True)
    axes[1, 1].axvline(np.mean(harmonized_raw), color='black', linestyle='--', 
                       linewidth=2, label=f'Mean={np.mean(harmonized_raw):.3f}')
    axes[1, 1].set_title('Raw Harmonized Histogram', fontsize=10)
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # Overlay histograms
    axes[1, 2].hist(harmonized_raw.flatten(), bins=50, alpha=0.5, 
                    color='red', label='Raw', density=True)
    axes[1, 2].hist(harmonized_clahe.flatten(), bins=50, alpha=0.5,
                    color='green', label='CLAHE', density=True)
    axes[1, 2].hist(harmonized_stretched.flatten(), bins=50, alpha=0.5,
                    color='purple', label='Stretched', density=True)
    axes[1, 2].set_title('Enhancement Comparison', fontsize=10)
    axes[1, 2].set_xlabel('Intensity')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    # Central profile
    center_row = harmonized_raw.shape[0] // 2
    axes[1, 3].plot(original[center_row, :], label='Original', linewidth=2, alpha=0.7)
    axes[1, 3].plot(harmonized_raw[center_row, :], label='Raw', linewidth=2, alpha=0.7)
    axes[1, 3].plot(harmonized_clahe[center_row, :], label='CLAHE', linewidth=2, alpha=0.7)
    axes[1, 3].set_title('Central Horizontal Profile', fontsize=10)
    axes[1, 3].set_xlabel('Position (pixels)')
    axes[1, 3].set_ylabel('Intensity')
    axes[1, 3].legend()
    axes[1, 3].grid(alpha=0.3)
    
    plt.suptitle('Contrast Compression Test Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved comparison figure: {output_path}")


def main():
    """Run the diagnostic test"""
    
    # Configuration
    test_data_path = 'processed_data_4slice_fixed/test_4slice_data.pkl'
    weight_dir = 'weights/cyclegan_2d'
    output_dir = Path('diagnostics_contrast_compression')
    output_dir.mkdir(exist_ok=True)
    
    # Load test data
    print(f"\nLoading test data from: {test_data_path}")
    try:
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        
        test_images = test_data['images'].astype(float) / 255.0
        test_ga = test_data['gestational_age']
        test_sites = test_data['site']
        
        print(f"  ✓ Loaded {len(test_images)} slices from {len(np.unique(test_sites))} sites")
    except FileNotFoundError:
        print(f"  ❌ Test data not found at: {test_data_path}")
        print("     Please update the path in the script.")
        return
    
    # Find a non-BCH sample
    non_bch_idx = np.where(test_sites != 'BCH_CHD')[0]
    if len(non_bch_idx) == 0:
        print("  ❌ No non-BCH samples found")
        return
    
    sample_idx = non_bch_idx[0]
    sample_site = test_sites[sample_idx]
    print(f"\n  Using sample from: {sample_site}")
    
    # Load generator (you'll need to import your model architecture)
    print(f"\nLoading generator model...")
    try:
        import tensorflow as tf
        from train_fetal_2d_cyclegan import build_2d_generator
        
        # Map site to generator name
        site_to_gen = {
            'BCH_Placenta': 'BCHPlacenta',
            'HBCD_Site5_Arkansas_UNC': 'HBCDSite5ArkansasUNC',
            'HBCD_Site6_Cincinnati': 'HBCDSite6Cincinnati',
            'VGH_Unknown': 'VGHUnknown',
            'VGH_Site4_GE': 'VGHUnknown',
            'dHCP': 'dHCP'
        }
        
        gen_name = site_to_gen.get(sample_site)
        if gen_name is None:
            print(f"  ❌ No generator mapping for site: {sample_site}")
            return
        
        gen = build_2d_generator((138, 176, 1), 16, name=f'gen_BCH2{gen_name}')
        weight_file = Path(weight_dir) / f'gen_BCH2{gen_name}_final.weights.h5'
        
        if not weight_file.exists():
            print(f"  ❌ Weights not found: {weight_file}")
            return
        
        gen.load_weights(str(weight_file))
        print(f"  ✓ Loaded weights: {weight_file.name}")
        
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return
    
    # Generate harmonized output
    print(f"\nGenerating harmonized output...")
    original = test_images[sample_idx]
    ga_value = test_ga[sample_idx].reshape(1, 1)
    img_batch = np.expand_dims(original, axis=0)
    
    harmonized = gen([img_batch, ga_value], training=False).numpy()[0]
    
    # Diagnose raw output
    print(f"\n{'='*80}")
    print("STEP 1: DIAGNOSE RAW HARMONIZED OUTPUT")
    print(f"{'='*80}")
    
    raw_stats = diagnose_output(harmonized, "Raw Harmonized")
    
    # Test enhancement methods
    harm_clahe, harm_stretched = test_enhancement_methods(harmonized, original)
    
    # Create comparison figure
    output_path = output_dir / 'contrast_compression_test.png'
    create_comparison_figure(
        original, harmonized, harm_clahe, harm_stretched, output_path
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if raw_stats['is_compressed']:
        print(f"\n⚠️  CONFIRMED: Output suffers from contrast compression")
        print(f"\n   Raw output statistics:")
        print(f"     - Std: {raw_stats['std']:.4f} (should be > 0.10)")
        print(f"     - Range: {raw_stats['range']:.4f} (should be > 0.5)")
        print(f"     - Edge strength: {raw_stats['edge_strength']:.4f}")
        
        if raw_stats['edge_strength'] > 0.01:
            print(f"\n   ✓ Structure IS present (edge strength = {raw_stats['edge_strength']:.4f})")
            print(f"   ✓ Enhancement methods reveal hidden details")
            print(f"\n   RECOMMENDED FIXES:")
            print(f"     1. Change output activation: tanh → sigmoid")
            print(f"     2. Increase loss weights (adv=2.0, cycle=10.0)")
            print(f"     3. Add contrast loss to penalize low std")
            print(f"     4. OR use CLAHE as post-processing")
        else:
            print(f"\n   ⚠️  Very weak edge strength - possible mode collapse")
            print(f"   RECOMMENDED FIXES:")
            print(f"     1. Complete retrain with sigmoid activation")
            print(f"     2. Add perceptual loss (VGG features)")
            print(f"     3. Increase adversarial loss weight significantly")
    else:
        print(f"\n✓ Output appears normal - no compression detected")
    
    print(f"\n{'='*80}")
    print(f"Test complete! Results saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
