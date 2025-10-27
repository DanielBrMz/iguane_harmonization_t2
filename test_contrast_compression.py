#!/usr/bin/env python3
"""
Auto-detecting Contrast Compression Diagnostic
Works with any site configuration
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import exposure
from scipy.ndimage import sobel

print("=" * 80)
print("AUTO-DETECTING CONTRAST COMPRESSION DIAGNOSTIC")
print("=" * 80)


def find_test_data():
    """Find test data file"""
    search_paths = [
        'processed_data_4slice_fixed/test_4slice_data.pkl',
        'processed_data_4slice/test_4slice_data.pkl',
        'test_4slice_data.pkl',
        'data/test_4slice_data.pkl',
    ]
    
    for path in search_paths:
        if Path(path).exists():
            return path
    
    return None


def find_generators(weight_dir='weights/cyclegan_2d'):
    """Find all available generators"""
    weight_path = Path(weight_dir)
    if not weight_path.exists():
        return {}
    
    generators = {}
    
    # Pattern: gen_BCH2{SITENAME}_final.weights.h5
    for weight_file in weight_path.glob('gen_BCH2*_final.weights.h5'):
        # Extract site name
        name = weight_file.stem
        # Remove 'gen_BCH2' prefix and '_final.weights' suffix
        site_name = name.replace('gen_BCH2', '').replace('_final.weights', '')
        generators[site_name] = str(weight_file)
    
    return generators


def normalize_site_name(site):
    """Normalize site name to match generator names"""
    # Remove special characters and convert to expected format
    normalized = site.replace('_', '').replace('-', '').replace(' ', '')
    return normalized


def diagnose_output(img, name="Image"):
    """Diagnose image statistics"""
    
    if img.ndim == 3:
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        elif img.shape[-1] == 3:
            img = img.mean(axis=2)
    
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
    range_narrow = (max_val - min_val) < 0.2
    
    print(f"\n{name}:")
    print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print(f"  Range: [{min_val:.4f}, {max_val:.4f}] (span: {max_val-min_val:.4f})")
    print(f"  Edge Strength: {edge_strength:.4f}")
    
    if is_compressed and is_gray:
        if edge_strength > 0.01:
            print(f"  ⚠️  CONTRAST COMPRESSED (but structure present!)")
        else:
            print(f"  ❌ POSSIBLE MODE COLLAPSE")
    elif is_compressed:
        print(f"  ⚠️  LOW CONTRAST")
    else:
        print(f"  ✓ Normal contrast")
    
    return {
        'mean': mean_val,
        'std': std_val,
        'range': max_val - min_val,
        'edge_strength': edge_strength,
        'is_compressed': is_compressed,
        'is_gray': is_gray
    }


def apply_clahe(img):
    """Apply CLAHE enhancement"""
    if img.ndim == 3:
        if img.shape[-1] == 1:
            img = img[:, :, 0]
    
    enhanced = exposure.equalize_adapthist(img, clip_limit=0.03)
    return enhanced


def create_visualization(original, harmonized, output_path):
    """Create diagnostic visualization"""
    
    # Ensure 2D
    if original.ndim == 3:
        original = original[:, :, 0] if original.shape[-1] == 1 else original.mean(axis=2)
    if harmonized.ndim == 3:
        harmonized = harmonized[:, :, 0] if harmonized.shape[-1] == 1 else harmonized.mean(axis=2)
    
    # Apply enhancements
    harm_clahe = apply_clahe(harmonized)
    
    # Contrast stretching
    mask = (harmonized > 0.05) & (harmonized < 0.95)
    if mask.sum() > 100:
        p2, p98 = np.percentile(harmonized[mask], (2, 98))
        if p98 > p2:
            harm_stretched = np.clip((harmonized - p2) / (p98 - p2), 0, 1)
        else:
            harm_stretched = harmonized.copy()
    else:
        harm_stretched = harmonized.copy()
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Images
    axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Original\nStd: {np.std(original):.4f}', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(harmonized, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Harmonized (Raw)\nStd: {np.std(harmonized):.4f}', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(harm_clahe, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'CLAHE Enhanced\nStd: {np.std(harm_clahe):.4f}', fontsize=12)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(harm_stretched, cmap='gray', vmin=0, vmax=1)
    axes[0, 3].set_title(f'Contrast Stretched\nStd: {np.std(harm_stretched):.4f}', fontsize=12)
    axes[0, 3].axis('off')
    
    # Row 2: Analysis
    axes[1, 0].hist(original.flatten(), bins=50, alpha=0.7, color='blue', density=True)
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlabel('Intensity')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].hist(harmonized.flatten(), bins=50, alpha=0.7, color='red', density=True)
    axes[1, 1].axvline(np.mean(harmonized), color='k', linestyle='--', linewidth=2)
    axes[1, 1].set_title(f'Harmonized Histogram\n(Mean={np.mean(harmonized):.3f})')
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].grid(alpha=0.3)
    
    # Comparison
    axes[1, 2].hist(harmonized.flatten(), bins=50, alpha=0.5, color='red', label='Raw', density=True)
    axes[1, 2].hist(harm_clahe.flatten(), bins=50, alpha=0.5, color='green', label='CLAHE', density=True)
    axes[1, 2].set_title('Enhancement Comparison')
    axes[1, 2].set_xlabel('Intensity')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    # Profile
    center = harmonized.shape[0] // 2
    axes[1, 3].plot(original[center, :], label='Original', linewidth=2, alpha=0.7)
    axes[1, 3].plot(harmonized[center, :], label='Harmonized', linewidth=2, alpha=0.7)
    axes[1, 3].plot(harm_clahe[center, :], label='CLAHE', linewidth=2, alpha=0.7)
    axes[1, 3].set_title('Central Profile')
    axes[1, 3].set_xlabel('Position')
    axes[1, 3].set_ylabel('Intensity')
    axes[1, 3].legend()
    axes[1, 3].grid(alpha=0.3)
    
    plt.suptitle('Contrast Compression Diagnostic', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualization: {output_path}")
    
    return harm_clahe, harm_stretched


def main():
    print("\n1. FINDING TEST DATA...")
    test_data_path = find_test_data()
    
    if test_data_path is None:
        print("  ❌ Could not find test data!")
        print("  Searched for:")
        print("    - processed_data_4slice_fixed/test_4slice_data.pkl")
        print("    - processed_data_4slice/test_4slice_data.pkl")
        print("\n  Please run this script from your project root directory.")
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
    
    print("\n2. FINDING GENERATORS...")
    generators = find_generators()
    
    if not generators:
        print("  ❌ No generators found in weights/cyclegan_2d/")
        return
    
    print(f"  ✓ Found {len(generators)} generators:")
    for gen_name in generators.keys():
        print(f"    - {gen_name}")
    
    print("\n3. MATCHING SITE TO GENERATOR...")
    
    # Try to find a non-BCH site
    non_bch_sites = [s for s in unique_sites if 'BCH' not in s]
    
    if not non_bch_sites:
        print("  ❌ No non-BCH sites in test data")
        return
    
    # Try to match a site to a generator
    test_site = None
    generator_name = None
    
    for site in non_bch_sites:
        # Try exact match first
        normalized = normalize_site_name(site)
        
        # Check all generators for a match
        for gen_name in generators.keys():
            if normalized.lower() in gen_name.lower() or gen_name.lower() in normalized.lower():
                test_site = site
                generator_name = gen_name
                break
        
        if test_site:
            break
    
    # If no match, just use first available
    if not test_site:
        test_site = non_bch_sites[0]
        generator_name = list(generators.keys())[0]
        print(f"  ⚠️  No exact match found, using first available generator")
    
    print(f"  ✓ Using site: {test_site}")
    print(f"  ✓ Using generator: {generator_name}")
    
    # Get sample
    site_idx = np.where(test_sites == test_site)[0]
    if len(site_idx) == 0:
        print(f"  ❌ No samples found for {test_site}")
        return
    
    sample_idx = site_idx[0]
    
    print("\n4. LOADING GENERATOR...")
    try:
        import tensorflow as tf
        from train_fetal_2d_cyclegan import build_2d_generator
        
        gen = build_2d_generator((138, 176, 1), 16, name=f'gen_BCH2{generator_name}')
        gen.load_weights(generators[generator_name])
        print(f"  ✓ Loaded weights successfully")
        
    except Exception as e:
        print(f"  ❌ Error loading generator: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n5. GENERATING HARMONIZED OUTPUT...")
    original = test_images[sample_idx]
    ga_value = test_ga[sample_idx].reshape(1, 1)
    img_batch = np.expand_dims(original, axis=0)
    
    harmonized = gen([img_batch, ga_value], training=False).numpy()[0]
    
    print("\n" + "="*80)
    print("DIAGNOSTIC RESULTS")
    print("="*80)
    
    orig_stats = diagnose_output(original, "Original")
    harm_stats = diagnose_output(harmonized, "Harmonized (Raw)")
    
    # Create visualization
    output_dir = Path('diagnostics_contrast')
    output_dir.mkdir(exist_ok=True)
    
    harm_clahe, harm_stretched = create_visualization(
        original, harmonized, 
        output_dir / 'contrast_diagnostic.png'
    )
    
    clahe_stats = diagnose_output(harm_clahe, "CLAHE Enhanced")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    mae_raw = np.mean(np.abs(original[:,:,0] - harmonized[:,:,0]))
    mae_clahe = np.mean(np.abs(original[:,:,0] - harm_clahe))
    
    print(f"\nQuantitative Metrics:")
    print(f"  MAE (raw): {mae_raw:.4f}")
    print(f"  MAE (CLAHE): {mae_clahe:.4f}")
    print(f"  Std (raw): {harm_stats['std']:.4f}")
    print(f"  Std (CLAHE): {clahe_stats['std']:.4f}")
    print(f"  Edge strength (raw): {harm_stats['edge_strength']:.4f}")
    
    if harm_stats['is_compressed'] and harm_stats['is_gray']:
        print(f"\n⚠️  DIAGNOSIS: CONTRAST COMPRESSION CONFIRMED")
        
        if harm_stats['edge_strength'] > 0.01:
            print(f"\n✓ Good news: Structure IS present (edge strength = {harm_stats['edge_strength']:.4f})")
            print(f"✓ CLAHE successfully reveals hidden structure")
            
            print(f"\n📋 RECOMMENDED ACTIONS:")
            print(f"\n  SHORT TERM (Use current model with post-processing):")
            print(f"    • Apply CLAHE enhancement to all harmonized outputs")
            print(f"    • This will give usable results immediately")
            
            print(f"\n  LONG TERM (Retrain for better results):")
            print(f"    1. Change final activation: tanh → sigmoid")
            print(f"    2. Increase loss weights:")
            print(f"         lambda_adversarial = 2.0")
            print(f"         lambda_cycle = 10.0")
            print(f"         lambda_identity = 5.0")
            print(f"    3. Add contrast loss:")
            print(f"         def contrast_loss(gen_img):")
            print(f"             std = tf.math.reduce_std(gen_img)")
            print(f"             return tf.maximum(0.0, 0.15 - std)")
            print(f"    4. Train for 50-100 epochs and monitor")
        else:
            print(f"\n⚠️  Weak edge strength - may need complete retrain")
            print(f"\n📋 RECOMMENDED ACTIONS:")
            print(f"    1. Complete retrain with sigmoid activation")
            print(f"    2. Add perceptual loss (VGG features)")
            print(f"    3. Significantly increase adversarial weight")
    else:
        print(f"\n✓ DIAGNOSIS: Output appears normal")
        print(f"  No contrast compression detected")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()