#!/usr/bin/env python3
"""
Generate presentation-ready results for Dr. Kiho Im - FIXED VERSION
Uses SAGITTAL views (existing trained model)
- Accurate histogram (no intensity filtering)  
- Excludes BCH_Placenta from visualization
- Consistent brain mask threshold (0.1)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.stats import gaussian_kde

# Add training directory to path (adjusted for new location)
sys.path.append(str(Path(__file__).parent.parent / 'training'))
from models import build_2d_generator


def load_data():
    """Load normalized sagittal training data"""
    print("Loading sagittal data...")
    # Adjust path relative to project root
    data_path = Path(__file__).parent.parent.parent / 'processed_data_4slice_fixed' / 'train_sagittal_only_normalized.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_model(epoch=5):
    """Load trained generator"""
    print(f"Loading model from epoch {epoch}...")
    gen = build_2d_generator((138, 176, 1), 16)
    # Adjust path relative to training directory
    weights_path = Path(__file__).parent.parent / 'training' / 'weights' / 'cyclegan_2d' / f'gen_site2BCH_epoch_{epoch}.weights.h5'
    gen.load_weights(str(weights_path))
    return gen


def compute_brain_stats(image, threshold=0.1):
    """Compute statistics for brain region only - CONSISTENT THRESHOLD"""
    brain_mask = image > threshold
    if brain_mask.sum() < 100:
        return None
    
    return {
        'mean': image[brain_mask].mean(),
        'std': image[brain_mask].std(),
        'min': image[brain_mask].min(),
        'max': image[brain_mask].max()
    }


def create_histogram_plot(data, gen, output_path, n_samples_bch=10, n_samples_other=6):
    """
    Create histogram plot - FIXED VERSION
    - No intensity filtering (uses all brain pixels)
    - Consistent brain mask (threshold=0.1)  
    - Excludes BCH_Placenta from visualization
    """
    print("\nCreating FIXED histogram visualization...")
    
    from matplotlib.patches import Patch
    
    # EXCLUDE BCH_Placenta as per Kiho's request
    sites = ['HBCD_Site5_Arkansas_UNC', 'VGH_Unknown']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax_before = axes[0]
    ax_after = axes[1]
    
    # X-axis range for KDE - cover full brain intensity range
    x_range = np.linspace(0.05, 0.80, 400)
    
    # Get BCH_CHD reference samples
    bch_mask = data['site'] == 'BCH_CHD'
    bch_images = data['images'][bch_mask]
    
    target_mean = None
    all_densities = []
    
    # Plot BCH reference (blue)
    print("  Processing BCH_CHD reference...")
    bch_count = 0
    for img in bch_images:
        if bch_count >= n_samples_bch:
            break
        
        # FIXED: Use same threshold as compute_brain_stats (0.1)
        brain_mask = img[:,:,0] > 0.1
        if brain_mask.sum() > 100:
            intensities = img[:,:,0][brain_mask]
            
            # FIXED: NO INTENSITY FILTERING - use all brain pixels
            if len(intensities) > 100:
                try:
                    kde = gaussian_kde(intensities, bw_method=0.06)
                    density = kde(x_range)
                    all_densities.append(density)
                    
                    ax_before.plot(x_range, density, color='blue', alpha=0.65, linewidth=2.2)
                    ax_after.plot(x_range, density, color='blue', alpha=0.65, linewidth=2.2)
                    
                    if target_mean is None:
                        target_mean = intensities.mean()
                        print(f"    Target mean: {target_mean:.3f}")
                    
                    bch_count += 1
                except Exception as e:
                    print(f"    KDE failed: {e}")
    
    print(f"  BCH_CHD: {bch_count} samples plotted")
    
    # Process non-reference sites
    colors = {'HBCD_Site5_Arkansas_UNC': 'green', 'VGH_Unknown': 'orange'}
    
    for site_name in sites:
        if site_name not in data['site']:
            print(f"  {site_name}: NOT FOUND in data")
            continue
        
        print(f"  Processing {site_name}...")
        color = colors[site_name]
        site_mask = data['site'] == site_name
        site_images = data['images'][site_mask]
        site_ga = data['gestational_age'][site_mask]
        
        if len(site_images) == 0:
            print(f"    No images found!")
            continue
        
        # BEFORE: Original distributions
        site_count = 0
        orig_means = []
        for img in site_images:
            if site_count >= n_samples_other:
                break
            
            brain_mask = img[:,:,0] > 0.1
            if brain_mask.sum() > 100:
                intensities = img[:,:,0][brain_mask]
                
                # NO FILTERING - use ALL brain pixels
                if len(intensities) > 100:
                    try:
                        kde = gaussian_kde(intensities, bw_method=0.06)
                        density = kde(x_range)
                        all_densities.append(density)
                        ax_before.plot(x_range, density, color=color, alpha=0.65, linewidth=2.2)
                        orig_means.append(intensities.mean())
                        site_count += 1
                    except:
                        pass
        
        if orig_means:
            print(f"    BEFORE: {site_count} samples, mean brain intensity: {np.mean(orig_means):.3f}")
        
        # AFTER: Harmonized distributions
        n_harm = min(n_samples_other, len(site_images))
        if n_harm > 0:
            harmonized = gen.predict([site_images[:n_harm], site_ga[:n_harm]], verbose=0, batch_size=4)
            
            harm_means = []
            for img in harmonized:
                brain_mask = img[:,:,0] > 0.1
                if brain_mask.sum() > 100:
                    intensities = img[:,:,0][brain_mask]
                    
                    # NO FILTERING
                    if len(intensities) > 100:
                        try:
                            kde = gaussian_kde(intensities, bw_method=0.06)
                            density = kde(x_range)
                            all_densities.append(density)
                            ax_after.plot(x_range, density, color=color, alpha=0.65, linewidth=2.2)
                            harm_means.append(intensities.mean())
                        except:
                            pass
            
            if harm_means:
                print(f"    AFTER: {len(harm_means)} samples, mean brain intensity: {np.mean(harm_means):.3f}")
                if orig_means and harm_means:
                    change = np.mean(orig_means) - np.mean(harm_means)
                    print(f"    CHANGE: {change:.3f} (shift toward target {target_mean:.3f})")
    
    # Calculate y-axis limit
    if all_densities:
        max_density = np.max([d.max() for d in all_densities])
        y_max = max_density * 1.15
    else:
        y_max = 6.0
    
    print(f"\n  Y-axis max: {y_max:.2f}")
    
    # Formatting
    for ax, title in [(ax_before, 'Before harmonization'), (ax_after, 'After harmonization')]:
        ax.set_xlabel('Image Intensity (a.u.)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count (all sites/all patients)', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlim(0.05, 0.75)
        ax.set_ylim(0, y_max)
        ax.tick_params(labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    
    # Add tissue markers to AFTER plot
    ax_after.axvline(x=0.30, color='gray', linestyle=':', linewidth=2.5, alpha=0.7)
    ax_after.text(0.30, y_max * 0.97, 'GM', fontsize=13, color='gray', fontweight='bold',
                 ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                          edgecolor='gray', linewidth=1.5))
    
    ax_after.axvline(x=0.45, color='navy', linestyle=':', linewidth=2.5, alpha=0.7)
    ax_after.text(0.45, y_max * 0.97, 'WM', fontsize=13, color='navy', fontweight='bold',
                 ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                          edgecolor='navy', linewidth=1.5))
    
    # Legend
    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', label='BCH CHD (Reference)', alpha=0.7),
        Patch(facecolor='green', edgecolor='green', label='HBCD Arkansas', alpha=0.7),
        Patch(facecolor='orange', edgecolor='orange', label='VGH', alpha=0.7)
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              fontsize=12, frameon=True, fancybox=True, shadow=True,
              bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def create_good_examples(data, gen, output_dir, n_examples=5):
    """Find and visualize best harmonization examples"""
    print("\nFinding good harmonization examples...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Only HBCD and VGH (exclude BCH_Placenta)
    sites = ['HBCD_Site5_Arkansas_UNC', 'VGH_Unknown']
    target_mean = 0.387
    
    all_examples = []
    
    for site_name in sites:
        if site_name not in data['site']:
            continue
        
        site_mask = data['site'] == site_name
        site_images = data['images'][site_mask]
        site_ga = data['gestational_age'][site_mask]
        
        for i, (img, ga) in enumerate(zip(site_images, site_ga)):
            orig_stats = compute_brain_stats(img[:,:,0])
            if orig_stats is None:
                continue
            
            harm = gen.predict([img[np.newaxis], ga[np.newaxis]], verbose=0)[0]
            harm_stats = compute_brain_stats(harm[:,:,0])
            if harm_stats is None:
                continue
            
            orig_dist = abs(orig_stats['mean'] - target_mean)
            harm_dist = abs(harm_stats['mean'] - target_mean)
            improvement = orig_dist - harm_dist
            
            if improvement > 0.01 and harm_stats['std'] > 0.05:
                all_examples.append({
                    'site': site_name,
                    'idx': i,
                    'orig_img': img,
                    'harm_img': harm,
                    'ga': ga,
                    'orig_mean': orig_stats['mean'],
                    'harm_mean': harm_stats['mean'],
                    'improvement': improvement,
                    'orig_std': orig_stats['std'],
                    'harm_std': harm_stats['std']
                })
    
    all_examples.sort(key=lambda x: x['improvement'], reverse=True)
    
    selected = []
    for site in sites:
        site_examples = [ex for ex in all_examples if ex['site'] == site]
        selected.extend(site_examples[:n_examples])
    
    print(f"Found {len(selected)} good examples")
    
    # Visualize
    for idx, example in enumerate(selected[:10]):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        orig = example['orig_img'][:,:,0]
        harm = example['harm_img'][:,:,0]
        diff = np.abs(orig - harm)
        
        # Original
        axes[0].imshow(orig, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Original ({example["site"].replace("_", " ")})\nBrain Intensity: {example["orig_mean"]:.3f}', 
                         fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        # Harmonized
        axes[1].imshow(harm, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Harmonized\nBrain Intensity: {example["harm_mean"]:.3f}', 
                         fontsize=11, fontweight='bold')
        axes[1].axis('off')
        
        # Difference
        im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[2].set_title(f'Difference Map\nIntensity Change: {example["improvement"]:.3f}', 
                         fontsize=11, fontweight='bold')
        axes[2].axis('off')
        cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label('Intensity Difference', fontsize=9)
        
        # Target reference
        bch_mask = data['site'] == 'BCH_CHD'
        bch_img = data['images'][bch_mask][idx % 10][:,:,0]
        axes[3].imshow(bch_img, cmap='gray', vmin=0, vmax=1)
        bch_stats = compute_brain_stats(bch_img)
        axes[3].set_title(f'Target (BCH_CHD)\nBrain Intensity: {bch_stats["mean"]:.3f}' if bch_stats else 'Target', 
                         fontsize=11, fontweight='bold')
        axes[3].axis('off')
        
        plt.suptitle(f'Sagittal Example {idx+1}: {example["site"].replace("_", " ")} → BCH CHD (GA={example["ga"]:.1f}w)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_dir / f'sagittal_example_{idx+1:02d}_{example["site"]}.png'
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  Example {idx+1}: {example['site']:.30s} - change {example['improvement']:.3f}")
    
    return selected


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5, help='Epoch to use')
    args = parser.parse_args()
    
    print("="*80)
    print("GENERATING FIXED RESULTS FOR DR. KIHO IM")
    print(f"Using Sagittal Model - Epoch {args.epoch}")
    print("FIXES: Accurate histogram, Exclude BCH_Placenta, Consistent brain mask")
    print("="*80)
    
    data = load_data()
    gen = load_model(epoch=args.epoch)
    
    output_dir = Path(f'presentation_results_sagittal_epoch{args.epoch}_FIXED')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Generate fixed histogram
    create_histogram_plot(
        data, gen, 
        output_dir / 'histogram_before_after_FIXED.png',
        n_samples_bch=10,
        n_samples_other=6
    )
    
    # 2. Generate examples
    selected = create_good_examples(
        data, gen,
        output_dir / 'sagittal_examples',
        n_examples=5
    )
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nResults: {output_dir}/")
    print(f"  - histogram_before_after_FIXED.png")
    print(f"  - sagittal_examples/ ({len(selected)} examples)")
    print("\nNote: Using SAGITTAL views (existing trained model)")
    print("All examples show consistent anatomical orientation")
    print("="*80)


if __name__ == '__main__':
    main()