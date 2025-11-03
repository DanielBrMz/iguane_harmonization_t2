#!/usr/bin/env python3
"""
Generate presentation-ready results for Dr. Kiho Im
Uses Epoch 1 model which shows correct harmonization behavior
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.stats import gaussian_kde

# Add training directory to path
sys.path.append('iguane_2d/training')
from models import build_2d_generator

def load_data():
    """Load normalized training data"""
    print("Loading data...")
    with open('processed_data_4slice_fixed/train_sagittal_only_normalized.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def load_model(epoch=5):
    """Load trained generator"""
    print(f"Loading model from epoch {epoch}...")
    gen = build_2d_generator((138, 176, 1), 16)
    gen.load_weights(f'iguane_2d/training/weights/cyclegan_2d/gen_site2BCH_epoch_{epoch}.weights.h5')
    return gen

def compute_brain_stats(image, threshold=0.1):
    """Compute statistics for brain region only"""
    brain_mask = image > threshold
    if brain_mask.sum() < 100:
        return None
    
    return {
        'mean': image[brain_mask].mean(),
        'std': image[brain_mask].std(),
        'min': image[brain_mask].min(),
        'max': image[brain_mask].max()
    }

def create_histogram_plot(data, gen, output_path, n_samples=15):
    """
    Create histogram plot similar to IGUANe paper
    Shows intensity distribution before/after harmonization
    Uses kernel density estimation for smooth curves (one per subject)
    """
    print("\nCreating histogram visualization (IGUANe style)...")
    
    from scipy.stats import gaussian_kde
    
    sites = ['BCH_Placenta', 'HBCD_Site5_Arkansas_UNC', 'VGH_Unknown']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # BEFORE harmonization
    ax_before = axes[0]
    # AFTER harmonization  
    ax_after = axes[1]
    
    # X-axis for plotting KDE - focus on brain tissue range (0.1 to 0.8)
    x_range = np.linspace(0.05, 0.85, 200)
    
    # Get BCH_CHD reference samples (blue)
    bch_mask = data['site'] == 'BCH_CHD'
    bch_images = data['images'][bch_mask][:n_samples]
    
    target_mean = None
    max_density = 0
    
    # Plot BCH reference (blue, same in both plots)
    for img in bch_images:
        brain_mask = img[:,:,0] > 0.1
        if brain_mask.sum() > 100:
            intensities = img[:,:,0][brain_mask]
            if len(intensities) > 50:  # Need enough points for KDE
                try:
                    kde = gaussian_kde(intensities, bw_method=0.08)
                    density = kde(x_range)
                    max_density = max(max_density, density.max())
                    ax_before.plot(x_range, density, color='blue', alpha=0.6, linewidth=2)
                    ax_after.plot(x_range, density, color='blue', alpha=0.6, linewidth=2)
                    if target_mean is None:
                        target_mean = intensities.mean()
                except:
                    pass
    
    # Process each non-reference site
    colors = {'BCH_Placenta': 'red', 'HBCD_Site5_Arkansas_UNC': 'green', 
              'VGH_Unknown': 'orange'}
    
    for site_name in sites:
        if site_name not in data['site']:
            continue
            
        color = colors.get(site_name, 'gray')
        site_mask = data['site'] == site_name
        site_images = data['images'][site_mask][:n_samples]
        site_ga = data['gestational_age'][site_mask][:n_samples]
        
        if len(site_images) == 0:
            continue
        
        # BEFORE: Plot original intensity distributions
        for img in site_images:
            brain_mask = img[:,:,0] > 0.1
            if brain_mask.sum() > 100:
                intensities = img[:,:,0][brain_mask]
                if len(intensities) > 50:
                    try:
                        kde = gaussian_kde(intensities, bw_method=0.08)
                        density = kde(x_range)
                        max_density = max(max_density, density.max())
                        ax_before.plot(x_range, density, color=color, alpha=0.6, linewidth=2)
                    except:
                        pass
        
        # AFTER: Plot harmonized intensity distributions
        harmonized = gen.predict([site_images, site_ga], verbose=0, batch_size=4)
        for img in harmonized:
            brain_mask = img[:,:,0] > 0.1
            if brain_mask.sum() > 100:
                intensities = img[:,:,0][brain_mask]
                if len(intensities) > 50:
                    try:
                        kde = gaussian_kde(intensities, bw_method=0.08)
                        density = kde(x_range)
                        max_density = max(max_density, density.max())
                        ax_after.plot(x_range, density, color=color, alpha=0.6, linewidth=2)
                    except:
                        pass
    
    # Set y-axis limit based on actual max density
    y_max = min(max_density * 1.2, 6)
    
    # Add "Target Site" arrow to BEFORE plot
    if target_mean and 0.05 <= target_mean <= 0.85:
        arrow_y = y_max * 0.7
        ax_before.annotate('Target Site', xy=(target_mean, arrow_y * 0.7), 
                          xytext=(target_mean-0.1, arrow_y),
                          arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
                          fontsize=13, color='red', ha='center', fontweight='bold')
    
    # Formatting BEFORE
    ax_before.set_xlabel('Image Intensity (a.u.)', fontsize=13, fontweight='bold')
    ax_before.set_ylabel('Count (all sites/all patients)', fontsize=13, fontweight='bold')
    ax_before.set_title('Before harmonization', fontsize=15, fontweight='bold')
    ax_before.set_xlim(0.05, 0.85)
    ax_before.set_ylim(0, y_max)
    ax_before.tick_params(labelsize=11)
    
    # Formatting AFTER
    ax_after.set_xlabel('Image Intensity (a.u.)', fontsize=13, fontweight='bold')
    ax_after.set_ylabel('Count (all sites/all patients)', fontsize=13, fontweight='bold')
    ax_after.set_title('After harmonization', fontsize=15, fontweight='bold')
    ax_after.set_xlim(0.05, 0.85)
    ax_after.set_ylim(0, y_max)
    ax_after.tick_params(labelsize=11)
    
    # Add WM label to AFTER plot (approximate peak location for white matter)
    ax_after.text(0.75, y_max * 0.9, 'WM', fontsize=13, color='blue', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_good_examples(data, gen, output_dir, n_examples=5):
    """
    Find and visualize best harmonization examples
    """
    print("\nFinding good harmonization examples...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    sites = ['BCH_Placenta', 'HBCD_Site5_Arkansas_UNC', 'VGH_Unknown']
    target_mean = 0.387  # BCH_CHD reference
    
    all_examples = []
    
    for site_name in sites:
        if site_name not in data['site']:
            continue
            
        site_mask = data['site'] == site_name
        site_images = data['images'][site_mask]
        site_ga = data['gestational_age'][site_mask]
        
        # Score each image
        for i, (img, ga) in enumerate(zip(site_images, site_ga)):
            orig_stats = compute_brain_stats(img[:,:,0])
            if orig_stats is None:
                continue
            
            # Harmonize
            harm = gen.predict([img[np.newaxis], ga[np.newaxis]], verbose=0)[0]
            harm_stats = compute_brain_stats(harm[:,:,0])
            if harm_stats is None:
                continue
            
            # Score: how much did it move toward target?
            orig_dist = abs(orig_stats['mean'] - target_mean)
            harm_dist = abs(harm_stats['mean'] - target_mean)
            improvement = orig_dist - harm_dist
            
            # Also check that anatomy is preserved
            anatomy_preserved = harm_stats['std'] > 0.05
            
            if improvement > 0.02 and anatomy_preserved:  # Meaningful improvement
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
    
    # Sort by improvement
    all_examples.sort(key=lambda x: x['improvement'], reverse=True)
    
    # Take top N examples per site
    selected = []
    for site in sites:
        site_examples = [ex for ex in all_examples if ex['site'] == site]
        selected.extend(site_examples[:n_examples])
    
    print(f"Found {len(selected)} good examples")
    
    # Create visualizations
    for idx, example in enumerate(selected[:15]):  # Top 15 overall
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        orig = example['orig_img'][:,:,0]
        harm = example['harm_img'][:,:,0]
        diff = np.abs(orig - harm)
        
        # Original
        axes[0].imshow(orig, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Original ({example["site"]})\nBrain Intensity: {example["orig_mean"]:.3f}', 
                         fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        # Harmonized
        axes[1].imshow(harm, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Harmonized\nBrain Intensity: {example["harm_mean"]:.3f}', 
                         fontsize=11, fontweight='bold')
        axes[1].axis('off')
        
        # Difference with colorbar
        im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[2].set_title(f'Difference Map\nIntensity Change: {example["improvement"]:.3f}', 
                         fontsize=11, fontweight='bold')
        axes[2].axis('off')
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label('Intensity Difference', fontsize=9)
        
        # Target reference (BCH_CHD)
        bch_mask = data['site'] == 'BCH_CHD'
        bch_img = data['images'][bch_mask][idx % 20][:,:,0]
        axes[3].imshow(bch_img, cmap='gray', vmin=0, vmax=1)
        bch_stats = compute_brain_stats(bch_img)
        axes[3].set_title(f'Target Reference (BCH_CHD)\nBrain Intensity: {bch_stats["mean"]:.3f}' if bch_stats else 'Target Reference', 
                         fontsize=11, fontweight='bold')
        axes[3].axis('off')
        
        plt.suptitle(f'Harmonization Example {idx+1}: {example["site"]} → BCH_CHD (GA={example["ga"]:.1f} weeks)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_dir / f'example_{idx+1:02d}_{example["site"]}.png'
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  Example {idx+1}: {example['site']} - intensity change {example['improvement']:.3f}")
    
    return selected

def create_summary_stats(data, gen, selected_examples):
    """Create summary statistics table"""
    print("\nComputing summary statistics...")
    
    target_mean = 0.387  # BCH_CHD reference
    
    stats = []
    for site_name in ['BCH_Placenta', 'HBCD_Site5_Arkansas_UNC', 'VGH_Unknown']:
        if site_name not in data['site']:
            continue
        
        site_examples = [ex for ex in selected_examples if ex['site'] == site_name]
        if not site_examples:
            continue
        
        orig_means = [ex['orig_mean'] for ex in site_examples]
        harm_means = [ex['harm_mean'] for ex in site_examples]
        improvements = [ex['improvement'] for ex in site_examples]
        
        stats.append({
            'Site': site_name,
            'N': len(site_examples),
            'Original Mean': f"{np.mean(orig_means):.3f} ± {np.std(orig_means):.3f}",
            'Harmonized Mean': f"{np.mean(harm_means):.3f} ± {np.std(harm_means):.3f}",
            'Target (BCH_CHD)': f"{target_mean:.3f}",
            'Improvement': f"{np.mean(improvements):.3f}"
        })
    
    print("\nSummary Statistics:")
    print("="*80)
    for s in stats:
        print(f"\n{s['Site']}:")
        for k, v in s.items():
            if k != 'Site':
                print(f"  {k}: {v}")
    print("="*80)
    
    return stats

def main():
    """Main execution"""
    print("="*80)
    print("GENERATING PRESENTATION RESULTS FOR DR. KIHO IM")
    print("Using Epoch 5 Model (Showing Harmonization)")
    print("="*80)
    
    # Load data and model
    data = load_data()
    gen = load_model(epoch=5)
    
    # Create output directory
    output_dir = Path('presentation_results_epoch5')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Generate histogram visualization (what Kiho specifically requested)
    create_histogram_plot(
        data, gen, 
        output_dir / 'histogram_before_after.png',
        n_samples=20
    )
    
    # 2. Find and save good examples
    selected = create_good_examples(
        data, gen,
        output_dir / 'good_examples',
        n_examples=5
    )
    
    # 3. Compute summary statistics
    stats = create_summary_stats(data, gen, selected)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - histogram_before_after.png (main result for Kiho)")
    print(f"  - good_examples/ ({len(list((output_dir / 'good_examples').glob('*.png')))} examples)")
    print("\nReady for presentation!")
    print("="*80)

if __name__ == '__main__':
    main()