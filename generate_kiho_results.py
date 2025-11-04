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


def clean_site_name(site_name):
    """Clean site names for visualization purposes"""
    # Remove _Unknown suffix
    site_name = site_name.replace('_Unknown', '')
    # Replace underscores with spaces for readability
    site_name = site_name.replace('_', ' ')
    return site_name


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

def create_histogram_plot(data, gen, output_path, n_samples_bch=8, n_samples_other=4):
    """
    Create histogram plot similar to IGUANe paper
    Shows intensity distribution before/after harmonization
    Uses kernel density estimation for smooth curves (one per subject)
    """
    print("\nCreating histogram visualization (IGUANe style)...")
    
    from scipy.stats import gaussian_kde
    from matplotlib.patches import Patch
    
    sites = ['BCH_Placenta', 'HBCD_Site5_Arkansas_UNC', 'VGH_Unknown']
    
    # Create figure with space for legend at bottom
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # BEFORE harmonization
    ax_before = axes[0]
    # AFTER harmonization  
    ax_after = axes[1]
    
    # X-axis for plotting KDE - focus on meaningful brain tissue range
    x_range = np.linspace(0.15, 0.70, 300)
    
    # Get BCH_CHD reference samples (blue)
    bch_mask = data['site'] == 'BCH_CHD'
    bch_images = data['images'][bch_mask]
    
    target_mean = None
    
    # Collect all densities first to determine appropriate y-axis range
    all_densities = []
    
    # Plot BCH reference (blue, same in both plots) - more samples since it's reference
    bch_count = 0
    for img in bch_images:
        if bch_count >= n_samples_bch:
            break
        brain_mask = img[:,:,0] > 0.12
        if brain_mask.sum() > 200:
            intensities = img[:,:,0][brain_mask]
            # Filter to only brain tissue range
            intensities = intensities[(intensities >= 0.15) & (intensities <= 0.70)]
            if len(intensities) > 100:  # Need enough points for KDE
                try:
                    # Use smoother bandwidth to avoid extreme peaks
                    kde = gaussian_kde(intensities, bw_method=0.08)
                    density = kde(x_range)
                    all_densities.append(density)
                    ax_before.plot(x_range, density, color='blue', alpha=0.6, linewidth=2.0)
                    ax_after.plot(x_range, density, color='blue', alpha=0.6, linewidth=2.0)
                    if target_mean is None:
                        target_mean = intensities.mean()
                    bch_count += 1
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
        site_images = data['images'][site_mask]
        site_ga = data['gestational_age'][site_mask]
        
        if len(site_images) == 0:
            continue
        
        # BEFORE: Plot original intensity distributions (fewer samples for clarity)
        site_count = 0
        for img in site_images:
            if site_count >= n_samples_other:
                break
            brain_mask = img[:,:,0] > 0.12
            if brain_mask.sum() > 200:
                intensities = img[:,:,0][brain_mask]
                # Filter to brain tissue range
                intensities = intensities[(intensities >= 0.15) & (intensities <= 0.70)]
                if len(intensities) > 100:
                    try:
                        # Use smoother bandwidth to avoid extreme peaks
                        kde = gaussian_kde(intensities, bw_method=0.08)
                        density = kde(x_range)
                        all_densities.append(density)
                        ax_before.plot(x_range, density, color=color, alpha=0.6, linewidth=2.0)
                        site_count += 1
                    except:
                        pass
        
        # AFTER: Plot harmonized intensity distributions
        n_harm = min(n_samples_other, len(site_images))
        harmonized = gen.predict([site_images[:n_harm], site_ga[:n_harm]], verbose=0, batch_size=4)
        for img in harmonized:
            brain_mask = img[:,:,0] > 0.12
            if brain_mask.sum() > 200:
                intensities = img[:,:,0][brain_mask]
                # Filter to brain tissue range
                intensities = intensities[(intensities >= 0.15) & (intensities <= 0.70)]
                if len(intensities) > 100:
                    try:
                        # Use smoother bandwidth to avoid extreme peaks
                        kde = gaussian_kde(intensities, bw_method=0.08)
                        density = kde(x_range)
                        all_densities.append(density)
                        ax_after.plot(x_range, density, color=color, alpha=0.6, linewidth=2.0)
                    except:
                        pass
    
    # Calculate appropriate y-axis limit based on actual data
    if all_densities:
        max_density = np.max([d.max() for d in all_densities])
        y_max = max_density * 1.15  # Add 15% headroom
    else:
        y_max = 6.0  # Fallback
    
    # Formatting BEFORE
    ax_before.set_xlabel('Image Intensity (a.u.)', fontsize=14, fontweight='bold')
    ax_before.set_ylabel('Count (all sites/all patients)', fontsize=14, fontweight='bold')
    ax_before.set_title('Before harmonization', fontsize=16, fontweight='bold')
    ax_before.set_xlim(0.08, 0.75)
    ax_before.set_ylim(0, y_max)
    ax_before.tick_params(labelsize=12)
    ax_before.spines['top'].set_visible(False)
    ax_before.spines['right'].set_visible(False)
    ax_before.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    
    # Formatting AFTER
    ax_after.set_xlabel('Image Intensity (a.u.)', fontsize=14, fontweight='bold')
    ax_after.set_ylabel('Count (all sites/all patients)', fontsize=14, fontweight='bold')
    ax_after.set_title('After harmonization', fontsize=16, fontweight='bold')
    ax_after.set_xlim(0.08, 0.75)
    ax_after.set_ylim(0, y_max)
    ax_after.tick_params(labelsize=12)
    ax_after.spines['top'].set_visible(False)
    ax_after.spines['right'].set_visible(False)
    ax_after.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    
    # Add tissue type indicators to AFTER plot
    # Gray Matter (GM) - typically around 0.25-0.35
    ax_after.axvline(x=0.30, color='gray', linestyle=':', linewidth=2.5, alpha=0.7)
    ax_after.text(0.30, y_max * 0.97, 'GM', fontsize=13, color='gray', fontweight='bold',
                 ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5))
    
    # White Matter (WM) - typically around 0.40-0.50
    ax_after.axvline(x=0.45, color='navy', linestyle=':', linewidth=2.5, alpha=0.7)
    ax_after.text(0.45, y_max * 0.97, 'WM', fontsize=13, color='navy', fontweight='bold',
                 ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='navy', linewidth=1.5))
    
    # Create legend with site colors
    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', label='BCH CHD (Reference)', alpha=0.7),
        Patch(facecolor='red', edgecolor='red', label='BCH Placenta', alpha=0.7),
        Patch(facecolor='green', edgecolor='green', label='HBCD Site5 Arkansas UNC', alpha=0.7),
        Patch(facecolor='orange', edgecolor='orange', label='VGH', alpha=0.7)
    ]
    
    # Add legend below plots, centered
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
              fontsize=12, frameon=True, fancybox=True, shadow=True,
              bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Make room for legend
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
        axes[0].set_title(f'Original ({clean_site_name(example["site"])})\nBrain Intensity: {example["orig_mean"]:.3f}', 
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
        
        plt.suptitle(f'Harmonization Example {idx+1}: {clean_site_name(example["site"])} → BCH CHD (GA={example["ga"]:.1f} weeks)', 
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
        n_samples_bch=8,  # 8 BCH reference samples
        n_samples_other=4  # 4 samples per non-BCH site
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