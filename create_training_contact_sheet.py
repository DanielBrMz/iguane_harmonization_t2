"""
Generate comprehensive contact sheet of all training images
Organized by site to visualize view distribution patterns
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches


def create_full_training_contact_sheet(data_path, output_path='training_data_contact_sheet.png',
                                       max_images_per_site=100, cols=20):
    """
    Create large contact sheet showing all training images organized by site
    
    Args:
        data_path: Path to training data pickle
        output_path: Where to save figure
        max_images_per_site: Maximum images to show per site (for readability)
        cols: Number of columns in grid
    """
    print("Loading data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    images = data['images']
    sites = data['site']
    ga = data.get('gestational_age', data.get('ga', None))
    
    if ga is None:
        print("Warning: No gestational age data found")
        ga = np.zeros(len(images))
    
    print(f"Total images: {len(images)}")
    print(f"Sites: {np.unique(sites)}")
    
    # Get unique sites
    unique_sites = np.unique(sites)
    n_sites = len(unique_sites)
    
    # Calculate layout
    site_samples = {}
    for site in unique_sites:
        site_mask = sites == site
        site_indices = np.where(site_mask)[0]
        n_site = len(site_indices)
        
        # Sample if too many
        if n_site > max_images_per_site:
            sampled_indices = np.random.choice(site_indices, max_images_per_site, replace=False)
        else:
            sampled_indices = site_indices
        
        site_samples[site] = sampled_indices
        print(f"  {site}: {len(sampled_indices)} images (total: {n_site})")
    
    # Create figure
    total_rows = sum([int(np.ceil(len(indices) / cols)) for indices in site_samples.values()])
    
    fig = plt.figure(figsize=(cols * 1.5, total_rows * 1.5))
    
    print("\nGenerating contact sheet...")
    
    current_row = 0
    
    for site_idx, site in enumerate(unique_sites):
        indices = site_samples[site]
        n_images = len(indices)
        n_rows = int(np.ceil(n_images / cols))
        
        print(f"  Processing {site}: {n_images} images, {n_rows} rows")
        
        for i, idx in enumerate(indices):
            row = current_row + i // cols
            col = i % cols
            
            ax = plt.subplot2grid((total_rows, cols), (row, col))
            
            # Show image
            img = images[idx, :, :, 0]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            # Add border color by site
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
            border_color = colors[site_idx % len(colors)]
            
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)
        
        # Add site label
        if n_rows > 0:
            ax_label = plt.subplot2grid((total_rows, cols), (current_row, 0), colspan=cols)
            ax_label.text(0.01, 0.5, f'{site} ({n_images} images)', 
                         fontsize=16, fontweight='bold',
                         verticalalignment='center',
                         bbox=dict(boxstyle='round', facecolor=colors[site_idx % len(colors)], alpha=0.3))
            ax_label.axis('off')
        
        current_row += n_rows
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=colors[i % len(colors)], 
                                     edgecolor='black', 
                                     label=site) 
                      for i, site in enumerate(unique_sites)]
    
    fig.legend(handles=legend_elements, loc='upper center', 
              ncol=n_sites, fontsize=12, frameon=True)
    
    plt.suptitle('Training Data: All Subjects by Site\n(Look for view consistency within/across sites)', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    print(f"\nSaving contact sheet to {output_path}...")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Done!")
    
    # Generate statistics
    print("\n" + "="*70)
    print("VISUAL INSPECTION GUIDE")
    print("="*70)
    print("\nLook for these patterns:")
    print("  1. Do images within each site look similar in view/orientation?")
    print("  2. Do different sites show different predominant views?")
    print("  3. Are there consistent anatomical features across sites?")
    print("\nView identification:")
    print("  - SAGITTAL: Side profile, one hemisphere, C-shaped")
    print("  - CORONAL: Front view, both hemispheres, butterfly-shaped")
    print("  - AXIAL: Top-down view, both hemispheres, oval/circular")
    print("\nIf different sites = different views → Harmonization confounded!")
    print("="*70)


def create_site_comparison_grid(data_path, output_path='site_comparison_grid.png',
                                samples_per_site=25):
    """
    Create side-by-side comparison grid for easier visual comparison
    Shows same number of samples from each site in aligned rows
    """
    print("Loading data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    images = data['images']
    sites = data['site']
    
    unique_sites = np.unique(sites)
    n_sites = len(unique_sites)
    
    print(f"\nCreating comparison grid: {n_sites} sites × {samples_per_site} samples")
    
    # Sample from each site
    site_images = {}
    for site in unique_sites:
        site_mask = sites == site
        site_indices = np.where(site_mask)[0]
        
        n_available = len(site_indices)
        n_sample = min(samples_per_site, n_available)
        
        sampled = np.random.choice(site_indices, n_sample, replace=False)
        site_images[site] = images[sampled]
        
        print(f"  {site}: {n_sample} samples")
    
    # Create figure
    fig, axes = plt.subplots(n_sites, samples_per_site, 
                            figsize=(samples_per_site * 1.2, n_sites * 1.5))
    
    if n_sites == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    for site_idx, site in enumerate(unique_sites):
        imgs = site_images[site]
        
        for img_idx in range(samples_per_site):
            ax = axes[site_idx, img_idx]
            
            if img_idx < len(imgs):
                ax.imshow(imgs[img_idx, :, :, 0], cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(np.zeros((138, 176)), cmap='gray')
            
            ax.axis('off')
            
            # Site label on first column
            if img_idx == 0:
                ax.text(-0.1, 0.5, site, 
                       transform=ax.transAxes,
                       fontsize=12, fontweight='bold',
                       verticalalignment='center',
                       horizontalalignment='right',
                       color=colors[site_idx % len(colors)])
            
            # Add colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(colors[site_idx % len(colors)])
                spine.set_linewidth(2)
    
    plt.suptitle('Site Comparison Grid: Random Samples from Each Site\n' + 
                 'Each row = one site. Look for view consistency within rows and differences between rows.',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved comparison grid to {output_path}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python create_training_contact_sheet.py <train_data.pkl>")
        print("\nThis will generate two visualizations:")
        print("  1. training_data_contact_sheet.png - All images by site")
        print("  2. site_comparison_grid.png - Side-by-side comparison")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    # Generate both visualizations
    print("="*70)
    print("GENERATING CONTACT SHEET (all images)")
    print("="*70)
    create_full_training_contact_sheet(data_path, 
                                       max_images_per_site=100,
                                       cols=20)
    
    print("\n" + "="*70)
    print("GENERATING COMPARISON GRID (aligned samples)")
    print("="*70)
    create_site_comparison_grid(data_path, samples_per_site=25)
    
    print("\n" + "="*70)
    print("REVIEW THE IMAGES")
    print("="*70)
    print("\nOpen the generated images and check:")
    print("  1. training_data_contact_sheet.png - See all data organized by site")
    print("  2. site_comparison_grid.png - Compare sites side-by-side")
    print("\nKey questions:")
    print("  - Do sites have consistent views within themselves?")
    print("  - Do different sites use different views?")
    print("  - Can you visually identify view patterns?")
    print("\nShare the images and we'll determine next steps.")