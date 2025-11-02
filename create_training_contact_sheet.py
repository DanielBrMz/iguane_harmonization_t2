"""
Fixed version - load images not masks
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt


def create_site_comparison_grid(data_path, output_path='site_comparison_grid.png',
                                samples_per_site=25):
    """
    Create side-by-side comparison grid
    """
    print("Loading data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Use 'images' not 'masks'
    images = data['images']  # This should be the actual brain images
    sites = data['site']
    
    print(f"Data shape: {images.shape}")
    print(f"Data range: {images.min():.3f} - {images.max():.3f}")
    
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
        
        print(f"  {site}: {n_sample} samples (total: {n_available})")
    
    # Create figure
    fig, axes = plt.subplots(n_sites, samples_per_site, 
                            figsize=(samples_per_site * 1.5, n_sites * 2))
    
    if n_sites == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    
    for site_idx, site in enumerate(unique_sites):
        imgs = site_images[site]
        
        for img_idx in range(samples_per_site):
            ax = axes[site_idx, img_idx]
            
            if img_idx < len(imgs):
                # Extract 2D slice
                img_2d = imgs[img_idx, :, :, 0]
                
                # Check if image is valid
                if img_2d.max() > 0.01:
                    ax.imshow(img_2d, cmap='gray', vmin=0, vmax=1)
                else:
                    # Mark as invalid
                    ax.imshow(np.zeros_like(img_2d), cmap='gray')
                    ax.text(0.5, 0.5, 'EMPTY', transform=ax.transAxes,
                           ha='center', va='center', color='red', fontsize=8)
            else:
                ax.imshow(np.zeros((138, 176)), cmap='gray')
            
            ax.axis('off')
            
            # Site label on first column
            if img_idx == 0:
                ax.text(-0.05, 0.5, site, 
                       transform=ax.transAxes,
                       fontsize=10, fontweight='bold',
                       verticalalignment='center',
                       horizontalalignment='right',
                       color=colors[site_idx % len(colors)],
                       rotation=0)
    
    plt.suptitle('Site Comparison: Random Training Samples\n' + 
                 'Look for view consistency within rows and differences between rows',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved to {output_path}")
    
    # Print statistics
    print("\n" + "="*70)
    print("IMAGE VALIDITY CHECK")
    print("="*70)
    for site in unique_sites:
        site_mask = sites == site
        site_imgs = images[site_mask]
        
        # Count valid images (max intensity > threshold)
        valid_mask = np.array([img[:,:,0].max() > 0.1 for img in site_imgs])
        n_valid = valid_mask.sum()
        n_total = len(site_imgs)
        pct_valid = 100 * n_valid / n_total
        
        print(f"{site}: {n_valid}/{n_total} valid ({pct_valid:.1f}%)")
        
        if pct_valid < 90:
            print(f"  ⚠️  WARNING: {site} has {100-pct_valid:.1f}% invalid images!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python create_site_comparison_fixed.py <train_data.pkl>")
        sys.exit(1)
    
    create_site_comparison_grid(sys.argv[1])