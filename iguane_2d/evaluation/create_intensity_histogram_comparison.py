"""
Generate intensity histogram comparison before/after harmonization
Similar to IGUANe paper Figure showing intensity distribution shifts
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add training directory to path (adjusted for new location)
sys.path.append(str(Path(__file__).parent.parent / 'training'))
from models import build_2d_generator


def compute_intensity_histogram(images, bins=50, range=(0, 1)):
    """Compute intensity histogram for brain tissue only"""
    all_intensities = []
    for img in images:
        brain_mask = img > 0.1
        brain_intensities = img[brain_mask]
        if len(brain_intensities) > 0:
            all_intensities.extend(brain_intensities.flatten())
    
    hist, bin_edges = np.histogram(all_intensities, bins=bins, range=range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist


def load_test_data(test_data_path):
    """Load test data"""
    with open(test_data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def generate_histogram_comparison(test_data_path, weight_path, output_path, 
                                  reference_site='BCH_CHD', max_samples=100):
    """
    Generate histogram comparison figure
    
    Args:
        test_data_path: Path to test data pickle
        weight_path: Path to generator weights
        output_path: Where to save figure
        reference_site: Reference site name
        max_samples: Maximum samples to include
    """
    print("Loading data...")
    data = load_test_data(test_data_path)
    
    images = data['images']
    ga = data['gestational_age']
    sites = data['site']
    
    # Get non-reference sites
    non_ref_mask = sites != reference_site
    non_ref_images = images[non_ref_mask][:max_samples]
    non_ref_ga = ga[non_ref_mask][:max_samples]
    non_ref_sites = sites[non_ref_mask][:max_samples]
    
    # Get reference site samples
    ref_mask = sites == reference_site
    ref_images = images[ref_mask][:max_samples]
    
    print(f"Non-reference samples: {len(non_ref_images)}")
    print(f"Reference samples: {len(ref_images)}")
    
    # Load generator
    print("Loading generator...")
    gen = build_2d_generator((138, 176, 1), 16)
    gen.load_weights(weight_path)
    
    # Generate harmonized images
    print("Generating harmonized images...")
    harmonized_images = gen.predict([non_ref_images, non_ref_ga], batch_size=16, verbose=1)
    
    # Compute histograms
    print("Computing histograms...")
    
    # Before harmonization: non-reference sites
    bins_before, hist_before = compute_intensity_histogram(non_ref_images)
    
    # After harmonization
    bins_after, hist_after = compute_intensity_histogram(harmonized_images)
    
    # Reference site (target distribution)
    bins_ref, hist_ref = compute_intensity_histogram(ref_images)
    
    # Normalize histograms
    hist_before = hist_before / hist_before.sum()
    hist_after = hist_after / hist_after.sum()
    hist_ref = hist_ref / hist_ref.sum()
    
    # Create figure
    print("Creating figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot before harmonization
    ax1.plot(bins_before, hist_before, linewidth=2, label='Source Sites', color='blue', alpha=0.7)
    ax1.plot(bins_ref, hist_ref, linewidth=2, label='Target (BCH)', color='red', alpha=0.7)
    ax1.set_xlabel('Image Intensity (a.u.)', fontsize=12)
    ax1.set_ylabel('Count (all sites/all patients)', fontsize=12)
    ax1.set_title('Before harmonization', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # Plot after harmonization
    ax2.plot(bins_after, hist_after, linewidth=2, label='Harmonized', color='green', alpha=0.7)
    ax2.plot(bins_ref, hist_ref, linewidth=2, label='Target (BCH)', color='red', alpha=0.7)
    ax2.set_xlabel('Image Intensity (a.u.)', fontsize=12)
    ax2.set_ylabel('Count (all sites/all patients)', fontsize=12)
    ax2.set_title('After harmonization', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()
    
    # Compute statistics
    print("\nIntensity Statistics:")
    print(f"Before - Mean: {np.mean([img[img>0.1].mean() for img in non_ref_images]):.4f}, "
          f"Std: {np.mean([img[img>0.1].std() for img in non_ref_images]):.4f}")
    print(f"After  - Mean: {np.mean([img[img>0.1].mean() for img in harmonized_images]):.4f}, "
          f"Std: {np.mean([img[img>0.1].std() for img in harmonized_images]):.4f}")
    print(f"Target - Mean: {np.mean([img[img>0.1].mean() for img in ref_images]):.4f}, "
          f"Std: {np.mean([img[img>0.1].std() for img in ref_images]):.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference_site', default='BCH_CHD')
    parser.add_argument('--max_samples', type=int, default=100)
    
    args = parser.parse_args()
    
    generate_histogram_comparison(
        args.test_data,
        args.weights,
        args.output,
        args.reference_site,
        args.max_samples
    )
