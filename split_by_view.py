"""
Split dataset into view-specific subsets
"""

import pickle
import numpy as np


def split_by_view(labeled_data_path):
    """
    Create separate datasets for sagittal and axial/coronal
    """
    print("Loading labeled data...")
    with open(labeled_data_path, 'rb') as f:
        data = pickle.load(f)
    
    images = data['images']
    sites = data['site']
    ga = data['gestational_age']
    views = data['view']
    
    if 'sex' in data:
        sex = data['sex']
    else:
        sex = np.array(['unknown'] * len(images))
    
    # Split by view
    sagittal_mask = views == 'sagittal'
    axial_mask = views == 'axial'
    
    print(f"\nTotal images: {len(images)}")
    print(f"Sagittal: {sagittal_mask.sum()}")
    print(f"Axial/Coronal: {axial_mask.sum()}")
    
    # Create sagittal dataset
    sagittal_data = {
        'images': images[sagittal_mask],
        'site': sites[sagittal_mask],
        'ga': ga[sagittal_mask],
        'sex': sex[sagittal_mask],
        'view': views[sagittal_mask]
    }
    
    # Create axial dataset
    axial_data = {
        'images': images[axial_mask],
        'site': sites[axial_mask],
        'ga': ga[axial_mask],
        'sex': sex[axial_mask],
        'view': views[axial_mask]
    }
    
    # Check site distribution per view
    print("\n" + "="*70)
    print("SAGITTAL DATASET - SITE DISTRIBUTION")
    print("="*70)
    unique_sites_sag, counts_sag = np.unique(sagittal_data['site'], return_counts=True)
    for site, count in zip(unique_sites_sag, counts_sag):
        print(f"{site}: {count}")
    
    print("\n" + "="*70)
    print("AXIAL/CORONAL DATASET - SITE DISTRIBUTION")
    print("="*70)
    unique_sites_ax, counts_ax = np.unique(axial_data['site'], return_counts=True)
    for site, count in zip(unique_sites_ax, counts_ax):
        print(f"{site}: {count}")
    
    # Save separate datasets
    base_path = labeled_data_path.replace('_with_views.pkl', '')
    
    sagittal_path = base_path + '_sagittal_only.pkl'
    axial_path = base_path + '_axial_only.pkl'
    
    with open(sagittal_path, 'wb') as f:
        pickle.dump(sagittal_data, f)
    print(f"\nSaved sagittal dataset: {sagittal_path}")
    
    with open(axial_path, 'wb') as f:
        pickle.dump(axial_data, f)
    print(f"Saved axial dataset: {axial_path}")
    
    # Recommendations
    print("\n" + "="*70)
    print("TRAINING RECOMMENDATIONS")
    print("="*70)
    
    if len(unique_sites_sag) >= 3 and min(counts_sag) >= 50:
        print("\n✓ Sagittal dataset suitable for training")
        print(f"  Sites: {len(unique_sites_sag)}, Min samples per site: {min(counts_sag)}")
    else:
        print("\n⚠️  Sagittal dataset may be limited")
        print(f"  Sites: {len(unique_sites_sag)}, Min samples per site: {min(counts_sag)}")
    
    if len(unique_sites_ax) >= 3 and min(counts_ax) >= 50:
        print("\n✓ Axial dataset suitable for training")
        print(f"  Sites: {len(unique_sites_ax)}, Min samples per site: {min(counts_ax)}")
    else:
        print("\n⚠️  Axial dataset may be limited")
        print(f"  Sites: {len(unique_sites_ax)}, Min samples per site: {min(counts_ax)}")
    
    return sagittal_data, axial_data


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python split_by_view.py <train_data_with_views.pkl>")
        sys.exit(1)
    
    split_by_view(sys.argv[1])