"""
Automatic anatomical view classification for fetal brain MRI
Based on structural features and symmetry patterns
"""

import numpy as np
import nibabel as nib
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
import pickle


def extract_view_features(image_2d):
    """
    Extract features that discriminate between anatomical views
    
    Axial: Horizontal brain slice, bilateral symmetry, circular/oval shape
    Coronal: Frontal view, bilateral symmetry, butterfly/horseshoe shape
    Sagittal: Side view, NO bilateral symmetry, C-shaped profile
    
    Returns:
        feature_vector: 15-dimensional feature vector
    """
    # Ensure 2D
    if image_2d.ndim == 3:
        image_2d = image_2d[:, :, 0]
    
    h, w = image_2d.shape
    
    # Feature 1-2: Aspect ratio and circularity
    brain_mask = image_2d > 0.1
    if brain_mask.sum() < 100:
        return np.zeros(15)
    
    coords = np.argwhere(brain_mask)
    y_range = coords[:, 0].max() - coords[:, 0].min()
    x_range = coords[:, 1].max() - coords[:, 1].min()
    aspect_ratio = y_range / (x_range + 1e-8)
    
    area = brain_mask.sum()
    perimeter = ndimage.binary_erosion(brain_mask).sum()
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
    
    # Feature 3-4: Left-right symmetry (critical for sagittal detection)
    left_half = image_2d[:, :w//2]
    right_half = np.fliplr(image_2d[:, w//2:])
    min_w = min(left_half.shape[1], right_half.shape[1])
    
    left_crop = left_half[:, :min_w]
    right_crop = right_half[:, :min_w]
    
    # Mask valid pixels
    valid_mask = (left_crop > 0.1) & (right_crop > 0.1)
    
    if valid_mask.sum() > 100:
        # Pearson correlation
        left_vals = left_crop[valid_mask]
        right_vals = right_crop[valid_mask]
        lr_correlation = np.corrcoef(left_vals, right_vals)[0, 1]
        
        # Mean absolute difference
        lr_difference = np.mean(np.abs(left_crop - right_crop))
    else:
        lr_correlation = 0
        lr_difference = 1
    
    # Feature 5-6: Top-bottom symmetry (distinguishes axial from coronal)
    top_half = image_2d[:h//2, :]
    bottom_half = np.flipud(image_2d[h//2:, :])
    min_h = min(top_half.shape[0], bottom_half.shape[0])
    
    top_crop = top_half[:min_h, :]
    bottom_crop = bottom_half[:min_h, :]
    
    valid_mask_tb = (top_crop > 0.1) & (bottom_crop > 0.1)
    
    if valid_mask_tb.sum() > 100:
        top_vals = top_crop[valid_mask_tb]
        bottom_vals = bottom_crop[valid_mask_tb]
        tb_correlation = np.corrcoef(top_vals, bottom_vals)[0, 1]
        tb_difference = np.mean(np.abs(top_crop - bottom_crop))
    else:
        tb_correlation = 0
        tb_difference = 1
    
    # Feature 7-8: Edge orientation (sagittal has strong vertical edges)
    edges_vertical = np.abs(np.diff(image_2d, axis=0)).sum()
    edges_horizontal = np.abs(np.diff(image_2d, axis=1)).sum()
    edge_ratio = edges_horizontal / (edges_vertical + 1e-8)
    edge_anisotropy = abs(edges_horizontal - edges_vertical) / (edges_horizontal + edges_vertical + 1e-8)
    
    # Feature 9-10: Shape moments
    from scipy import ndimage
    moments = ndimage.moments(brain_mask.astype(float), order=2)
    mu20 = moments[2, 0] / (moments[0, 0] + 1e-8) - (moments[1, 0] / (moments[0, 0] + 1e-8))**2
    mu02 = moments[0, 2] / (moments[0, 0] + 1e-8) - (moments[0, 1] / (moments[0, 0] + 1e-8))**2
    orientation = 0.5 * np.arctan2(2 * moments[1, 1], mu20 - mu02)
    eccentricity = np.sqrt(1 - min(mu20, mu02) / (max(mu20, mu02) + 1e-8))
    
    # Feature 11-12: Intensity distribution shape
    brain_intensities = image_2d[brain_mask]
    intensity_std = np.std(brain_intensities)
    intensity_skew = np.mean((brain_intensities - brain_intensities.mean())**3) / (intensity_std**3 + 1e-8)
    
    # Feature 13: Compactness
    compactness = area / (np.sqrt(area) * np.pi)
    
    # Feature 14-15: Center of mass position
    com = ndimage.center_of_mass(brain_mask)
    com_y_norm = com[0] / h
    com_x_norm = com[1] / w
    
    features = np.array([
        aspect_ratio,           # 0
        circularity,           # 1
        lr_correlation,        # 2 - HIGH for axial/coronal, LOW for sagittal
        lr_difference,         # 3 - LOW for axial/coronal, HIGH for sagittal
        tb_correlation,        # 4 - Varies by view
        tb_difference,         # 5
        edge_ratio,            # 6 - Different for each view
        edge_anisotropy,       # 7
        orientation,           # 8
        eccentricity,          # 9
        intensity_std,         # 10
        intensity_skew,        # 11
        compactness,           # 12
        com_y_norm,            # 13
        com_x_norm             # 14
    ])
    
    # Replace NaN/Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features


def create_pseudo_labels_clustering(data_path, n_manual_verify=50):
    """
    Create pseudo-labels using unsupervised clustering + manual verification
    
    Strategy:
    1. Extract features for all images
    2. Cluster into 2-3 groups (likely sagittal vs axial/coronal)
    3. Show representative images from each cluster for manual verification
    4. User labels clusters as 'sagittal', 'coronal', or 'axial'
    """
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    print("Loading data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    images = data['images']
    n_images = len(images)
    
    print(f"Extracting features from {n_images} images...")
    features = []
    valid_indices = []
    
    for i, img in enumerate(images):
        if i % 100 == 0:
            print(f"  {i}/{n_images}")
        
        feat = extract_view_features(img[:, :, 0])
        
        # Only include images with valid features
        if not np.all(feat == 0):
            features.append(feat)
            valid_indices.append(i)
    
    features = np.array(features)
    valid_indices = np.array(valid_indices)
    
    print(f"\nValid images: {len(valid_indices)}/{n_images}")
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Cluster into 2 groups (sagittal vs coronal/axial)
    # Start with 2 clusters, can increase to 3 if needed
    print("\nClustering images...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Analyze clusters
    print("\nCluster analysis:")
    for c in range(2):
        cluster_mask = cluster_labels == c
        cluster_features = features[cluster_mask]
        
        print(f"\nCluster {c}: {cluster_mask.sum()} images")
        print(f"  LR correlation: {cluster_features[:, 2].mean():.3f} ± {cluster_features[:, 2].std():.3f}")
        print(f"  LR difference:  {cluster_features[:, 3].mean():.3f} ± {cluster_features[:, 3].std():.3f}")
        print(f"  Edge ratio:     {cluster_features[:, 6].mean():.3f} ± {cluster_features[:, 6].std():.3f}")
    
    # Show representative images for manual verification
    print(f"\nShowing {n_manual_verify} representative images for manual labeling...")
    
    # Sample images from each cluster
    samples_per_cluster = n_manual_verify // 2
    
    fig, axes = plt.subplots(4, samples_per_cluster // 2, figsize=(20, 12))
    axes = axes.flatten()
    
    manual_labels = {}
    
    for c in range(2):
        cluster_indices = np.where(cluster_labels == c)[0]
        sampled = np.random.choice(cluster_indices, 
                                   size=min(samples_per_cluster, len(cluster_indices)), 
                                   replace=False)
        
        for i, idx in enumerate(sampled):
            ax_idx = c * samples_per_cluster + i
            if ax_idx < len(axes):
                img_idx = valid_indices[idx]
                axes[ax_idx].imshow(images[img_idx, :, :, 0], cmap='gray')
                axes[ax_idx].set_title(f'Cluster {c} - Sample {i}')
                axes[ax_idx].axis('off')
    
    plt.suptitle('Manual View Verification\nTop 2 rows: Cluster 0 | Bottom 2 rows: Cluster 1', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_verification.png', dpi=150, bbox_inches='tight')
    print("\nSaved cluster_verification.png")
    print("\nPlease examine the images and determine:")
    print("  Cluster 0 view type (sagittal/coronal/axial)?")
    print("  Cluster 1 view type (sagittal/coronal/axial)?")
    
    # Interactive labeling
    cluster_0_label = input("\nCluster 0 is (s=sagittal, c=coronal, a=axial): ").strip().lower()
    cluster_1_label = input("Cluster 1 is (s=sagittal, c=coronal, a=axial): ").strip().lower()
    
    label_map = {'s': 'sagittal', 'c': 'coronal', 'a': 'axial'}
    cluster_0_view = label_map.get(cluster_0_label, 'unknown')
    cluster_1_view = label_map.get(cluster_1_label, 'unknown')
    
    # Assign labels
    view_labels = np.array(['unknown'] * n_images, dtype=object)
    view_labels[valid_indices[cluster_labels == 0]] = cluster_0_view
    view_labels[valid_indices[cluster_labels == 1]] = cluster_1_view
    
    # Statistics
    print("\nFinal view distribution:")
    unique, counts = np.unique(view_labels, return_counts=True)
    for view, count in zip(unique, counts):
        print(f"  {view}: {count} ({100*count/n_images:.1f}%)")
    
    # Per-site distribution
    sites = data['site']
    print("\nPer-site view distribution:")
    for site in np.unique(sites):
        site_mask = sites == site
        site_views = view_labels[site_mask]
        print(f"\n  {site}:")
        unique_site, counts_site = np.unique(site_views, return_counts=True)
        for view, count in zip(unique_site, counts_site):
            print(f"    {view}: {count} ({100*count/site_mask.sum():.1f}%)")
    
    # Save results
    output_data = {
        'view_labels': view_labels,
        'features': features,
        'valid_indices': valid_indices,
        'cluster_labels': cluster_labels,
        'scaler': scaler,
        'kmeans': kmeans
    }
    
    output_path = data_path.replace('.pkl', '_with_view_labels.pkl')
    
    # Add view labels to original data
    data['view'] = view_labels
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nSaved labeled data to: {output_path}")
    
    return view_labels, features, cluster_labels


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python classify_views.py <train_data.pkl>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    view_labels, features, clusters = create_pseudo_labels_clustering(data_path)