"""
Semi-automated view labeling with manual correction interface
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def extract_view_features(image_2d):
    """
    Extract discriminative features for view classification
    """
    if image_2d.ndim == 3:
        image_2d = image_2d[:, :, 0]
    
    h, w = image_2d.shape
    
    # Feature 1: Left-right symmetry (key discriminator)
    left_half = image_2d[:, :w//2]
    right_half = np.fliplr(image_2d[:, w//2:])
    min_w = min(left_half.shape[1], right_half.shape[1])
    
    left_crop = left_half[:, :min_w]
    right_crop = right_half[:, :min_w]
    
    valid_mask = (left_crop > 0.1) & (right_crop > 0.1)
    
    if valid_mask.sum() > 100:
        left_vals = left_crop[valid_mask]
        right_vals = right_crop[valid_mask]
        lr_correlation = np.corrcoef(left_vals, right_vals)[0, 1]
    else:
        lr_correlation = 0
    
    # Feature 2: Aspect ratio
    brain_mask = image_2d > 0.1
    if brain_mask.sum() < 100:
        return np.array([0, 0, 0])
    
    coords = np.argwhere(brain_mask)
    y_range = coords[:, 0].max() - coords[:, 0].min()
    x_range = coords[:, 1].max() - coords[:, 1].min()
    aspect_ratio = y_range / (x_range + 1e-8)
    
    # Feature 3: Horizontal vs vertical edge strength
    edges_y = np.abs(np.diff(image_2d, axis=0)).sum()
    edges_x = np.abs(np.diff(image_2d, axis=1)).sum()
    edge_ratio = edges_x / (edges_y + 1e-8)
    
    return np.array([lr_correlation, aspect_ratio, edge_ratio])


def auto_classify_with_manual_correction(data_path, output_path, n_verify=50):
    """
    Auto-classify views, then allow manual correction
    """
    print("Loading data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    images = data['images']
    sites = data['site']
    n_images = len(images)
    
    print(f"Extracting features from {n_images} images...")
    features = []
    valid_indices = []
    
    for i, img in enumerate(images):
        if i % 100 == 0:
            print(f"  {i}/{n_images}")
        
        feat = extract_view_features(img[:, :, 0])
        
        if not np.all(feat == 0):
            features.append(feat)
            valid_indices.append(i)
    
    features = np.array(features)
    valid_indices = np.array(valid_indices)
    
    print(f"\nValid images: {len(valid_indices)}/{n_images}")
    
    # Normalize and cluster
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-means with 2 clusters (sagittal vs axial/coronal)
    print("\nClustering into 2 view groups...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Determine which cluster is sagittal
    # Sagittal has LOW lr_correlation, axial/coronal has HIGH
    cluster_0_lr_corr = features[cluster_labels == 0, 0].mean()
    cluster_1_lr_corr = features[cluster_labels == 1, 0].mean()
    
    if cluster_0_lr_corr < cluster_1_lr_corr:
        # Cluster 0 is sagittal, cluster 1 is axial/coronal
        sagittal_cluster = 0
        axial_cluster = 1
    else:
        sagittal_cluster = 1
        axial_cluster = 0
    
    print(f"\nCluster {sagittal_cluster}: SAGITTAL (LR corr = {features[cluster_labels == sagittal_cluster, 0].mean():.3f})")
    print(f"Cluster {axial_cluster}: AXIAL/CORONAL (LR corr = {features[cluster_labels == axial_cluster, 0].mean():.3f})")
    
    # Create initial labels
    view_labels = np.array(['unknown'] * n_images, dtype=object)
    view_labels[valid_indices[cluster_labels == sagittal_cluster]] = 'sagittal'
    view_labels[valid_indices[cluster_labels == axial_cluster]] = 'axial'
    
    # Manual verification interface
    print(f"\n{'='*70}")
    print("MANUAL VERIFICATION")
    print(f"{'='*70}")
    print(f"\nShowing {n_verify} random samples for verification...")
    print("This will help assess clustering accuracy.\n")
    
    # Sample from each cluster
    samples_per_cluster = n_verify // 2
    
    fig, axes = plt.subplots(2, samples_per_cluster, figsize=(20, 6))
    
    for c, view_name in [(sagittal_cluster, 'SAGITTAL'), (axial_cluster, 'AXIAL/CORONAL')]:
        cluster_indices = valid_indices[cluster_labels == c]
        sampled = np.random.choice(cluster_indices, 
                                   min(samples_per_cluster, len(cluster_indices)), 
                                   replace=False)
        
        row = 0 if c == sagittal_cluster else 1
        
        for i, idx in enumerate(sampled[:samples_per_cluster]):
            if i < samples_per_cluster:
                axes[row, i].imshow(images[idx, :, :, 0], cmap='gray')
                axes[row, i].set_title(f'{view_name}\nSample {i}', fontsize=8)
                axes[row, i].axis('off')
    
    plt.suptitle('Automated View Classification - Verify Accuracy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('view_classification_verification.png', dpi=150, bbox_inches='tight')
    print("Saved: view_classification_verification.png")
    plt.close()
    
    # Ask for confirmation
    print("\nPlease examine 'view_classification_verification.png'")
    print("Top row should be SAGITTAL (side views)")
    print("Bottom row should be AXIAL/CORONAL (top-down/front views)")
    
    while True:
        response = input("\nDoes the clustering look correct? (y/n): ").strip().lower()
        if response in ['y', 'n']:
            break
    
    if response == 'n':
        print("\nClustering needs adjustment. Swapping labels...")
        view_labels[valid_indices[cluster_labels == sagittal_cluster]] = 'axial'
        view_labels[valid_indices[cluster_labels == axial_cluster]] = 'sagittal'
    
    # Statistics
    print("\n" + "="*70)
    print("VIEW DISTRIBUTION")
    print("="*70)
    
    unique_views, counts = np.unique(view_labels, return_counts=True)
    for view, count in zip(unique_views, counts):
        print(f"{view}: {count} ({100*count/n_images:.1f}%)")
    
    # Per-site distribution
    print("\n" + "="*70)
    print("VIEW DISTRIBUTION BY SITE")
    print("="*70)
    
    for site in np.unique(sites):
        site_mask = sites == site
        site_views = view_labels[site_mask]
        
        n_total = len(site_views)
        n_sagittal = (site_views == 'sagittal').sum()
        n_axial = (site_views == 'axial').sum()
        n_unknown = (site_views == 'unknown').sum()
        
        print(f"\n{site}: {n_total} total")
        print(f"  Sagittal: {n_sagittal} ({100*n_sagittal/n_total:.1f}%)")
        print(f"  Axial/Coronal: {n_axial} ({100*n_axial/n_total:.1f}%)")
        if n_unknown > 0:
            print(f"  Unknown: {n_unknown} ({100*n_unknown/n_total:.1f}%)")
    
    # Save labeled data
    data['view'] = view_labels
    data['view_features'] = np.zeros((n_images, 3))
    data['view_features'][valid_indices] = features
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n{'='*70}")
    print(f"Saved labeled data to: {output_path}")
    print(f"{'='*70}")
    
    return view_labels, features, cluster_labels


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python auto_classify_views.py <train_data.pkl>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = input_path.replace('.pkl', '_with_views.pkl')
    
    view_labels, features, clusters = auto_classify_with_manual_correction(
        input_path, 
        output_path,
        n_verify=50
    )