import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from train_fetal_2d_cyclegan import build_2d_generator

print("="*80)
print("EVALUATING CYCLEGAN HARMONIZATION RESULTS")
print("="*80)

# Load test data
print("\nLoading test data...")
with open('processed_data_4slice_fixed/test_4slice_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

test_images = test_data['images'][:20] / 255.0  # First 20 slices
test_ga = test_data['gestational_age'][:20]
test_site = test_data['site'][:20]

print(f"Test images: {test_images.shape}")
print(f"Test sites: {np.unique(test_site)}")

# Create results directory
Path('results/cyclegan_2d/evaluation').mkdir(parents=True, exist_ok=True)

# Test different checkpoints
epochs_to_test = [50, 100, 150, 'final']

for epoch in epochs_to_test:
    print(f"\n{'='*80}")
    print(f"EVALUATING EPOCH {epoch}")
    print(f"{'='*80}")
    
    # Build generator
    gen_B2A = build_2d_generator((138, 176, 1), 16, name='gen_B2A')
    
    # Load weights
    try:
        if epoch == 'final':
            weight_file = 'weights/cyclegan_2d/gen_B2A_final.weights.h5'
        else:
            weight_file = f'weights/cyclegan_2d/gen_B2A_epoch_{epoch}.weights.h5'
        
        gen_B2A.load_weights(weight_file)
        print(f"✓ Loaded: {weight_file}")
    except Exception as e:
        print(f"✗ Could not load weights: {e}")
        continue
    
    # Harmonize test images
    print("Generating harmonized images...")
    harmonized_images = []
    
    for i in range(len(test_images)):
        img = test_images[i:i+1]
        ga = test_ga[i:i+1].reshape(-1, 1)
        
        harm = gen_B2A([img, ga], training=False).numpy()
        harmonized_images.append(harm[0])
    
    harmonized_images = np.array(harmonized_images)
    
    # Calculate statistics
    differences = np.abs(test_images - harmonized_images)
    mean_diff = differences.mean()
    max_diff = differences.max()
    
    print(f"\nStatistics:")
    print(f"  Mean pixel difference: {mean_diff:.4f}")
    print(f"  Max pixel difference: {max_diff:.4f}")
    
    if mean_diff < 0.01:
        print(f"  ⚠️  WARNING: Model is mostly copying input (mean diff < 0.01)!")
        status = "IDENTITY_MAPPING"
    elif mean_diff < 0.05:
        print(f"  ⚠️  Subtle changes only (mean diff < 0.05)")
        status = "SUBTLE_CHANGES"
    else:
        print(f"  ✓ Significant harmonization detected!")
        status = "HARMONIZING"
    
    # Visualize samples
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    fig.suptitle(f'Epoch {epoch} - Status: {status}', fontsize=16, y=0.995)
    
    for i in range(5):
        # Original
        axes[i, 0].imshow(test_images[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Original\n{test_site[i]}\nGA: {test_ga[i]:.1f}')
        axes[i, 0].axis('off')
        
        # Harmonized
        axes[i, 1].imshow(harmonized_images[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Harmonized\n→ BCH_CHD')
        axes[i, 1].axis('off')
        
        # Difference
        diff = np.abs(test_images[i, :, :, 0] - harmonized_images[i, :, :, 0])
        im = axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[i, 2].set_title(f'Difference\nMean: {diff.mean():.4f}')
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)
    
    plt.tight_layout()
    output_file = f'results/cyclegan_2d/evaluation/epoch_{epoch}_samples.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")
    
    # Histogram of differences
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(differences.flatten(), bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Absolute Pixel Difference')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Pixel Differences - Epoch {epoch}\nMean: {mean_diff:.4f}, Status: {status}')
    ax.axvline(mean_diff, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_diff:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_file = f'results/cyclegan_2d/evaluation/epoch_{epoch}_histogram.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print("\nCheck the results in: results/cyclegan_2d/evaluation/")
print("\nRECOMMENDATION:")
print("  - If mean_diff < 0.01: Model failed (just copying)")
print("  - If mean_diff 0.01-0.05: Subtle harmonization (might be useful)")
print("  - If mean_diff > 0.05: Good harmonization")
print("\nCompare all epochs and pick the best one!")