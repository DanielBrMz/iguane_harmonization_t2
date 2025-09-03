#!/usr/bin/env python3
"""
Create Visual Quality Comparison Figures for Multi-Site Harmonization
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import seaborn as sns
from skimage import exposure

import tensorflow as tf
from tensorflow.keras import layers, Model

print("=" * 80)
print("FETAL BRAIN HARMONIZATION - VISUAL COMPARISON")
print("=" * 80)


# ============================================================================
# MODEL ARCHITECTURE (same as evaluation script)
# ============================================================================

def build_2d_generator(input_shape=(138, 176, 1), ga_embedding_dim=16, name='generator'):
    """2D U-Net Generator with GA Conditioning"""
    
    img_input = layers.Input(shape=input_shape, name='image_input')
    ga_input = layers.Input(shape=(1,), name='ga_input')
    
    # GA embedding
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_input)
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_embedding)

    # Encoder
    x = layers.Conv2D(32, 3, padding='same')(img_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    skip1 = x
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    skip2 = x
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    skip3 = x
    x = layers.MaxPooling2D(2)(x)
    
    # Bottleneck
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Inject GA
    ga_spatial = layers.RepeatVector(x.shape[1] * x.shape[2])(ga_embedding)
    ga_spatial = layers.Reshape((x.shape[1], x.shape[2], ga_embedding_dim))(ga_spatial)
    x = layers.Concatenate()([x, ga_spatial])
    
    # Decoder
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    if x.shape[1] != skip3.shape[1] or x.shape[2] != skip3.shape[2]:
        x = layers.Resizing(skip3.shape[1], skip3.shape[2])(x)
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    if x.shape[1] != skip2.shape[1] or x.shape[2] != skip2.shape[2]:
        x = layers.Resizing(skip2.shape[1], skip2.shape[2])(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    if x.shape[1] != skip1.shape[1] or x.shape[2] != skip1.shape[2]:
        x = layers.Resizing(skip1.shape[1], skip1.shape[2])(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Final
    if x.shape[1] != input_shape[0] or x.shape[2] != input_shape[1]:
        x = layers.Resizing(input_shape[0], input_shape[1])(x)
    
    output = layers.Conv2D(1, 1, padding='same', activation='tanh')(x)
    output = layers.Lambda(lambda x: (x + 1.0) / 2.0)(output)
    
    model = Model(inputs=[img_input, ga_input], outputs=output, name=name)
    
    return model


# ============================================================================
# CONTRAST ENHANCEMENT FUNCTIONS
# ============================================================================

def apply_clahe_single(img, clip_limit=0.03):
    """Apply CLAHE to a single image"""
    if img.shape[-1] == 1:
        img_2d = img[:, :, 0]
    else:
        img_2d = img
    
    # Apply CLAHE
    img_clahe = exposure.equalize_adapthist(img_2d, clip_limit=clip_limit)
    
    # Restore channel dimension if needed
    if len(img.shape) == 3 and img.shape[-1] == 1:
        img_clahe = img_clahe[:, :, np.newaxis]
    
    return img_clahe


def contrast_stretch_single(img, percentile_low=2, percentile_high=98):
    """Apply contrast stretching to a single image"""
    mask = (img > 0.05) & (img < 0.95)
    
    if mask.sum() > 100:
        brain_pixels = img[mask]
        p_low = np.percentile(brain_pixels, percentile_low)
        p_high = np.percentile(brain_pixels, percentile_high)
        
        if p_high > p_low:
            img_stretched = np.clip((img - p_low) / (p_high - p_low), 0, 1)
        else:
            img_stretched = img
    else:
        img_stretched = img
    
    return img_stretched


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comparison_panel(original, harmonized, site_name, ga, output_path, epoch='final', show_enhanced=True):
    """
    Create a comparison panel for a single subject
    Showing: Original | Harmonized (Raw) | Enhanced | Difference
    """
    
    n_cols = 5 if show_enhanced else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    # Squeeze out channel dimension
    if original.shape[-1] == 1:
        original = original[:, :, 0]
        harmonized_display = harmonized[:, :, 0]
    else:
        harmonized_display = harmonized
    
    col_idx = 0
    
    # Original
    axes[col_idx].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[col_idx].set_title(f'Original\n{site_name}\nGA: {ga:.1f}w', fontsize=12)
    axes[col_idx].axis('off')
    col_idx += 1
    
    # Harmonized (Raw)
    axes[col_idx].imshow(harmonized_display, cmap='gray', vmin=0, vmax=1)
    harm_std = np.std(harmonized_display)
    axes[col_idx].set_title(f'Harmonized (Raw)\nEpoch: {epoch}\nStd: {harm_std:.4f}', fontsize=12)
    axes[col_idx].axis('off')
    col_idx += 1
    
    if show_enhanced:
        # CLAHE Enhanced
        harmonized_clahe = apply_clahe_single(harmonized, clip_limit=0.03)
        if harmonized_clahe.shape[-1] == 1:
            harmonized_clahe = harmonized_clahe[:, :, 0]
        axes[col_idx].imshow(harmonized_clahe, cmap='gray', vmin=0, vmax=1)
        axes[col_idx].set_title(f'CLAHE Enhanced\nStd: {np.std(harmonized_clahe):.4f}', fontsize=12)
        axes[col_idx].axis('off')
        col_idx += 1
        
        # Contrast Stretched
        harmonized_stretched = contrast_stretch_single(harmonized, percentile_low=2, percentile_high=98)
        if harmonized_stretched.shape[-1] == 1:
            harmonized_stretched = harmonized_stretched[:, :, 0]
        axes[col_idx].imshow(harmonized_stretched, cmap='gray', vmin=0, vmax=1)
        axes[col_idx].set_title(f'Contrast Stretched\nStd: {np.std(harmonized_stretched):.4f}', fontsize=12)
        axes[col_idx].axis('off')
        col_idx += 1
    
    # Difference (using raw harmonized)
    diff = np.abs(original - harmonized_display)
    im = axes[col_idx].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    axes[col_idx].set_title(f'Difference (Raw)\nMAE: {diff.mean():.4f}', fontsize=12)
    axes[col_idx].axis('off')
    plt.colorbar(im, ax=axes[col_idx], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_multisite_comparison(site_images, output_path, epoch='final'):
    """
    Create a comprehensive multi-site comparison figure
    One row per site, showing multiple examples
    """
    
    n_sites = len(site_images)
    n_examples = 5  # Show 5 examples per site
    
    fig = plt.figure(figsize=(20, 4 * n_sites))
    gs = gridspec.GridSpec(n_sites, n_examples * 3, figure=fig, hspace=0.3, wspace=0.1)
    
    for site_idx, (site_name, images_dict) in enumerate(site_images.items()):
        originals = images_dict['originals'][:n_examples]
        harmonized = images_dict['harmonized'][:n_examples]
        gas = images_dict['gas'][:n_examples]
        
        for ex_idx in range(len(originals)):
            # Squeeze channel dimension
            orig = originals[ex_idx]
            harm = harmonized[ex_idx]
            if orig.shape[-1] == 1:
                orig = orig[:, :, 0]
                harm = harm[:, :, 0]
            
            # Original
            ax_orig = fig.add_subplot(gs[site_idx, ex_idx * 3])
            ax_orig.imshow(orig, cmap='gray', vmin=0, vmax=1)
            if ex_idx == 0:
                ax_orig.set_ylabel(site_name, fontsize=12, fontweight='bold')
            if site_idx == 0:
                ax_orig.set_title(f'Original\nGA:{gas[ex_idx]:.1f}w', fontsize=10)
            ax_orig.axis('off')
            
            # Harmonized
            ax_harm = fig.add_subplot(gs[site_idx, ex_idx * 3 + 1])
            ax_harm.imshow(harm, cmap='gray', vmin=0, vmax=1)
            if site_idx == 0:
                ax_harm.set_title(f'Harmonized', fontsize=10)
            ax_harm.axis('off')
            
            # Difference
            diff = np.abs(orig - harm)
            ax_diff = fig.add_subplot(gs[site_idx, ex_idx * 3 + 2])
            im = ax_diff.imshow(diff, cmap='hot', vmin=0, vmax=0.3)
            if site_idx == 0:
                ax_diff.set_title(f'Difference', fontsize=10)
            ax_diff.axis('off')
    
    plt.suptitle(f'Multi-Site Harmonization Comparison (Epoch {epoch})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved histogram comparison: {output_path}")


def create_diagnostic_panel(site_images, output_path, epoch='final'):
    """
    Create diagnostic panel to detect contrast compression
    Shows statistics and enhancement comparisons
    """
    n_sites = len(site_images)
    fig, axes = plt.subplots(n_sites, 4, figsize=(20, 5 * n_sites))
    
    # Handle single site case
    if n_sites == 1:
        axes = axes.reshape(1, -1)
    
    for site_idx, (site_name, images_dict) in enumerate(site_images.items()):
        harmonized = images_dict['harmonized']
        
        # Take first example for visualization
        harm_img = harmonized[0]
        if harm_img.shape[-1] == 1:
            harm_img = harm_img[:, :, 0]
        
        # Calculate statistics
        mean_val = np.mean(harm_img)
        std_val = np.std(harm_img)
        min_val = np.min(harm_img)
        max_val = np.max(harm_img)
        
        # Detect compression
        is_compressed = std_val < 0.05
        is_gray = (mean_val > 0.45) and (mean_val < 0.55)
        
        # 1. Raw harmonized
        axes[site_idx, 0].imshow(harm_img, cmap='gray', vmin=0, vmax=1)
        status = '⚠️ COMPRESSED' if is_compressed else '✓ Normal'
        axes[site_idx, 0].set_title(
            f'{site_name}\nRaw Harmonized\n'
            f'Mean: {mean_val:.3f}, Std: {std_val:.4f}\n{status}',
            fontsize=10
        )
        axes[site_idx, 0].axis('off')
        
        # 2. CLAHE Enhanced
        harm_clahe = apply_clahe_single(harmonized[0], clip_limit=0.03)
        if harm_clahe.shape[-1] == 1:
            harm_clahe = harm_clahe[:, :, 0]
        axes[site_idx, 1].imshow(harm_clahe, cmap='gray', vmin=0, vmax=1)
        axes[site_idx, 1].set_title(
            f'CLAHE Enhanced\nStd: {np.std(harm_clahe):.4f}',
            fontsize=10
        )
        axes[site_idx, 1].axis('off')
        
        # 3. Intensity histogram
        axes[site_idx, 2].hist(harm_img.flatten(), bins=50, alpha=0.7, 
                               color='blue', label='Raw', density=True)
        axes[site_idx, 2].hist(harm_clahe.flatten(), bins=50, alpha=0.7,
                               color='red', label='CLAHE', density=True)
        axes[site_idx, 2].axvline(mean_val, color='blue', linestyle='--', 
                                  linewidth=2, label=f'Raw Mean={mean_val:.3f}')
        axes[site_idx, 2].set_xlabel('Intensity')
        axes[site_idx, 2].set_ylabel('Density')
        axes[site_idx, 2].legend(fontsize=8)
        axes[site_idx, 2].grid(alpha=0.3)
        
        # 4. Intensity profile (central horizontal line)
        center_row = harm_img.shape[0] // 2
        profile_raw = harm_img[center_row, :]
        profile_clahe = harm_clahe[center_row, :]
        
        axes[site_idx, 3].plot(profile_raw, label='Raw', linewidth=2)
        axes[site_idx, 3].plot(profile_clahe, label='CLAHE', linewidth=2)
        axes[site_idx, 3].axhline(mean_val, color='gray', linestyle='--', 
                                  alpha=0.5, label=f'Mean={mean_val:.3f}')
        axes[site_idx, 3].set_xlabel('Position (pixels)')
        axes[site_idx, 3].set_ylabel('Intensity')
        axes[site_idx, 3].set_title(f'Central Horizontal Profile\nRange: [{min_val:.3f}, {max_val:.3f}]', 
                                    fontsize=10)
        axes[site_idx, 3].legend(fontsize=8)
        axes[site_idx, 3].grid(alpha=0.3)
    
    plt.suptitle(f'Contrast Compression Diagnostics (Epoch {epoch})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved diagnostic panel: {output_path}")


def create_histogram_comparison(site_images, output_path):
    """Create intensity histogram comparison before/after harmonization"""
    
    n_sites = len(site_images)
    fig, axes = plt.subplots(2, (n_sites + 1) // 2, figsize=(15, 8))
    # Always flatten axes to handle single or multiple subplots
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for idx, (site_name, images_dict) in enumerate(site_images.items()):
        originals = images_dict['originals']
        harmonized = images_dict['harmonized']
        
        # Flatten all images
        orig_flat = np.concatenate([img.flatten() for img in originals])
        harm_flat = np.concatenate([img.flatten() for img in harmonized])
        
        # Plot histograms
        axes[idx].hist(orig_flat, bins=50, alpha=0.5, label='Original', 
                      color='blue', density=True)
        axes[idx].hist(harm_flat, bins=50, alpha=0.5, label='Harmonized', 
                      color='red', density=True)
        axes[idx].set_title(site_name)
        axes[idx].set_xlabel('Intensity')
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(site_images), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Intensity Distribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved histogram comparison: {output_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def create_visual_comparison(args):
    """Main function to create all visual comparisons"""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load test data
    print(f"\nLoading test data from: {args.test_data}")
    with open(args.test_data, 'rb') as f:
        test_data = pickle.load(f)
    
    test_images = test_data['images'].astype(float) / 255.0
    test_ga = test_data['gestational_age']
    test_sites = test_data['site']
    
    print(f"  Loaded {len(test_images)} slices from {len(np.unique(test_sites))} sites")
    
    # Site mapping for generators
    site_to_gen = {
        'BCH_CHD': None,
        'BCH_Placenta': 'BCHPlacenta',
        'HBCD_Site5_Arkansas_UNC': 'HBCDSite5ArkansasUNC',
        'HBCD_Site6_Cincinnati': 'HBCDSite6Cincinnati',
        'VGH_Unknown': 'VGHUnknown',
        'VGH_Site4_GE': 'VGHUnknown',
        'dHCP': 'dHCP'
    }
    
    # Process each epoch
    for epoch in args.epochs:
        print(f"\n{'='*80}")
        print(f"CREATING VISUALS FOR EPOCH {epoch}")
        print(f"{'='*80}")
        
        site_images = {}
        
        for site in np.unique(test_sites):
            if site == 'BCH_CHD':
                continue  # Skip reference site
            
            print(f"\n--- Processing site: {site} ---")
            
            # Get samples for this site
            site_idx = np.where(test_sites == site)[0]
            n_samples = min(len(site_idx), args.n_samples_per_site)
            site_idx = site_idx[:n_samples]
            
            # Get generator name
            gen_name = site_to_gen.get(site)
            if gen_name is None:
                print(f"  Warning: No generator for {site}")
                continue
            
            # Load generator
            gen = build_2d_generator((138, 176, 1), 16, name=f'gen_BCH2{gen_name}')
            
            if epoch == 'final':
                weight_file = Path(args.weight_dir) / f'gen_BCH2{gen_name}_final.weights.h5'
            else:
                weight_file = Path(args.weight_dir) / f'gen_BCH2{gen_name}_epoch_{epoch}.weights.h5'
            
            if not weight_file.exists():
                print(f"  Warning: Weights not found: {weight_file}")
                continue
            
            gen.load_weights(str(weight_file))
            print(f"  ✓ Loaded weights: {weight_file.name}")
            
            # Generate harmonized images
            originals = []
            harmonized_list = []
            gas = []
            
            for idx in site_idx:
                original = test_images[idx]
                ga_value = test_ga[idx].reshape(1, 1)
                
                # Generate
                img_batch = np.expand_dims(original, axis=0)
                harmonized = gen([img_batch, ga_value], training=False).numpy()[0]
                
                originals.append(original)
                harmonized_list.append(harmonized)
                gas.append(test_ga[idx])
            
            site_images[site] = {
                'originals': originals,
                'harmonized': harmonized_list,
                'gas': gas
            }
            
            # Create individual comparison panels
            individual_dir = output_dir / f'epoch_{epoch}_individual'
            individual_dir.mkdir(exist_ok=True)
            
            for i in range(min(3, len(originals))):
                panel_path = individual_dir / f'{site}_sample_{i+1}.png'
                create_comparison_panel(
                    originals[i], 
                    harmonized_list[i], 
                    site, 
                    gas[i],
                    panel_path,
                    epoch
                )
            
            print(f"  ✓ Processed {len(originals)} samples")
        
        # Create multi-site comparison
        if site_images:
            multisite_path = output_dir / f'multisite_comparison_epoch_{epoch}_{timestamp}.png'
            create_multisite_comparison(site_images, multisite_path, epoch)
            
            # Create histogram comparison
            histogram_path = output_dir / f'intensity_histograms_epoch_{epoch}_{timestamp}.png'
            create_histogram_comparison(site_images, histogram_path)
            
            # Create diagnostic panel (NEW!)
            diagnostic_path = output_dir / f'diagnostic_panel_epoch_{epoch}_{timestamp}.png'
            create_diagnostic_panel(site_images, diagnostic_path, epoch)
    
    print(f"\n{'='*80}")
    print("VISUAL COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"  Results saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Create visual comparison figures for harmonization'
    )
    
    parser.add_argument(
        '--test_data',
        default='processed_data_4slice_fixed/test_4slice_data.pkl',
        help='Path to test data pickle file'
    )
    
    parser.add_argument(
        '--weight_dir',
        default='weights/cyclegan_2d',
        help='Directory containing model weights'
    )
    
    parser.add_argument(
        '--output_dir',
        default='visual_comparison',
        help='Output directory for figures'
    )
    
    parser.add_argument(
        '--epochs',
        nargs='+',
        default=['150', '200', 'final'],
        help='Epochs to visualize (e.g., --epochs 150 final)'
    )
    
    parser.add_argument(
        '--n_samples_per_site',
        type=int,
        default=10,
        help='Number of samples per site to visualize'
    )
    
    args = parser.parse_args()
    
    # Convert epoch strings to appropriate format
    args.epochs = [int(e) if e.isdigit() else e for e in args.epochs]
    
    create_visual_comparison(args)


if __name__ == '__main__':
    main()