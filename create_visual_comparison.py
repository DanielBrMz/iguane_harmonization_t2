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
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comparison_panel(original, harmonized, site_name, ga, output_path, epoch='final'):
    """
    Create a comparison panel for a single subject
    Showing: Original | Harmonized | Difference
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Squeeze out channel dimension
    if original.shape[-1] == 1:
        original = original[:, :, 0]
        harmonized = harmonized[:, :, 0]
    
    # Original
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Original\n{site_name}\nGA: {ga:.1f}w', fontsize=12)
    axes[0].axis('off')
    
    # Harmonized
    axes[1].imshow(harmonized, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Harmonized → BCH\nEpoch: {epoch}', fontsize=12)
    axes[1].axis('off')
    
    # Difference
    diff = np.abs(original - harmonized)
    im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    axes[2].set_title(f'Difference\nMAE: {diff.mean():.4f}', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
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
    print(f"✓ Saved multi-site comparison: {output_path}")


def create_histogram_comparison(site_images, output_path):
    """Create intensity histogram comparison before/after harmonization"""
    
    n_sites = len(site_images)
    fig, axes = plt.subplots(2, (n_sites + 1) // 2, figsize=(15, 8))
    axes = axes.flatten() if n_sites > 1 else [axes]
    
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