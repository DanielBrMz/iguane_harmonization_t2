#!/usr/bin/env python3
"""
Complete Harmonization Pipeline with Post-Processing Enhancement
================================================================

This script does EVERYTHING in one go:
1. Loads test data
2. Applies CycleGAN harmonization 
3. Post-processes to enhance visual quality
4. Creates comparison figures
5. Saves all outputs

The goal: Harmonized images that look like the originals but with site-specific 
differences removed - NOT washed out!

Usage:
    python harmonize_and_enhance.py --model_weights weights/cyclegan_2d/ --test_data processed_data_4slice/test_4slice_data.pkl
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, Model
from skimage import exposure
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
from skimage.filters import threshold_otsu

print("=" * 80)
print("COMPLETE HARMONIZATION + ENHANCEMENT PIPELINE")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
print("=" * 80)


# ============================================================================
# GENERATOR ARCHITECTURE (must match training)
# ============================================================================

def build_2d_generator(input_shape=(138, 176, 1), ga_embedding_dim=16, name='generator'):
    """
    2D U-Net Generator - matches training architecture
    """
    
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
    
    if x.shape[1] != input_shape[0] or x.shape[2] != input_shape[1]:
        x = layers.Resizing(input_shape[0], input_shape[1])(x)
    
    # Output with tanh, then scale to [0, 1]
    output = layers.Conv2D(1, 1, padding='same', activation='tanh')(x)
    output = layers.Lambda(lambda x: (x + 1.0) / 2.0)(output)
    
    model = Model(inputs=[img_input, ga_input], outputs=output, name=name)
    
    return model


# ============================================================================
# BRAIN MASK GENERATION
# ============================================================================

def generate_brain_mask(image, method='adaptive'):
    """
    Generate brain mask from image
    
    Args:
        image: 2D image [0, 1]
        method: 'adaptive', 'otsu', or 'threshold'
    
    Returns:
        brain_mask: boolean array
    """
    if method == 'otsu':
        try:
            thresh = threshold_otsu(image)
            brain_mask = image > thresh
        except:
            brain_mask = image > (image.max() * 0.1)
            
    elif method == 'adaptive':
        # Multi-level thresholding
        thresh1 = image.max() * 0.05
        thresh2 = image.max() * 0.15
        
        mask1 = image > thresh1
        mask2 = image > thresh2
        
        # Use connected components
        from skimage.measure import label
        from scipy.ndimage import binary_fill_holes
        
        labeled = label(mask2)
        if labeled.max() > 0:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            largest = sizes.argmax()
            brain_mask = labeled == largest
            brain_mask = binary_fill_holes(brain_mask)
        else:
            brain_mask = mask1
    else:
        # Simple threshold
        brain_mask = image > (image.max() * 0.1)
    
    # Morphological cleanup
    brain_mask = binary_opening(brain_mask, structure=np.ones((3, 3)))
    brain_mask = binary_closing(brain_mask, structure=np.ones((5, 5)))
    
    return brain_mask


# ============================================================================
# POST-PROCESSING ENHANCEMENT
# ============================================================================

def enhance_contrast_clahe(image, brain_mask, clip_limit=0.03):
    """
    Apply CLAHE only to brain region
    """
    enhanced = image.copy()
    
    if brain_mask.sum() == 0:
        return enhanced
    
    brain_region = image[brain_mask]
    brain_min, brain_max = brain_region.min(), brain_region.max()
    
    if brain_max - brain_min < 1e-6:
        return enhanced
    
    # Normalize to [0, 1]
    brain_normalized = (image - brain_min) / (brain_max - brain_min + 1e-8)
    brain_normalized = np.clip(brain_normalized, 0, 1)
    
    # Apply CLAHE
    brain_uint8 = (brain_normalized * 255).astype(np.uint8)
    enhanced_uint8 = exposure.equalize_adapthist(
        brain_uint8, 
        clip_limit=clip_limit,
        nbins=256
    )
    
    # Convert back
    enhanced_brain = enhanced_uint8 * (brain_max - brain_min) + brain_min
    enhanced[brain_mask] = enhanced_brain[brain_mask]
    
    return enhanced


def preserve_details(harmonized, original, brain_mask, alpha=0.25, sigma=1.5):
    """
    Add back high-frequency details from original
    """
    # Extract high-frequency component
    original_smooth = gaussian_filter(original, sigma=sigma)
    original_highfreq = original - original_smooth
    
    # Add to harmonized
    enhanced = harmonized.copy()
    enhanced[brain_mask] = (
        harmonized[brain_mask] + alpha * original_highfreq[brain_mask]
    )
    
    enhanced = np.clip(enhanced, 0, 1)
    
    return enhanced


def match_contrast_statistics(harmonized, original, brain_mask):
    """
    Match mean and std of harmonized to original (brain region only)
    """
    matched = harmonized.copy()
    
    if brain_mask.sum() == 0:
        return matched
    
    harm_brain = harmonized[brain_mask]
    orig_brain = original[brain_mask]
    
    orig_mean = orig_brain.mean()
    orig_std = orig_brain.std()
    harm_mean = harm_brain.mean()
    harm_std = harm_brain.std()
    
    if harm_std < 1e-6:
        return matched
    
    # Standardize and rescale
    matched_brain = (harm_brain - harm_mean) / harm_std
    matched_brain = matched_brain * orig_std + orig_mean
    matched_brain = np.clip(matched_brain, 0, 1)
    
    matched[brain_mask] = matched_brain
    
    return matched


def enhance_harmonized_image(harmonized, original, 
                            clahe_clip=0.03,
                            detail_alpha=0.25,
                            detail_sigma=1.5,
                            match_stats=True):
    """
    Complete enhancement pipeline to make harmonized look natural
    
    Args:
        harmonized: raw harmonized output [0, 1]
        original: original image [0, 1]
        clahe_clip: CLAHE enhancement strength
        detail_alpha: high-frequency detail weight
        detail_sigma: Gaussian sigma for detail extraction
        match_stats: match contrast statistics to original
    
    Returns:
        enhanced: enhanced harmonized image [0, 1]
        brain_mask: generated brain mask
    """
    # Step 1: Generate brain mask from original
    brain_mask = generate_brain_mask(original, method='adaptive')
    
    # Step 2: CLAHE enhancement
    enhanced = enhance_contrast_clahe(harmonized, brain_mask, clip_limit=clahe_clip)
    
    # Step 3: Preserve high-frequency details
    enhanced = preserve_details(enhanced, original, brain_mask, 
                               alpha=detail_alpha, sigma=detail_sigma)
    
    # Step 4: Match contrast statistics
    if match_stats:
        enhanced = match_contrast_statistics(enhanced, original, brain_mask)
    
    # Step 5: Ensure background is zero
    enhanced[~brain_mask] = 0
    
    return enhanced, brain_mask


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comparison_figure(original, raw_harmonized, enhanced_harmonized, 
                            brain_mask, site_name, subject_id, save_path=None):
    """
    Create comprehensive comparison figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Images
    axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Original ({site_name})\nMean: {original[brain_mask].mean():.3f}, Std: {original[brain_mask].std():.3f}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(raw_harmonized, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Harmonized (Raw)\nMean: {raw_harmonized[brain_mask].mean():.3f}, Std: {raw_harmonized[brain_mask].std():.3f}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(enhanced_harmonized, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Harmonized (Enhanced)\nMean: {enhanced_harmonized[brain_mask].mean():.3f}, Std: {enhanced_harmonized[brain_mask].std():.3f}')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(brain_mask, cmap='gray')
    axes[0, 3].set_title(f'Brain Mask\nPixels: {brain_mask.sum()}')
    axes[0, 3].axis('off')
    
    # Row 2: Difference maps
    diff_orig_raw = np.abs(original - raw_harmonized)
    im1 = axes[1, 0].imshow(diff_orig_raw, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 0].set_title(f'|Original - Raw|\nMAE: {diff_orig_raw[brain_mask].mean():.4f}')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    diff_orig_enh = np.abs(original - enhanced_harmonized)
    im2 = axes[1, 1].imshow(diff_orig_enh, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 1].set_title(f'|Original - Enhanced|\nMAE: {diff_orig_enh[brain_mask].mean():.4f}')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    # Histograms
    bins = np.linspace(0, 1, 50)
    axes[1, 2].hist(original[brain_mask], bins=bins, alpha=0.5, label='Original', density=True)
    axes[1, 2].hist(raw_harmonized[brain_mask], bins=bins, alpha=0.5, label='Raw', density=True)
    axes[1, 2].hist(enhanced_harmonized[brain_mask], bins=bins, alpha=0.5, label='Enhanced', density=True)
    axes[1, 2].set_xlabel('Intensity')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Intensity Distribution (Brain)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Central profile
    center_row = original.shape[0] // 2
    axes[1, 3].plot(original[center_row, :], label='Original', linewidth=2, alpha=0.7)
    axes[1, 3].plot(raw_harmonized[center_row, :], label='Raw', linewidth=2, alpha=0.7)
    axes[1, 3].plot(enhanced_harmonized[center_row, :], label='Enhanced', linewidth=2, alpha=0.7)
    axes[1, 3].set_xlabel('Column')
    axes[1, 3].set_ylabel('Intensity')
    axes[1, 3].set_title(f'Central Profile (Row {center_row})')
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.suptitle(f'Harmonization Pipeline - Subject {subject_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_test_data(test_data_path, model_weights_dir, output_dir,
                     clahe_clip=0.03, detail_alpha=0.25, n_samples_per_site=10):
    """
    Complete processing pipeline
    
    Args:
        test_data_path: path to test_4slice_data.pkl
        model_weights_dir: directory with model weights
        output_dir: output directory for results
        clahe_clip: CLAHE enhancement strength
        detail_alpha: detail preservation weight
        n_samples_per_site: number of samples to process per site
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'raw_harmonized').mkdir(exist_ok=True)
    (output_dir / 'enhanced_harmonized').mkdir(exist_ok=True)
    (output_dir / 'comparison_figures').mkdir(exist_ok=True)
    (output_dir / 'brain_masks').mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    with open(test_data_path, 'rb') as f:
        data = pickle.load(f)
    
    images = data['images']
    if images.max() > 1.0:
        images = images / 255.0
    
    ga = data['gestational_age']
    sites = data['site']
    subject_ids = data.get('subject_id', np.arange(len(images)))
    
    print(f"Loaded {len(images)} test images")
    print(f"Sites: {np.unique(sites)}")
    
    # Replace NaN GA values
    ga_mean = np.nanmean(ga)
    ga = np.where(np.isnan(ga), ga_mean, ga)
    
    print("\n" + "="*80)
    print("STEP 2: LOADING MODEL")
    print("="*80)
    
    # Build generator
    generator = build_2d_generator(input_shape=images.shape[1:])
    
    # Load weights
    model_weights_dir = Path(model_weights_dir)
    weight_files = list(model_weights_dir.glob('gen_site2BCH_final.weights.h5'))
    
    if not weight_files:
        weight_files = list(model_weights_dir.glob('gen_site2BCH_epoch_*.weights.h5'))
        if weight_files:
            # Use latest epoch
            weight_files.sort()
            weight_file = weight_files[-1]
            print(f"Using checkpoint: {weight_file.name}")
        else:
            raise FileNotFoundError(f"No generator weights found in {model_weights_dir}")
    else:
        weight_file = weight_files[0]
        print(f"Using final weights: {weight_file.name}")
    
    generator.load_weights(str(weight_file))
    print("✓ Model loaded successfully")
    
    print("\n" + "="*80)
    print("STEP 3: PROCESSING IMAGES")
    print("="*80)
    print(f"Enhancement parameters:")
    print(f"  CLAHE clip limit: {clahe_clip}")
    print(f"  Detail alpha: {detail_alpha}")
    print(f"  Samples per site: {n_samples_per_site}")
    print("="*80)
    
    # Get unique sites
    unique_sites = np.unique(sites)
    
    # Process samples from each site
    all_results = []
    
    for site in unique_sites:
        print(f"\n--- Processing site: {site} ---")
        
        # Get indices for this site
        site_indices = np.where(sites == site)[0]
        
        if len(site_indices) == 0:
            print(f"  No samples found for {site}")
            continue
        
        # Select random samples
        if len(site_indices) > n_samples_per_site:
            selected_indices = np.random.choice(site_indices, n_samples_per_site, replace=False)
        else:
            selected_indices = site_indices
        
        print(f"  Processing {len(selected_indices)} samples")
        
        for idx in tqdm(selected_indices, desc=f"  {site}"):
            original = images[idx, :, :, 0]
            ga_value = ga[idx].reshape(1, 1)
            subject_id = subject_ids[idx]
            
            # Step A: Harmonize with CycleGAN
            img_batch = np.expand_dims(images[idx], axis=0)
            raw_harmonized = generator([img_batch, ga_value], training=False).numpy()[0, :, :, 0]
            
            # Step B: Enhance post-processing
            enhanced_harmonized, brain_mask = enhance_harmonized_image(
                raw_harmonized, 
                original,
                clahe_clip=clahe_clip,
                detail_alpha=detail_alpha,
                detail_sigma=1.5,
                match_stats=True
            )
            
            # Save outputs
            filename_base = f"{site}_{subject_id}_idx{idx}"
            
            np.save(output_dir / 'raw_harmonized' / f'{filename_base}_raw.npy', raw_harmonized)
            np.save(output_dir / 'enhanced_harmonized' / f'{filename_base}_enhanced.npy', enhanced_harmonized)
            np.save(output_dir / 'brain_masks' / f'{filename_base}_mask.npy', brain_mask)
            
            # Create comparison figure
            fig = create_comparison_figure(
                original, 
                raw_harmonized, 
                enhanced_harmonized,
                brain_mask,
                site,
                subject_id,
                save_path=output_dir / 'comparison_figures' / f'{filename_base}_comparison.png'
            )
            
            # Store results
            if brain_mask.sum() > 0:
                result = {
                    'site': site,
                    'subject_id': subject_id,
                    'idx': idx,
                    'original_mean': original[brain_mask].mean(),
                    'original_std': original[brain_mask].std(),
                    'raw_mean': raw_harmonized[brain_mask].mean(),
                    'raw_std': raw_harmonized[brain_mask].std(),
                    'enhanced_mean': enhanced_harmonized[brain_mask].mean(),
                    'enhanced_std': enhanced_harmonized[brain_mask].std(),
                    'contrast_improvement': enhanced_harmonized[brain_mask].std() / (raw_harmonized[brain_mask].std() + 1e-8),
                    'mae_raw': np.abs(original - raw_harmonized)[brain_mask].mean(),
                    'mae_enhanced': np.abs(original - enhanced_harmonized)[brain_mask].mean()
                }
                all_results.append(result)
    
    print("\n" + "="*80)
    print("STEP 4: CREATING SUMMARY")
    print("="*80)
    
    if all_results:
        # Create summary DataFrame
        import pandas as pd
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / 'enhancement_summary.csv', index=False)
        
        print("\nOverall Statistics:")
        print(f"  Processed: {len(results_df)} images")
        print(f"\nContrast (Std):")
        print(f"  Original:  {results_df['original_std'].mean():.4f} ± {results_df['original_std'].std():.4f}")
        print(f"  Raw:       {results_df['raw_std'].mean():.4f} ± {results_df['raw_std'].std():.4f}")
        print(f"  Enhanced:  {results_df['enhanced_std'].mean():.4f} ± {results_df['enhanced_std'].std():.4f}")
        print(f"\nContrast Improvement:")
        print(f"  Average:   {results_df['contrast_improvement'].mean():.2f}x")
        print(f"  Median:    {results_df['contrast_improvement'].median():.2f}x")
        print(f"\nMAE (Mean Absolute Error):")
        print(f"  Raw:       {results_df['mae_raw'].mean():.4f}")
        print(f"  Enhanced:  {results_df['mae_enhanced'].mean():.4f}")
        
        # Per-site summary
        print("\nPer-Site Statistics:")
        for site in results_df['site'].unique():
            site_df = results_df[results_df['site'] == site]
            print(f"\n  {site}:")
            print(f"    Samples: {len(site_df)}")
            print(f"    Contrast improvement: {site_df['contrast_improvement'].mean():.2f}x")
            print(f"    Enhanced std: {site_df['enhanced_std'].mean():.4f}")
        
        # Create summary visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Contrast comparison
        sites_list = results_df['site'].unique()
        x = np.arange(len(sites_list))
        width = 0.25
        
        orig_stds = [results_df[results_df['site']==s]['original_std'].mean() for s in sites_list]
        raw_stds = [results_df[results_df['site']==s]['raw_std'].mean() for s in sites_list]
        enh_stds = [results_df[results_df['site']==s]['enhanced_std'].mean() for s in sites_list]
        
        axes[0, 0].bar(x - width, orig_stds, width, label='Original', alpha=0.8)
        axes[0, 0].bar(x, raw_stds, width, label='Raw Harmonized', alpha=0.8)
        axes[0, 0].bar(x + width, enh_stds, width, label='Enhanced', alpha=0.8)
        axes[0, 0].set_xlabel('Site')
        axes[0, 0].set_ylabel('Standard Deviation')
        axes[0, 0].set_title('Contrast Comparison by Site')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(sites_list, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Contrast improvement
        improvements = [results_df[results_df['site']==s]['contrast_improvement'].mean() for s in sites_list]
        axes[0, 1].bar(sites_list, improvements, color='green', alpha=0.7)
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', label='No improvement')
        axes[0, 1].set_xlabel('Site')
        axes[0, 1].set_ylabel('Contrast Improvement (x)')
        axes[0, 1].set_title('Contrast Improvement Factor')
        axes[0, 1].set_xticklabels(sites_list, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: MAE comparison
        mae_raw = [results_df[results_df['site']==s]['mae_raw'].mean() for s in sites_list]
        mae_enh = [results_df[results_df['site']==s]['mae_enhanced'].mean() for s in sites_list]
        
        axes[1, 0].bar(x - width/2, mae_raw, width, label='Raw', alpha=0.8)
        axes[1, 0].bar(x + width/2, mae_enh, width, label='Enhanced', alpha=0.8)
        axes[1, 0].set_xlabel('Site')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('Mean Absolute Error to Original')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(sites_list, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Distribution of contrast improvements
        axes[1, 1].hist(results_df['contrast_improvement'], bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='No improvement')
        axes[1, 1].axvline(x=results_df['contrast_improvement'].mean(), color='blue', linestyle='-', linewidth=2, label=f"Mean: {results_df['contrast_improvement'].mean():.2f}x")
        axes[1, 1].set_xlabel('Contrast Improvement Factor')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Contrast Improvements')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Harmonization Enhancement Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'summary_statistics.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Summary visualization saved: {output_dir / 'summary_statistics.png'}")
    
    print("\n" + "="*80)
    print("✓ PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - Raw harmonized: {output_dir / 'raw_harmonized'}")
    print(f"  - Enhanced harmonized: {output_dir / 'enhanced_harmonized'}")
    print(f"  - Brain masks: {output_dir / 'brain_masks'}")
    print(f"  - Comparison figures: {output_dir / 'comparison_figures'}")
    print(f"  - Summary: {output_dir / 'enhancement_summary.csv'}")
    print(f"  - Summary plot: {output_dir / 'summary_statistics.png'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Complete harmonization pipeline with post-processing enhancement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python harmonize_and_enhance.py \\
      --test_data processed_data_4slice/test_4slice_data.pkl \\
      --model_weights weights/cyclegan_2d/ \\
      --output harmonized_results/ \\
      --n_samples 10 \\
      --clahe_clip 0.03 \\
      --detail_alpha 0.25
        """
    )
    
    parser.add_argument('--test_data', required=True,
                       help='Path to test_4slice_data.pkl')
    parser.add_argument('--model_weights', required=True,
                       help='Directory with model weights')
    parser.add_argument('--output', default='./harmonized_results',
                       help='Output directory')
    parser.add_argument('--n_samples', type=int, default=10,
                       help='Number of samples per site to process')
    parser.add_argument('--clahe_clip', type=float, default=0.03,
                       help='CLAHE clip limit for contrast enhancement')
    parser.add_argument('--detail_alpha', type=float, default=0.25,
                       help='Weight for high-frequency detail preservation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    print("="*80)
    print("HARMONIZATION + ENHANCEMENT PIPELINE")
    print("="*80)
    print(f"Test data: {args.test_data}")
    print(f"Model weights: {args.model_weights}")
    print(f"Output: {args.output}")
    print(f"Samples per site: {args.n_samples}")
    print(f"CLAHE clip: {args.clahe_clip}")
    print(f"Detail alpha: {args.detail_alpha}")
    print(f"Random seed: {args.seed}")
    print("="*80)
    
    # Process
    process_test_data(
        args.test_data,
        args.model_weights,
        args.output,
        clahe_clip=args.clahe_clip,
        detail_alpha=args.detail_alpha,
        n_samples_per_site=args.n_samples
    )


if __name__ == "__main__":
    main()