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
    python harmonize_and_enhance.py --model_weights weights/cyclegan_2d/ --test_data processed_data_4slice/test_4slice_data.pkl --validation_csv stackQC_50stacks.csv
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
import numpy as np
import pickle
import pandas as pd
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
    
    if brain_region.max() - brain_region.min() < 1e-6:
        return enhanced
    
    # Normalize to [0, 1] for CLAHE
    brain_min = brain_region.min()
    brain_max = brain_region.max()
    brain_norm = (brain_region - brain_min) / (brain_max - brain_min + 1e-8)
    
    # Apply CLAHE
    brain_equalized = exposure.equalize_adapthist(
        brain_norm.reshape(-1, 1), 
        clip_limit=clip_limit
    ).flatten()
    
    # Rescale to original range
    brain_enhanced = brain_equalized * (brain_max - brain_min) + brain_min
    
    # Update only brain region
    enhanced[brain_mask] = brain_enhanced
    
    return enhanced


def preserve_high_frequency_details(original, harmonized, brain_mask, alpha=0.25):
    """
    Preserve high-frequency details from original
    """
    if brain_mask.sum() == 0:
        return harmonized
    
    # Extract high-frequency details
    sigma = 1.0
    original_smooth = gaussian_filter(original, sigma=sigma)
    harmonized_smooth = gaussian_filter(harmonized, sigma=sigma)
    
    original_details = original - original_smooth
    harmonized_details = harmonized - harmonized_smooth
    
    # Blend details
    blended_details = alpha * original_details + (1 - alpha) * harmonized_details
    
    # Add back to harmonized
    enhanced = harmonized_smooth + blended_details
    
    # Apply only to brain
    result = harmonized.copy()
    result[brain_mask] = enhanced[brain_mask]
    
    return result


def enhance_harmonized_image(original, harmonized, clahe_clip=0.03, detail_alpha=0.25):
    """
    Complete enhancement pipeline
    """
    # Generate brain mask
    brain_mask = generate_brain_mask(original, method='adaptive')
    
    # Step 1: Enhance contrast with CLAHE
    enhanced = enhance_contrast_clahe(harmonized, brain_mask, clip_limit=clahe_clip)
    
    # Step 2: Preserve high-frequency details
    enhanced = preserve_high_frequency_details(original, enhanced, brain_mask, alpha=detail_alpha)
    
    # Step 3: Match intensity statistics to original
    if brain_mask.sum() > 0:
        original_brain = original[brain_mask]
        enhanced_brain = enhanced[brain_mask]
        
        # Match mean and std
        orig_mean = original_brain.mean()
        orig_std = original_brain.std()
        enh_mean = enhanced_brain.mean()
        enh_std = enhanced_brain.std()
        
        if enh_std > 1e-6:
            enhanced_brain = (enhanced_brain - enh_mean) / enh_std
            enhanced_brain = enhanced_brain * orig_std + orig_mean
            enhanced[brain_mask] = enhanced_brain
    
    # Clip to [0, 1]
    enhanced = np.clip(enhanced, 0, 1)
    
    return enhanced, brain_mask


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_multi_subject_grid(subjects_data, save_path):
    """
    Create a grid showing all subjects in a single image
    subjects_data: list of dicts with keys: 
        'original', 'target_bch', 'enhanced', 'brain_mask', 'site', 'subject_id', 'split'
    """
    n_subjects = len(subjects_data)
    n_cols = 4  # Original, Target BCH, Harmonized, Difference
    n_rows = n_subjects
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, data in enumerate(subjects_data):
        original = data['original']
        target_bch = data['target_bch']
        enhanced = data['enhanced']
        brain_mask = data['brain_mask']
        site = data['site']
        subject_id = data['subject_id']
        split = data['split']
        
        # Calculate difference map
        difference = np.abs(original - enhanced)
        
        # Calculate statistics
        orig_mean = original.mean()
        orig_std = original.std()
        target_mean = target_bch.mean()
        target_std = target_bch.std()
        enh_mean = enhanced.mean()
        enh_std = enhanced.std()
        
        # Calculate difference statistics in brain region only
        if brain_mask.sum() > 0:
            diff_mean = difference[brain_mask].mean()
            diff_max = difference[brain_mask].max()
        else:
            diff_mean = difference.mean()
            diff_max = difference.max()
        
        # Original
        axes[i, 0].imshow(original, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Original ({site})\nMean: {orig_mean:.3f}, Std: {orig_std:.3f}', fontsize=10)
        axes[i, 0].axis('off')
        
        # Target BCH
        axes[i, 1].imshow(target_bch, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Target BCH\nMean: {target_mean:.3f}, Std: {target_std:.3f}', fontsize=10)
        axes[i, 1].axis('off')
        
        # Harmonized
        axes[i, 2].imshow(enhanced, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Harmonized\nMean: {enh_mean:.3f}, Std: {enh_std:.3f}', fontsize=10)
        axes[i, 2].axis('off')
        
        # Difference map with blue-white-red colormap
        im = axes[i, 3].imshow(difference, cmap='hot', vmin=0, vmax=0.3)
        axes[i, 3].set_title(f'Difference |Orig - Harm|\nMean: {diff_mean:.4f}, Max: {diff_max:.4f}', fontsize=10)
        axes[i, 3].axis('off')
        
        # Add colorbar only to the first row
        if i == 0:
            cbar = plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
            cbar.set_label('Intensity Difference', rotation=270, labelpad=15, fontsize=9)
            cbar.ax.tick_params(labelsize=8)
        
        # Add row label
        axes[i, 0].text(-0.1, 0.5, f'Subject {subject_id}\n({split})', 
                        transform=axes[i, 0].transAxes,
                        fontsize=12, fontweight='bold',
                        verticalalignment='center',
                        rotation=90)
    
    plt.suptitle(f'Harmonization Pipeline - All Subjects', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-subject grid saved: {save_path}")


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_test_data(test_data_path, model_weights_dir, output_dir, 
                     validation_csv_path=None,
                     clahe_clip=0.03, detail_alpha=0.25, 
                     n_samples_per_site=10):
    """
    Complete processing pipeline
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    (output_dir / 'raw_harmonized').mkdir(exist_ok=True)
    (output_dir / 'enhanced_harmonized').mkdir(exist_ok=True)
    (output_dir / 'brain_masks').mkdir(exist_ok=True)
    (output_dir / 'comparison_figures').mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    with open(test_data_path, 'rb') as f:
        data = pickle.load(f)
    
    images = data['images']
    sites = data['site']
    subject_ids = data['subject_id']
    ga = data['gestational_age']
    
    print(f"Loaded {len(images)} test samples")
    print(f"Image shape: {images.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Image range: [{images.min()}, {images.max()}]")
    
    # Normalize images to [0, 1] if they're in [0, 255]
    if images.max() > 1.0:
        print(f"Normalizing images from [0, 255] to [0, 1]")
        images = images.astype(np.float32) / 255.0
    else:
        images = images.astype(np.float32)
    
    print(f"After normalization: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Sites: {np.unique(sites)}")
    
    # Handle NaN GA values
    ga_mean = np.nanmean(ga)
    ga = np.where(np.isnan(ga), ga_mean, ga)
    
    # Load validation subjects if provided
    validation_subjects = []
    validation_data = {}
    if validation_csv_path and Path(validation_csv_path).exists():
        print(f"\nLoading validation subjects from {validation_csv_path}")
        val_df = pd.read_csv(validation_csv_path)
        validation_subjects = val_df['file'].astype(int).tolist()
        validation_data = {int(row['file']): row['Site'] for _, row in val_df.iterrows()}
        print(f"Found {len(validation_subjects)} validation subjects")
    
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
    print("Model loaded successfully")
    
    print("\n" + "="*80)
    print("STEP 3: PROCESSING IMAGES")
    print("="*80)
    print(f"Enhancement parameters:")
    print(f"  CLAHE clip limit: {clahe_clip}")
    print(f"  Detail alpha: {detail_alpha}")
    print(f"  Samples per site: {n_samples_per_site}")
    print("="*80)
    
    # Separate test and validation subjects
    test_mask = np.array([sid not in validation_subjects for sid in subject_ids])
    val_mask = np.array([sid in validation_subjects for sid in subject_ids])
    
    # Get BCH samples for target visualization
    bch_mask = sites == 'BCH'
    bch_indices = np.where(bch_mask)[0]
    
    if len(bch_indices) == 0:
        print("\nWARNING: No BCH samples found in test set!")
        print(f"Available sites: {np.unique(sites)}")
        # Use any available samples as "target"
        bch_indices = np.arange(min(10, len(images)))
    else:
        print(f"\nFound {len(bch_indices)} BCH samples for target visualization")
    
    # Process by site
    unique_sites = np.unique(sites)
    all_subjects_data = []
    all_results = []
    
    for site in unique_sites:
        if site == 'BCH':
            continue  # Skip BCH as it's the target
            
        print(f"\n--- Processing site: {site} ---")
        
        # Get test samples for this site
        site_test_mask = (sites == site) & test_mask
        site_test_indices = np.where(site_test_mask)[0]
        
        # Get validation samples for this site
        site_val_mask = (sites == site) & val_mask
        site_val_indices = np.where(site_val_mask)[0]
        
        # Select samples
        if len(site_test_indices) > n_samples_per_site:
            selected_test = np.random.choice(site_test_indices, n_samples_per_site, replace=False)
        else:
            selected_test = site_test_indices
        
        selected_val = site_val_indices
        
        print(f"  Test samples: {len(selected_test)}")
        print(f"  Validation samples: {len(selected_val)}")
        
        # Process test samples
        for idx in tqdm(selected_test, desc=f"  Test {site}"):
            original = images[idx, :, :, 0]
            ga_value = ga[idx].reshape(1, 1)
            subject_id = subject_ids[idx]
            
            # Get corresponding BCH target
            if len(bch_indices) > 0:
                bch_idx = np.random.choice(bch_indices)
                target_bch = images[bch_idx, :, :, 0]
            else:
                target_bch = np.zeros_like(original)
            
            # Harmonize
            img_batch = np.expand_dims(images[idx], axis=0)
            raw_harmonized = generator([img_batch, ga_value], training=False).numpy()[0, :, :, 0]
            
            # Enhance
            enhanced_harmonized, brain_mask = enhance_harmonized_image(
                original, raw_harmonized, 
                clahe_clip=clahe_clip, 
                detail_alpha=detail_alpha
            )
            
            # Save outputs
            filename_base = f'{site}_{subject_id}'
            np.save(output_dir / 'raw_harmonized' / f'{filename_base}_raw.npy', raw_harmonized)
            np.save(output_dir / 'enhanced_harmonized' / f'{filename_base}_enhanced.npy', enhanced_harmonized)
            np.save(output_dir / 'brain_masks' / f'{filename_base}_mask.npy', brain_mask)
            
            # Store for grid
            all_subjects_data.append({
                'original': original,
                'target_bch': target_bch,
                'enhanced': enhanced_harmonized,
                'brain_mask': brain_mask,
                'site': site,
                'subject_id': subject_id,
                'split': 'Test'
            })
            
            # Store metrics
            if brain_mask.sum() > 0:
                result = {
                    'site': site,
                    'subject_id': subject_id,
                    'split': 'Test',
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
        
        # Process validation samples
        for idx in tqdm(selected_val, desc=f"  Validation {site}"):
            original = images[idx, :, :, 0]
            ga_value = ga[idx].reshape(1, 1)
            subject_id = subject_ids[idx]
            
            # Get corresponding BCH target
            if len(bch_indices) > 0:
                bch_idx = np.random.choice(bch_indices)
                target_bch = images[bch_idx, :, :, 0]
            else:
                target_bch = np.zeros_like(original)
            
            # Harmonize
            img_batch = np.expand_dims(images[idx], axis=0)
            raw_harmonized = generator([img_batch, ga_value], training=False).numpy()[0, :, :, 0]
            
            # Enhance
            enhanced_harmonized, brain_mask = enhance_harmonized_image(
                original, raw_harmonized, 
                clahe_clip=clahe_clip, 
                detail_alpha=detail_alpha
            )
            
            # Save outputs
            filename_base = f'{site}_{subject_id}_val'
            np.save(output_dir / 'raw_harmonized' / f'{filename_base}_raw.npy', raw_harmonized)
            np.save(output_dir / 'enhanced_harmonized' / f'{filename_base}_enhanced.npy', enhanced_harmonized)
            np.save(output_dir / 'brain_masks' / f'{filename_base}_mask.npy', brain_mask)
            
            # Store for grid
            all_subjects_data.append({
                'original': original,
                'target_bch': target_bch,
                'enhanced': enhanced_harmonized,
                'brain_mask': brain_mask,
                'site': site,
                'subject_id': subject_id,
                'split': 'Validation'
            })
            
            # Store metrics
            if brain_mask.sum() > 0:
                result = {
                    'site': site,
                    'subject_id': subject_id,
                    'split': 'Validation',
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
    print("STEP 4: CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create multi-subject grid
    if all_subjects_data:
        create_multi_subject_grid(
            all_subjects_data,
            save_path=output_dir / 'comparison_figures' / 'all_subjects_grid.png'
        )
    
    print("\n" + "="*80)
    print("STEP 5: CREATING SUMMARY")
    print("="*80)
    
    if all_results:
        # Create summary DataFrame
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / 'enhancement_summary.csv', index=False)
        
        print("\nOverall Statistics:")
        print(f"  Processed: {len(results_df)} images")
        print(f"  Test: {len(results_df[results_df['split']=='Test'])}")
        print(f"  Validation: {len(results_df[results_df['split']=='Validation'])}")
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
            print(f"    Samples: {len(site_df)} (Test: {len(site_df[site_df['split']=='Test'])}, Val: {len(site_df[site_df['split']=='Validation'])})")
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
        
        print(f"\nSummary visualization saved: {output_dir / 'summary_statistics.png'}")
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
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
  python harmonize_and_enhance.py \
      --test_data processed_data_4slice/test_4slice_data.pkl \
      --model_weights weights/cyclegan_2d/ \
      --output harmonized_results/ \
      --validation_csv stackQC_50stacks.csv \
      --n_samples 10 \
      --clahe_clip 0.03 \
      --detail_alpha 0.25
        """
    )
    
    parser.add_argument('--test_data', required=True,
                       help='Path to test_4slice_data.pkl')
    parser.add_argument('--model_weights', required=True,
                       help='Directory with model weights')
    parser.add_argument('--output', default='./harmonized_results',
                       help='Output directory')
    parser.add_argument('--validation_csv', default=None,
                       help='Path to CSV with validation subjects')
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
    print(f"Validation CSV: {args.validation_csv}")
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
        validation_csv_path=args.validation_csv,
        clahe_clip=args.clahe_clip,
        detail_alpha=args.detail_alpha,
        n_samples_per_site=args.n_samples
    )


if __name__ == "__main__":
    main()