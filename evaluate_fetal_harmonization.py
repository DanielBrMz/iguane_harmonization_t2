#!/usr/bin/env python3
"""
Comprehensive Evaluation of Fetal Brain 2D CycleGAN Harmonization
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json

import tensorflow as tf
from tensorflow.keras import layers, Model
import nibabel as nib

# Image quality metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from scipy.stats import entropy
from scipy.ndimage import sobel

print("=" * 80)
print("FETAL BRAIN HARMONIZATION EVALUATION")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
print("=" * 80)


# ============================================================================
# MODEL ARCHITECTURE (must match training)
# ============================================================================

def build_2d_generator(input_shape=(138, 176, 1), ga_embedding_dim=16, name='generator'):
    """
    2D U-Net Generator with GA Conditioning
    Architecture: 32→64→128→256
    ** MUST MATCH TRAINING ARCHITECTURE EXACTLY **
    """
    
    img_input = layers.Input(shape=input_shape, name='image_input')
    ga_input = layers.Input(shape=(1,), name='ga_input')
    
    # GA embedding
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_input)
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_embedding)

    # Encoder: 32→64→128→256
    # Block 1
    x = layers.Conv2D(32, 3, padding='same')(img_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    skip1 = x
    x = layers.MaxPooling2D(2)(x)
    
    # Block 2
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    skip2 = x
    x = layers.MaxPooling2D(2)(x)
    
    # Block 3
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
    # Block 5
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    if x.shape[1] != skip3.shape[1] or x.shape[2] != skip3.shape[2]:
        x = layers.Resizing(skip3.shape[1], skip3.shape[2])(x)
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Block 6
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    if x.shape[1] != skip2.shape[1] or x.shape[2] != skip2.shape[2]:
        x = layers.Resizing(skip2.shape[1], skip2.shape[2])(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Block 7
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
    
    # Output with tanh (range -1 to 1)
    output = layers.Conv2D(1, 1, padding='same', activation='tanh')(x)
    
    # Scale to 0-1
    output = layers.Lambda(lambda x: (x + 1.0) / 2.0)(output)
    
    model = Model(inputs=[img_input, ga_input], outputs=output, name=name)
    
    return model


# ============================================================================
# QUANTITATIVE METRICS
# ============================================================================

def calculate_ssim(img1, img2, data_range=1.0):
    """Structural Similarity Index"""
    try:
        return ssim(img1, img2, data_range=data_range)
    except:
        return np.nan


def calculate_psnr(img1, img2, data_range=1.0):
    """Peak Signal-to-Noise Ratio"""
    try:
        return psnr(img1, img2, data_range=data_range)
    except:
        return np.nan


def calculate_mae(img1, img2):
    """Mean Absolute Error"""
    return np.mean(np.abs(img1 - img2))


def calculate_mse(img1, img2):
    """Mean Squared Error"""
    return np.mean((img1 - img2) ** 2)


def calculate_nmse(img1, img2):
    """Normalized Mean Squared Error"""
    mse_val = calculate_mse(img1, img2)
    return mse_val / (np.std(img1) ** 2 + 1e-8)


def calculate_contrast(img):
    """Contrast metric using standard deviation"""
    return np.std(img)


def calculate_snr(img):
    """Signal-to-Noise Ratio estimate"""
    signal = np.mean(img)
    noise = np.std(img)
    return signal / (noise + 1e-8)


def calculate_cnr(img, roi1_mask=None, roi2_mask=None):
    """
    Contrast-to-Noise Ratio
    For fetal brain: GM vs WM contrast
    """
    if roi1_mask is None or roi2_mask is None:
        # Simplified: use intensity thresholding
        thresh = np.percentile(img, 50)
        roi1 = img[img > thresh]
        roi2 = img[img <= thresh]
    else:
        roi1 = img[roi1_mask]
        roi2 = img[roi2_mask]
    
    if len(roi1) == 0 or len(roi2) == 0:
        return np.nan
    
    mean1 = np.mean(roi1)
    mean2 = np.mean(roi2)
    std_pooled = np.sqrt((np.std(roi1)**2 + np.std(roi2)**2) / 2)
    
    return np.abs(mean1 - mean2) / (std_pooled + 1e-8)


def calculate_sharpness(img):
    """Edge sharpness using Sobel operator"""
    sx = sobel(img, axis=0)
    sy = sobel(img, axis=1)
    return np.mean(np.sqrt(sx**2 + sy**2))


def calculate_histogram_similarity(img1, img2, bins=256):
    """Histogram-based similarity"""
    hist1, _ = np.histogram(img1.flatten(), bins=bins, range=(0, 1))
    hist2, _ = np.histogram(img2.flatten(), bins=bins, range=(0, 1))
    
    # Normalize
    hist1 = hist1 / (hist1.sum() + 1e-8)
    hist2 = hist2 / (hist2.sum() + 1e-8)
    
    # KL divergence (lower is better)
    kl_div = entropy(hist1 + 1e-10, hist2 + 1e-10)
    
    # Correlation
    corr = np.corrcoef(hist1, hist2)[0, 1]
    
    return {'kl_divergence': kl_div, 'correlation': corr}


def comprehensive_metrics(original, harmonized):
    """Calculate all metrics for a pair of images"""
    metrics = {}
    
    # Ensure float [0, 1] range
    if original.max() > 1:
        original = original.astype(float) / 255.0
    if harmonized.max() > 1:
        harmonized = harmonized.astype(float) / 255.0
    
    # Tier 1: Standard similarity metrics
    metrics['ssim'] = calculate_ssim(original, harmonized, data_range=1.0)
    metrics['psnr'] = calculate_psnr(original, harmonized, data_range=1.0)
    metrics['mae'] = calculate_mae(original, harmonized)
    metrics['mse'] = calculate_mse(original, harmonized)
    metrics['nmse'] = calculate_nmse(original, harmonized)
    
    # Tier 2: Perceptual metrics
    hist_sim = calculate_histogram_similarity(original, harmonized)
    metrics['hist_kl_div'] = hist_sim['kl_divergence']
    metrics['hist_corr'] = hist_sim['correlation']
    
    # Tier 3: Clinical relevance metrics
    metrics['contrast_original'] = calculate_contrast(original)
    metrics['contrast_harmonized'] = calculate_contrast(harmonized)
    metrics['contrast_change'] = np.abs(metrics['contrast_original'] - metrics['contrast_harmonized'])
    metrics['contrast_ratio'] = metrics['contrast_harmonized'] / (metrics['contrast_original'] + 1e-8)
    
    metrics['snr_original'] = calculate_snr(original)
    metrics['snr_harmonized'] = calculate_snr(harmonized)
    metrics['snr_change'] = np.abs(metrics['snr_original'] - metrics['snr_harmonized'])
    
    metrics['cnr_original'] = calculate_cnr(original)
    metrics['cnr_harmonized'] = calculate_cnr(harmonized)
    metrics['cnr_preservation'] = metrics['cnr_harmonized'] / (metrics['cnr_original'] + 1e-8)
    
    metrics['sharpness_original'] = calculate_sharpness(original)
    metrics['sharpness_harmonized'] = calculate_sharpness(harmonized)
    metrics['sharpness_change'] = np.abs(metrics['sharpness_original'] - metrics['sharpness_harmonized'])
    
    return metrics


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data(test_pickle_path):
    """Load test data from pickle file"""
    print(f"\nLoading test data from: {test_pickle_path}")
    
    with open(test_pickle_path, 'rb') as f:
        test_data = pickle.load(f)
    
    # Convert uint8 to float [0, 1]
    images = test_data['images'].astype(float) / 255.0
    ga = test_data['gestational_age']
    sites = test_data['site']
    
    print(f"  Loaded {len(images)} slices")
    print(f"  Shape: {images.shape}")
    print(f"  Sites: {np.unique(sites)}")
    print(f"  GA range: {ga.min():.1f} - {ga.max():.1f} weeks")
    
    return images, ga, sites, test_data


# ============================================================================
# HARMONIZATION EVALUATION
# ============================================================================

def evaluate_harmonization(args):
    """Main evaluation function"""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load test data
    test_images, test_ga, test_sites, test_data = load_test_data(args.test_data)
    
    # Site mapping for generator names
    site_to_gen = {
        'BCH_CHD': None,  # Reference site
        'BCH_Placenta': 'BCHPlacenta',
        'HBCD_Site5_Arkansas_UNC': 'HBCDSite5ArkansasUNC',
        'HBCD_Site6_Cincinnati': 'HBCDSite6Cincinnati',
        'VGH_Unknown': 'VGHUnknown',
        'VGH_Site4_GE': 'VGHUnknown',  # Map to VGHUnknown generator
        'dHCP': 'dHCP'
    }
    
    # Epochs to evaluate
    epochs_to_test = args.epochs if args.epochs else [150, 200]
    if 'final' not in epochs_to_test:
        epochs_to_test.append('final')
    
    all_results = []
    
    for epoch in epochs_to_test:
        print(f"\n{'='*80}")
        print(f"EVALUATING EPOCH {epoch}")
        print(f"{'='*80}")
        
        epoch_results = {}
        
        # Process each unique site
        for site in np.unique(test_sites):
            print(f"\n--- Site: {site} ---")
            
            # Get indices for this site
            site_idx = np.where(test_sites == site)[0]
            n_samples = min(len(site_idx), args.max_samples_per_site)
            site_idx = site_idx[:n_samples]
            
            print(f"  Processing {len(site_idx)} samples")
            
            # Skip if reference site
            if site == 'BCH_CHD':
                print("  Skipping reference site (BCH_CHD)")
                continue
            
            # Get generator name
            gen_name = site_to_gen.get(site)
            if gen_name is None:
                print(f"  Warning: No generator mapping for {site}, skipping")
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
            
            try:
                gen.load_weights(str(weight_file))
                print(f"  ✓ Loaded weights: {weight_file.name}")
            except Exception as e:
                print(f"  ✗ Error loading weights: {e}")
                continue
            
            # Harmonize images
            site_metrics = []
            
            for idx in tqdm(site_idx, desc=f"  Harmonizing {site}"):
                original = test_images[idx]
                ga_value = test_ga[idx].reshape(1, 1)
                
                # Add batch dimension
                img_batch = np.expand_dims(original, axis=0)
                
                # Generate harmonized image
                harmonized = gen([img_batch, ga_value], training=False).numpy()[0]
                
                # Calculate metrics
                metrics = comprehensive_metrics(original, harmonized)
                metrics['site'] = site
                metrics['epoch'] = epoch
                metrics['ga'] = test_ga[idx]
                
                site_metrics.append(metrics)
            
            epoch_results[site] = site_metrics
            
            # Print summary for this site
            df_site = pd.DataFrame(site_metrics)
            print(f"\n  Site {site} Summary (epoch {epoch}):")
            print(f"    SSIM: {df_site['ssim'].mean():.4f} ± {df_site['ssim'].std():.4f}")
            print(f"    PSNR: {df_site['psnr'].mean():.2f} ± {df_site['psnr'].std():.2f}")
            print(f"    MAE: {df_site['mae'].mean():.4f} ± {df_site['mae'].std():.4f}")
            print(f"    CNR Preservation: {df_site['cnr_preservation'].mean():.4f}")
        
        all_results.append(epoch_results)
        
        # Save epoch results
        epoch_df_list = []
        for site, metrics_list in epoch_results.items():
            epoch_df_list.extend(metrics_list)
        
        if epoch_df_list:
            epoch_df = pd.DataFrame(epoch_df_list)
            output_csv = output_dir / f'metrics_epoch_{epoch}_{timestamp}.csv'
            epoch_df.to_csv(output_csv, index=False)
            print(f"\n✓ Saved results: {output_csv}")
    
    # Compare epochs
    print(f"\n{'='*80}")
    print("EPOCH COMPARISON")
    print(f"{'='*80}")
    
    comparison = []
    for epoch_results in all_results:
        for site, metrics_list in epoch_results.items():
            df_site = pd.DataFrame(metrics_list)
            comparison.append({
                'epoch': df_site['epoch'].iloc[0],
                'site': site,
                'ssim_mean': df_site['ssim'].mean(),
                'ssim_std': df_site['ssim'].std(),
                'psnr_mean': df_site['psnr'].mean(),
                'psnr_std': df_site['psnr'].std(),
                'mae_mean': df_site['mae'].mean(),
                'mae_std': df_site['mae'].std(),
                'cnr_preservation_mean': df_site['cnr_preservation'].mean(),
                'n_samples': len(df_site)
            })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_csv = output_dir / f'epoch_comparison_{timestamp}.csv'
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\n✓ Saved epoch comparison: {comparison_csv}")
    
    # Print comparison
    print("\nEpoch Comparison (Mean ± Std):")
    print(comparison_df.to_string(index=False))
    
    # Save configuration
    config = {
        'timestamp': timestamp,
        'test_data': str(args.test_data),
        'weight_dir': str(args.weight_dir),
        'epochs_evaluated': epochs_to_test,
        'max_samples_per_site': args.max_samples_per_site,
        'n_test_samples': len(test_images),
        'sites_evaluated': list(np.unique(test_sites))
    }
    
    config_file = output_dir / f'evaluation_config_{timestamp}.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_dir}/")
    
    return comparison_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate fetal brain 2D CycleGAN harmonization'
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
        default='evaluation_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--epochs',
        nargs='+',
        type=int,
        default=[150, 200],
        help='Epochs to evaluate (e.g., --epochs 100 150 200)'
    )
    
    parser.add_argument(
        '--max_samples_per_site',
        type=int,
        default=100,
        help='Maximum number of samples to evaluate per site'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_harmonization(args)


if __name__ == '__main__':
    main()