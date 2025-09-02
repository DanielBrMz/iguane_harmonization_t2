"""
Evaluation and checkpoint utilities
Handles model evaluation and collapse detection
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def evaluate_checkpoint(
    generator,
    site_data: Dict,
    reference_site: str,
    result_dir: Path,
    epoch: int,
    batch_size: int = 4
) -> Tuple[bool, pd.DataFrame]:
    """
    Comprehensive checkpoint evaluation with multi-level collapse detection
    
    Args:
        generator: Forward generator model (site->BCH)
        site_data: Dictionary of site data
        reference_site: Name of reference site
        result_dir: Directory to save evaluation results
        epoch: Current epoch number
        batch_size: Number of samples to evaluate
    
    Returns:
        Tuple of (collapse_detected, stats_dataframe)
    """
    print(f"\n  Evaluating checkpoint (epoch {epoch})...")
    
    eval_dir = result_dir / f'eval_epoch_{epoch}'
    eval_dir.mkdir(exist_ok=True, parents=True)
    
    collapse_detected = False
    stats_summary = []
    
    # Evaluate each non-reference site
    for site_name, site_dict in site_data.items():
        if site_name == reference_site or site_dict['n_slices'] < batch_size:
            continue
        
        # Sample random images
        indices = np.random.choice(
            site_dict['n_slices'],
            min(batch_size, 8),
            replace=False
        )
        samples_img = site_dict['images'][indices]
        samples_ga = site_dict['ga'][indices]
        
        # Generate harmonized images
        import tensorflow as tf
        samples_ga_tf = tf.expand_dims(samples_ga, axis=-1)
        harmonized = generator.predict([samples_img, samples_ga_tf], verbose=0)
        
        # Compute comprehensive statistics
        stats = compute_harmonization_stats(samples_img, harmonized, samples_ga)
        
        # Detect collapse
        is_collapsed, collapse_reasons = detect_collapse(stats)
        
        # Add metadata
        stats['site'] = site_name
        stats['collapsed'] = is_collapsed
        stats['reasons'] = ', '.join(collapse_reasons) if collapse_reasons else 'none'
        stats_summary.append(stats)
        
        # Print status
        status = "COLLAPSED" if is_collapsed else "OK"
        print(f"    {site_name}: std={stats['harm_std']:.4f}, "
              f"mean={stats['harm_mean']:.4f}, "
              f"diff={stats['mean_diff']:.4f} [{status}]")
        
        if is_collapsed:
            print(f"      Reasons: {', '.join(collapse_reasons)}")
            collapse_detected = True
        
        # Create visualization
        save_evaluation_figure(
            samples_img, harmonized, samples_ga,
            eval_dir / f'{site_name}.png'
        )
    
    # Save statistics
    stats_df = pd.DataFrame(stats_summary)
    stats_df.to_csv(eval_dir / 'evaluation_stats.csv', index=False)
    
    print(f"  Evaluation complete")
    
    return collapse_detected, stats_df


def compute_harmonization_stats(
    original: np.ndarray,
    harmonized: np.ndarray,
    ga: np.ndarray
) -> Dict:
    """
    Compute comprehensive statistics for harmonization quality
    
    Args:
        original: Original images
        harmonized: Harmonized images
        ga: Gestational ages
    
    Returns:
        Dictionary of statistics
    """
    stats = {
        'orig_std': np.std(original),
        'harm_std': np.std(harmonized),
        'orig_mean': np.mean(original),
        'harm_mean': np.mean(harmonized),
        'mean_diff': np.mean(np.abs(original - harmonized)),
        'max_diff': np.max(np.abs(original - harmonized)),
        'min_per_image_std': np.min([np.std(harmonized[i]) for i in range(len(harmonized))]),
        'per_image_mean_std': np.std([np.mean(harmonized[i]) for i in range(len(harmonized))])
    }
    
    return stats


def detect_collapse(stats: Dict) -> Tuple[bool, list]:
    """
    Multi-level collapse detection
    
    Args:
        stats: Statistics dictionary from compute_harmonization_stats
    
    Returns:
        Tuple of (is_collapsed, list_of_reasons)
    """
    is_collapsed = False
    collapse_reasons = []
    
    # Check 1: Near-zero standard deviation
    if stats['harm_std'] < 0.01:
        is_collapsed = True
        collapse_reasons.append(f"near-zero std ({stats['harm_std']:.4f})")
    
    # Check 2: Minimum per-image standard deviation
    if stats['min_per_image_std'] < 0.005:
        is_collapsed = True
        collapse_reasons.append(f"min per-image std ({stats['min_per_image_std']:.4f})")
    
    # Check 3: Extreme mean values
    if stats['harm_mean'] < 0.05 or stats['harm_mean'] > 0.95:
        is_collapsed = True
        collapse_reasons.append(f"extreme mean ({stats['harm_mean']:.3f})")
    
    # Check 4: Identity mapping
    if stats['mean_diff'] < 0.001 and stats['max_diff'] < 0.01:
        is_collapsed = True
        collapse_reasons.append("identity mapping")
    
    # Check 5: Near-zero maximum output
    harm_max = np.max(np.abs(stats['harm_mean']))  # Approximate check
    if harm_max < 0.1:
        is_collapsed = True
        collapse_reasons.append("near-zero max output")
    
    # Check 6: Constant output across images
    if stats['per_image_mean_std'] < 0.001:
        is_collapsed = True
        collapse_reasons.append("constant output")
    
    return is_collapsed, collapse_reasons


def save_evaluation_figure(
    original: np.ndarray,
    harmonized: np.ndarray,
    ga: np.ndarray,
    output_path: Path
):
    """
    Create and save visualization comparing original and harmonized images
    
    Args:
        original: Original images [N, H, W, C]
        harmonized: Harmonized images [N, H, W, C]
        ga: Gestational ages [N]
        output_path: Path to save figure
    """
    n_show = min(4, len(original))
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 3*n_show))
    
    if n_show == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_show):
        orig = original[i, :, :, 0]
        harm = harmonized[i, :, :, 0]
        diff = np.abs(orig - harm)
        
        # Original image
        axes[i, 0].imshow(orig, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Original (GA:{ga[i]:.1f}w)')
        axes[i, 0].axis('off')
        
        # Harmonized image
        axes[i, 1].imshow(harm, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Harmonized (std:{np.std(harm):.4f})')
        axes[i, 1].axis('off')
        
        # Difference map
        axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[i, 2].set_title(f'Diff (max:{np.max(diff):.4f})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_training_history(history: Dict, output_path: Path):
    """
    Save training history to CSV
    
    Args:
        history: Dictionary of loss histories
        output_path: Path to save CSV
    """
    df = pd.DataFrame(history)
    df.to_csv(output_path, index=False)
    print(f"\nSaved training history to {output_path}")


def generate_quality_report(history: Dict):
    """
    Generate and print training quality report
    
    Args:
        history: Dictionary of loss histories
    """
    print("\nTRAINING QUALITY REPORT:")
    
    if len(history.get('disc_BCH_loss', [])) == 0:
        print("  No training history available")
        return
    
    # Discriminator assessment
    final_disc_loss = history['disc_BCH_loss'][-1]
    if final_disc_loss < 0.01:
        print("  Discriminator: Collapsed")
    elif final_disc_loss < 0.3:
        print("  Discriminator: Weak")
    elif final_disc_loss > 2.0:
        print("  Discriminator: Too strong")
    else:
        print("  Discriminator: Healthy")
    
    # Generator assessment
    final_gen_loss = history['gen_loss'][-1]
    if final_gen_loss > 50:
        print("  Generator: Struggling")
    elif final_gen_loss < 5:
        print("  Generator: Possibly collapsed")
    else:
        print("  Generator: Learning")
    
    # Cycle consistency assessment
    final_cycle = history['cycle_loss'][-1]
    if final_cycle < 0.2:
        print("  Cycle consistency: Excellent")
    elif final_cycle < 1.0:
        print("  Cycle consistency: Good")
    else:
        print("  Cycle consistency: Weak")
