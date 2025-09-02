"""
Main training script for 2D CycleGAN
Orchestrates the complete training pipeline
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np
import gc
from pathlib import Path
from tqdm import tqdm

# Local imports
from config import parse_args, TrainingConfig
from gpu_utils import configure_gpu, print_gpu_usage, enable_xla_optimization
from data_loader import (
    load_preprocessed_data,
    create_site_datasets,
    create_all_datasets
)
from cyclegan import CycleGAN2D_MultiSite
from evaluation import (
    evaluate_checkpoint,
    save_training_history,
    generate_quality_report
)
from losses import check_for_nan_loss


def print_header():
    """Print training header"""
    print("="*80)
    print("FETAL BRAIN 2D CYCLEGAN - IGUANE TRAINING")
    print("="*80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    print("="*80)


def setup_environment(config: TrainingConfig):
    """
    Setup training environment
    
    Args:
        config: Training configuration
    
    Returns:
        List of configured GPUs
    """
    # Enable optimizations
    enable_xla_optimization()
    
    # Configure GPUs
    gpus = configure_gpu(config.gpu, memory_growth=config.memory_growth)
    
    # Create output directories
    config.create_directories()
    
    return gpus


def load_and_prepare_data(config: TrainingConfig):
    """
    Load and prepare training data
    
    Args:
        config: Training configuration
    
    Returns:
        Tuple of (site_data, reference_site, site_datasets, target_sites)
    """
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load preprocessed data
    train_images, train_ga, train_sex, train_site = load_preprocessed_data(
        config.train_data
    )
    
    # Create site datasets
    train_site_data, ref_site = create_site_datasets(
        train_images, train_ga, train_sex, train_site,
        config.reference_site
    )
    
    # Get target sites (non-reference)
    target_sites = [s for s in train_site_data.keys() if s != ref_site]
    
    # Create TensorFlow datasets
    print("\n" + "="*80)
    print("CREATING TENSORFLOW DATASETS")
    print("="*80)
    
    site_datasets = create_all_datasets(
        train_site_data,
        batch_size=config.batch_size,
        shuffle=True,
        augment=config.use_augmentation
    )
    
    return train_site_data, ref_site, site_datasets, target_sites


def build_model(config: TrainingConfig, target_sites: list):
    """
    Build and compile CycleGAN model
    
    Args:
        config: Training configuration
        target_sites: List of target site names
    
    Returns:
        Compiled CycleGAN model
    """
    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)
    
    cyclegan = CycleGAN2D_MultiSite(
        img_shape=config.img_shape,
        ga_embedding_dim=config.ga_embedding_dim,
        target_sites=target_sites,
        use_multi_gpu=(config.multi_gpu_strategy != 'single'),
        lambda_cycle=config.lambda_cycle,
        lambda_identity=config.lambda_identity,
        gradient_clip_norm=config.gradient_clip_norm
    )
    
    cyclegan.compile(
        lr_gen=config.lr_gen,
        lr_disc=config.lr_disc,
        beta_1=config.beta_1
    )
    
    return cyclegan


def train_epoch(
    cyclegan: CycleGAN2D_MultiSite,
    site_datasets: dict,
    train_site_data: dict,
    config: TrainingConfig,
    epoch: int
) -> dict:
    """
    Train for one epoch
    
    Args:
        cyclegan: CycleGAN model
        site_datasets: Dictionary of TensorFlow datasets
        train_site_data: Dictionary of site data
        config: Training configuration
        epoch: Current epoch number
    
    Returns:
        Dictionary of epoch losses
    """
    print(f"\nEpoch {epoch+1}/{config.epochs}")
    
    # Initialize loss tracking
    epoch_losses = {
        'gen_loss': [],
        'disc_BCH_loss': [],
        'cycle_loss': [],
        'identity_loss': []
    }
    for site in cyclegan.target_sites:
        epoch_losses[f'disc_{site}_loss'] = []
    
    # Create iterators
    site_iters = {name: iter(dataset) for name, dataset in site_datasets.items()}
    
    # Calculate steps per epoch
    steps_per_epoch = max([
        len(train_site_data[s]['images']) // config.batch_size
        for s in train_site_data.keys()
    ])
    
    # Training loop with progress bar
    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
    
    for step in pbar:
        # Prepare batches for all sites
        site_batches = {}
        for site_name, site_iter in site_iters.items():
            try:
                images, ga = next(site_iter)
                if len(ga.shape) == 1:
                    ga = tf.expand_dims(ga, axis=-1)
                site_batches[site_name] = (images, ga)
            except StopIteration:
                # Restart iterator if exhausted
                site_iters[site_name] = iter(site_datasets[site_name])
                images, ga = next(site_iters[site_name])
                if len(ga.shape) == 1:
                    ga = tf.expand_dims(ga, axis=-1)
                site_batches[site_name] = (images, ga)
        
        try:
            # Training step
            losses = cyclegan.train_step(site_batches)
            
            # Check for NaN
            if check_for_nan_loss(losses):
                print(f"\n  NaN detected, skipping step {step}")
                continue
            
            # Collapse warning
            if losses.get('collapse_warning', 0) > config.collapse_threshold:
                print(f"\n  WARNING: Persistent discriminator saturation")
            
            # Record losses
            for k, v in losses.items():
                if k in epoch_losses and not isinstance(v, int):
                    epoch_losses[k].append(v)
            
            # Update progress bar
            pbar.set_postfix({
                'G': f"{losses['gen_loss']:.3f}",
                'D_BCH': f"{losses['disc_BCH_loss']:.3f}",
                'Cyc': f"{losses['cycle_loss']:.3f}",
                'Collapse': losses.get('collapse_warning', 0)
            })
            
        except Exception as e:
            print(f"\n  Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute average losses
    avg_losses = {}
    for k in epoch_losses.keys():
        if epoch_losses[k]:
            avg_losses[k] = np.mean(epoch_losses[k])
        else:
            avg_losses[k] = 0.0
    
    # Print summary
    print(f"\n  Gen: {avg_losses['gen_loss']:.4f} | "
          f"Disc BCH: {avg_losses['disc_BCH_loss']:.4f} | "
          f"Cycle: {avg_losses['cycle_loss']:.4f} | "
          f"Identity: {avg_losses['identity_loss']:.4f}")
    
    return avg_losses


def save_checkpoint(
    cyclegan: CycleGAN2D_MultiSite,
    epoch: int,
    config: TrainingConfig
):
    """
    Save model checkpoint
    
    Args:
        cyclegan: CycleGAN model
        epoch: Current epoch
        config: Training configuration
    """
    print(f"\n  Saving checkpoint at epoch {epoch+1}")
    
    weight_dir = Path(config.weight_dir)
    
    # Save forward generator
    cyclegan.gen_site2BCH.save_weights(
        weight_dir / f'gen_site2BCH_epoch_{epoch+1}.weights.h5'
    )
    
    # Save backward generators
    for site_name, gen_bwd in cyclegan.gen_BCH2site.items():
        safe_name = cyclegan._sanitize_name(site_name)
        gen_bwd.save_weights(
            weight_dir / f'gen_BCH2{safe_name}_epoch_{epoch+1}.weights.h5'
        )
    
    # Save BCH discriminator
    cyclegan.disc_BCH.save_weights(
        weight_dir / f'disc_BCH_epoch_{epoch+1}.weights.h5'
    )
    
    # Save site discriminators
    for site_name, disc in cyclegan.disc_sites.items():
        safe_name = cyclegan._sanitize_name(site_name)
        disc.save_weights(
            weight_dir / f'disc_{safe_name}_epoch_{epoch+1}.weights.h5'
        )


def train(config: TrainingConfig):
    """
    Main training function
    
    Args:
        config: Training configuration
    """
    # Setup environment
    gpus = setup_environment(config)
    
    # Load data
    train_site_data, ref_site, site_datasets, target_sites = load_and_prepare_data(config)
    
    # Build model
    cyclegan = build_model(config, target_sites)
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Generator LR: {config.lr_gen}")
    print(f"Discriminator LR: {config.lr_disc}")
    print(f"Lambda cycle: {config.lambda_cycle}")
    print(f"Lambda identity: {config.lambda_identity}")
    print("="*80)
    
    # Initialize history
    history = {
        'gen_loss': [],
        'disc_BCH_loss': [],
        'cycle_loss': [],
        'identity_loss': []
    }
    for site in target_sites:
        history[f'disc_{site}_loss'] = []
    
    # Training loop
    for epoch in range(config.epochs):
        # Train one epoch
        avg_losses = train_epoch(
            cyclegan, site_datasets, train_site_data,
            config, epoch
        )
        
        # Record history
        for k in history.keys():
            if k in avg_losses:
                history[k].append(avg_losses[k])
        
        # Memory cleanup
        tf.keras.backend.clear_session()
        gc.collect()
        
        # GPU monitoring
        if (epoch + 1) % 10 == 0:
            print_gpu_usage()
        
        # Early stopping check
        if (history['disc_BCH_loss'][-1] < config.early_stop_disc_loss and
            epoch > config.early_stop_min_epoch):
            print("\n  CRITICAL: Discriminator collapse detected")
            print("  Training stopped early")
            break
        
        # Save checkpoint
        if (epoch + 1) % config.save_freq == 0:
            save_checkpoint(cyclegan, epoch, config)
            
            # Evaluate
            collapse_detected, stats_df = evaluate_checkpoint(
                cyclegan.gen_site2BCH,
                train_site_data,
                ref_site,
                Path(config.result_dir),
                epoch + 1
            )
            
            if collapse_detected:
                print(f"\n  WARNING: Collapse detected at epoch {epoch+1}")
    
    # Save final models
    print("\n  Saving final models...")
    save_checkpoint(cyclegan, config.epochs - 1, config)
    
    # Save history
    save_training_history(history, Path(config.log_dir) / 'training_history.csv')
    
    # Print final report
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    generate_quality_report(history)


def main():
    """Main entry point"""
    # Print header
    print_header()
    
    # Parse configuration
    config = parse_args()
    print(config)
    
    # Train
    train(config)


if __name__ == '__main__':
    main()
