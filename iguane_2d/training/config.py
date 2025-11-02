"""
Configuration management for 2D CycleGAN training
Handles all hyperparameters and path settings
"""

import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults"""
    
    # Data paths
    train_data: str = 'processed_data_4slice/train_4slice_data.pkl'
    val_data: str = 'processed_data_4slice/val_4slice_data.pkl'
    reference_site: str = 'BCH_CHD'
    
    # Model architecture
    ga_embedding_dim: int = 16
    img_height: int = 138
    img_width: int = 176
    img_channels: int = 1
    
    # Training hyperparameters
    epochs: int = 200
    batch_size: int = 16
    lr_gen: float = 0.0002
    lr_disc: float = 0.0002
    beta_1: float = 0.5
    lambda_cycle: float = 30.0
    lambda_identity: float = 0.5
    
    # Output directories
    weight_dir: str = './weights/cyclegan_2d'
    result_dir: str = './results/cyclegan_2d'
    log_dir: str = './logs/cyclegan_2d'
    save_freq: int = 25
    
    # Hardware
    gpu: str = '0,1,2'
    multi_gpu_strategy: str = 'model_parallel'
    memory_growth: bool = True
    
    # Augmentation
    use_augmentation: bool = True
    augment_flip: float = 0.5
    augment_brightness: float = 0.05
    augment_contrast_lower: float = 0.95
    augment_contrast_upper: float = 1.05
    
    # Training stability
    gradient_clip_norm: float = 5.0
    label_smoothing: float = 0.1
    use_spectral_norm: bool = True
    discriminator_dropout: float = 0.4
    
    # Collapse detection
    collapse_threshold: int = 30
    early_stop_disc_loss: float = 0.001
    early_stop_min_epoch: int = 10
    
    @property
    def img_shape(self):
        """Return image shape as tuple"""
        return (self.img_height, self.img_width, self.img_channels)
    
    def create_directories(self):
        """Create all output directories"""
        Path(self.weight_dir).mkdir(exist_ok=True, parents=True)
        Path(self.result_dir).mkdir(exist_ok=True, parents=True)
        Path(self.log_dir).mkdir(exist_ok=True, parents=True)
    
    def __str__(self):
        """Pretty print configuration"""
        lines = ["=" * 80, "CONFIGURATION", "=" * 80]
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                lines.append(f"  {key}: {value}")
        lines.append("=" * 80)
        return "\n".join(lines)


def parse_args():
    """Parse command line arguments and return TrainingConfig"""
    parser = argparse.ArgumentParser(
        description='Train 2D CycleGAN for fetal brain harmonization (IGUANe-style)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--train_data', type=str,
                           default='processed_data_4slice/train_4slice_data.pkl',
                           help='Path to training data pickle file')
    data_group.add_argument('--val_data', type=str,
                           default='processed_data_4slice/val_4slice_data.pkl',
                           help='Path to validation data pickle file')
    data_group.add_argument('--reference_site', type=str, default='BCH_CHD',
                           help='Reference site for harmonization')
    
    # Model arguments
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--ga_embedding_dim', type=int, default=16,
                            help='Gestational age embedding dimension')
    model_group.add_argument('--img_height', type=int, default=138,
                            help='Input image height')
    model_group.add_argument('--img_width', type=int, default=176,
                            help='Input image width')
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=200,
                            help='Number of training epochs')
    train_group.add_argument('--batch_size', type=int, default=16,
                            help='Batch size')
    train_group.add_argument('--lr_gen', type=float, default=0.0002,
                            help='Generator learning rate')
    train_group.add_argument('--lr_disc', type=float, default=0.0002,
                            help='Discriminator learning rate')
    train_group.add_argument('--beta_1', type=float, default=0.5,
                            help='Adam optimizer beta_1 parameter')
    train_group.add_argument('--lambda_cycle', type=float, default=30.0,
                            help='Cycle consistency loss weight')
    train_group.add_argument('--lambda_identity', type=float, default=0.5,
                            help='Identity loss weight')
    train_group.add_argument('--gradient_clip_norm', type=float, default=5.0,
                            help='Gradient clipping norm')
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--weight_dir', type=str, default='./weights/cyclegan_2d',
                             help='Directory to save model weights')
    output_group.add_argument('--result_dir', type=str, default='./results/cyclegan_2d',
                             help='Directory to save results')
    output_group.add_argument('--log_dir', type=str, default='./logs/cyclegan_2d',
                             help='Directory to save logs')
    output_group.add_argument('--save_freq', type=int, default=25,
                             help='Save checkpoint every N epochs')
    
    # Hardware arguments
    hardware_group = parser.add_argument_group('Hardware')
    hardware_group.add_argument('--gpu', type=str, default='0,1,2',
                               help='GPU IDs to use (comma-separated)')
    hardware_group.add_argument('--multi_gpu_strategy', type=str, default='model_parallel',
                               choices=['model_parallel', 'single'],
                               help='Multi-GPU strategy')
    hardware_group.add_argument('--no_memory_growth', action='store_true',
                               help='Disable GPU memory growth')
    
    args = parser.parse_args()
    
    # Create config from args
    config = TrainingConfig(
        train_data=args.train_data,
        val_data=args.val_data,
        reference_site=args.reference_site,
        ga_embedding_dim=args.ga_embedding_dim,
        img_height=args.img_height,
        img_width=args.img_width,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_gen=args.lr_gen,
        lr_disc=args.lr_disc,
        beta_1=args.beta_1,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        weight_dir=args.weight_dir,
        result_dir=args.result_dir,
        log_dir=args.log_dir,
        save_freq=args.save_freq,
        gpu=args.gpu,
        multi_gpu_strategy=args.multi_gpu_strategy,
        memory_growth=not args.no_memory_growth,
        gradient_clip_norm=args.gradient_clip_norm
    )
    
    return config
