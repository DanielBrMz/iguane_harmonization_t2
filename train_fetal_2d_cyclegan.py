"""
2D CycleGAN for Fetal Brain MRI Harmonization
Following the agreed approach with Hyeokjin:
- 4 central slices per stack
- 2D slice-wise processing
- Gestational age conditioning
- BCH as reference site
- Multi-site harmonization

Author: Daniel Barreras Meraz
Date: October 15, 2025
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from tensorflow.keras import backend as K

print("="*80)
print("FETAL BRAIN 2D CYCLEGAN HARMONIZATION")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
print("="*80)


# ============================================================================
# GPU CONFIGURATION
# ============================================================================

def configure_gpu(gpu_id='0', memory_growth=True):
    """Configure GPU settings"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Configured {len(gpus)} GPU(s) with memory growth: {memory_growth}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("⚠ No GPUs found, using CPU")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_preprocessed_data(data_path):
    """Load preprocessed 4-slice data"""
    print(f"Loading data from: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    images = data['images'].astype(np.float32) / 255.0  # Normalize to [0, 1]
    ga = data['gestational_age'].astype(np.float32)
    sex = data['sex'].astype(np.float32)
    site = data['site']
    
    print(f"  Images: {images.shape}, dtype: {images.dtype}")
    print(f"  GA range: {ga.min():.1f} - {ga.max():.1f} weeks")
    print(f"  Sites: {np.unique(site)}")
    
    return images, ga, sex, site


def create_site_datasets(images, ga, sex, site, reference_site='BCH_CHD'):
    """
    Create separate datasets for each site
    Reference site vs all other sites for CycleGAN
    """
    site_data = {}
    
    unique_sites = np.unique(site)
    print(f"\nCreating datasets for {len(unique_sites)} sites:")
    
    for s in unique_sites:
        mask = site == s
        site_data[s] = {
            'images': images[mask],
            'ga': ga[mask],
            'sex': sex[mask],
            'n_slices': np.sum(mask)
        }
        print(f"  {s}: {site_data[s]['n_slices']} slices")
    
    # Verify reference site exists
    if reference_site not in site_data:
        print(f"⚠ Reference site {reference_site} not found!")
        print(f"  Available sites: {list(site_data.keys())}")
        reference_site = list(site_data.keys())[0]
        print(f"  Using {reference_site} as reference instead")
    
    return site_data, reference_site


# ============================================================================
# NETWORK ARCHITECTURES
# ============================================================================

def build_2d_generator(input_shape=(138, 176, 1), ga_embedding_dim=16, name='generator'):
    """
    2D U-Net Generator with Gestational Age Conditioning
    """
    
    # Image input
    img_input = layers.Input(shape=input_shape, name='image_input')
    
    # GA conditioning input
    ga_input = layers.Input(shape=(1,), name='ga_input')
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_input)
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_embedding)
    
    # Encoder (reduced channels: 32, 64, 128, 256 instead of 64, 128, 256, 512)
    # Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(img_input)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    skip1 = x
    x = layers.MaxPooling2D(2)(x)
    
    # Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    skip2 = x
    x = layers.MaxPooling2D(2)(x)
    
    # Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    skip3 = x
    x = layers.MaxPooling2D(2)(x)
    
    # Block 4 (Bottleneck)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    
    # Inject GA conditioning at bottleneck
    ga_spatial = layers.RepeatVector(x.shape[1] * x.shape[2])(ga_embedding)
    ga_spatial = layers.Reshape((x.shape[1], x.shape[2], ga_embedding_dim))(ga_spatial)
    x = layers.Concatenate()([x, ga_spatial])
    
    # Decoder
    # Block 5
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    
    # Match dimensions for skip3
    if x.shape[1] != skip3.shape[1] or x.shape[2] != skip3.shape[2]:
        x = layers.Resizing(skip3.shape[1], skip3.shape[2])(x)
    
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    
    # Block 6
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    
    # Match dimensions for skip2
    if x.shape[1] != skip2.shape[1] or x.shape[2] != skip2.shape[2]:
        x = layers.Resizing(skip2.shape[1], skip2.shape[2])(x)
    
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    
    # Block 7
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    
    # Match dimensions for skip1
    if x.shape[1] != skip1.shape[1] or x.shape[2] != skip1.shape[2]:
        x = layers.Resizing(skip1.shape[1], skip1.shape[2])(x)
    
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    
    # Final resize to match input exactly
    if x.shape[1] != input_shape[0] or x.shape[2] != input_shape[1]:
        x = layers.Resizing(input_shape[0], input_shape[1])(x)
    
    # Output
    output = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    
    # Residual connection
    output = layers.Add()([img_input, output])
    output = layers.Activation('sigmoid')(output)
    
    model = Model(inputs=[img_input, ga_input], outputs=output, name=name)
    
    return model


def build_2d_discriminator(input_shape=(138, 176, 1), ga_embedding_dim=16, name='discriminator'):
    """
    2D PatchGAN Discriminator with GA Conditioning
    """
    
    # Image input
    img_input = layers.Input(shape=input_shape, name='image_input')
    
    # GA conditioning input
    ga_input = layers.Input(shape=(1,), name='ga_input')
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_input)
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_embedding)
    
    # Initial conv
    x = layers.Conv2D(64, 4, strides=2, padding='same')(img_input)
    x = layers.LeakyReLU(0.2)(x)
    
    # Conv blocks
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(512, 4, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Inject GA conditioning
    ga_spatial = layers.RepeatVector(x.shape[1] * x.shape[2])(ga_embedding)
    ga_spatial = layers.Reshape((x.shape[1], x.shape[2], ga_embedding_dim))(ga_spatial)
    x = layers.Concatenate()([x, ga_spatial])
    
    # Output (PatchGAN)
    output = layers.Conv2D(1, 4, strides=1, padding='same')(x)
    
    model = Model(inputs=[img_input, ga_input], outputs=output, name=name)
    
    return model


# ============================================================================
# LOSSES
# ============================================================================

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss for age conditioning"""
    error = y_pred - y_true
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * K.square(quadratic) + delta * linear


def cycle_consistency_loss(real_img, cycled_img):
    """L1 loss for cycle consistency"""
    return tf.reduce_mean(tf.abs(real_img - cycled_img))


def identity_loss(real_img, same_img):
    """L1 loss for identity mapping"""
    return tf.reduce_mean(tf.abs(real_img - same_img))


def discriminator_loss(real_output, fake_output):
    """Standard GAN discriminator loss"""
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(real_output), logits=real_output
    ))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(fake_output), logits=fake_output
    ))
    return real_loss + fake_loss


def generator_loss(fake_output):
    """Standard GAN generator loss"""
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(fake_output), logits=fake_output
    ))


# ============================================================================
# CYCLEGAN MODEL
# ============================================================================

class CycleGAN2D_MultiSite:
    """
    2D CycleGAN with ONE generator but MULTIPLE discriminators (one per site)
    Following the paper: BCH harmonizes to all sites using site-specific discriminators
    """
    
    def __init__(self, img_shape=(138, 176, 1), ga_embedding_dim=16, target_sites=None):
        self.img_shape = img_shape
        self.ga_embedding_dim = ga_embedding_dim
        self.target_sites = target_sites or []
        
        print(f"\nBuilding CycleGAN with {len(self.target_sites)} site-specific discriminators:")
        for site in self.target_sites:
            print(f"  - {site}")
        
        # Build ONE generator for BCH → harmonized
        self.gen_harmonize = build_2d_generator(img_shape, ga_embedding_dim, name='gen_harmonize')
        
        # Build ONE discriminator for BCH (reference)
        self.disc_BCH = build_2d_discriminator(img_shape, ga_embedding_dim, name='disc_BCH')
        
        # Build ONE discriminator PER target site
        self.disc_sites = {}
        for site in self.target_sites:
            site_name = site.replace('_', '').replace('-', '')[:20]  # Clean name for TF
            self.disc_sites[site] = build_2d_discriminator(
                img_shape, ga_embedding_dim, name=f'disc_{site_name}'
            )
        
        # Loss weights
        self.lambda_cycle = 10.0
        self.lambda_identity = 5.0
        
    def compile(self, lr=0.0002, beta_1=0.5):
        """Compile model with optimizers"""
        self.gen_optimizer = Adam(learning_rate=lr, beta_1=beta_1)
        self.disc_optimizer = Adam(learning_rate=lr, beta_1=beta_1)
        
        # Build generator optimizer
        self.gen_optimizer.build(self.gen_harmonize.trainable_variables)
        
        # Build discriminator optimizer with ALL discriminators
        all_disc_vars = self.disc_BCH.trainable_variables
        for disc in self.disc_sites.values():
            all_disc_vars += disc.trainable_variables
        self.disc_optimizer.build(all_disc_vars)
        
        print("✓ Model compiled")
        print(f"  Generator parameters: {self.gen_harmonize.count_params():,}")
        print(f"  BCH Discriminator parameters: {self.disc_BCH.count_params():,}")
        print(f"  Per-site Discriminator parameters: {self.disc_sites[self.target_sites[0]].count_params():,} × {len(self.target_sites)}")
    
    def train_step(self, site_batches):
        """
        Single training step with multiple site batches
        
        site_batches: dict {
            'BCH_CHD': (images, ga),
            'dHCP': (images, ga),
            'BCH_Placenta': (images, ga),
            ...
        }
        """
        
        with tf.GradientTape(persistent=True) as tape:
            
            # Get BCH reference batch
            real_BCH, ga_BCH = site_batches['BCH_CHD']
            
            # Initialize loss accumulators
            total_gen_loss = 0.0
            total_disc_BCH_loss = 0.0
            total_disc_site_losses = {}
            total_cycle_loss = 0.0
            total_identity_loss = 0.0
            
            # For each target site
            for site_name in self.target_sites:
                if site_name not in site_batches:
                    continue
                
                real_site, ga_site = site_batches[site_name]
                disc_site = self.disc_sites[site_name]
                
                # Forward: Site → Harmonized (to BCH style)
                harmonized = self.gen_harmonize([real_site, ga_site], training=True)
                
                # Cycle: Site → Harmonized → Site (identity, should preserve)
                # For simplicity, we assume harmonization should be reversible
                cycle_site = self.gen_harmonize([harmonized, ga_site], training=True)
                
                # Identity: BCH → Harmonized (should stay same)
                identity_BCH = self.gen_harmonize([real_BCH, ga_BCH], training=True)
                
                # Discriminator predictions
                # BCH discriminator: should distinguish real BCH from harmonized images
                disc_real_BCH = self.disc_BCH([real_BCH, ga_BCH], training=True)
                disc_fake_BCH = self.disc_BCH([harmonized, ga_site], training=True)
                
                # Site discriminator: should distinguish real site images from anything
                disc_real_site = disc_site([real_site, ga_site], training=True)
                disc_fake_site = disc_site([harmonized, ga_site], training=True)
                
                # Generator loss: fool BCH discriminator
                gen_loss = generator_loss(disc_fake_BCH)
                
                # Cycle consistency loss
                cycle_loss = cycle_consistency_loss(real_site, cycle_site)
                
                # Identity loss
                identity_loss_val = identity_loss(real_BCH, identity_BCH)
                
                # Discriminator losses
                disc_BCH_loss = discriminator_loss(disc_real_BCH, disc_fake_BCH)
                disc_site_loss = discriminator_loss(disc_real_site, disc_fake_site)
                
                # Accumulate
                total_gen_loss += gen_loss + self.lambda_cycle * cycle_loss + self.lambda_identity * identity_loss_val
                total_disc_BCH_loss += disc_BCH_loss
                total_disc_site_losses[site_name] = disc_site_loss
                total_cycle_loss += cycle_loss
                total_identity_loss += identity_loss_val
            
            # Average losses across sites
            n_sites = len([s for s in self.target_sites if s in site_batches])
            if n_sites > 0:
                total_gen_loss /= n_sites
                total_disc_BCH_loss /= n_sites
                total_cycle_loss /= n_sites
                total_identity_loss /= n_sites
        
        # Calculate gradients for generator
        gen_gradients = tape.gradient(
            total_gen_loss, self.gen_harmonize.trainable_variables
        )
        
        # Calculate gradients for BCH discriminator
        disc_BCH_gradients = tape.gradient(
            total_disc_BCH_loss, self.disc_BCH.trainable_variables
        )
        
        # Calculate gradients for each site discriminator
        disc_site_gradients = {}
        for site_name, disc in self.disc_sites.items():
            if site_name in total_disc_site_losses:
                disc_site_gradients[site_name] = tape.gradient(
                    total_disc_site_losses[site_name], disc.trainable_variables
                )
        
        # Apply gradients to generator
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.gen_harmonize.trainable_variables)
        )
        
        # Apply gradients to BCH discriminator
        self.disc_optimizer.apply_gradients(
            zip(disc_BCH_gradients, self.disc_BCH.trainable_variables)
        )
        
        # Apply gradients to site discriminators
        for site_name, gradients in disc_site_gradients.items():
            self.disc_optimizer.apply_gradients(
                zip(gradients, self.disc_sites[site_name].trainable_variables)
            )
        
        del tape
        
        # Return losses
        losses = {
            'gen_loss': total_gen_loss.numpy(),
            'disc_BCH_loss': total_disc_BCH_loss.numpy(),
            'cycle_loss': total_cycle_loss.numpy(),
            'identity_loss': total_identity_loss.numpy()
        }
        
        # Add individual site discriminator losses
        for site_name, loss in total_disc_site_losses.items():
            losses[f'disc_{site_name}_loss'] = loss.numpy()
        
        return losses


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class DataAugmenter:
    """Data augmentation for 2D slices"""
    
    @staticmethod
    @tf.function
    def augment(image, ga):
        """Apply random augmentations - NO ROTATION to avoid dimension issues"""
        
        # Random flip (these preserve dimensions)
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
        
        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, ga


# ============================================================================
# TRAINING LOOP
# ============================================================================

def create_tf_dataset(images, ga, batch_size, shuffle=True, augment=True):
    """Create TensorFlow dataset"""
    dataset = tf.data.Dataset.from_tensor_slices((images, ga))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    if augment:
        dataset = dataset.map(
            DataAugmenter.augment,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def train(args):
    """Main training function with multi-site discriminators"""
    
    # Configure GPU
    configure_gpu(args.gpu, memory_growth=True)
    
    # Create output directories
    weight_dir = Path(args.weight_dir)
    result_dir = Path(args.result_dir)
    log_dir = Path(args.log_dir)
    
    weight_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    train_images, train_ga, train_sex, train_site = load_preprocessed_data(
        args.train_data
    )
    
    # Check and fix NaN values
    if np.isnan(train_ga).any():
        print(f"Replacing {np.isnan(train_ga).sum()} NaN GA values with median")
        train_ga_median = np.nanmedian(train_ga)
        train_ga = np.where(np.isnan(train_ga), train_ga_median, train_ga)
    
    # Create site datasets
    train_site_data, ref_site = create_site_datasets(
        train_images, train_ga, train_sex, train_site, args.reference_site
    )
    
    # Get target sites (all sites except reference)
    target_sites = [s for s in train_site_data.keys() if s != ref_site]
    
    print(f"\nReference site: {ref_site}")
    print(f"Target sites: {target_sites}")
    
    # Build model with multi-site discriminators
    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)
    
    cyclegan = CycleGAN2D_MultiSite(
        img_shape=(138, 176, 1), 
        ga_embedding_dim=args.ga_embedding_dim,
        target_sites=target_sites
    )
    cyclegan.compile(lr=args.lr, beta_1=args.beta_1)
    
    # Create TF datasets for each site
    print("\n" + "="*80)
    print("CREATING DATASETS")
    print("="*80)
    
    site_datasets = {}
    for site_name, site_data in train_site_data.items():
        dataset = create_tf_dataset(
            site_data['images'], 
            site_data['ga'],
            args.batch_size,
            shuffle=True,
            augment=True
        )
        site_datasets[site_name] = dataset
        print(f"  {site_name}: {site_data['n_slices']} slices")
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*80)
    
    # Training history
    history = {
        'gen_loss': [],
        'disc_BCH_loss': [],
        'cycle_loss': [],
        'identity_loss': []
    }
    for site in target_sites:
        history[f'disc_{site}_loss'] = []
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        epoch_losses = {k: [] for k in history.keys()}
        
        # Create iterators for each site
        site_iters = {name: iter(dataset) for name, dataset in site_datasets.items()}
        
        # Calculate steps per epoch (minimum across all sites)
        steps_per_epoch = min([
            len(train_site_data[s]['images']) // args.batch_size 
            for s in train_site_data.keys()
        ])
        
        # Progress bar
        from tqdm import tqdm
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
        
        # Iterate over batches
        for step in pbar:
            # Get batch from each site
            site_batches = {}
            
            for site_name, site_iter in site_iters.items():
                try:
                    images, ga = next(site_iter)
                    
                    # Ensure GA has correct shape
                    if len(ga.shape) == 1:
                        ga = tf.expand_dims(ga, axis=-1)
                    
                    site_batches[site_name] = (images, ga)
                    
                except StopIteration:
                    # Reset iterator
                    site_iters[site_name] = iter(site_datasets[site_name])
                    images, ga = next(site_iters[site_name])
                    if len(ga.shape) == 1:
                        ga = tf.expand_dims(ga, axis=-1)
                    site_batches[site_name] = (images, ga)
            
            # Train step with all site batches
            try:
                losses = cyclegan.train_step(site_batches)
                
                # Check for NaN
                if any(np.isnan(v) for v in losses.values()):
                    print(f"\nWarning: NaN detected at step {step}")
                    continue
                
                # Accumulate losses
                for k, v in losses.items():
                    if k in epoch_losses:
                        epoch_losses[k].append(v)
                
                # Update progress bar
                pbar.set_postfix({
                    'G': f"{losses['gen_loss']:.3f}",
                    'D_BCH': f"{losses['disc_BCH_loss']:.3f}",
                    'Cycle': f"{losses['cycle_loss']:.3f}"
                })
                
            except Exception as e:
                print(f"\nError at step {step}: {e}")
                continue
        
        # Average losses
        if not epoch_losses['gen_loss']:
            print(f"Warning: No valid losses for epoch {epoch+1}")
            continue
        
        for k in history.keys():
            if epoch_losses[k]:
                avg_loss = np.mean(epoch_losses[k])
                history[k].append(avg_loss)
        
        # Print epoch summary
        print(f"  Gen: {history['gen_loss'][-1]:.4f} | "
              f"Disc BCH: {history['disc_BCH_loss'][-1]:.4f} | "
              f"Cycle: {history['cycle_loss'][-1]:.4f} | "
              f"Identity: {history['identity_loss'][-1]:.4f}")
        
        # Save checkpoints
        if (epoch + 1) % args.save_freq == 0:
            print(f"  Saving checkpoint at epoch {epoch+1}")
            cyclegan.gen_harmonize.save_weights(
                weight_dir / f'gen_harmonize_epoch_{epoch+1}.weights.h5'
            )
            cyclegan.disc_BCH.save_weights(
                weight_dir / f'disc_BCH_epoch_{epoch+1}.weights.h5'
            )
            for site_name, disc in cyclegan.disc_sites.items():
                safe_name = site_name.replace('_', '').replace('-', '')[:20]
                disc.save_weights(
                    weight_dir / f'disc_{safe_name}_epoch_{epoch+1}.weights.h5'
                )
    
    # Save final models
    print("\nSaving final models...")
    cyclegan.gen_harmonize.save_weights(weight_dir / 'gen_harmonize_final.weights.h5')
    cyclegan.disc_BCH.save_weights(weight_dir / 'disc_BCH_final.weights.h5')
    for site_name, disc in cyclegan.disc_sites.items():
        safe_name = site_name.replace('_', '').replace('-', '')[:20]
        disc.save_weights(weight_dir / f'disc_{safe_name}_final.weights.h5')
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(log_dir / 'training_history.csv', index=False)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train 2D CycleGAN for fetal brain harmonization'
    )
    
    # Data
    parser.add_argument('--train_data', default='processed_data_4slice/train_4slice_data.pkl')
    parser.add_argument('--val_data', default='processed_data_4slice/val_4slice_data.pkl')
    parser.add_argument('--reference_site', default='BCH_CHD', help='Reference site for harmonization')
    
    # Model
    parser.add_argument('--ga_embedding_dim', type=int, default=16, help='GA embedding dimension')
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Cycle consistency weight')
    parser.add_argument('--lambda_identity', type=float, default=5.0, help='Identity loss weight')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Adam beta_1')
    
    # Output
    parser.add_argument('--weight_dir', default='./weights/cyclegan_2d')
    parser.add_argument('--result_dir', default='./results/cyclegan_2d')
    parser.add_argument('--log_dir', default='./logs/cyclegan_2d')
    parser.add_argument('--save_freq', type=int, default=50, help='Save checkpoint every N epochs')
    
    # Hardware
    parser.add_argument('--gpu', default='0', help='GPU ID')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("="*80)
    
    train(args)


if __name__ == '__main__':
    main()