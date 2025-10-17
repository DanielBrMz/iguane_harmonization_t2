"""
2D CycleGAN for Fetal Brain MRI Harmonization
With proper two-generator architecture and safety checks
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

print("="*80)
print("FIXED FETAL BRAIN 2D CYCLEGAN - MULTI-DISCRIMINATOR")
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


# ============================================================================
# DATA LOADING
# ============================================================================

def load_preprocessed_data(data_path):
    """Load preprocessed 4-slice data"""
    print(f"Loading data from: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    images = data['images'].astype(np.float32) / 255.0
    ga = data['gestational_age'].astype(np.float32)
    sex = data['sex'].astype(np.float32)
    site = data['site']
    
    print(f"  Images: {images.shape}, dtype: {images.dtype}")
    print(f"  GA range: {ga.min():.1f} - {ga.max():.1f} weeks")
    print(f"  Sites: {np.unique(site)}")
    
    return images, ga, sex, site


def create_site_datasets(images, ga, sex, site, reference_site='BCH_CHD'):
    """Create separate datasets for each site"""
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
    
    if reference_site not in site_data:
        print(f" Reference site {reference_site} not found!")
        print(f"  Available sites: {list(site_data.keys())}")
        reference_site = list(site_data.keys())[0]
        print(f"  Using {reference_site} as reference instead")
    
    return site_data, reference_site


# ============================================================================
# NETWORK ARCHITECTURES (Lighter for stability)
# ============================================================================

def build_2d_generator(input_shape=(138, 176, 1), ga_embedding_dim=16, name='generator'):
    """
    Lighter 2D U-Net Generator with Gestational Age Conditioning
    Reduced capacity to prevent discriminator collapse
    """
    
    img_input = layers.Input(shape=input_shape, name='image_input')
    ga_input = layers.Input(shape=(1,), name='ga_input')
    
    # GA embedding
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_input)
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_embedding)
    
    # Encoder (even lighter: 16, 32, 64, 128)
    # Block 1
    x = layers.Conv2D(16, 3, padding='same')(img_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(16, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    skip1 = x
    x = layers.MaxPooling2D(2)(x)
    
    # Block 2
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    skip2 = x
    x = layers.MaxPooling2D(2)(x)
    
    # Block 3
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    skip3 = x
    x = layers.MaxPooling2D(2)(x)
    
    # Bottleneck
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
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
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Block 6
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    if x.shape[1] != skip2.shape[1] or x.shape[2] != skip2.shape[2]:
        x = layers.Resizing(skip2.shape[1], skip2.shape[2])(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Block 7
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    if x.shape[1] != skip1.shape[1] or x.shape[2] != skip1.shape[2]:
        x = layers.Resizing(skip1.shape[1], skip1.shape[2])(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(16, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(16, 3, padding='same')(x)
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


def build_2d_discriminator(input_shape=(138, 176, 1), ga_embedding_dim=16, name='discriminator'):
    """
    2D PatchGAN Discriminator with GA Conditioning and Label Smoothing
    """
    
    img_input = layers.Input(shape=input_shape, name='image_input')
    ga_input = layers.Input(shape=(1,), name='ga_input')
    
    # GA embedding
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_input)
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_embedding)
    
    # Discriminator path
    x = layers.Conv2D(64, 4, strides=2, padding='same')(img_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)  # Add dropout for stability
    
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(512, 4, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Inject GA
    ga_spatial = layers.RepeatVector(x.shape[1] * x.shape[2])(ga_embedding)
    ga_spatial = layers.Reshape((x.shape[1], x.shape[2], ga_embedding_dim))(ga_spatial)
    x = layers.Concatenate()([x, ga_spatial])
    
    # Output
    output = layers.Conv2D(1, 4, strides=1, padding='same')(x)
    
    model = Model(inputs=[img_input, ga_input], outputs=output, name=name)
    
    return model


# ============================================================================
# LOSSES with Label Smoothing
# ============================================================================

def cycle_consistency_loss(real_img, cycled_img):
    """L1 loss for cycle consistency"""
    return tf.reduce_mean(tf.abs(real_img - cycled_img))


def identity_loss(real_img, same_img):
    """L1 loss for identity mapping"""
    return tf.reduce_mean(tf.abs(real_img - same_img))


def discriminator_loss_smooth(real_output, fake_output, label_smoothing=0.1):
    """Discriminator loss with label smoothing to prevent collapse"""
    # Smooth labels: real = 0.9, fake = 0.1
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(real_output) * (1.0 - label_smoothing), 
        logits=real_output
    ))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(fake_output) + label_smoothing,
        logits=fake_output
    ))
    return real_loss + fake_loss


def generator_loss(fake_output):
    """Standard GAN generator loss"""
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(fake_output), logits=fake_output
    ))


# ============================================================================
# CYCLEGAN MODEL - FIXED with TWO GENERATORS
# ============================================================================

class CycleGAN2D_MultiSite:
    """
    CORRECT 2D CycleGAN with TWO generators and multiple discriminators
    """
    
    def __init__(self, img_shape=(138, 176, 1), ga_embedding_dim=16, target_sites=None):
        self.img_shape = img_shape
        self.ga_embedding_dim = ga_embedding_dim
        self.target_sites = target_sites or []
        
        print(f"\nBuilding CORRECTED CycleGAN:")
        print(f"  - 2 generators (forward & backward)")
        print(f"  - {len(self.target_sites) + 1} discriminators (BCH + {len(self.target_sites)} sites)")
        
        # Build TWO generators (THIS IS THE FIX!)
        self.gen_site2BCH = build_2d_generator(img_shape, ga_embedding_dim, name='gen_site2BCH')
        self.gen_BCH2site = build_2d_generator(img_shape, ga_embedding_dim, name='gen_BCH2site')
        
        # Build discriminators
        self.disc_BCH = build_2d_discriminator(img_shape, ga_embedding_dim, name='disc_BCH')
        
        self.disc_sites = {}
        for site in self.target_sites:
            site_name = site.replace('_', '').replace('-', '')[:20]
            self.disc_sites[site] = build_2d_discriminator(
                img_shape, ga_embedding_dim, name=f'disc_{site_name}'
            )
        
        # Loss weights (REDUCED for stability)
        self.lambda_cycle = 5.0  # Reduced from 10
        self.lambda_identity = 2.5  # Reduced from 5
        
        # Safety tracking
        self.collapse_counter = 0
        
    def compile(self, lr=0.0001, beta_1=0.5):  # LOWER learning rate
        """Compile model with optimizers"""
        self.gen_optimizer = Adam(learning_rate=lr, beta_1=beta_1)
        self.disc_optimizer = Adam(learning_rate=lr, beta_1=beta_1)
        
        # Build optimizers
        self.gen_optimizer.build(
            self.gen_site2BCH.trainable_variables + self.gen_BCH2site.trainable_variables
        )
        
        all_disc_vars = self.disc_BCH.trainable_variables
        for disc in self.disc_sites.values():
            all_disc_vars += disc.trainable_variables
        self.disc_optimizer.build(all_disc_vars)
        
        print("✓ Model compiled")
        print(f"  Gen site→BCH parameters: {self.gen_site2BCH.count_params():,}")
        print(f"  Gen BCH→site parameters: {self.gen_BCH2site.count_params():,}")
        print(f"  Disc BCH parameters: {self.disc_BCH.count_params():,}")
        print(f"  Disc per-site parameters: {list(self.disc_sites.values())[0].count_params():,} × {len(self.target_sites)}")
    
    def train_step(self, site_batches):
        """
        CORRECTED training step with proper cycle consistency
        """
        
        with tf.GradientTape(persistent=True) as tape:
            
            real_BCH, ga_BCH = site_batches['BCH_CHD']
            
            total_gen_loss = 0.0
            total_disc_BCH_loss = 0.0
            total_disc_site_losses = {}
            total_cycle_loss = 0.0
            total_identity_loss = 0.0
            
            for site_name in self.target_sites:
                if site_name not in site_batches:
                    continue
                
                real_site, ga_site = site_batches[site_name]
                disc_site = self.disc_sites[site_name]
                
                # FORWARD CYCLE: Site → BCH → Site (THIS IS THE FIX!)
                fake_BCH = self.gen_site2BCH([real_site, ga_site], training=True)
                cycled_site = self.gen_BCH2site([fake_BCH, ga_site], training=True)
                
                # BACKWARD CYCLE: BCH → Site → BCH (THIS IS THE FIX!)
                fake_site = self.gen_BCH2site([real_BCH, ga_BCH], training=True)
                cycled_BCH = self.gen_site2BCH([fake_site, ga_BCH], training=True)
                
                # IDENTITY: BCH through BCH→site should stay same (THIS IS THE FIX!)
                identity_BCH = self.gen_BCH2site([real_BCH, ga_BCH], training=True)
                # IDENTITY: Site through site→BCH should stay same
                identity_site = self.gen_site2BCH([real_site, ga_site], training=True)
                
                # Discriminator predictions
                disc_real_BCH = self.disc_BCH([real_BCH, ga_BCH], training=True)
                disc_fake_BCH = self.disc_BCH([fake_BCH, ga_site], training=True)
                
                disc_real_site = disc_site([real_site, ga_site], training=True)
                disc_fake_site = disc_site([fake_site, ga_BCH], training=True)
                
                # Generator losses
                gen_site2BCH_loss = generator_loss(disc_fake_BCH)
                gen_BCH2site_loss = generator_loss(disc_fake_site)
                
                # Cycle losses (THIS IS THE FIX!)
                cycle_loss_forward = cycle_consistency_loss(real_site, cycled_site)
                cycle_loss_backward = cycle_consistency_loss(real_BCH, cycled_BCH)
                cycle_loss_total = cycle_loss_forward + cycle_loss_backward
                
                # Identity losses (THIS IS THE FIX!)
                identity_loss_BCH = identity_loss(real_BCH, identity_BCH)
                identity_loss_site = identity_loss(real_site, identity_site)
                identity_loss_total = identity_loss_BCH + identity_loss_site
                
                # Discriminator losses with label smoothing
                disc_BCH_loss = discriminator_loss_smooth(disc_real_BCH, disc_fake_BCH)
                disc_site_loss = discriminator_loss_smooth(disc_real_site, disc_fake_site)
                
                # Accumulate
                total_gen_loss += (gen_site2BCH_loss + gen_BCH2site_loss + 
                                  self.lambda_cycle * cycle_loss_total + 
                                  self.lambda_identity * identity_loss_total)
                total_disc_BCH_loss += disc_BCH_loss
                total_disc_site_losses[site_name] = disc_site_loss
                total_cycle_loss += cycle_loss_total
                total_identity_loss += identity_loss_total
            
            # Average
            n_sites = len([s for s in self.target_sites if s in site_batches])
            if n_sites > 0:
                total_gen_loss /= n_sites
                total_disc_BCH_loss /= n_sites
                total_cycle_loss /= n_sites
                total_identity_loss /= n_sites
        
        # Apply gradients
        gen_gradients = tape.gradient(
            total_gen_loss, 
            self.gen_site2BCH.trainable_variables + self.gen_BCH2site.trainable_variables
        )
        
        disc_BCH_gradients = tape.gradient(
            total_disc_BCH_loss, self.disc_BCH.trainable_variables
        )
        
        disc_site_gradients = {}
        for site_name, disc in self.disc_sites.items():
            if site_name in total_disc_site_losses:
                disc_site_gradients[site_name] = tape.gradient(
                    total_disc_site_losses[site_name], disc.trainable_variables
                )
        
        # SAFETY CHECK: Clip gradients to prevent explosion
        gen_gradients, _ = tf.clip_by_global_norm(gen_gradients, 5.0)
        disc_BCH_gradients, _ = tf.clip_by_global_norm(disc_BCH_gradients, 5.0)
        
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, 
                self.gen_site2BCH.trainable_variables + self.gen_BCH2site.trainable_variables)
        )
        
        self.disc_optimizer.apply_gradients(
            zip(disc_BCH_gradients, self.disc_BCH.trainable_variables)
        )
        
        for site_name, gradients in disc_site_gradients.items():
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.disc_optimizer.apply_gradients(
                zip(gradients, self.disc_sites[site_name].trainable_variables)
            )
        
        del tape
        
        # SAFETY CHECK: Detect collapse
        if total_disc_BCH_loss < 0.01:
            self.collapse_counter += 1
        else:
            self.collapse_counter = 0
        
        losses = {
            'gen_loss': total_gen_loss.numpy(),
            'disc_BCH_loss': total_disc_BCH_loss.numpy(),
            'cycle_loss': total_cycle_loss.numpy(),
            'identity_loss': total_identity_loss.numpy(),
            'collapse_warning': self.collapse_counter
        }
        
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
        """Apply random augmentations"""
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)  # Reduced
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, ga


def create_tf_dataset(images, ga, batch_size, shuffle=True, augment=True):
    """Create TensorFlow dataset"""
    dataset = tf.data.Dataset.from_tensor_slices((images, ga))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    if augment:
        dataset = dataset.map(DataAugmenter.augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ============================================================================
# TRAINING LOOP with Safety Checks
# ============================================================================

def train(args):
    """Main training function with safety checks"""
    
    configure_gpu(args.gpu, memory_growth=True)
    
    weight_dir = Path(args.weight_dir)
    result_dir = Path(args.result_dir)
    log_dir = Path(args.log_dir)
    
    weight_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    train_images, train_ga, train_sex, train_site = load_preprocessed_data(args.train_data)
    
    if np.isnan(train_ga).any():
        print(f"Replacing {np.isnan(train_ga).sum()} NaN GA values")
        train_ga = np.where(np.isnan(train_ga), np.nanmedian(train_ga), train_ga)
    
    train_site_data, ref_site = create_site_datasets(
        train_images, train_ga, train_sex, train_site, args.reference_site
    )
    
    target_sites = [s for s in train_site_data.keys() if s != ref_site]
    
    print(f"\nReference: {ref_site}")
    print(f"Targets: {target_sites}")
    
    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)
    
    cyclegan = CycleGAN2D_MultiSite(
        img_shape=(138, 176, 1), 
        ga_embedding_dim=args.ga_embedding_dim,
        target_sites=target_sites
    )
    cyclegan.compile(lr=args.lr, beta_1=args.beta_1)
    
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
    
    print("\n" + "="*80)
    print("STARTING TRAINING WITH SAFETY CHECKS")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*80)
    
    history = {
        'gen_loss': [],
        'disc_BCH_loss': [],
        'cycle_loss': [],
        'identity_loss': []
    }
    for site in target_sites:
        history[f'disc_{site}_loss'] = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        epoch_losses = {k: [] for k in history.keys()}
        
        site_iters = {name: iter(dataset) for name, dataset in site_datasets.items()}
        
        steps_per_epoch = min([
            len(train_site_data[s]['images']) // args.batch_size 
            for s in train_site_data.keys()
        ])
        
        from tqdm import tqdm
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
        
        for step in pbar:
            site_batches = {}
            
            for site_name, site_iter in site_iters.items():
                try:
                    images, ga = next(site_iter)
                    if len(ga.shape) == 1:
                        ga = tf.expand_dims(ga, axis=-1)
                    site_batches[site_name] = (images, ga)
                except StopIteration:
                    site_iters[site_name] = iter(site_datasets[site_name])
                    images, ga = next(site_iters[site_name])
                    if len(ga.shape) == 1:
                        ga = tf.expand_dims(ga, axis=-1)
                    site_batches[site_name] = (images, ga)
            
            try:
                losses = cyclegan.train_step(site_batches)
                
                # SAFETY CHECK: NaN detection
                if any(np.isnan(v) if not isinstance(v, int) else False for v in losses.values()):
                    print(f"\n NaN detected at step {step}, skipping...")
                    continue
                
                # SAFETY CHECK: Collapse detection
                if losses.get('collapse_warning', 0) > 10:
                    print(f"\n WARNING: Discriminator collapse detected! (counter={losses['collapse_warning']})")
                    print("  Consider: reducing learning rate, increasing lambda weights, or restarting")
                
                for k, v in losses.items():
                    if k in epoch_losses and not isinstance(v, int):
                        epoch_losses[k].append(v)
                
                pbar.set_postfix({
                    'G': f"{losses['gen_loss']:.3f}",
                    'D_BCH': f"{losses['disc_BCH_loss']:.3f}",
                    'Cyc': f"{losses['cycle_loss']:.3f}",
                    'Collapse': losses.get('collapse_warning', 0)
                })
                
            except Exception as e:
                print(f"\n Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Average losses
        if not epoch_losses['gen_loss']:
            print(f" No valid losses for epoch {epoch+1}, skipping...")
            continue
        
        for k in history.keys():
            if epoch_losses[k]:
                avg_loss = np.mean(epoch_losses[k])
                history[k].append(avg_loss)
        
        # Print epoch summary
        print(f"\n  Gen: {history['gen_loss'][-1]:.4f} | "
              f"Disc BCH: {history['disc_BCH_loss'][-1]:.4f} | "
              f"Cycle: {history['cycle_loss'][-1]:.4f} | "
              f"Identity: {history['identity_loss'][-1]:.4f}")
        
        # SAFETY CHECK: Early stopping if discriminator collapses
        if history['disc_BCH_loss'][-1] < 0.001 and epoch > 10:
            print("\n DISCRIMINATOR COLLAPSE DETECTED ")
            print("Training stopped early to prevent wasted computation.")
            print("Recommendations:")
            print("  1. Reduce learning rate (try 0.00005)")
            print("  2. Increase lambda_cycle and lambda_identity")
            print("  3. Add more dropout to discriminator")
            break
        
        # Save checkpoints
        if (epoch + 1) % args.save_freq == 0:
            print(f"\n  Saving checkpoint at epoch {epoch+1}")
            cyclegan.gen_site2BCH.save_weights(
                weight_dir / f'gen_site2BCH_epoch_{epoch+1}.weights.h5'
            )
            cyclegan.gen_BCH2site.save_weights(
                weight_dir / f'gen_BCH2site_epoch_{epoch+1}.weights.h5'
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
    print("\n Saving final models...")
    cyclegan.gen_site2BCH.save_weights(weight_dir / 'gen_site2BCH_final.weights.h5')
    cyclegan.gen_BCH2site.save_weights(weight_dir / 'gen_BCH2site_final.weights.h5')
    cyclegan.disc_BCH.save_weights(weight_dir / 'disc_BCH_final.weights.h5')
    for site_name, disc in cyclegan.disc_sites.items():
        safe_name = site_name.replace('_', '').replace('-', '')[:20]
        disc.save_weights(weight_dir / f'disc_{safe_name}_final.weights.h5')
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(log_dir / 'training_history.csv', index=False)
    
    print("\n" + "="*80)
    print(" TRAINING COMPLETE!")
    print("="*80)
    
    # Final safety report
    print("\n TRAINING QUALITY REPORT:")
    if len(history['disc_BCH_loss']) > 0:
        final_disc_loss = history['disc_BCH_loss'][-1]
        if final_disc_loss < 0.01:
            print("   POOR: Discriminator collapsed (loss < 0.01)")
        elif final_disc_loss < 0.1:
            print("   FAIR: Discriminator weak (loss < 0.1)")
        elif final_disc_loss > 2.0:
            print("   POOR: Discriminator too strong (loss > 2.0)")
        else:
            print("   GOOD: Discriminator in healthy range")
        
        final_gen_loss = history['gen_loss'][-1]
        if final_gen_loss > 50:
            print("   POOR: Generator struggling (loss > 50)")
        elif final_gen_loss < 5:
            print("   SUSPICIOUS: Generator may have collapsed (loss < 5)")
        else:
            print("   GOOD: Generator learning")
        
        final_cycle = history['cycle_loss'][-1]
        if final_cycle < 0.1:
            print("   EXCELLENT: Strong cycle consistency")
        elif final_cycle < 1.0:
            print("   GOOD: Decent cycle consistency")
        else:
            print("   POOR: Weak cycle consistency (loss > 1.0)")


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
    parser.add_argument('--reference_site', default='BCH_CHD')
    
    # Model
    parser.add_argument('--ga_embedding_dim', type=int, default=16)
    
    # Training 
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)  # Reduced from 32
    parser.add_argument('--lr', type=float, default=0.0001)  # Reduced from 0.0002
    parser.add_argument('--beta_1', type=float, default=0.5)
    
    # Output
    parser.add_argument('--weight_dir', default='./weights/cyclegan_2d')
    parser.add_argument('--result_dir', default='./results/cyclegan_2d')
    parser.add_argument('--log_dir', default='./logs/cyclegan_2d')
    parser.add_argument('--save_freq', type=int, default=25)  # More frequent saves
    
    # Hardware
    parser.add_argument('--gpu', default='0')
    
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