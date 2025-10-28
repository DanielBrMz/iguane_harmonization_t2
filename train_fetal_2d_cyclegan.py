"""
2D CycleGAN for Fetal Brain MRI Harmonization
IGUANe-Compliant Implementation with Critical Architectural Fixes

Key Changes from Previous Version:
1. MSE (LSGAN) loss instead of sigmoid cross-entropy (IGUANe paper requirement)
2. InstanceNormalization with MAD instead of BatchNormalization
3. N forward discriminators (one per source site) - TRUE IGUANe architecture
4. Removed Spectral Normalization (not in IGUANe)
5. Linear learning rate decay (0.0002 → 0.00002)
6. Same LR for generators and discriminators
7. Reduced batch sizes (closer to IGUANe: gen=1-2, disc=2-4)
8. Removed dropout from discriminator
9. Removed label smoothing
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

print("="*80)
print("FETAL BRAIN 2D CYCLEGAN - IGUANE-COMPLIANT TRAINING")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
print("="*80)


# ============================================================================
# GPU CONFIGURATION
# ============================================================================

def print_gpu_usage():
    """Print current GPU memory usage"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("\n  GPU Memory Usage:")
        for i, gpu in enumerate(gpus):
            try:
                memory_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                current_mb = memory_info['current'] / 1024**2
                peak_mb = memory_info['peak'] / 1024**2
                print(f"    GPU {i}: Current={current_mb:.0f}MB, Peak={peak_mb:.0f}MB")
            except Exception as e:
                print(f"    GPU {i}: Unable to get memory info")


def configure_gpu(gpu_ids='0,1,2', memory_growth=True):
    """Configure multiple GPU settings"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    tf.keras.backend.clear_session()
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    return gpus


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
        print(f"  Reference site {reference_site} not found!")
        print(f"  Available sites: {list(site_data.keys())}")
        reference_site = list(site_data.keys())[0]
        print(f"  Using {reference_site} as reference instead")
    
    return site_data, reference_site


# ============================================================================
# INSTANCE NORMALIZATION WITH MAD (IGUANe-Compliant)
# ============================================================================

class InstanceNormalizationMAD(layers.Layer):
    """
    Instance Normalization with Mean Absolute Deviation
    As specified in IGUANe paper (Appendix B.4):
    'Both generators and discriminators use instance normalization with mean 
    absolute deviation instead of standard deviation (Wu et al., 2019)'
    """
    
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalizationMAD, self).__init__(**kwargs)
        self.epsilon = epsilon
    
    def build(self, input_shape):
        # Learnable scale and shift parameters
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True,
            name='gamma'
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='beta'
        )
        super(InstanceNormalizationMAD, self).build(input_shape)
    
    def call(self, x):
        # Compute mean per instance (across spatial dimensions)
        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        
        # Compute Mean Absolute Deviation (MAD) instead of std
        mad = tf.reduce_mean(tf.abs(x - mean), axis=[1, 2], keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / (mad + self.epsilon)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta


# ============================================================================
# LEARNING RATE SCHEDULE (IGUANe-Compliant)
# ============================================================================

class LinearDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear learning rate decay from initial_lr to final_lr
    As specified in IGUANe paper (Appendix B.5):
    'linear learning rate decay from 0.0002 to 0.00002'
    """
    
    def __init__(self, initial_lr=0.0002, final_lr=0.00002, total_steps=20000):
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.final_lr = tf.cast(final_lr, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Linear interpolation
        alpha = tf.minimum(step / self.total_steps, 1.0)
        lr = self.initial_lr + alpha * (self.final_lr - self.initial_lr)
        return lr
    
    def get_config(self):
        return {
            'initial_lr': float(self.initial_lr),
            'final_lr': float(self.final_lr),
            'total_steps': float(self.total_steps)
        }


# ============================================================================
# NETWORK ARCHITECTURES (IGUANe-Compliant)
# ============================================================================

def build_2d_generator_iguane(input_shape=(138, 176, 1), ga_embedding_dim=16, name='generator'):
    """
    2D U-Net Generator with GA Conditioning - IGUANe-Compliant
    
    Key IGUANe features:
    - InstanceNormalization with MAD (not BatchNorm!)
    - Residual learning with tanh activation
    - Multiple skip connections
    - Output clipped to [0, 1]
    """
    
    img_input = layers.Input(shape=input_shape, name='image_input')
    ga_input = layers.Input(shape=(1,), name='ga_input')
    
    # GA embedding (dropout for regularization)
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_input)
    ga_embedding = layers.Dropout(0.2)(ga_embedding)
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_embedding)

    # Encoder with InstanceNorm (IGUANe style)
    x = layers.Conv2D(32, 3, padding='same')(img_input)
    x = InstanceNormalizationMAD()(x)  # IGUANe: InstanceNorm with MAD!
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    skip1 = x
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    skip2 = x
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    skip3 = x
    x = layers.MaxPooling2D(2)(x)
    
    # Bottleneck with GA injection
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Inject GA embedding
    ga_spatial = layers.RepeatVector(x.shape[1] * x.shape[2])(ga_embedding)
    ga_spatial = layers.Reshape((x.shape[1], x.shape[2], ga_embedding_dim))(ga_spatial)
    x = layers.Concatenate()([x, ga_spatial])
    
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Decoder with skip connections
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    if x.shape[1] != skip3.shape[1] or x.shape[2] != skip3.shape[2]:
        x = layers.Resizing(skip3.shape[1], skip3.shape[2])(x)
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    if x.shape[1] != skip2.shape[1] or x.shape[2] != skip2.shape[2]:
        x = layers.Resizing(skip2.shape[1], skip2.shape[2])(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    if x.shape[1] != skip1.shape[1] or x.shape[2] != skip1.shape[2]:
        x = layers.Resizing(skip1.shape[1], skip1.shape[2])(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Final output with shape guarantee
    if x.shape[1] != input_shape[0] or x.shape[2] != input_shape[1]:
        x = layers.Resizing(input_shape[0], input_shape[1])(x)
    
    # IGUANe residual learning: tanh activation, add to input
    # "The last activation function before the addition is tanh" (Appendix B.4)
    residual = layers.Conv2D(1, 1, padding='same', activation='tanh')(x)
    
    # Add residual to input (residual learning)
    output = layers.Add()([img_input, residual])
    
    # Clip negative values (IGUANe: "negative voxels clipped to background value")
    output = layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0))(output)
    
    model = Model(inputs=[img_input, ga_input], outputs=output, name=name)
    
    return model


def build_2d_discriminator_iguane(input_shape=(138, 176, 1), ga_embedding_dim=16, name='discriminator'):
    """
    2D PatchGAN Discriminator with GA Conditioning - IGUANe-Compliant
    
    Key IGUANe features:
    - InstanceNormalization with MAD (not SpectralNorm!)
    - NO dropout (not in IGUANe)
    - LeakyReLU(0.2)
    - Architecture: C64S2K4 → C128S2K4 → C256S2K4 → C1S1K3
    """
    
    img_input = layers.Input(shape=input_shape, name='image_input')
    ga_input = layers.Input(shape=(1,), name='ga_input')
    
    # GA embedding
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_input)
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_embedding)
    
    # IGUANe discriminator architecture (2D adaptation)
    # NO Spectral Normalization! (not in IGUANe)
    # YES InstanceNormalization with MAD!
    
    # C64S2K4
    x = layers.Conv2D(64, 4, strides=2, padding='same')(img_input)
    x = InstanceNormalizationMAD()(x)  # IGUANe style!
    x = layers.LeakyReLU(0.2)(x)
    # NO dropout! (not in IGUANe)
    
    # C128S2K4
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # C256S2K4
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Additional layer for 2D (to increase receptive field)
    x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
    x = InstanceNormalizationMAD()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # GA injection
    ga_spatial = layers.RepeatVector(x.shape[1] * x.shape[2])(ga_embedding)
    ga_spatial = layers.Reshape((x.shape[1], x.shape[2], ga_embedding_dim))(ga_spatial)
    x = layers.Concatenate()([x, ga_spatial])
    
    # C1S1K3 - final classification
    x = layers.Conv2D(1, 3, strides=1, padding='same')(x)
    # NO final activation (returns logits for MSE loss)
    
    model = Model(inputs=[img_input, ga_input], outputs=x, name=name)
    
    return model


# ============================================================================
# LOSSES (IGUANe-Compliant MSE/LSGAN)
# ============================================================================

def discriminator_loss_mse(real_output, fake_output):
    """
    IGUANe discriminator loss using Mean Squared Error (LSGAN)
    Paper (Section 3.2.3): 'Mean squared error is used as the adversarial loss (Mao et al., 2017)'
    
    For real images: minimize (D(real) - 1)²
    For fake images: minimize (D(fake) - 0)²
    """
    real_loss = tf.reduce_mean(tf.square(real_output - 1.0))
    fake_loss = tf.reduce_mean(tf.square(fake_output))
    return real_loss + fake_loss


def generator_loss_mse(fake_output):
    """
    IGUANe generator loss using Mean Squared Error (LSGAN)
    Generator tries to make discriminator output 1 for fake images
    minimize (D(G(x)) - 1)²
    """
    return tf.reduce_mean(tf.square(fake_output - 1.0))


def cycle_consistency_loss(real_img, cycled_img):
    """
    L1 loss for cycle consistency (IGUANe Appendix B.3)
    L_cyc = |x - x'|₁ / N
    """
    return tf.reduce_mean(tf.abs(real_img - cycled_img))


def identity_loss(real_img, same_img):
    """
    L1 loss for identity mapping (IGUANe Appendix B.3)
    L_id = |x - x'|₁ / N
    """
    return tf.reduce_mean(tf.abs(real_img - same_img))


# ============================================================================
# DATA PIPELINE
# ============================================================================

def create_tf_dataset(images, ga, batch_size=1, shuffle=True, augment=False):
    """
    Create TensorFlow dataset with IGUANe-style augmentation
    
    IGUANe augmentation (Appendix B.5):
    - Random translation: ±5 voxels
    - Random rotation: ±10° (probability 1/2)
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, ga))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, len(images)))
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    if augment:
        def augment_fn(img_batch, ga_batch):
            # IGUANe-style augmentation
            # Random rotation ±10° with probability 0.5
            if tf.random.uniform([]) > 0.5:
                angle = tf.random.uniform([], -10, 10) * (3.14159 / 180.0)
                img_batch = tfa.image.rotate(img_batch, angle, interpolation='bilinear')
            
            # Random translation ±5 pixels (adapted from ±5 voxels)
            dx = tf.random.uniform([], -5, 5, dtype=tf.int32)
            dy = tf.random.uniform([], -5, 5, dtype=tf.int32)
            img_batch = tfa.image.translate(img_batch, [dx, dy])
            
            return img_batch, ga_batch
        
        try:
            import tensorflow_addons as tfa
            dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        except ImportError:
            print("WARNING: tensorflow_addons not available, using basic augmentation")
            # Fallback to basic augmentation
            dataset = dataset.map(
                lambda img, ga: (tf.image.random_flip_left_right(img), ga),
                num_parallel_calls=tf.data.AUTOTUNE
            )
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ============================================================================
# CYCLEGAN MODEL - TRUE IGUANE ARCHITECTURE
# ============================================================================

class CycleGAN2D_IGUANe:
    """
    2D CycleGAN following TRUE IGUANe architecture
    
    Key IGUANe features:
    1. N forward discriminators (one per source site) - CRITICAL!
    2. MSE (LSGAN) loss instead of cross-entropy
    3. InstanceNormalization with MAD
    4. Linear LR decay
    5. Batch size 1-2 for generators, 2-4 for discriminators
    6. Same LR for all optimizers
    """
    
    def __init__(self, img_shape=(138, 176, 1), ga_embedding_dim=16, target_sites=None,
                 lambda_cycle=30.0, lambda_identity=15.0):
        self.img_shape = img_shape
        self.ga_embedding_dim = ga_embedding_dim
        self.target_sites = target_sites if target_sites else []
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.collapse_counter = 0
        
        print("\nBuilding IGUANe-compliant CycleGAN architecture...")
        
        # Forward generator (GenFwd): All sites → Reference
        print("  Building forward generator (site→BCH)...")
        self.gen_site2BCH = build_2d_generator_iguane(
            img_shape, ga_embedding_dim, name='gen_site2BCH'
        )
        
        # Backward generators (GenBwd): Reference → Each site
        print("  Building backward generators (BCH→site)...")
        self.gen_BCH2site = {}
        for site in self.target_sites:
            site_name = site.replace('_', '').replace('-', '')[:20]
            self.gen_BCH2site[site] = build_2d_generator_iguane(
                img_shape, ga_embedding_dim, name=f'gen_BCH2{site_name}'
            )
        
        # CRITICAL: N forward discriminators (one per source site!)
        # This is a KEY difference from your previous implementation!
        print("  Building N forward discriminators (one per source site)...")
        self.disc_fwd_sites = {}
        for site in self.target_sites:
            site_name = site.replace('_', '').replace('-', '')[:20]
            self.disc_fwd_sites[site] = build_2d_discriminator_iguane(
                img_shape, ga_embedding_dim, name=f'disc_fwd_{site_name}'
            )
        
        # Backward discriminators (DiscBwd): One per source site
        print("  Building backward discriminators (site discriminators)...")
        self.disc_sites = {}
        for site in self.target_sites:
            site_name = site.replace('_', '').replace('-', '')[:20]
            self.disc_sites[site] = build_2d_discriminator_iguane(
                img_shape, ga_embedding_dim, name=f'disc_{site_name}'
            )
        
        # Reference discriminator (for backward path)
        print("  Building reference discriminator (BCH)...")
        self.disc_BCH = build_2d_discriminator_iguane(
            img_shape, ga_embedding_dim, name='disc_BCH'
        )
        
        print("  ✓ Architecture built successfully")
    
    def compile(self, initial_lr=0.0002, final_lr=0.00002, total_steps=20000):
        """
        Compile with IGUANe-compliant optimizers
        - Same LR for all optimizers (IGUANe doesn't differentiate)
        - Linear decay from 0.0002 to 0.00002
        """
        
        lr_schedule = LinearDecaySchedule(initial_lr, final_lr, total_steps)
        
        # All optimizers use same LR schedule (IGUANe style)
        self.gen_optimizer = Adam(learning_rate=lr_schedule, beta_1=0.5, beta_2=0.999)
        self.disc_BCH_optimizer = Adam(learning_rate=lr_schedule, beta_1=0.5, beta_2=0.999)
        
        # Forward discriminator optimizers (one per site)
        self.disc_fwd_optimizers = {}
        for site in self.target_sites:
            self.disc_fwd_optimizers[site] = Adam(
                learning_rate=lr_schedule, beta_1=0.5, beta_2=0.999
            )
        
        # Backward discriminator optimizers
        self.disc_site_optimizers = {}
        for site in self.target_sites:
            self.disc_site_optimizers[site] = Adam(
                learning_rate=lr_schedule, beta_1=0.5, beta_2=0.999
            )
        
        # Backward generator optimizers
        self.gen_bwd_optimizers = {}
        for site in self.target_sites:
            self.gen_bwd_optimizers[site] = Adam(
                learning_rate=lr_schedule, beta_1=0.5, beta_2=0.999
            )
        
        print("\n✓ Model compiled with IGUANe-compliant optimizers:")
        print(f"  Initial LR: {initial_lr}, Final LR: {final_lr}")
        print(f"  Total steps: {total_steps}")
        print(f"  Gen params: {self.gen_site2BCH.count_params():,}")
        print(f"  Disc params: {self.disc_BCH.count_params():,}")
        print(f"  N source sites: {len(self.target_sites)}")
        print(f"  Total optimizers: {1 + len(self.gen_bwd_optimizers) + 1 + len(self.disc_fwd_optimizers) + len(self.disc_site_optimizers)}")
    
    @tf.function
    def train_step(self, site_batches, step_counter):
        """
        IGUANe training step - TRUE architecture
        
        For EACH source site (in random order):
        1. Update forward discriminator (DiscFwdᵢ)
        2. Update backward discriminator (DiscBwdᵢ)
        3. Update GenFwd (accumulated) and GenBwdᵢ (immediate)
        """
        
        total_gen_loss = 0.0
        total_disc_BCH_loss = 0.0
        total_cycle_loss = 0.0
        total_identity_loss = 0.0
        total_disc_site_losses = {}
        n_sites = 0.0
        
        # Accumulate forward generator gradients
        accumulated_gen_fwd_grads = None
        
        # Iterate through each source site
        for site_name, (real_site, ga_site) in site_batches.items():
            if site_name == 'BCH_CHD':  # Skip reference site
                continue
            
            # Get corresponding backward generator and discriminators
            gen_BCH2site_i = self.gen_BCH2site[site_name]
            disc_fwd_i = self.disc_fwd_sites[site_name]  # Forward disc for this site!
            disc_site = self.disc_sites[site_name]  # Backward disc for this site
            
            # Sample reference site data (randomly from batch)
            ref_site_name = 'BCH_CHD'
            if ref_site_name in site_batches:
                real_BCH, ga_BCH = site_batches[ref_site_name]
            else:
                # If BCH not in batch, use any other site as fallback
                real_BCH, ga_BCH = real_site, ga_site
            
            # ============================================================
            # STEP 1: UPDATE FORWARD DISCRIMINATOR (DiscFwdᵢ)
            # ============================================================
            with tf.GradientTape() as disc_fwd_tape:
                # Generate fake reference images from this source site
                fake_BCH = self.gen_site2BCH([real_site, ga_site], training=True)
                
                # Forward discriminator judges: real BCH vs fake BCH (from sitei)
                disc_real_BCH = disc_fwd_i([real_BCH, ga_BCH], training=True)
                disc_fake_BCH = disc_fwd_i([fake_BCH, ga_site], training=True)
                
                # MSE loss (LSGAN) - IGUANe style!
                disc_fwd_loss = discriminator_loss_mse(disc_real_BCH, disc_fake_BCH)
            
            disc_fwd_grads = disc_fwd_tape.gradient(
                disc_fwd_loss, disc_fwd_i.trainable_variables
            )
            disc_fwd_grads, _ = tf.clip_by_global_norm(disc_fwd_grads, 5.0)
            self.disc_fwd_optimizers[site_name].apply_gradients(
                zip(disc_fwd_grads, disc_fwd_i.trainable_variables)
            )
            
            # ============================================================
            # STEP 2: UPDATE BACKWARD DISCRIMINATOR (DiscBwdᵢ)
            # ============================================================
            with tf.GradientTape() as disc_site_tape:
                # Generate fake source images from reference
                fake_site = gen_BCH2site_i([real_BCH, ga_BCH], training=True)
                
                # Backward discriminator judges: real sitei vs fake sitei
                disc_real_site = disc_site([real_site, ga_site], training=True)
                disc_fake_site = disc_site([fake_site, ga_BCH], training=True)
                
                # MSE loss (LSGAN) - IGUANe style!
                disc_site_loss = discriminator_loss_mse(disc_real_site, disc_fake_site)
            
            disc_site_grads = disc_site_tape.gradient(
                disc_site_loss, disc_site.trainable_variables
            )
            disc_site_grads, _ = tf.clip_by_global_norm(disc_site_grads, 5.0)
            self.disc_site_optimizers[site_name].apply_gradients(
                zip(disc_site_grads, disc_site.trainable_variables)
            )
            
            # ============================================================
            # STEP 3: UPDATE GENERATORS
            # ============================================================
            with tf.GradientTape() as fwd_tape, tf.GradientTape() as bwd_tape:
                # Forward cycle: site → BCH → site
                fake_BCH_gen = self.gen_site2BCH([real_site, ga_site], training=True)
                cycled_site = gen_BCH2site_i([fake_BCH_gen, ga_site], training=True)
                
                # Backward cycle: BCH → site → BCH
                fake_site_gen = gen_BCH2site_i([real_BCH, ga_BCH], training=True)
                cycled_BCH = self.gen_site2BCH([fake_site_gen, ga_BCH], training=True)
                
                # Identity
                identity_BCH = gen_BCH2site_i([real_BCH, ga_BCH], training=True)
                identity_site = self.gen_site2BCH([real_site, ga_site], training=True)
                
                # Discriminator predictions for generator training
                disc_fake_BCH_gen = disc_fwd_i([fake_BCH_gen, ga_site], training=False)
                disc_fake_site_gen = disc_site([fake_site_gen, ga_BCH], training=False)
                
                # Generator losses (MSE/LSGAN style)
                gen_site2BCH_loss = generator_loss_mse(disc_fake_BCH_gen)
                gen_BCH2site_loss = generator_loss_mse(disc_fake_site_gen)
                
                # Cycle consistency losses
                cycle_loss_forward = cycle_consistency_loss(real_site, cycled_site)
                cycle_loss_backward = cycle_consistency_loss(real_BCH, cycled_BCH)
                cycle_loss_total = cycle_loss_forward + cycle_loss_backward
                
                # Identity losses
                identity_loss_BCH = identity_loss(real_BCH, identity_BCH)
                identity_loss_site = identity_loss(real_site, identity_site)
                identity_loss_total = identity_loss_BCH + identity_loss_site
                
                # Combined generator loss (IGUANe formula: L = L_adv + λ*L_cyc + (λ/2)*L_id)
                gen_loss = (gen_site2BCH_loss + gen_BCH2site_loss + 
                           self.lambda_cycle * cycle_loss_total + 
                           (self.lambda_cycle / 2) * identity_loss_total)  # λ/2 for identity!
            
            # Accumulate forward generator gradients
            gen_fwd_vars = self.gen_site2BCH.trainable_variables
            gen_fwd_grads = fwd_tape.gradient(gen_loss, gen_fwd_vars)
            gen_fwd_grads, _ = tf.clip_by_global_norm(gen_fwd_grads, 5.0)
            
            if accumulated_gen_fwd_grads is None:
                accumulated_gen_fwd_grads = gen_fwd_grads
            else:
                accumulated_gen_fwd_grads = [
                    ag + g for ag, g in zip(accumulated_gen_fwd_grads, gen_fwd_grads)
                ]
            
            # Update backward generator immediately
            gen_bwd_vars = gen_BCH2site_i.trainable_variables
            gen_bwd_grads = bwd_tape.gradient(gen_loss, gen_bwd_vars)
            gen_bwd_grads, _ = tf.clip_by_global_norm(gen_bwd_grads, 5.0)
            self.gen_bwd_optimizers[site_name].apply_gradients(
                zip(gen_bwd_grads, gen_bwd_vars)
            )
            
            # Accumulate metrics
            total_gen_loss += gen_loss
            total_disc_BCH_loss += disc_fwd_loss  # Note: using forward disc loss
            total_cycle_loss += cycle_loss_total
            total_identity_loss += identity_loss_total
            total_disc_site_losses[site_name] = disc_site_loss
            n_sites += 1.0
        
        # ============================================================
        # UPDATE FORWARD GENERATOR (once per step, averaged gradients)
        # ============================================================
        if n_sites > 0 and accumulated_gen_fwd_grads is not None:
            # Average accumulated gradients
            accumulated_gen_fwd_grads = [g / n_sites for g in accumulated_gen_fwd_grads]
            
            self.gen_optimizer.apply_gradients(
                zip(accumulated_gen_fwd_grads, self.gen_site2BCH.trainable_variables)
            )
            
            # Average losses
            total_gen_loss /= n_sites
            total_disc_BCH_loss /= n_sites
            total_cycle_loss /= n_sites
            total_identity_loss /= n_sites
        
        # Collapse detection
        if total_disc_BCH_loss < 0.01:
            self.collapse_counter += 1
        else:
            self.collapse_counter = tf.maximum(0, self.collapse_counter - 1)
        
        losses = {
            'gen_loss': total_gen_loss,
            'disc_BCH_loss': total_disc_BCH_loss,
            'cycle_loss': total_cycle_loss,
            'identity_loss': total_identity_loss,
            'collapse_warning': self.collapse_counter
        }
        
        for site_name, loss in total_disc_site_losses.items():
            losses[f'disc_{site_name}_loss'] = loss
        
        return losses


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args):
    """Main training function"""
    
    # Configure GPU
    gpus = configure_gpu(args.gpu)
    
    # Setup directories
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
    train_images, train_ga, train_sex, train_site = load_preprocessed_data(args.train_data)
    
    # Create site datasets
    train_site_data, ref_site = create_site_datasets(
        train_images, train_ga, train_sex, train_site, args.reference_site
    )
    
    target_sites = [s for s in train_site_data.keys() if s != ref_site]
    
    # Create model
    print("\n" + "="*80)
    print("CREATING IGUANE-COMPLIANT MODEL")
    print("="*80)
    
    cyclegan = CycleGAN2D_IGUANe(
        img_shape=(138, 176, 1),
        ga_embedding_dim=args.ga_embedding_dim,
        target_sites=target_sites,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity
    )
    
    # Calculate total training steps for LR schedule
    steps_per_epoch = max([
        len(train_site_data[s]['images']) // args.batch_size 
        for s in train_site_data.keys()
    ])
    total_steps = args.epochs * steps_per_epoch
    
    cyclegan.compile(
        initial_lr=args.initial_lr,
        final_lr=args.final_lr,
        total_steps=total_steps
    )
    
    # Create datasets - IGUANe uses small batch sizes!
    print("\n" + "="*80)
    print("CREATING DATASETS")
    print("="*80)
    print(f"  Generator batch size: {args.batch_size_gen} (IGUANe uses 1)")
    print(f"  Discriminator batch size: {args.batch_size_disc} (IGUANe uses 2)")
    
    site_datasets = {}
    for site_name, site_data in train_site_data.items():
        dataset = create_tf_dataset(
            site_data['images'],
            site_data['ga'],
            batch_size=args.batch_size_gen,  # Use small batch size!
            shuffle=True,
            augment=True
        )
        site_datasets[site_name] = dataset
        print(f"  {site_name}: {site_data['n_slices']} slices")
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING IGUANE-COMPLIANT TRAINING")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Lambda cycle: {args.lambda_cycle}")
    print(f"Lambda identity: {args.lambda_cycle / 2}")  # IGUANe uses λ/2!
    print("="*80)
    
    history = {
        'gen_loss': [],
        'disc_BCH_loss': [],
        'cycle_loss': [],
        'identity_loss': []
    }
    for site in target_sites:
        history[f'disc_{site}_loss'] = []
    
    step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        epoch_losses = {k: [] for k in history.keys()}
        
        site_iters = {name: iter(dataset) for name, dataset in site_datasets.items()}
        
        from tqdm import tqdm
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
        
        for step in pbar:
            site_batches = {}
            
            # Sample from each site
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
                losses = cyclegan.train_step(site_batches, step_counter)
                step_counter.assign_add(1)
                
                # Convert losses to numpy
                losses_np = {k: float(v.numpy()) if tf.is_tensor(v) else v 
                           for k, v in losses.items()}
                
                # NaN detection
                if any(np.isnan(v) for v in losses_np.values() if isinstance(v, float)):
                    print(f"\n  NaN detected, skipping step {step}")
                    continue
                
                # Track losses
                for k, v in losses_np.items():
                    if k in epoch_losses and isinstance(v, float):
                        epoch_losses[k].append(v)
                
                pbar.set_postfix({
                    'G': f"{losses_np['gen_loss']:.3f}",
                    'D': f"{losses_np['disc_BCH_loss']:.3f}",
                    'Cyc': f"{losses_np['cycle_loss']:.3f}",
                    'Collapse': losses_np.get('collapse_warning', 0)
                })
                
            except Exception as e:
                print(f"\n  Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Average losses
        if not epoch_losses['gen_loss']:
            print(f"  No valid losses for epoch {epoch+1}")
            continue
        
        for k in history.keys():
            if epoch_losses[k]:
                avg_loss = np.mean(epoch_losses[k])
                history[k].append(avg_loss)
        
        # Print summary
        print(f"\n  Gen: {history['gen_loss'][-1]:.4f} | "
              f"Disc: {history['disc_BCH_loss'][-1]:.4f} | "
              f"Cycle: {history['cycle_loss'][-1]:.4f} | "
              f"Identity: {history['identity_loss'][-1]:.4f}")
        
        # Save checkpoints
        if (epoch + 1) % args.save_freq == 0:
            print(f"\n  Saving checkpoint at epoch {epoch+1}")
            cyclegan.gen_site2BCH.save_weights(
                weight_dir / f'gen_site2BCH_epoch_{epoch+1}.weights.h5'
            )
            cyclegan.disc_BCH.save_weights(
                weight_dir / f'disc_BCH_epoch_{epoch+1}.weights.h5'
            )
    
    # Save final models
    print("\n  Saving final models...")
    cyclegan.gen_site2BCH.save_weights(weight_dir / 'gen_site2BCH_final.weights.h5')
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(log_dir / 'training_history.csv', index=False)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train IGUANe-compliant 2D CycleGAN for fetal brain harmonization'
    )
    
    # Data
    parser.add_argument('--train_data', default='processed_data_4slice/train_4slice_data.pkl')
    parser.add_argument('--reference_site', default='BCH_CHD')
    
    # Model
    parser.add_argument('--ga_embedding_dim', type=int, default=16)
    
    # Training - IGUANe-compliant hyperparameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='IGUANe uses 100 epochs')
    parser.add_argument('--batch_size_gen', type=int, default=1,
                       help='Generator batch size (IGUANe uses 1)')
    parser.add_argument('--batch_size_disc', type=int, default=2,
                       help='Discriminator batch size (IGUANe uses 2)')
    parser.add_argument('--initial_lr', type=float, default=0.0002,
                       help='Initial learning rate (IGUANe: 0.0002)')
    parser.add_argument('--final_lr', type=float, default=0.00002,
                       help='Final learning rate (IGUANe: 0.00002)')
    parser.add_argument('--lambda_cycle', type=float, default=30.0,
                       help='Cycle consistency loss weight (IGUANe: 30)')
    parser.add_argument('--lambda_identity', type=float, default=15.0,
                       help='Identity loss weight (IGUANe: λ/2 = 15)')
    
    # Output
    parser.add_argument('--weight_dir', default='./weights/cyclegan_2d_iguane')
    parser.add_argument('--result_dir', default='./results/cyclegan_2d_iguane')
    parser.add_argument('--log_dir', default='./logs/cyclegan_2d_iguane')
    parser.add_argument('--save_freq', type=int, default=25)
    
    # Hardware
    parser.add_argument('--gpu', default='0,1,2')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("IGUANE-COMPLIANT CONFIGURATION")
    print("="*80)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("="*80)
    
    train(args)


if __name__ == '__main__':
    main()