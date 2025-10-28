"""
2D CycleGAN for Fetal Brain MRI Harmonization
IGUANe-style training with gradient accumulation and comprehensive collapse prevention
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
print("FETAL BRAIN 2D CYCLEGAN - IGUANE TRAINING")
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
# SPECTRAL NORMALIZATION
# ============================================================================

class SpectralNormalization(layers.Wrapper):
    """
    Spectral normalization for discriminator stability
    Constrains discriminator Lipschitz constant to prevent mode collapse
    """
    
    def __init__(self, layer, iteration=1, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.iteration = iteration
    
    def build(self, input_shape):
        self.layer.build(input_shape)
        
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        
        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name='sn_u',
            dtype=tf.float32
        )
        
        super(SpectralNormalization, self).build()
    
    def call(self, inputs, training=None):
        self.update_weights()
        output = self.layer(inputs)
        return output
    
    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        
        u_hat = self.u
        v_hat = None
        
        for _ in range(self.iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
            v_hat = v_ / (tf.norm(v_) + 1e-12)
            
            u_ = tf.matmul(v_hat, w_reshaped)
            u_hat = u_ / (tf.norm(u_) + 1e-12)
        
        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        
        self.u.assign(u_hat)
        self.layer.kernel.assign(self.w / sigma)


# ============================================================================
# NETWORK ARCHITECTURES
# ============================================================================

def build_2d_generator(input_shape=(138, 176, 1), ga_embedding_dim=16, name='generator'):
    """
    2D U-Net Generator with GA Conditioning
    
    Anti-collapse features:
    - Residual learning with conservative scaling (0.2)
    - Batch normalization in decoder for stable gradients
    - Tanh activation allowing positive AND negative corrections
    - Shape correction for odd dimensions
    """
    
    img_input = layers.Input(shape=input_shape, name='image_input')
    ga_input = layers.Input(shape=(1,), name='ga_input')
    
    # GA embedding with dropout
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_input)
    ga_embedding = layers.Dropout(0.2)(ga_embedding)
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
    
    # Bottleneck with GA injection
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    ga_spatial = layers.RepeatVector(x.shape[1] * x.shape[2])(ga_embedding)
    ga_spatial = layers.Reshape((x.shape[1], x.shape[2], ga_embedding_dim))(ga_spatial)
    x = layers.Concatenate()([x, ga_spatial])
    
    # Decoder with batch normalization AND shape correction
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    if x.shape[1] != skip3.shape[1] or x.shape[2] != skip3.shape[2]:
        x = layers.Resizing(skip3.shape[1], skip3.shape[2])(x)
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    if x.shape[1] != skip2.shape[1] or x.shape[2] != skip2.shape[2]:
        x = layers.Resizing(skip2.shape[1], skip2.shape[2])(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    if x.shape[1] != skip1.shape[1] or x.shape[2] != skip1.shape[2]:
        x = layers.Resizing(skip1.shape[1], skip1.shape[2])(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Final output with shape guarantee
    if x.shape[1] != input_shape[0] or x.shape[2] != input_shape[1]:
        x = layers.Resizing(input_shape[0], input_shape[1])(x)
    
    # Residual connection
    residual = layers.Conv2D(1, 1, padding='same', activation='tanh')(x)
    residual = layers.Lambda(lambda x: x * 0.2)(residual)  # Scale residuals
    output = layers.Add()([img_input, residual])
    output = layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0))(output)
    
    model = Model(inputs=[img_input, ga_input], outputs=output, name=name)
    
    return model

def build_2d_discriminator(input_shape=(138, 176, 1), ga_embedding_dim=16, 
                           use_spectral_norm=True, name='discriminator'):
    """
    2D PatchGAN Discriminator with GA Conditioning
    
    Anti-collapse features:
    - Spectral normalization
    - Higher dropout (0.4)
    - No batch norm (unstable with spectral norm)
    """
    
    img_input = layers.Input(shape=input_shape, name='image_input')
    ga_input = layers.Input(shape=(1,), name='ga_input')
    
    # GA embedding
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_input)
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_embedding)
    
    # Discriminator path with spectral normalization
    conv_layer = layers.Conv2D(64, 4, strides=2, padding='same')
    if use_spectral_norm:
        conv_layer = SpectralNormalization(conv_layer)
    x = conv_layer(img_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.4)(x)
    
    conv_layer = layers.Conv2D(128, 4, strides=2, padding='same')
    if use_spectral_norm:
        conv_layer = SpectralNormalization(conv_layer)
    x = conv_layer(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.4)(x)
    
    conv_layer = layers.Conv2D(256, 4, strides=2, padding='same')
    if use_spectral_norm:
        conv_layer = SpectralNormalization(conv_layer)
    x = conv_layer(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.4)(x)
    
    conv_layer = layers.Conv2D(512, 4, strides=1, padding='same')
    if use_spectral_norm:
        conv_layer = SpectralNormalization(conv_layer)
    x = conv_layer(x)
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
# CHECKPOINT EVALUATION
# ============================================================================

def evaluate_checkpoint(generator, site_data, reference_site, result_dir, epoch, batch_size=4):
    """Comprehensive checkpoint evaluation with multi-level collapse detection"""
    print(f"\n  Evaluating checkpoint (epoch {epoch})...")
    
    eval_dir = result_dir / f'eval_epoch_{epoch}'
    eval_dir.mkdir(exist_ok=True, parents=True)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    collapse_detected = False
    stats_summary = []
    
    for site_name, site_dict in site_data.items():
        if site_name == reference_site or site_dict['n_slices'] < batch_size:
            continue
        
        indices = np.random.choice(site_dict['n_slices'], min(batch_size, 8), replace=False)
        samples_img = site_dict['images'][indices]
        samples_ga = site_dict['ga'][indices]
        
        samples_ga_tf = tf.expand_dims(samples_ga, axis=-1)
        harmonized = generator.predict([samples_img, samples_ga_tf], verbose=0)
        
        # Comprehensive statistics
        orig_std = np.std(samples_img)
        harm_std = np.std(harmonized)
        orig_mean = np.mean(samples_img)
        harm_mean = np.mean(harmonized)
        mean_diff = np.mean(np.abs(samples_img - harmonized))
        max_diff = np.max(np.abs(samples_img - harmonized))
        
        per_image_stds = [np.std(harmonized[i]) for i in range(len(harmonized))]
        min_per_image_std = np.min(per_image_stds)
        
        # Multi-level collapse detection
        is_collapsed = False
        collapse_reasons = []
        
        if harm_std < 0.01:
            is_collapsed = True
            collapse_reasons.append(f"near-zero std ({harm_std:.4f})")
        
        if min_per_image_std < 0.005:
            is_collapsed = True
            collapse_reasons.append(f"min per-image std ({min_per_image_std:.4f})")
        
        if harm_mean < 0.05 or harm_mean > 0.95:
            is_collapsed = True
            collapse_reasons.append(f"extreme mean ({harm_mean:.3f})")
        
        if mean_diff < 0.001 and max_diff < 0.01:
            is_collapsed = True
            collapse_reasons.append("identity mapping")
        
        if np.max(np.abs(harmonized)) < 0.1:
            is_collapsed = True
            collapse_reasons.append("near-zero max output")
        
        if np.std([np.mean(harmonized[i]) for i in range(len(harmonized))]) < 0.001:
            is_collapsed = True
            collapse_reasons.append("constant output")
        
        stats_summary.append({
            'site': site_name,
            'orig_std': orig_std,
            'harm_std': harm_std,
            'orig_mean': orig_mean,
            'harm_mean': harm_mean,
            'mean_diff': mean_diff,
            'max_diff': max_diff,
            'min_per_image_std': min_per_image_std,
            'collapsed': is_collapsed,
            'reasons': ', '.join(collapse_reasons) if collapse_reasons else 'none'
        })
        
        status = "COLLAPSED" if is_collapsed else "OK"
        print(f"    {site_name}: std={harm_std:.4f}, mean={harm_mean:.4f}, diff={mean_diff:.4f} [{status}]")
        
        if is_collapsed:
            print(f"      Reasons: {', '.join(collapse_reasons)}")
            collapse_detected = True
        
        # Visualization
        n_show = min(4, len(indices))
        fig, axes = plt.subplots(n_show, 3, figsize=(12, 3*n_show))
        if n_show == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_show):
            orig = samples_img[i, :, :, 0]
            harm = harmonized[i, :, :, 0]
            diff = np.abs(orig - harm)
            
            axes[i, 0].imshow(orig, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title(f'Original (GA:{samples_ga[i]:.1f}w)')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(harm, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(f'Harmonized (std:{np.std(harm):.4f})')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
            axes[i, 2].set_title(f'Diff (max:{np.max(diff):.4f})')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(eval_dir / f'{site_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    stats_df = pd.DataFrame(stats_summary)
    stats_df.to_csv(eval_dir / 'evaluation_stats.csv', index=False)
    
    print(f"  Evaluation complete")
    
    return collapse_detected, stats_df


# ============================================================================
# LOSSES
# ============================================================================

def cycle_consistency_loss(real_img, cycled_img):
    """L1 loss for cycle consistency"""
    return tf.reduce_mean(tf.abs(real_img - cycled_img))


def identity_loss(real_img, same_img):
    """L1 loss for identity mapping"""
    return tf.reduce_mean(tf.abs(real_img - same_img))


def discriminator_loss_smooth(real_output, fake_output, label_smoothing=0.1):
    """Discriminator loss with label smoothing"""
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
# CYCLEGAN MODEL - IGUANE STYLE
# ============================================================================

class CycleGAN2D_MultiSite:
    """
    2D CycleGAN following IGUANe training procedure
    GenFwd accumulates gradients from all sites, updated once per step
    """
    
    def __init__(self, img_shape=(138, 176, 1), ga_embedding_dim=16, target_sites=None, 
                 use_multi_gpu=True, lambda_cycle=30.0, lambda_identity=15.0):
        self.img_shape = img_shape
        self.ga_embedding_dim = ga_embedding_dim
        self.target_sites = target_sites or []
        self.use_multi_gpu = use_multi_gpu
        
        self.gpus = tf.config.list_physical_devices('GPU')
        self.n_gpus = len(self.gpus)
        
        print(f"\nBuilding CycleGAN (IGUANe-style):")
        print(f"  - 1 universal forward generator (site->BCH)")
        print(f"  - {len(self.target_sites)} site-specific backward generators")
        print(f"  - {len(self.target_sites) + 1} discriminators")
        print(f"  - Lambda cycle: {lambda_cycle}")
        print(f"  - Lambda identity: {lambda_identity}")
        
        # GPU assignment
        if self.use_multi_gpu and self.n_gpus >= 2:
            self.gpu_assignments = {
                'generators': '/GPU:0',
                'disc_BCH': '/GPU:1' if self.n_gpus >= 2 else '/GPU:0',
                'disc_sites': '/GPU:2' if self.n_gpus >= 3 else '/GPU:1' if self.n_gpus >= 2 else '/GPU:0'
            }
        else:
            self.gpu_assignments = {
                'generators': '/GPU:0',
                'disc_BCH': '/GPU:0',
                'disc_sites': '/GPU:0'
            }
        
        # Build universal forward generator
        with tf.device(self.gpu_assignments['generators']):
            self.gen_site2BCH = build_2d_generator(img_shape, ga_embedding_dim, name='gen_site2BCH')
        
        # Build site-specific backward generators
        self.gen_BCH2site = {}
        for site in self.target_sites:
            with tf.device(self.gpu_assignments['generators']):
                site_name = site.replace('_', '').replace('-', '')[:20]
                self.gen_BCH2site[site] = build_2d_generator(
                    img_shape, ga_embedding_dim, name=f'gen_BCH2{site_name}'
                )
        
        # Build BCH discriminator
        with tf.device(self.gpu_assignments['disc_BCH']):
            self.disc_BCH = build_2d_discriminator(img_shape, ga_embedding_dim, name='disc_BCH')
        
        # Build site discriminators
        self.disc_sites = {}
        for i, site in enumerate(self.target_sites):
            if self.use_multi_gpu and self.n_gpus >= 3:
                gpu_id = i % self.n_gpus
                device = f'/GPU:{gpu_id}'
            else:
                device = self.gpu_assignments['disc_sites']
            
            with tf.device(device):
                site_name = site.replace('_', '').replace('-', '')[:20]
                self.disc_sites[site] = build_2d_discriminator(
                    img_shape, ga_embedding_dim, name=f'disc_{site_name}'
                )
        
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.collapse_counter = 0
        
    def compile(self, lr_gen=0.0002, lr_disc=0.0001, beta_1=0.5):
        """Compile with separate optimizers for each discriminator"""
        
        self.gen_optimizer = Adam(learning_rate=lr_gen, beta_1=beta_1)
        
        self.disc_BCH_optimizer = Adam(learning_rate=lr_disc, beta_1=beta_1)
        
        self.disc_site_optimizers = {}
        for site in self.target_sites:
            self.disc_site_optimizers[site] = Adam(learning_rate=lr_disc, beta_1=beta_1)
        
        print("\nModel compiled:")
        print(f"  Gen site->BCH parameters: {self.gen_site2BCH.count_params():,}")
        print(f"  Gen BCH->site parameters: {list(self.gen_BCH2site.values())[0].count_params():,} x {len(self.target_sites)}")
        print(f"  Disc BCH parameters: {self.disc_BCH.count_params():,}")
        if self.target_sites:
            print(f"  Disc per-site parameters: {list(self.disc_sites.values())[0].count_params():,} x {len(self.target_sites)}")
        print(f"  Generator LR: {lr_gen}")
        print(f"  Discriminator LR: {lr_disc}")
        print(f"  Created {1 + len(self.disc_site_optimizers)} discriminator optimizers")
    
    def train_step(self, site_batches):
        """
        IGUANe-style training with gradient accumulation
        
        Per step:
        1. Iterate through sites in random order
        2. Update discriminators for each site
        3. Accumulate GenFwd gradients, update GenBwd_i immediately
        4. After all sites: Update GenFwd once with accumulated gradients
        """
        
        # Shuffle sites for random order
        sites = list(self.target_sites)
        np.random.shuffle(sites)
        
        # Accumulator for universal forward generator
        accumulated_gen_fwd_grads = None
        n_sites_processed = 0
        
        # Track losses
        total_gen_loss = 0.0
        total_disc_BCH_loss = 0.0
        total_disc_site_losses = {}
        total_cycle_loss = 0.0
        total_identity_loss = 0.0
        
        # Process each site
        for site_name in sites:
            if site_name not in site_batches or 'BCH_CHD' not in site_batches:
                continue
            
            real_site, ga_site = site_batches[site_name]
            real_BCH, ga_BCH = site_batches['BCH_CHD']
            disc_site = self.disc_sites[site_name]
            gen_BCH2site_i = self.gen_BCH2site[site_name]
            
            # Generate fake images
            fake_BCH = self.gen_site2BCH([real_site, ga_site], training=False)
            fake_site = gen_BCH2site_i([real_BCH, ga_BCH], training=False)
            
            # Update BCH discriminator
            with tf.device(self.gpu_assignments['disc_BCH']):
                with tf.GradientTape() as disc_BCH_tape:
                    disc_real_BCH = self.disc_BCH([real_BCH, ga_BCH], training=True)
                    disc_fake_BCH = self.disc_BCH([fake_BCH, ga_site], training=True)
                    disc_BCH_loss = discriminator_loss_smooth(disc_real_BCH, disc_fake_BCH)
                
                disc_BCH_grads = disc_BCH_tape.gradient(disc_BCH_loss, self.disc_BCH.trainable_variables)
                disc_BCH_grads, _ = tf.clip_by_global_norm(disc_BCH_grads, 5.0)
                self.disc_BCH_optimizer.apply_gradients(zip(disc_BCH_grads, self.disc_BCH.trainable_variables))
            
            # Update site discriminator
            with tf.GradientTape() as disc_site_tape:
                disc_real_site = disc_site([real_site, ga_site], training=True)
                disc_fake_site = disc_site([fake_site, ga_BCH], training=True)
                disc_site_loss = discriminator_loss_smooth(disc_real_site, disc_fake_site)
            
            disc_site_grads = disc_site_tape.gradient(disc_site_loss, disc_site.trainable_variables)
            disc_site_grads, _ = tf.clip_by_global_norm(disc_site_grads, 5.0)
            self.disc_site_optimizers[site_name].apply_gradients(zip(disc_site_grads, disc_site.trainable_variables))
            
            # Compute generator gradients
            with tf.device(self.gpu_assignments['generators']):
                with tf.GradientTape(persistent=True) as gen_tape:
                    # Forward cycle
                    fake_BCH = self.gen_site2BCH([real_site, ga_site], training=True)
                    cycled_site = gen_BCH2site_i([fake_BCH, ga_site], training=True)
                    
                    # Backward cycle
                    fake_site = gen_BCH2site_i([real_BCH, ga_BCH], training=True)
                    cycled_BCH = self.gen_site2BCH([fake_site, ga_BCH], training=True)
                    
                    # Identity
                    identity_BCH = gen_BCH2site_i([real_BCH, ga_BCH], training=True)
                    identity_site = self.gen_site2BCH([real_site, ga_site], training=True)
                    
                    # Discriminator predictions
                    disc_fake_BCH = self.disc_BCH([fake_BCH, ga_site], training=False)
                    disc_fake_site = disc_site([fake_site, ga_BCH], training=False)
                    
                    # Losses
                    gen_site2BCH_loss = generator_loss(disc_fake_BCH)
                    gen_BCH2site_loss = generator_loss(disc_fake_site)
                    
                    cycle_loss_forward = cycle_consistency_loss(real_site, cycled_site)
                    cycle_loss_backward = cycle_consistency_loss(real_BCH, cycled_BCH)
                    cycle_loss_total = cycle_loss_forward + cycle_loss_backward
                    
                    identity_loss_BCH = identity_loss(real_BCH, identity_BCH)
                    identity_loss_site = identity_loss(real_site, identity_site)
                    identity_loss_total = identity_loss_BCH + identity_loss_site
                    
                    gen_loss = (gen_site2BCH_loss + gen_BCH2site_loss + 
                               self.lambda_cycle * cycle_loss_total + 
                               self.lambda_identity * identity_loss_total)
                
                # Accumulate GenFwd gradients
                gen_fwd_vars = self.gen_site2BCH.trainable_variables
                gen_fwd_grads = gen_tape.gradient(gen_loss, gen_fwd_vars)
                gen_fwd_grads, _ = tf.clip_by_global_norm(gen_fwd_grads, 5.0)
                
                if accumulated_gen_fwd_grads is None:
                    accumulated_gen_fwd_grads = gen_fwd_grads
                else:
                    accumulated_gen_fwd_grads = [
                        ag + g for ag, g in zip(accumulated_gen_fwd_grads, gen_fwd_grads)
                    ]
                
                # Update GenBwd_i immediately
                gen_bwd_vars = gen_BCH2site_i.trainable_variables
                gen_bwd_grads = gen_tape.gradient(gen_loss, gen_bwd_vars)
                gen_bwd_grads, _ = tf.clip_by_global_norm(gen_bwd_grads, 5.0)
                self.gen_optimizer.apply_gradients(zip(gen_bwd_grads, gen_bwd_vars))
                
                del gen_tape
            
            # Accumulate losses
            total_gen_loss += gen_loss.numpy()
            total_disc_BCH_loss += disc_BCH_loss.numpy()
            total_disc_site_losses[site_name] = disc_site_loss.numpy()
            total_cycle_loss += cycle_loss_total.numpy()
            total_identity_loss += identity_loss_total.numpy()
            n_sites_processed += 1
            
            # Clean up
            del fake_BCH, cycled_site, fake_site, cycled_BCH
            del identity_BCH, identity_site
        
        # Update universal forward generator
        if n_sites_processed > 0 and accumulated_gen_fwd_grads is not None:
            # Average accumulated gradients
            accumulated_gen_fwd_grads = [g / n_sites_processed for g in accumulated_gen_fwd_grads]
            
            # Apply once per training step
            with tf.device(self.gpu_assignments['generators']):
                self.gen_optimizer.apply_gradients(
                    zip(accumulated_gen_fwd_grads, self.gen_site2BCH.trainable_variables)
                )
            
            # Average losses
            total_gen_loss /= n_sites_processed
            total_disc_BCH_loss /= n_sites_processed
            total_cycle_loss /= n_sites_processed
            total_identity_loss /= n_sites_processed
        
        # Collapse detection
        if total_disc_BCH_loss < 0.01:
            self.collapse_counter += 1
        else:
            self.collapse_counter = max(0, self.collapse_counter - 1)
        
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
            image = tf.image.random_brightness(image, max_delta=0.05)
            image = tf.clip_by_value(image, 0.0, 1.0)
        
        if tf.random.uniform(()) > 0.5:
            image = tf.image.random_contrast(image, lower=0.95, upper=1.05)
            image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, ga


def create_tf_dataset(images, ga, batch_size=16, shuffle=True, augment=False):
    """Create TensorFlow dataset"""
    dataset = tf.data.Dataset.from_tensor_slices((images, ga))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, len(images)))
    
    if augment:
        dataset = dataset.map(DataAugmenter.augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args):
    """Main training loop"""
    
    gpus = configure_gpu(args.gpu, memory_growth=True)
    
    weight_dir = Path(args.weight_dir)
    result_dir = Path(args.result_dir)
    log_dir = Path(args.log_dir)
    
    weight_dir.mkdir(exist_ok=True, parents=True)
    result_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    train_images, train_ga, train_sex, train_site = load_preprocessed_data(args.train_data)
    
    # Create site datasets
    train_site_data, ref_site = create_site_datasets(
        train_images, train_ga, train_sex, train_site, args.reference_site
    )
    
    target_sites = [s for s in train_site_data.keys() if s != ref_site]
    
    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)
    
    # Build CycleGAN
    cyclegan = CycleGAN2D_MultiSite(
        img_shape=(138, 176, 1),
        ga_embedding_dim=args.ga_embedding_dim,
        target_sites=target_sites,
        use_multi_gpu=(args.multi_gpu_strategy != 'single'),
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity
    )
    
    cyclegan.compile(lr_gen=args.lr_gen, lr_disc=args.lr_disc, beta_1=args.beta_1)
    
    # Create datasets
    print("\n" + "="*80)
    print("CREATING DATASETS")
    print("="*80)
    
    site_datasets = {}
    for site_name, site_data in train_site_data.items():
        dataset = create_tf_dataset(
            site_data['images'],
            site_data['ga'],
            batch_size=args.batch_size,
            shuffle=True,
            augment=True
        )
        site_datasets[site_name] = dataset
        print(f"  {site_name}: {site_data['n_slices']} slices")
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Generator LR: {args.lr_gen}")
    print(f"Discriminator LR: {args.lr_disc}")
    print(f"Lambda cycle: {args.lambda_cycle}")
    print(f"Lambda identity: {args.lambda_identity}")
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
        
        steps_per_epoch = max([
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
                
                # NaN detection
                if any(np.isnan(v) if not isinstance(v, int) else False for v in losses.values()):
                    print(f"\n  NaN detected, skipping step {step}")
                    continue
                
                # Collapse warning
                if losses.get('collapse_warning', 0) > 30:
                    print(f"\n  WARNING: Persistent discriminator saturation")
                
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
        
        # Memory cleanup
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        
        # Print summary
        print(f"\n  Gen: {history['gen_loss'][-1]:.4f} | "
              f"Disc BCH: {history['disc_BCH_loss'][-1]:.4f} | "
              f"Cycle: {history['cycle_loss'][-1]:.4f} | "
              f"Identity: {history['identity_loss'][-1]:.4f}")
        
        # GPU monitoring
        if (epoch + 1) % 10 == 0:
            print_gpu_usage()
        
        # Early stopping
        if history['disc_BCH_loss'][-1] < 0.001 and epoch > 10:
            print("\n  CRITICAL: Discriminator collapse detected")
            print("  Training stopped early")
            break
        
        # Save checkpoints
        if (epoch + 1) % args.save_freq == 0:
            print(f"\n  Saving checkpoint at epoch {epoch+1}")
            cyclegan.gen_site2BCH.save_weights(
                weight_dir / f'gen_site2BCH_epoch_{epoch+1}.weights.h5'
            )
            for site_name, gen_bwd in cyclegan.gen_BCH2site.items():
                safe_name = site_name.replace('_', '').replace('-', '')[:20]
                gen_bwd.save_weights(
                    weight_dir / f'gen_BCH2{safe_name}_epoch_{epoch+1}.weights.h5'
                )
            cyclegan.disc_BCH.save_weights(
                weight_dir / f'disc_BCH_epoch_{epoch+1}.weights.h5'
            )
            for site_name, disc in cyclegan.disc_sites.items():
                safe_name = site_name.replace('_', '').replace('-', '')[:20]
                disc.save_weights(
                    weight_dir / f'disc_{safe_name}_epoch_{epoch+1}.weights.h5'
                )
            
            # Evaluate
            collapse_detected, stats_df = evaluate_checkpoint(
                cyclegan.gen_site2BCH, train_site_data, ref_site, result_dir, epoch+1
            )
            
            if collapse_detected:
                print(f"\n  WARNING: Collapse detected at epoch {epoch+1}")
    
    # Save final models
    print("\n  Saving final models...")
    cyclegan.gen_site2BCH.save_weights(weight_dir / 'gen_site2BCH_final.weights.h5')
    for site_name, gen_bwd in cyclegan.gen_BCH2site.items():
        safe_name = site_name.replace('_', '').replace('-', '')[:20]
        gen_bwd.save_weights(weight_dir / f'gen_BCH2{safe_name}_final.weights.h5')
    cyclegan.disc_BCH.save_weights(weight_dir / 'disc_BCH_final.weights.h5')
    for site_name, disc in cyclegan.disc_sites.items():
        safe_name = site_name.replace('_', '').replace('-', '')[:20]
        disc.save_weights(weight_dir / f'disc_{safe_name}_final.weights.h5')
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(log_dir / 'training_history.csv', index=False)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    # Quality report
    print("\nTRAINING QUALITY REPORT:")
    if len(history['disc_BCH_loss']) > 0:
        final_disc_loss = history['disc_BCH_loss'][-1]
        if final_disc_loss < 0.01:
            print("  Discriminator: Collapsed")
        elif final_disc_loss < 0.3:
            print("  Discriminator: Weak")
        elif final_disc_loss > 2.0:
            print("  Discriminator: Too strong")
        else:
            print("  Discriminator: Healthy")
        
        final_gen_loss = history['gen_loss'][-1]
        if final_gen_loss > 50:
            print("  Generator: Struggling")
        elif final_gen_loss < 5:
            print("  Generator: Possibly collapsed")
        else:
            print("  Generator: Learning")
        
        final_cycle = history['cycle_loss'][-1]
        if final_cycle < 0.2:
            print("  Cycle consistency: Excellent")
        elif final_cycle < 1.0:
            print("  Cycle consistency: Good")
        else:
            print("  Cycle consistency: Weak")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train 2D CycleGAN for fetal brain harmonization (IGUANe-style)'
    )
    
    # Data
    parser.add_argument('--train_data', default='processed_data_4slice/train_4slice_data.pkl')
    parser.add_argument('--val_data', default='processed_data_4slice/val_4slice_data.pkl')
    parser.add_argument('--reference_site', default='BCH_CHD')
    
    # Model
    parser.add_argument('--ga_embedding_dim', type=int, default=16)
    
    # Training 
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_gen', type=float, default=0.0002, 
                       help='Generator learning rate')
    parser.add_argument('--lr_disc', type=float, default=0.0001, 
                       help='Discriminator learning rate')
    parser.add_argument('--beta_1', type=float, default=0.5)
    parser.add_argument('--lambda_cycle', type=float, default=30.0, 
                       help='Cycle consistency loss weight')
    parser.add_argument('--lambda_identity', type=float, default=15.0, 
                       help='Identity loss weight')
    
    # Output
    parser.add_argument('--weight_dir', default='./weights/cyclegan_2d')
    parser.add_argument('--result_dir', default='./results/cyclegan_2d')
    parser.add_argument('--log_dir', default='./logs/cyclegan_2d')
    parser.add_argument('--save_freq', type=int, default=25)
    
    # Hardware
    parser.add_argument('--gpu', default='0,1,2')
    parser.add_argument('--multi_gpu_strategy', default='model_parallel', 
                        choices=['model_parallel', 'single'])
    
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