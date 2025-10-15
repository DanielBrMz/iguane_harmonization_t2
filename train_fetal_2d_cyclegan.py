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
    Following IGUANe architecture adapted for 2D
    Fixed to handle dimension mismatches in skip connections
    """
    
    # Image input
    img_input = layers.Input(shape=input_shape, name='image_input')
    
    # GA conditioning input
    ga_input = layers.Input(shape=(1,), name='ga_input')
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_input)
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu')(ga_embedding)
    
    # Encoder
    # Block 1 (138x176 -> 138x176)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(img_input)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    skip1 = x
    x = layers.MaxPooling2D(2)(x)  # 69x88
    
    # Block 2 (69x88 -> 69x88)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    skip2 = x
    x = layers.MaxPooling2D(2)(x)  # 34x44
    
    # Block 3 (34x44 -> 34x44)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    skip3 = x
    x = layers.MaxPooling2D(2)(x)  # 17x22
    
    # Block 4 (Bottleneck) (17x22 -> 17x22)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    
    # Inject GA conditioning at bottleneck
    # Tile GA embedding to match spatial dimensions
    ga_spatial = layers.RepeatVector(x.shape[1] * x.shape[2])(ga_embedding)
    ga_spatial = layers.Reshape((x.shape[1], x.shape[2], ga_embedding_dim))(ga_spatial)
    x = layers.Concatenate()([x, ga_spatial])
    
    # Decoder
    # Block 5 (17x22 -> 34x44)
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    
    # Crop or pad skip3 to match x spatial dimensions
    if x.shape[1] != skip3.shape[1] or x.shape[2] != skip3.shape[2]:
        # Calculate cropping/padding needed
        height_diff = skip3.shape[1] - x.shape[1]
        width_diff = skip3.shape[2] - x.shape[2]
        
        if height_diff > 0 or width_diff > 0:
            # Crop skip3
            crop_h = height_diff // 2
            crop_w = width_diff // 2
            skip3 = layers.Cropping2D(cropping=((crop_h, height_diff - crop_h), 
                                               (crop_w, width_diff - crop_w)))(skip3)
        elif height_diff < 0 or width_diff < 0:
            # Pad x
            pad_h = abs(height_diff) // 2
            pad_w = abs(width_diff) // 2
            x = layers.ZeroPadding2D(padding=((pad_h, abs(height_diff) - pad_h),
                                             (pad_w, abs(width_diff) - pad_w)))(x)
    
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    
    # Block 6 (34x44 -> 68x88)
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    
    # Crop or pad skip2 to match x spatial dimensions
    if x.shape[1] != skip2.shape[1] or x.shape[2] != skip2.shape[2]:
        height_diff = skip2.shape[1] - x.shape[1]
        width_diff = skip2.shape[2] - x.shape[2]
        
        if height_diff > 0 or width_diff > 0:
            crop_h = height_diff // 2
            crop_w = width_diff // 2
            skip2 = layers.Cropping2D(cropping=((crop_h, height_diff - crop_h),
                                               (crop_w, width_diff - crop_w)))(skip2)
        elif height_diff < 0 or width_diff < 0:
            pad_h = abs(height_diff) // 2
            pad_w = abs(width_diff) // 2
            x = layers.ZeroPadding2D(padding=((pad_h, abs(height_diff) - pad_h),
                                             (pad_w, abs(width_diff) - pad_w)))(x)
    
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    
    # Block 7 (68x88 -> 136x176)
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    
    # Crop or pad skip1 to match x spatial dimensions
    if x.shape[1] != skip1.shape[1] or x.shape[2] != skip1.shape[2]:
        height_diff = skip1.shape[1] - x.shape[1]
        width_diff = skip1.shape[2] - x.shape[2]
        
        if height_diff > 0 or width_diff > 0:
            crop_h = height_diff // 2
            crop_w = width_diff // 2
            skip1 = layers.Cropping2D(cropping=((crop_h, height_diff - crop_h),
                                               (crop_w, width_diff - crop_w)))(skip1)
        elif height_diff < 0 or width_diff < 0:
            pad_h = abs(height_diff) // 2
            pad_w = abs(width_diff) // 2
            x = layers.ZeroPadding2D(padding=((pad_h, abs(height_diff) - pad_h),
                                             (pad_w, abs(width_diff) - pad_w)))(x)
    
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    
    # Final resize to match input exactly (in case of any remaining mismatch)
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

class CycleGAN2D:
    """2D CycleGAN for multi-site harmonization"""
    
    def __init__(self, img_shape=(138, 176, 1), ga_embedding_dim=16):
        self.img_shape = img_shape
        self.ga_embedding_dim = ga_embedding_dim
        
        # Build generators
        self.gen_A2B = build_2d_generator(img_shape, ga_embedding_dim, name='gen_A2B')
        self.gen_B2A = build_2d_generator(img_shape, ga_embedding_dim, name='gen_B2A')
        
        # Build discriminators
        self.disc_A = build_2d_discriminator(img_shape, ga_embedding_dim, name='disc_A')
        self.disc_B = build_2d_discriminator(img_shape, ga_embedding_dim, name='disc_B')
        
        # Loss weights
        self.lambda_cycle = 10.0
        self.lambda_identity = 5.0
        
    def compile(self, lr=0.0002, beta_1=0.5):
        """Compile model with optimizers"""
        self.gen_optimizer = Adam(learning_rate=lr, beta_1=beta_1)
        self.disc_optimizer = Adam(learning_rate=lr, beta_1=beta_1)
        
        print("✓ Model compiled")
    
    @tf.function
    def train_step(self, real_A, real_B, ga_A, ga_B):
        """Single training step"""
        
        with tf.GradientTape(persistent=True) as tape:
            # Forward cycle: A -> B -> A
            fake_B = self.gen_A2B([real_A, ga_A], training=True)
            cycled_A = self.gen_B2A([fake_B, ga_A], training=True)
            
            # Backward cycle: B -> A -> B
            fake_A = self.gen_B2A([real_B, ga_B], training=True)
            cycled_B = self.gen_A2B([fake_A, ga_B], training=True)
            
            # Identity mapping
            same_A = self.gen_B2A([real_A, ga_A], training=True)
            same_B = self.gen_A2B([real_B, ga_B], training=True)
            
            # Discriminator outputs
            disc_real_A = self.disc_A([real_A, ga_A], training=True)
            disc_fake_A = self.disc_A([fake_A, ga_B], training=True)
            
            disc_real_B = self.disc_B([real_B, ga_B], training=True)
            disc_fake_B = self.disc_B([fake_B, ga_A], training=True)
            
            # Generator losses
            gen_A2B_loss = generator_loss(disc_fake_B)
            gen_B2A_loss = generator_loss(disc_fake_A)
            
            # Cycle consistency losses
            cycle_loss_A = cycle_consistency_loss(real_A, cycled_A)
            cycle_loss_B = cycle_consistency_loss(real_B, cycled_B)
            total_cycle_loss = cycle_loss_A + cycle_loss_B
            
            # Identity losses
            identity_loss_A = identity_loss(real_A, same_A)
            identity_loss_B = identity_loss(real_B, same_B)
            total_identity_loss = identity_loss_A + identity_loss_B
            
            # Total generator loss
            total_gen_A2B_loss = (gen_A2B_loss + 
                                  self.lambda_cycle * total_cycle_loss +
                                  self.lambda_identity * total_identity_loss)
            total_gen_B2A_loss = (gen_B2A_loss + 
                                  self.lambda_cycle * total_cycle_loss +
                                  self.lambda_identity * total_identity_loss)
            
            # Discriminator losses
            disc_A_loss = discriminator_loss(disc_real_A, disc_fake_A)
            disc_B_loss = discriminator_loss(disc_real_B, disc_fake_B)
        
        # Calculate gradients
        gen_A2B_gradients = tape.gradient(
            total_gen_A2B_loss, self.gen_A2B.trainable_variables
        )
        gen_B2A_gradients = tape.gradient(
            total_gen_B2A_loss, self.gen_B2A.trainable_variables
        )
        disc_A_gradients = tape.gradient(
            disc_A_loss, self.disc_A.trainable_variables
        )
        disc_B_gradients = tape.gradient(
            disc_B_loss, self.disc_B.trainable_variables
        )
        
        # Apply gradients
        self.gen_optimizer.apply_gradients(
            zip(gen_A2B_gradients, self.gen_A2B.trainable_variables)
        )
        self.gen_optimizer.apply_gradients(
            zip(gen_B2A_gradients, self.gen_B2A.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_A_gradients, self.disc_A.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_B_gradients, self.disc_B.trainable_variables)
        )
        
        del tape
        
        return {
            'gen_A2B_loss': total_gen_A2B_loss,
            'gen_B2A_loss': total_gen_B2A_loss,
            'disc_A_loss': disc_A_loss,
            'disc_B_loss': disc_B_loss,
            'cycle_loss': total_cycle_loss,
            'identity_loss': total_identity_loss
        }


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class DataAugmenter:
    """Data augmentation for 2D slices"""
    
    @staticmethod
    @tf.function
    def augment(image, ga):
        """Apply random augmentations"""
        # Random rotation
        if tf.random.uniform(()) > 0.5:
            k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
            image = tf.image.rot90(image, k=k)
        
        # Random flip
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
    """Main training function"""
    
    # Configure GPU
    configure_gpu(args.gpu, memory_growth=True)
    
    # Create output directories
    weight_dir = Path(args.weight_dir)
    result_dir = Path(args.result_dir)
    log_dir = Path(args.log_dir)
    
    weight_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directories:")
    print(f"  Weights: {weight_dir}")
    print(f"  Results: {result_dir}")
    print(f"  Logs: {log_dir}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    train_images, train_ga, train_sex, train_site = load_preprocessed_data(
        args.train_data
    )
    val_images, val_ga, val_sex, val_site = load_preprocessed_data(
        args.val_data
    )
    
    # Create site datasets
    train_site_data, ref_site = create_site_datasets(
        train_images, train_ga, train_sex, train_site, args.reference_site
    )
    val_site_data, _ = create_site_datasets(
        val_images, val_ga, val_sex, val_site, args.reference_site
    )
    
    print(f"\nUsing {ref_site} as reference site")
    
    # Build model
    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)
    
    cyclegan = CycleGAN2D(img_shape=(138, 176, 1), ga_embedding_dim=args.ga_embedding_dim)
    cyclegan.compile(lr=args.lr, beta_1=args.beta_1)
    
    print(f"\nGenerator A2B parameters: {cyclegan.gen_A2B.count_params():,}")
    print(f"Generator B2A parameters: {cyclegan.gen_B2A.count_params():,}")
    print(f"Discriminator A parameters: {cyclegan.disc_A.count_params():,}")
    print(f"Discriminator B parameters: {cyclegan.disc_B.count_params():,}")
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*80)
    
    # Create datasets for reference site vs other sites
    ref_data = train_site_data[ref_site]
    
    # For simplicity, combine all non-reference sites
    other_images = []
    other_ga = []
    for site_name, site_data in train_site_data.items():
        if site_name != ref_site:
            other_images.append(site_data['images'])
            other_ga.append(site_data['ga'])
    
    other_images = np.concatenate(other_images, axis=0)
    other_ga = np.concatenate(other_ga, axis=0)
    
    print(f"\nTraining split:")
    print(f"  Reference ({ref_site}): {len(ref_data['images'])} slices")
    print(f"  Other sites: {len(other_images)} slices")
    
    # Create TF datasets
    ref_dataset = create_tf_dataset(
        ref_data['images'], ref_data['ga'], 
        args.batch_size, shuffle=True, augment=True
    )
    other_dataset = create_tf_dataset(
        other_images, other_ga,
        args.batch_size, shuffle=True, augment=True
    )
    
    # Training history
    history = {
        'gen_A2B_loss': [],
        'gen_B2A_loss': [],
        'disc_A_loss': [],
        'disc_B_loss': [],
        'cycle_loss': [],
        'identity_loss': []
    }
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        epoch_losses = {k: [] for k in history.keys()}
        
        # Iterate over batches
        for (real_A, ga_A), (real_B, ga_B) in zip(ref_dataset, other_dataset):
            
            losses = cyclegan.train_step(real_A, real_B, ga_A, ga_B)
            
            for k, v in losses.items():
                epoch_losses[k].append(v.numpy())
        
        # Average losses
        for k in history.keys():
            avg_loss = np.mean(epoch_losses[k])
            history[k].append(avg_loss)
        
        # Print progress
        print(f"  Gen A2B: {history['gen_A2B_loss'][-1]:.4f} | "
              f"Gen B2A: {history['gen_B2A_loss'][-1]:.4f} | "
              f"Disc A: {history['disc_A_loss'][-1]:.4f} | "
              f"Disc B: {history['disc_B_loss'][-1]:.4f} | "
              f"Cycle: {history['cycle_loss'][-1]:.4f} | "
              f"Identity: {history['identity_loss'][-1]:.4f}")
        
        # Save checkpoints
        if (epoch + 1) % args.save_freq == 0:
            print(f"  Saving checkpoint at epoch {epoch+1}")
            cyclegan.gen_A2B.save_weights(weight_dir / f'gen_A2B_epoch_{epoch+1}.h5')
            cyclegan.gen_B2A.save_weights(weight_dir / f'gen_B2A_epoch_{epoch+1}.h5')
            cyclegan.disc_A.save_weights(weight_dir / f'disc_A_epoch_{epoch+1}.h5')
            cyclegan.disc_B.save_weights(weight_dir / f'disc_B_epoch_{epoch+1}.h5')
    
    # Save final models
    print("\nSaving final models...")
    cyclegan.gen_A2B.save_weights(weight_dir / 'gen_A2B_final.h5')
    cyclegan.gen_B2A.save_weights(weight_dir / 'gen_B2A_final.h5')
    cyclegan.disc_A.save_weights(weight_dir / 'disc_A_final.h5')
    cyclegan.disc_B.save_weights(weight_dir / 'disc_B_final.h5')
    
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