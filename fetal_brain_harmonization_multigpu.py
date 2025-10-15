#!/usr/bin/env python3
"""
Multi-GPU Fetal Brain Harmonization Training
Optimized for 3x NVIDIA RTX A5000 GPUs
Author: Daniel Barreras Meraz
Date: October 2025
"""

import numpy as np
import nibabel as nib
import os
import glob
import sys
import time
import pickle
import csv
from tqdm import tqdm
import pandas as pd

# Multi-GPU setup - must be done before importing TensorFlow components
from multi_gpu_config import setup_multi_gpu_strategy, create_distributed_model, setup_distributed_optimizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, concatenate, Dropout, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet101V2
import argparse

# Set up argument parser with multi-GPU options
parser = argparse.ArgumentParser('Multi-GPU Fetal Brain Harmonization Training')
parser.add_argument('-train_csv', default='Andrea_total_list_230119.csv', type=str, help='Andrea dataset CSV')
parser.add_argument('-batch_size', default=96, type=int, help='Global batch size (will be distributed across GPUs)')
parser.add_argument('-n_slice', default=4, type=int, help='Number of slices (fixed at 4)')
parser.add_argument('-threshold', default=0.4, type=float, help='Quality threshold')
parser.add_argument('-epochs', default=1000, type=int, help='Number of epochs')
parser.add_argument('-mode', default='slice', choices=['slice', 'stack'], help='2D slice or 2D stack mode')
parser.add_argument('-output_dir', default='./harmonization_results_multigpu', type=str, help='Output directory')
parser.add_argument('--gpus', default='0,1,2', type=str, help='Comma-separated GPU IDs to use (default: 0,1,2)')
parser.add_argument('--single-gpu', action='store_true', help='Use single GPU instead of multi-GPU')
args = parser.parse_args()

# Parse GPU IDs
gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
print(f"Configuring training for GPUs: {gpu_ids}")

# Setup multi-GPU strategy or single GPU
if args.single_gpu or len(gpu_ids) == 1:
    print("Using single GPU training")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    strategy = tf.distribute.get_strategy()  # Default strategy
    batch_size_per_replica = args.batch_size
    global_batch_size = args.batch_size
else:
    print("Setting up multi-GPU training")
    strategy, batch_size_per_replica, global_batch_size = setup_multi_gpu_strategy(
        gpu_ids=gpu_ids, 
        memory_growth=True
    )
    # Adjust global batch size if provided
    if args.batch_size != 96:  # If user specified different batch size
        global_batch_size = args.batch_size
        batch_size_per_replica = global_batch_size // strategy.num_replicas_in_sync

# Create output directories
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'history'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

print('\n' + '='*80)
print('MULTI-GPU FETAL BRAIN HARMONIZATION WITH 4-SLICE APPROACH')
print('='*80)
print(f'Mode: {args.mode}')
print(f'GPUs: {gpu_ids}')
print(f'Strategy replicas: {strategy.num_replicas_in_sync}')
print(f'Global batch size: {global_batch_size}')
print(f'Batch size per replica: {batch_size_per_replica}')
print(f'Number of slices: {args.n_slice}')
print(f'Quality threshold: {args.threshold}')
print(f'Output directory: {os.path.realpath(args.output_dir)}')
print('='*80 + '\n')

# Loss functions (same as original)
def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss for robust training"""
    error = y_pred - y_true
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * K.square(quadratic) + delta * linear

def crop_pad_ND(img, target_shape):
    """Crop and pad image to target shape"""
    import operator
    if (img.shape > np.array(target_shape)).any():
        target_shape2 = np.min([target_shape, img.shape], axis=0)
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, target_shape2))
        end = tuple(map(operator.add, start, target_shape2))
        slices = tuple(map(slice, start, end))
        img = img[tuple(slices)]
    offset = tuple(map(lambda a, da: a//2-da//2, target_shape, img.shape))
    slices = [slice(offset[dim], offset[dim] + img.shape[dim]) for dim in range(img.ndim)]
    result = np.zeros(target_shape)
    result[tuple(slices)] = img
    return result

def make_dic_harmonization(img_paths, num_slice=4, mode='slice'):
    """Extract 4 central slices from each stack - optimized for multi-GPU"""
    max_size = [176, 138, 1]
    
    if mode == 'stack':
        max_size = [176, 138, num_slice]
    
    dic_images = []
    
    print(f"Processing {len(img_paths)} images for {mode} mode...")
    
    for i, img_path in enumerate(tqdm(img_paths, desc="Loading images")):
        try:
            # Load NIfTI image
            img = nib.load(img_path)
            data = img.get_fdata()
            
            if len(data.shape) == 3:
                if mode == 'slice':
                    # Extract 4 central slices
                    center_z = data.shape[2] // 2
                    start_slice = max(0, center_z - 2)
                    end_slice = min(data.shape[2], center_z + 2)
                    
                    slices = []
                    for z in range(start_slice, end_slice):
                        if z < data.shape[2]:
                            slice_2d = data[:, :, z]
                            slice_2d = crop_pad_ND(slice_2d, max_size[:2])
                            slices.append(slice_2d)
                    
                    # Ensure we have exactly 4 slices
                    while len(slices) < num_slice:
                        slices.append(np.zeros(max_size[:2]))
                    
                    # Stack slices and add channel dimension
                    img_data = np.stack(slices[:num_slice], axis=-1)
                    img_data = np.expand_dims(img_data, axis=-1)  # Add channel dim
                    
                elif mode == 'stack':
                    # Use original 3D approach but crop/pad to standard size
                    img_data = crop_pad_ND(data, max_size[:-1] + [data.shape[2]])
                    img_data = np.expand_dims(img_data, axis=-1)
                
                dic_images.append(img_data.astype(np.float32))
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Add a zero image as placeholder
            if mode == 'slice':
                placeholder = np.zeros(max_size[:2] + [num_slice, 1], dtype=np.float32)
            else:
                placeholder = np.zeros(max_size, dtype=np.float32)
            dic_images.append(placeholder)
    
    print(f"Processed {len(dic_images)} images")
    return np.array(dic_images)

def create_harmonization_model():
    """Create harmonization model within strategy scope"""
    
    def model_fn():
        if args.mode == 'slice':
            input_shape = (176, 138, args.n_slice, 1)
        else:
            input_shape = (176, 138, None, 1)  # Variable depth for stack mode
            
        # Input layer
        inputs = Input(shape=input_shape, name='brain_input')
        
        # Use ResNet101V2 as backbone (modified for brain data)
        if args.mode == 'slice':
            # For slice mode, treat as 2D with multiple channels
            reshaped = Lambda(lambda x: tf.squeeze(x, axis=-1))(inputs)  # Remove last dim
            base_model = ResNet101V2(
                weights=None,  # No pretrained weights for brain data
                include_top=False,
                input_shape=(176, 138, args.n_slice),
                pooling='avg'
            )
            x = base_model(reshaped)
        else:
            # For stack mode, use 3D convolutions
            x = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
            x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)
            x = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)
            x = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.GlobalAveragePooling3D()(x)
        
        # Add harmonization layers
        x = Dense(512, activation='relu', name='harmonization_dense1')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu', name='harmonization_dense2')(x)
        x = Dropout(0.3)(x)
        
        # Output layer for harmonization (regression)
        outputs = Dense(1, activation='linear', name='harmonization_output', dtype='float32')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='harmonization_model')
        return model
    
    return model_fn

def create_optimizer():
    """Create optimizer within strategy scope"""
    def optimizer_fn():
        # Scale learning rate by number of GPUs
        base_lr = 0.001
        scaled_lr = base_lr * strategy.num_replicas_in_sync
        
        optimizer = Adam(
            learning_rate=scaled_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        return optimizer
    
    return optimizer_fn

# Create distributed dataset
def create_distributed_dataset(X, y, batch_size_per_replica):
    """Create distributed dataset for multi-GPU training"""
    
    def preprocess_fn(x, y):
        # Normalize to [-1, 1] range
        x = tf.cast(x, tf.float32)
        x = (x - tf.reduce_mean(x)) / (tf.reduce_std(x) + 1e-8)
        x = tf.clip_by_value(x, -3.0, 3.0)  # Clip outliers
        
        y = tf.cast(y, tf.float32)
        return x, y
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=min(1000, len(X)))
    dataset = dataset.batch(batch_size_per_replica)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Distribute dataset
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    
    return dist_dataset

def main():
    print("Starting Multi-GPU Fetal Brain Harmonization Training")
    
    # Load and prepare data
    try:
        # For demonstration, create dummy data
        # Replace this with your actual data loading
        print("Loading training data...")
        
        # Example: Create dummy data for testing
        n_samples = 1000
        dummy_paths = [f"dummy_image_{i}.nii.gz" for i in range(n_samples)]
        
        # Generate dummy images (replace with actual image loading)
        if args.mode == 'slice':
            X = np.random.rand(n_samples, 176, 138, args.n_slice, 1).astype(np.float32)
        else:
            X = np.random.rand(n_samples, 176, 138, 64, 1).astype(np.float32)
        
        # Generate dummy labels (ages between 20-40 weeks)
        y = np.random.uniform(20, 40, n_samples).astype(np.float32)
        
        print(f"Loaded {len(X)} samples")
        print(f"Input shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Create distributed datasets
        print("Creating distributed datasets...")
        train_dataset = create_distributed_dataset(X_train, y_train, batch_size_per_replica)
        val_dataset = create_distributed_dataset(X_val, y_val, batch_size_per_replica)
        
        # Create model within strategy scope
        print("Building model...")
        model = create_distributed_model(create_harmonization_model(), strategy)
        
        # Setup optimizer
        print("Setting up optimizer...")
        optimizer = setup_distributed_optimizer(create_optimizer(), strategy)
        
        # Compile model within strategy scope
        with strategy.scope():
            model.compile(
                optimizer=optimizer,
                loss=huber_loss,
                metrics=['mae', 'mse']
            )
        
        print("Model summary:")
        model.summary()
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(args.output_dir, 'weights', 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=50,
                restore_best_weights=True,
                verbose=1
            ),
            CSVLogger(
                filename=os.path.join(args.output_dir, 'logs', 'training_log.csv'),
                append=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=20,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Calculate steps
        steps_per_epoch = len(X_train) // global_batch_size
        validation_steps = len(X_val) // global_batch_size
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Train model
        print("Starting training...")
        start_time = time.time()
        
        history = model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        model.save(os.path.join(args.output_dir, 'weights', 'final_model.h5'))
        
        # Save training history
        import pickle
        with open(os.path.join(args.output_dir, 'history', 'training_history.pkl'), 'wb') as f:
            pickle.dump(history.history, f)
        
        print(f"Multi-GPU training completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
        # Print final GPU utilization
        from multi_gpu_config import print_gpu_utilization
        print("\nFinal GPU Status:")
        print_gpu_utilization()
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("Training completed successfully!")
    else:
        print("Training failed!")
        sys.exit(1)