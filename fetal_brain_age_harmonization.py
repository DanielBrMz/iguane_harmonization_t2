#!/usr/bin/env python3
"""
Fetal Brain Harmonization with Age Conditioning
Based on brain age prediction model with 4 central slices
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

# Set up argument parser
parser = argparse.ArgumentParser('Fetal Brain Harmonization Training - 4 Slice Approach')
parser.add_argument('-train_csv', default='Andrea_total_list_230119.csv', type=str, help='Andrea dataset CSV')
parser.add_argument('-batch_size', default=128, type=int, help='Batch size')
parser.add_argument('-n_slice', default=4, type=int, help='Number of slices (fixed at 4)')
parser.add_argument('-threshold', default=0.4, type=float, help='Quality threshold')
parser.add_argument('-gpu', default='0', type=str, help='GPU selection')
parser.add_argument('-epochs', default=1000, type=int, help='Number of epochs')
parser.add_argument('-mode', default='slice', choices=['slice', 'stack'], help='2D slice or 2D stack mode')
parser.add_argument('-output_dir', default='./harmonization_results', type=str, help='Output directory')
args = parser.parse_args()

# GPU configuration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Create output directories
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'history'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

print('\n' + '='*80)
print('FETAL BRAIN HARMONIZATION WITH 4-SLICE APPROACH')
print('='*80)
print(f'Mode: {args.mode}')
print(f'Batch size: {args.batch_size}')
print(f'Number of slices: {args.n_slice}')
print(f'Quality threshold: {args.threshold}')
print(f'Output directory: {os.path.realpath(args.output_dir)}')
print('='*80 + '\n')

# Functions from original brain age code
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
    """Extract 4 central slices from each stack"""
    max_size = [176, 138, 1]  # Standard size from brain age model
    
    if mode == 'stack':
        # Stack mode: 4 slices as channels
        dic = np.zeros([len(img_paths), max_size[1], max_size[0], num_slice], dtype=np.float32)
    else:
        # Slice mode: individual slices
        dic = np.zeros([len(img_paths)*num_slice, max_size[1], max_size[0], 1], dtype=np.float32)
    
    for i, img_path in enumerate(tqdm(img_paths, desc='Loading images')):
        try:
            img = np.squeeze(nib.load(img_path).get_fdata())
            img = crop_pad_ND(img, np.max(np.vstack((max_size, img.shape)), axis=0))
            
            # Normalize intensity
            img = (img - np.mean(img)) / (np.std(img) + 1e-8)
            
            # Extract central slices
            center = img.shape[-1] // 2
            start_idx = center - num_slice // 2
            end_idx = center + num_slice // 2
            
            if mode == 'stack':
                dic[i, :, :, :] = np.swapaxes(img[:, :, start_idx:end_idx], 0, 1)
            else:
                slices = np.swapaxes(img[:, :, start_idx:end_idx], 0, 2)
                dic[i*num_slice:(i+1)*num_slice, :, :, 0] = slices
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    return dic

def encode_site(site_name):
    """Encode site information for conditioning"""
    site_mapping = {
        'BCH_CHD': 0,
        'BCH_Placenta': 1, 
        'dHCP': 2,
        'HBCD': 3,
        'TMC': 4,
        'VGH': 5
    }
    return site_mapping.get(site_name, 0)

def create_harmonization_model(img_shape, num_sites=6):
    """Create harmonization model with site conditioning"""
    # Base ResNet101V2 model
    base_model = ResNet101V2(input_shape=img_shape, include_top=False, weights=None, pooling='avg')
    
    # Site conditioning input (one-hot encoded)
    site_input = Input(shape=(num_sites,), name='site_input')
    
    # GA conditioning input
    ga_input = Input(shape=(1,), name='ga_input')
    
    # Sex conditioning input (binary)
    sex_input = Input(shape=(1,), name='sex_input')
    
    # Concatenate all inputs with image features
    concat_layer = concatenate([
        base_model.layers[-1].output,
        site_input,
        ga_input,
        sex_input
    ])
    
    # Harmonization layers
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(concat_layer)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.3)(x)
    
    # Output: harmonized age prediction
    age_output = Dense(1, activation='linear', name='age_output')(x)
    
    # Site discriminator branch (adversarial)
    site_pred = Dense(num_sites, activation='softmax', name='site_output')(x)
    
    # Combined model
    model = Model(
        inputs=[base_model.layers[0].input, site_input, ga_input, sex_input],
        outputs=[age_output, site_pred]
    )
    
    return model

# Load and prepare data
print("Loading Andrea's dataset...")
df = pd.read_csv(args.train_csv)

# Parse dataset and determine sites
def get_site_from_path(mr_path):
    """Determine site from MR path"""
    if 'CHD' in mr_path:
        return 'BCH_CHD'
    elif 'Placenta' in mr_path or 'Normative' in mr_path:
        return 'BCH_Placenta'
    elif 'dHCP' in mr_path:
        return 'dHCP'
    elif 'HBCD' in mr_path:
        return 'HBCD'
    elif 'TMC' in mr_path:
        return 'TMC'
    elif 'VGH' in mr_path or 'TVGH' in mr_path:
        return 'VGH'
    else:
        return 'Unknown'

# Add site information
df['Site'] = df['MR'].apply(get_site_from_path)

# Split data according to Hyeokjin's instructions
train_sites = ['BCH_CHD', 'BCH_Placenta', 'dHCP', 'HBCD']
test_sites = ['TMC', 'VGH']

train_df = df[df['Site'].isin(train_sites)]
test_df = df[df['Site'].isin(test_sites)]

# Further split training into train/validation (90/10)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    train_df, 
    test_size=0.1, 
    stratify=train_df['Site'],
    random_state=42
)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# Prepare data dictionaries
print("Preparing training data...")
train_images = make_dic_harmonization(train_df['MR'].values, args.n_slice, args.mode)
val_images = make_dic_harmonization(val_df['MR'].values, args.n_slice, args.mode)

# Prepare labels
if args.mode == 'slice':
    # Expand labels for each slice
    train_ga = np.repeat(train_df['GA'].values, args.n_slice)
    train_sex = np.repeat(train_df['Sex'].values, args.n_slice)
    train_sites = np.repeat(train_df['Site'].apply(encode_site).values, args.n_slice)
    
    val_ga = np.repeat(val_df['GA'].values, args.n_slice)
    val_sex = np.repeat(val_df['Sex'].values, args.n_slice)
    val_sites = np.repeat(val_df['Site'].apply(encode_site).values, args.n_slice)
else:
    train_ga = train_df['GA'].values
    train_sex = train_df['Sex'].values
    train_sites = train_df['Site'].apply(encode_site).values
    
    val_ga = val_df['GA'].values
    val_sex = val_df['Sex'].values
    val_sites = val_df['Site'].apply(encode_site).values

# One-hot encode sites
from tensorflow.keras.utils import to_categorical
train_sites_oh = to_categorical(train_sites, num_classes=6)
val_sites_oh = to_categorical(val_sites, num_classes=6)

# Create data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    vertical_flip=False
)

# Build model
print("Building model...")
if args.mode == 'stack':
    input_shape = (138, 176, 4)
else:
    input_shape = (138, 176, 1)

model = create_harmonization_model(input_shape)

# Compile with multiple losses
model.compile(
    optimizer=Adam(learning_rate=0.001, decay=0.0001),
    loss={
        'age_output': huber_loss,
        'site_output': 'categorical_crossentropy'
    },
    loss_weights={
        'age_output': 1.0,
        'site_output': -0.5  # Negative for adversarial training
    },
    metrics={
        'age_output': 'mae',
        'site_output': 'accuracy'
    }
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_age_output_mae', patience=50, restore_best_weights=True),
    ModelCheckpoint(
        filepath=os.path.join(args.output_dir, 'weights', 'best_model.h5'),
        monitor='val_age_output_mae',
        save_best_only=True,
        save_weights_only=True
    ),
    CSVLogger(os.path.join(args.output_dir, 'logs', 'training_log.csv'))
]

# Custom training loop for adversarial training
class AdversarialTraining(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        # Flip site loss sign for adversarial training
        if 'site_output_loss' in logs:
            logs['site_output_loss'] *= -1

callbacks.append(AdversarialTraining())

# Training
print("Starting training...")
history = model.fit(
    x=[train_images, train_sites_oh, train_ga.reshape(-1, 1), train_sex.reshape(-1, 1)],
    y={
        'age_output': train_ga,
        'site_output': train_sites_oh
    },
    validation_data=(
        [val_images, val_sites_oh, val_ga.reshape(-1, 1), val_sex.reshape(-1, 1)],
        {
            'age_output': val_ga,
            'site_output': val_sites_oh
        }
    ),
    batch_size=args.batch_size,
    epochs=args.epochs,
    callbacks=callbacks,
    verbose=1
)

# Save history
with open(os.path.join(args.output_dir, 'history', 'training_history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

print(f"\nTraining complete! Results saved to {args.output_dir}")