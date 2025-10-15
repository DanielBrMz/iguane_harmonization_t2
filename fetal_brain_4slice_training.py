#!/usr/bin/env python3
"""
Fetal Brain Harmonization with 4-Slice Approach
"""

import numpy as np
import nibabel as nib
import os
import sys
import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Set TF logging before import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
import argparse

print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Parse arguments
parser = argparse.ArgumentParser('Fetal Brain Harmonization - 4 Slice Implementation')
parser.add_argument('-train_csv', default='Andrea_total_list_230119.csv', type=str, help='Dataset CSV')
parser.add_argument('-batch_size', default=128, type=int, help='Batch size')
parser.add_argument('-n_slice', default=4, type=int, help='Number of slices (fixed at 4 per Hyeokjin)')
parser.add_argument('-threshold', default=0.4, type=float, help='IQA threshold')
parser.add_argument('-d_huber', default=1.0, type=float, help='Delta for Huber loss')
parser.add_argument('-gpu', default='0', type=str, help='GPU selection')
parser.add_argument('-rl', '--result_save_location', default='./result_harmonization', type=str)
parser.add_argument('-wl', '--weight_save_location', default='./weight_harmonization', type=str)
parser.add_argument('-hl', '--history_save_location', default='./hist_harmonization', type=str)
parser.add_argument('-output', default='harmonization_4slice', type=str, help='Output name')
parser.add_argument('-epochs', default=1000, type=int, help='Number of epochs')
args = parser.parse_args()

# Create directories
result_loc = args.result_save_location
weight_loc = args.weight_save_location
hist_loc = args.history_save_location

for loc in [result_loc, weight_loc, hist_loc]:
    os.makedirs(loc, exist_ok=True)

print('\n' + '='*80)
print('FETAL BRAIN HARMONIZATION WITH 4-SLICE APPROACH')
print('='*80)
print(f'Result save location: {os.path.realpath(result_loc)}')
print(f'Weight save location: {os.path.realpath(weight_loc)}')
print(f'History save location: {os.path.realpath(hist_loc)}')
print(f'Number of training slices: {args.n_slice}')
print(f'Batch size: {args.batch_size}')
print(f'Delta of Huber loss: {args.d_huber}')
print(f'GPU number: {args.gpu}')
print('='*80 + '\n')

# GPU configuration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Set random seeds
tf.random.set_seed(1234)
np.random.seed(1234)

# Functions from original brain age code
def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss function"""
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

def make_dic(img_list, num_slice, slice_mode=0, desc=''):
    """Extract central slices from stacks - exactly as in brain age code"""
    max_size = [176, 138, 1]
    if slice_mode:
        dic = np.zeros([len(img_list), max_size[1], max_size[0], num_slice], dtype=np.uint8)
    else:
        dic = np.zeros([len(img_list)*num_slice, max_size[1], max_size[0], 1], dtype=np.uint8)
    
    for i in tqdm(range(len(img_list)), desc=desc):
        try:
            # Check if path exists
            img_path = img_list[i]
            if not os.path.exists(img_path):
                # Try to construct full path
                base_path = Path('/neuro/labs/grantlab/users/mri.team/fetal_mri/Data')
                possible_paths = [
                    base_path / 'CHD_protocol' / 'Data' / Path(img_path).name,
                    base_path / 'Normative' / 'Data' / Path(img_path).name,
                    base_path / 'dHCP' / 'Seungyoon' / Path(img_path).name,
                    img_path  # Try original path
                ]
                for p in possible_paths:
                    if p.exists():
                        img_path = str(p)
                        break
            
            img = np.squeeze(nib.load(img_path).get_fdata())
            orig_img = img
            img = crop_pad_ND(img, np.max(np.vstack((max_size, img.shape)), axis=0))
            
            # Extract central slices
            center_slice = img.shape[-1] // 2
            start = center_slice - num_slice // 2
            end = center_slice + num_slice // 2
            
            if slice_mode:
                dic[i,:,:,:] = np.swapaxes(img[:,:,start:end], 0, 1)
            else:
                dic[i*num_slice:(i+1)*num_slice,:,:,0] = np.swapaxes(img[:,:,start:end], 0, 2)
        except Exception as e:
            print(f"Error loading {img_list[i]}: {e}")
            continue
    return dic

def age_predic_network(img_shape):
    """Original brain age network from Hyeokjin's code"""
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.applications import ResNet101V2
    
    model = ResNet101V2(input_shape=img_shape, include_top=False, weights=None, pooling='avg')
    o = Dropout(0.4)(model.layers[-1].output)
    o = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.01))(o)
    model = Model(model.layers[0].output, o)
    model.compile(
        optimizer=Adam(learning_rate=0.05, decay=0.001),
        loss=huber_loss,
        metrics=['mae']
    )
    return model

def conditioned_brain_age_model(img_shape, num_sites=6):
    """Modified model with site conditioning for harmonization"""
    from tensorflow.keras.layers import Dense, Input, concatenate, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.applications import ResNet101V2
    
    # Base ResNet model
    base_model = ResNet101V2(input_shape=img_shape, include_top=False, weights=None, pooling='avg')
    
    # Conditioning inputs
    cov_input = Input(shape=(2,), name='sex_input')  # Sex as binary
    site_input = Input(shape=(num_sites,), name='site_input')  # One-hot encoded sites
    
    # Concatenate features
    concat_layer = concatenate([base_model.layers[-1].output, cov_input, site_input])
    o = Dropout(0.4)(concat_layer)
    o = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.01))(o)
    
    brain_age_model = Model([base_model.layers[0].input, cov_input, site_input], o)
    brain_age_model.compile(
        optimizer=Adam(learning_rate=0.05, decay=0.001),
        loss=huber_loss,
        metrics=['mae']
    )
    return brain_age_model

# Load Andrea's dataset
print("Loading dataset...")
df = pd.read_csv(args.train_csv)
print(f"Loaded {len(df)} subjects")

# Determine site from path
def get_site_label(path):
    if 'CHD' in path:
        return 0  # BCH CHD
    elif 'Normative' in path or 'Placenta' in path:
        return 1  # BCH Normative/Placenta  
    elif 'dHCP' in path:
        return 2  # dHCP
    elif 'TMC' in path:
        return 3  # TMC
    elif 'VGH' in path or 'TVGH' in path:
        return 4  # VGH
    elif 'HBCD' in path:
        return 5  # HBCD
    else:
        return 0  # Default to BCH

df['Site'] = df['MR'].apply(get_site_label)

# Split data according to Hyeokjin's instructions
# Training: BCH, dHCP, HBCD
# Testing: TMC, VGH (held out)
train_sites = [0, 1, 2, 5]  # BCH variants, dHCP, HBCD
test_sites = [3, 4]  # TMC, VGH

train_df = df[df['Site'].isin(train_sites)].copy()
valid_df = df[df['Site'].isin(test_sites)].copy()

# For validation, use 10% of training data
from sklearn.model_selection import train_test_split
train_df, val_split_df = train_test_split(train_df, test_size=0.1, stratify=train_df['Site'], random_state=42)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_split_df)}")
print(f"Test samples (held out): {len(valid_df)}")

# Create data dictionaries using the exact make_dic function
num_slice = args.n_slice
train_dic = make_dic(train_df['MR'].values, num_slice, slice_mode=0, desc='Loading training data')
valid_dic = make_dic(val_split_df['MR'].values, num_slice, slice_mode=0, desc='Loading validation data')

# Prepare labels - expand for each slice
train_GW = train_df['GA'].values
train_sex = train_df['Sex'].values
train_sites = train_df['Site'].values

valid_GW = val_split_df['GA'].values
valid_sex = val_split_df['Sex'].values
valid_sites = val_split_df['Site'].values

# Expand labels for slices
b_train_GW = np.repeat(train_GW, num_slice)
b_train_sex = np.repeat(train_sex, num_slice)
b_train_sites = np.repeat(train_sites, num_slice)

b_valid_GW = np.repeat(valid_GW, num_slice)
b_valid_sex = np.repeat(valid_sex, num_slice)
b_valid_sites = np.repeat(valid_sites, num_slice)

# Convert sex to binary (0/1)
b_train_cov = np.zeros([len(b_train_sex), 2])
b_train_cov[:, 0] = (b_train_sex == 0).astype(int)  # Female
b_train_cov[:, 1] = (b_train_sex == 1).astype(int)  # Male

b_valid_cov = np.zeros([len(b_valid_sex), 2])
b_valid_cov[:, 0] = (b_valid_sex == 0).astype(int)
b_valid_cov[:, 1] = (b_valid_sex == 1).astype(int)

# One-hot encode sites
b_train_sites_oh = to_categorical(b_train_sites, num_classes=6)
b_valid_sites_oh = to_categorical(b_valid_sites, num_classes=6)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.5, 1],
    vertical_flip=True,
    horizontal_flip=True
)

# Choose model type
use_site_conditioning = True  # Set to True for harmonization

if use_site_conditioning:
    print("Building conditioned model with site information...")
    model = conditioned_brain_age_model([138, 176, 1], num_sites=6)
else:
    print("Building standard brain age model...")
    model = age_predic_network([138, 176, 1])

# Callbacks
callbacks = [
    EarlyStopping(monitor='mae', patience=150, verbose=1, mode='min', restore_best_weights=True),
    ModelCheckpoint(
        filepath=os.path.join(weight_loc, 'best_fold.h5'),
        monitor='mae',
        save_best_only=True,
        mode='min',
        save_weights_only=True,
        verbose=0
    ),
    CSVLogger(os.path.join(hist_loc, f'log_{args.output}.csv'), separator=",", append=True)
]

# Custom saver for specific epochs
class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch in [200, 400, 600, 800, 1000]:
            self.model.save_weights(os.path.join(weight_loc, f'weights{epoch}.h5'))

callbacks.append(CustomSaver())

# Training
print("Starting training...")
if use_site_conditioning:
    # Train with site conditioning
    history = model.fit(
        x=[train_dic, b_train_cov, b_train_sites_oh],
        y=b_train_GW,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=([valid_dic, b_valid_cov, b_valid_sites_oh], b_valid_GW),
        callbacks=callbacks,
        verbose=2
    )
else:
    # Standard training
    history = model.fit(
        datagen.flow(train_dic, b_train_GW, batch_size=args.batch_size, shuffle=True),
        steps_per_epoch=len(train_dic) / args.batch_size,
        epochs=args.epochs,
        validation_data=datagen.flow(valid_dic, b_valid_GW, batch_size=args.batch_size, shuffle=True),
        validation_steps=len(valid_dic) / args.batch_size,
        callbacks=callbacks,
        verbose=2
    )

# Save history
with open(os.path.join(hist_loc, 'history_fold.pkl'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print(f"\nTraining complete! Results saved to {result_loc}")