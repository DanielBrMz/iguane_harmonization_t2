#!/usr/bin/env python3
"""
Normalize axial dataset - same process as sagittal
"""

import pickle
import numpy as np
from pathlib import Path

print("="*80)
print("NORMALIZING AXIAL DATASET")
print("="*80)

# Load axial data
input_file = 'processed_data_4slice_fixed/train_4slice_data_labeled_no_dHCP.pkl_axial_only.pkl'
print(f"\nLoading: {input_file}")

with open(input_file, 'rb') as f:
    data = pickle.load(f)

print(f"\nOriginal data keys: {data.keys()}")
print(f"Images shape: {data['images'].shape}")
print(f"Sites: {np.unique(data['site'], return_counts=True)}")

# Normalize images to [0, 1]
print("\nNormalizing images to [0, 1]...")
images = data['images'].astype(np.float32)

# Per-image normalization
for i in range(len(images)):
    img = images[i, :, :, 0]
    img_min = img.min()
    img_max = img.max()
    
    if img_max > img_min:
        images[i, :, :, 0] = (img - img_min) / (img_max - img_min)
    else:
        images[i, :, :, 0] = 0.0

print(f"Normalized range: [{images.min():.3f}, {images.max():.3f}]")

# Create normalized dataset
normalized_data = {
    'images': images,
    'gestational_age': data['ga'].astype(np.float32),  # 'ga' -> 'gestational_age'
    'site': data['site'],
}

if 'subject_id' in data:
    normalized_data['subject_id'] = data['subject_id']
if 'sex' in data:
    normalized_data['sex'] = data['sex']

print("\nNormalized data structure:")
for key, val in normalized_data.items():
    if isinstance(val, np.ndarray):
        print(f"  {key}: {val.shape} {val.dtype}")
    else:
        print(f"  {key}: {type(val)}")

# Verify gestational age range
print(f"\nGestational age range: [{normalized_data['gestational_age'].min():.1f}, {normalized_data['gestational_age'].max():.1f}] weeks")

# Save
output_file = 'processed_data_4slice_fixed/train_axial_only_normalized.pkl'
print(f"\nSaving to: {output_file}")

with open(output_file, 'wb') as f:
    pickle.dump(normalized_data, f)

print("\n" + "="*80)
print("✓ NORMALIZATION COMPLETE")
print("="*80)
print(f"\nNormalized axial dataset saved: {output_file}")
print(f"Total images: {len(normalized_data['images'])}")
print("\nSites:")
sites, counts = np.unique(normalized_data['site'], return_counts=True)
for site, count in zip(sites, counts):
    print(f"  {site}: {count}")
print("\nReady for training!")