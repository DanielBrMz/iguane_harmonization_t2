#!/usr/bin/env python3
"""
Normalize dataset by view (sagittal, axial, or coronal)
"""

import pickle
import numpy as np
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Normalize fetal brain MRI dataset by view',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--view',
        type=str,
        choices=['sagittal', 'axial', 'coronal'],
        required=True,
        help='Brain MRI view to normalize (sagittal, axial, or coronal)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input pickle file (default: processed_data_4slice_fixed/train_4slice_data_labeled_no_dHCP.pkl_{view}_only.pkl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output pickle file (default: processed_data_4slice_fixed/train_{view}_only_normalized.pkl)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"NORMALIZING {args.view.upper()} DATASET")
    print("="*80)
    
    # Determine input file
    if args.input:
        input_file = args.input
    else:
        input_file = f'processed_data_4slice_fixed/train_4slice_data_labeled_no_dHCP.pkl_{args.view}_only.pkl'
    
    print(f"\nLoading: {input_file}")
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"ERROR: Input file not found: {input_file}")
        return
    
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
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = f'processed_data_4slice_fixed/train_{args.view}_only_normalized.pkl'
    
    print(f"\nSaving to: {output_file}")
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(normalized_data, f)
    
    print("\n" + "="*80)
    print("✓ NORMALIZATION COMPLETE")
    print("="*80)
    print(f"\nNormalized {args.view} dataset saved: {output_file}")
    print(f"Total images: {len(normalized_data['images'])}")
    print("\nSites:")
    sites, counts = np.unique(normalized_data['site'], return_counts=True)
    for site, count in zip(sites, counts):
        print(f"  {site}: {count}")
    print("\nReady for training!")

if __name__ == '__main__':
    main()