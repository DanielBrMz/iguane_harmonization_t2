"""
Prepare fetal brain data with 4-slice approach for 2D CycleGAN harmonization
Following the brain age paper methodology (Hong et al., 2021)
"""

import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
import argparse

def crop_pad_ND(img, target_shape):
    """
    Crop and pad image to target shape
    From brain age prediction code
    """
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


def extract_4_central_slices(img_path, target_size=[176, 138]):
    """
    Extract 4 central slices from a 3D volume
    Following brain age paper: 4 slices covering ~2cm anatomically
    
    Returns: array of shape [4, 138, 176, 1] ready for 2D training
    """
    try:
        img = nib.load(str(img_path)).get_fdata()
        img = np.squeeze(img)
        
        if img.ndim != 3:
            print(f"  Warning: Image has {img.ndim} dimensions, expected 3")
            return None
        
        # Crop/pad to consistent in-plane size
        max_size = target_size + [img.shape[2]]
        img = crop_pad_ND(img, max_size)
        
        # Extract 4 central slices
        z_center = img.shape[2] // 2
        slice_start = z_center - 2
        slice_end = z_center + 2
        
        if slice_start < 0 or slice_end > img.shape[2]:
            print(f"  Warning: Not enough slices in z-direction")
            return None
        
        # Get slices [z_center-2, z_center-1, z_center, z_center+1]
        central_slices = img[:, :, slice_start:slice_end]
        
        # Transpose to [4, 138, 176] and add channel dimension
        slices_2d = np.transpose(central_slices, (2, 1, 0))  # [4, 138, 176]
        slices_2d = np.expand_dims(slices_2d, axis=-1)  # [4, 138, 176, 1]
        
        # Normalize to uint8 range [0, 255] for memory efficiency
        if slices_2d.max() > 0:
            slices_2d = (slices_2d / slices_2d.max() * 255).astype(np.uint8)
        else:
            slices_2d = slices_2d.astype(np.uint8)
        
        return slices_2d
        
    except Exception as e:
        print(f"  Error processing {img_path}: {e}")
        return None


def prepare_dataset_from_csv(csv_path, output_dir, dataset_name):
    """
    Prepare dataset with 4-slice approach
    CSV columns: ID, MR (path), GA (gestational age), Sex, Site, Quality
    """
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {dataset_name}: {len(df)} stacks from {df['ID'].nunique()} subjects")
    print(f"Sites: {df['Site'].value_counts().to_dict()}")
    
    all_slices = []
    all_ga = []
    all_sex = []
    all_site = []
    all_subject_id = []
    all_stack_id = []
    
    successful = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {dataset_name}"):
        img_path = Path(row['MR'])
        
        if not img_path.exists():
            print(f"  File not found: {img_path}")
            failed += 1
            continue
            
        # Extract 4 central slices
        slices = extract_4_central_slices(img_path)
        
        if slices is not None:
            all_slices.append(slices)
            
            # Replicate metadata for each of the 4 slices
            all_ga.extend([row['GA']] * 4)
            all_sex.extend([row['Sex']] * 4)
            all_site.extend([row['Site']] * 4)
            all_subject_id.extend([row['ID']] * 4)
            all_stack_id.extend([f"{row['ID']}_{idx}"] * 4)
            
            successful += 1
    
    # Convert to numpy arrays
    if len(all_slices) > 0:
        all_slices = np.concatenate(all_slices, axis=0)  # [N*4, 138, 176, 1]
        all_ga = np.array(all_ga)
        all_sex = np.array(all_sex)
        all_site = np.array(all_site)
        all_subject_id = np.array(all_subject_id)
        all_stack_id = np.array(all_stack_id)
        
        # Save processed data
        data_dict = {
            'images': all_slices,
            'gestational_age': all_ga,
            'sex': all_sex,
            'site': all_site,
            'subject_id': all_subject_id,
            'stack_id': all_stack_id,
            'normalization': '0-255 uint8'
        }
        
        output_file = output_dir / f"{dataset_name}_4slice_data.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(data_dict, f, protocol=4)
        
        print(f"\n{dataset_name} Summary:")
        print(f"  Successful: {successful} stacks")
        print(f"  Failed: {failed} stacks")
        print(f"  Total slices: {len(all_slices)}")
        print(f"  Unique subjects: {len(np.unique(all_subject_id))}")
        print(f"  Data shape: {all_slices.shape}")
        print(f"  Memory: {all_slices.nbytes / 1024**2:.1f} MB")
        print(f"  Saved to: {output_file}")
        
        return data_dict
    else:
        print(f"No valid data found for {dataset_name}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Prepare fetal brain data with 4-slice approach'
    )
    parser.add_argument(
        '--train_csv', 
        default='train_fetal_4slice.csv',
        help='Training CSV file'
    )
    parser.add_argument(
        '--val_csv', 
        default='val_fetal_4slice.csv',
        help='Validation CSV file'
    )
    parser.add_argument(
        '--test_csv',
        default='test_fetal_4slice.csv',
        help='Test CSV file'
    )
    parser.add_argument(
        '--output_dir', 
        default='./processed_data_4slice',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("FETAL BRAIN 4-SLICE DATA PREPARATION")
    print("="*80)
    print(f"Target size: 138 x 176 x 4 slices")
    print(f"Normalization: uint8 [0-255]")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    # Process training data
    print("\n" + "="*80)
    print("TRAINING DATA")
    print("="*80)
    train_data = prepare_dataset_from_csv(
        args.train_csv, 
        args.output_dir, 
        'train'
    )
    
    # Process validation data
    print("\n" + "="*80)
    print("VALIDATION DATA")
    print("="*80)
    val_data = prepare_dataset_from_csv(
        args.val_csv, 
        args.output_dir, 
        'val'
    )
    
    # Process test data
    print("\n" + "="*80)
    print("TEST DATA")
    print("="*80)
    test_data = prepare_dataset_from_csv(
        args.test_csv,
        args.output_dir,
        'test'
    )
    
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE!")
    print("="*80)
    
    if train_data:
        print(f"\nTraining: {len(train_data['images'])} slices")
        print(f"  Sites: {np.unique(train_data['site'], return_counts=True)}")
    if val_data:
        print(f"\nValidation: {len(val_data['images'])} slices")
        print(f"  Sites: {np.unique(val_data['site'], return_counts=True)}")
    if test_data:
        print(f"\nTest: {len(test_data['images'])} slices")
        print(f"  Sites: {np.unique(test_data['site'], return_counts=True)}")


if __name__ == "__main__":
    main()