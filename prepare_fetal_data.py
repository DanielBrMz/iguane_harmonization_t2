#!/usr/bin/env python3
"""
Prepare fetal brain data for harmonization training
Handles the specific requirements for 4-slice extraction and site organization
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
import json

# Base paths
BASE_DATA_PATH = Path('/neuro/labs/grantlab/users/mri.team/fetal_mri/Data')
ANDREA_CSV = 'Andrea_total_list_230119.csv'

def validate_and_prepare_dataset():
    """Validate paths and prepare dataset with proper site assignments"""
    
    # Load Andrea's CSV
    df = pd.read_csv(ANDREA_CSV)
    print(f"Loaded {len(df)} subjects from Andrea's list")
    
    # Define site mappings based on paths
    site_mappings = {
        'CHD_protocol': 'BCH_CHD',
        'Normative': 'BCH_Normative', 
        'Placenta_protocol': 'BCH_Placenta',
        'dHCP': 'dHCP',
        'HBCD': 'HBCD',
        'TMC_data': 'TMC',
        'VGH_data': 'VGH',
        'TVGH': 'VGH'
    }
    
    # Process each entry
    processed_data = []
    missing_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating paths"):
        mr_path = row['MR']
        
        # Determine site from path
        site = 'Unknown'
        for key, value in site_mappings.items():
            if key in mr_path:
                site = value
                break
        
        # Check if file exists
        if not os.path.exists(mr_path):
            # Try to construct full path
            for dataset_key in site_mappings.keys():
                potential_path = BASE_DATA_PATH / dataset_key / 'Data' / os.path.basename(mr_path)
                if potential_path.exists():
                    mr_path = str(potential_path)
                    break
        
        if os.path.exists(mr_path):
            # Load and check dimensions
            try:
                img = nib.load(mr_path)
                shape = img.shape
                
                # Check if we have enough slices
                if len(shape) >= 3 and shape[2] >= 4:
                    processed_data.append({
                        'ID': row['ID'],
                        'PID': row['PID'],
                        'MR_path': mr_path,
                        'Study': row['Study'],
                        'GA': row['GA'],
                        'GA_round': row['GA_round'],
                        'Sex': row['Sex'],
                        'Site': site,
                        'Shape': shape,
                        'Num_slices': shape[2] if len(shape) >= 3 else 0
                    })
                else:
                    print(f"Skipping {mr_path}: insufficient slices ({shape})")
            except Exception as e:
                print(f"Error loading {mr_path}: {e}")
        else:
            missing_files.append(mr_path)
    
    # Create processed dataframe
    processed_df = pd.DataFrame(processed_data)
    
    # Print statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"Total valid subjects: {len(processed_df)}")
    print(f"Missing files: {len(missing_files)}")
    print("\nSite distribution:")
    print(processed_df['Site'].value_counts())
    print("\nGA distribution:")
    print(processed_df['GA'].describe())
    print("\nSex distribution:")
    print(processed_df['Sex'].value_counts())
    
    # Save processed data
    processed_df.to_csv('processed_fetal_data.csv', index=False)
    print(f"\nSaved processed data to processed_fetal_data.csv")
    
    # Save configuration for training
    config = {
        'train_sites': ['BCH_CHD', 'BCH_Normative', 'BCH_Placenta', 'dHCP', 'HBCD'],
        'test_sites': ['TMC', 'VGH'],
        'num_slices': 4,
        'input_shape': [138, 176, 1],
        'quality_threshold': 0.4,
        'total_subjects': len(processed_df),
        'site_counts': processed_df['Site'].value_counts().to_dict()
    }
    
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved training configuration to training_config.json")
    
    return processed_df

if __name__ == "__main__":
    validate_and_prepare_dataset()