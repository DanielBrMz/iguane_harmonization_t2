# save as: create_training_csvs_from_best_images.py

import pandas as pd
from pathlib import Path
import numpy as np
import re
from collections import defaultdict

BASE_PATH = Path('/neuro/labs/grantlab/users/mri.team/fetal_mri/Data')

DATASETS = {
    'CHD_protocol': {
        'path': 'CHD_protocol/Data',
        'excel': 'CHD_protocol/CHD_updated_2021.12.30.xlsx',
        'site': 'BCH_CHD'
    },
    'Placenta_protocol': {
        'path': 'Placenta_protocol/Data',
        'excel': 'Placenta_protocol/Placenta_list_updated_2023.02.06.xlsx',
        'site': 'BCH_Placenta'
    },
    'HBCD': {
        'path': 'HBCD/Data_from_Hyukjin',
        'excel': None,  # Will need to find demographics
        'site_prefix': 'HBCD'  # Will determine specific site from subject ID
    },
    'TMC_data': {
        'path': 'TMC_data/Data',
        'excel': 'TMC_data/Data/TMC_TD_list.xlsx',
        'site': 'TMC'
    },
    'VGH_data': {
        'path': 'VGH_data/Data',
        'excel': None,
        'site_prefix': 'VGH'  # Will determine Siemens vs GE
    }
}


def find_best_images_dirs(base_path):
    """Find all Best_Images_crop directories"""
    best_img_dirs = []
    
    # Subject ID patterns
    subject_patterns = [
        lambda d: d.isdigit() and len(d) >= 4,  # VGH: 0019, Normative: 0158485
        lambda d: d.startswith(('FCB', 'FNB')),  # CHD/Placenta/Normative
        lambda d: d.startswith('sub-'),  # dHCP
        lambda d: re.match(r'^\d{3}[-_]\d{3}[-_]?\d{1,2}$', d),  # HBCD: 100_002_01
        lambda d: re.match(r'^\d{3}[-_]\d{2,3}$', d),  # HBCD: 200_001, 672_02
        lambda d: d.startswith('BM'),  # TMC
    ]
    
    def is_subject_dir(dirname):
        return any(pattern(dirname) for pattern in subject_patterns)
    
    try:
        immediate_dirs = list(base_path.iterdir())
    except (PermissionError, OSError):
        return best_img_dirs
    
    # Pattern 1: Direct subject directories
    potential_subjects = [d for d in immediate_dirs if d.is_dir() and is_subject_dir(d.name)]
    
    for subj_dir in potential_subjects:
        try:
            best_img_candidate = subj_dir / 'Best_Images_crop'
            if best_img_candidate.exists():
                best_img_dirs.append(best_img_candidate)
                continue
            
            # Pattern 2: Date-stamped subdirectories (Normative)
            for date_dir in subj_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                best_img_candidate = date_dir / 'Best_Images_crop'
                if best_img_candidate.exists():
                    best_img_dirs.append(best_img_candidate)
                    break
        except (PermissionError, OSError):
            continue
    
    # Pattern 3: Nested organization
    if not potential_subjects:
        for subdir in immediate_dirs:
            if not subdir.is_dir():
                continue
            try:
                for subj_dir in subdir.iterdir():
                    if not subj_dir.is_dir():
                        continue
                    best_img_candidate = subj_dir / 'Best_Images_crop'
                    if best_img_candidate.exists():
                        best_img_dirs.append(best_img_candidate)
            except (PermissionError, OSError):
                continue
    
    return best_img_dirs


def extract_subject_id(file_path):
    """Extract subject ID from path"""
    parts = file_path.parts
    
    for i, part in enumerate(parts):
        if part == 'Best_Images_crop' and i > 0:
            candidate = parts[i - 1]
            if re.match(r'^\d{4}\.\d{2}\.\d{2}', candidate) and i > 1:
                candidate = parts[i - 2]
            return candidate
    
    return 'unknown'


def determine_hbcd_site(subject_id):
    """Determine HBCD site from subject ID"""
    if subject_id.startswith('100') or subject_id.startswith('300'):
        return 'HBCD_Site5_Arkansas_UNC'  # Siemens 3T Prisma
    elif subject_id.startswith('200') or subject_id.startswith('672'):
        return 'HBCD_Site6_Cincinnati'  # Philips 3T Ingenia
    else:
        return 'HBCD_Unknown'


def determine_vgh_site(scanner_manufacturer):
    """Determine VGH site from scanner"""
    if scanner_manufacturer == 'Siemens':
        return 'VGH_Site3_Siemens'
    elif scanner_manufacturer == 'GE':
        return 'VGH_Site4_GE'
    else:
        return 'VGH_Unknown'


def extract_scanner_from_filename(filename):
    """Extract scanner info from filename"""
    filename_upper = filename.upper()
    
    manufacturer = None
    if any(kw in filename_upper for kw in ['HASTE', 'T2_HASTE']):
        manufacturer = 'Siemens'
    elif any(kw in filename_upper for kw in ['SSFSE', 'FIESTA']):
        manufacturer = 'GE'
    elif any(kw in filename_upper for kw in ['BFFE', 'TSE']):
        manufacturer = 'Philips'
    
    return manufacturer


def load_quality_scores(qa_file):
    """Load quality assessment scores"""
    if not qa_file.exists():
        return {}
    
    try:
        df = pd.read_csv(qa_file)
        quality_dict = {}
        
        for _, row in df.iterrows():
            # Handle both full paths and filenames
            file_key = Path(str(row.iloc[0])).name
            quality_dict[file_key] = pd.to_numeric(row.iloc[1], errors='coerce')
        
        return quality_dict
    except:
        return {}


def load_demographics(excel_path):
    """Load demographics from Excel file"""
    if not excel_path or not Path(excel_path).exists():
        return {}
    
    try:
        df = pd.read_excel(excel_path)
        demo_dict = {}
        
        # Find ID column
        id_col = None
        for col in df.columns:
            if any(kw in col.upper() for kw in ['ID', 'SUBJECT', 'PATIENT']):
                id_col = col
                break
        
        if not id_col:
            return {}
        
        # Find GA column
        ga_col = None
        for col in df.columns:
            if any(kw in col.upper() for kw in ['GA', 'GESTATIONAL']) and \
               'MATERNAL' not in col.upper():
                ga_col = col
                break
        
        # Find Sex column
        sex_col = None
        for col in df.columns:
            if any(kw in col.upper() for kw in ['SEX', 'GENDER']):
                sex_col = col
                break
        
        for _, row in df.iterrows():
            subject_id = str(row[id_col])
            demo_dict[subject_id] = {
                'GA': pd.to_numeric(row[ga_col], errors='coerce') if ga_col else None,
                'Sex': row[sex_col] if sex_col else None
            }
        
        return demo_dict
    except Exception as e:
        print(f"Error loading demographics from {excel_path}: {e}")
        return {}


def create_dataset_csv(dataset_name, dataset_info):
    """Create CSV for a single dataset"""
    print(f"\nProcessing {dataset_name}...")
    
    base_path = BASE_PATH / dataset_info['path']
    if not base_path.exists():
        print(f"  Path not found: {base_path}")
        return pd.DataFrame()
    
    # Find all Best_Images_crop directories
    best_img_dirs = find_best_images_dirs(base_path)
    print(f"  Found {len(best_img_dirs)} Best_Images_crop directories")
    
    # Load demographics
    demo_dict = {}
    if dataset_info.get('excel'):
        excel_path = BASE_PATH / dataset_info['excel']
        demo_dict = load_demographics(excel_path)
        print(f"  Loaded demographics for {len(demo_dict)} subjects")
    
    # Process each directory
    rows = []
    for best_img_dir in best_img_dirs:
        subject_id = extract_subject_id(best_img_dir)
        
        # Load quality scores
        qa_file = best_img_dir.parent / 'quality_assessment.csv'
        quality_map = load_quality_scores(qa_file)
        
        # Get all .nii files (NOT .nii.gz)
        nii_files = sorted(best_img_dir.glob('*.nii'))
        
        for nii_file in nii_files:
            quality = quality_map.get(nii_file.name, None)
            
            # Filter by quality threshold
            if quality is None or quality > 0.4:
                # Get demographics
                demo = demo_dict.get(subject_id, {})
                ga = demo.get('GA', 25.0)  # Default GA
                sex = demo.get('Sex', 0)  # Default sex
                
                # Convert sex to binary if needed
                if isinstance(sex, str):
                    sex = 1 if sex.upper() in ['M', 'MALE', '1'] else 0
                
                # Determine site
                site = dataset_info.get('site')
                if 'site_prefix' in dataset_info:
                    if dataset_info['site_prefix'] == 'HBCD':
                        site = determine_hbcd_site(subject_id)
                    elif dataset_info['site_prefix'] == 'VGH':
                        scanner = extract_scanner_from_filename(nii_file.name)
                        site = determine_vgh_site(scanner)
                
                rows.append({
                    'ID': subject_id,
                    'MR': str(nii_file.absolute()),
                    'GA': float(ga) if ga is not None else 25.0,
                    'Sex': int(sex) if pd.notna(sex) else 0,
                    'Site': site,
                    'Quality': float(quality) if quality is not None else None,
                    'Dataset': dataset_name
                })
    
    df = pd.DataFrame(rows)
    print(f"  Created {len(df)} stack entries from {len(df['ID'].unique())} subjects")
    
    return df


def main():
    """Create complete training and validation CSVs"""
    print("="*80)
    print("Creating Training CSVs for 4-Slice Approach")
    print("="*80)
    
    all_dfs = []
    
    # Process each dataset
    for dataset_name, dataset_info in DATASETS.items():
        df = create_dataset_csv(dataset_name, dataset_info)
        if not df.empty:
            all_dfs.append(df)
    
    # Combine all datasets
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined: {len(combined_df)} stacks from {len(combined_df['ID'].unique())} subjects")
    
    # Site distribution
    print("\nSite distribution:")
    print(combined_df['Site'].value_counts())
    
    # Split by site for held-out test sets
    # TMC and VGH are held-out test sites (as per email)
    test_sites = ['TMC', 'VGH_Site3_Siemens', 'VGH_Site4_GE']
    
    test_df = combined_df[combined_df['Site'].isin(test_sites)].copy()
    train_val_df = combined_df[~combined_df['Site'].isin(test_sites)].copy()
    
    print(f"\nHeld-out test set: {len(test_df)} stacks from sites {test_sites}")
    print(f"Train+Val set: {len(train_val_df)} stacks")
    
    # Stratified split for train/val
    from sklearn.model_selection import train_test_split
    
    # Group by subject to avoid data leakage
    subject_sites = train_val_df.groupby('ID')['Site'].first()
    subject_gas = train_val_df.groupby('ID')['GA'].first()
    
    # Create stratification labels (site + GA bins)
    ga_bins = pd.cut(subject_gas, bins=5, labels=False)
    strat_labels = [f"{site}_{ga_bin}" for site, ga_bin in zip(subject_sites, ga_bins)]
    
    train_subjects, val_subjects = train_test_split(
        subject_sites.index,
        test_size=0.1,  # 10% validation
        stratify=strat_labels,
        random_state=42
    )
    
    train_df = train_val_df[train_val_df['ID'].isin(train_subjects)].copy()
    val_df = train_val_df[train_val_df['ID'].isin(val_subjects)].copy()
    
    # Save CSVs
    train_df.to_csv('train_fetal_4slice.csv', index=False)
    val_df.to_csv('val_fetal_4slice.csv', index=False)
    test_df.to_csv('test_fetal_4slice.csv', index=False)
    
    print("\n" + "="*80)
    print("FINAL SPLIT SUMMARY")
    print("="*80)
    print(f"Training: {len(train_df)} stacks, {len(train_df['ID'].unique())} subjects")
    print(f"  Site distribution:")
    for site, count in train_df['Site'].value_counts().items():
        print(f"    {site}: {count} stacks")
    
    print(f"\nValidation: {len(val_df)} stacks, {len(val_df['ID'].unique())} subjects")
    print(f"  Site distribution:")
    for site, count in val_df['Site'].value_counts().items():
        print(f"    {site}: {count} stacks")
    
    print(f"\nTest (held-out): {len(test_df)} stacks, {len(test_df['ID'].unique())} subjects")
    print(f"  Site distribution:")
    for site, count in test_df['Site'].value_counts().items():
        print(f"    {site}: {count} stacks")
    
    print("\n" + "="*80)
    print("CSV files created:")
    print("  - train_fetal_4slice.csv")
    print("  - val_fetal_4slice.csv")
    print("  - test_fetal_4slice.csv")
    print("="*80)
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = main()