# save as: create_training_csvs_from_best_images.py (UPDATED)

import pandas as pd
from pathlib import Path
import numpy as np
import re
from collections import defaultdict

BASE_PATH = Path('/neuro/labs/grantlab/users/mri.team/fetal_mri/Data')

# dHCP special path from Hyeokjin's email
DHCP_PATH = Path('/neuro/labs/grantlab/research/MRI_processing/sungmin.you/Study/Brain_feat_Age/Data_analysis_dHCP/nuc')
DHCP_CSV = Path('/neuro/labs/grantlab/research/MRI_processing/sungmin.you/Study/Brain_feat_Age/Data_analysis_dHCP/dHCP_fetal_t2_summary_BCH.xlsx')

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
        'excel': None,
        'site_prefix': 'HBCD'
    },
    'TMC_data': {
        'path': 'TMC_data/Data',
        'excel': 'TMC_data/Data/TMC_TD_list.xlsx',
        'site': 'TMC'
    },
    'VGH_data': {
        'path': 'VGH_data/Data',
        'path_alt': 'VGH_data/Data_from_Andrea',  # Try alternative path
        'excel': None,
        'site_prefix': 'VGH'
    }
}


def find_best_images_dirs(base_path):
    """Find all Best_Images_crop directories"""
    best_img_dirs = []
    
    subject_patterns = [
        lambda d: d.isdigit() and len(d) >= 4,
        lambda d: d.startswith(('FCB', 'FNB')),
        lambda d: d.startswith('sub-'),
        lambda d: re.match(r'^\d{3}[-_]\d{3}[-_]?\d{1,2}$', d),
        lambda d: re.match(r'^\d{3}[-_]\d{2,3}$', d),
        lambda d: d.startswith('BM'),
    ]
    
    def is_subject_dir(dirname):
        return any(pattern(dirname) for pattern in subject_patterns)
    
    try:
        immediate_dirs = list(base_path.iterdir())
    except (PermissionError, OSError):
        return best_img_dirs
    
    potential_subjects = [d for d in immediate_dirs if d.is_dir() and is_subject_dir(d.name)]
    
    for subj_dir in potential_subjects:
        try:
            best_img_candidate = subj_dir / 'Best_Images_crop'
            if best_img_candidate.exists():
                best_img_dirs.append(best_img_candidate)
                continue
            
            for date_dir in subj_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                best_img_candidate = date_dir / 'Best_Images_crop'
                if best_img_candidate.exists():
                    best_img_dirs.append(best_img_candidate)
                    break
        except (PermissionError, OSError):
            continue
    
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
    
    # For dHCP files with sub-CC pattern
    for part in parts:
        if part.startswith('sub-'):
            return part.split('_ses-')[0]
    
    return 'unknown'


def determine_hbcd_site(subject_id):
    """Determine HBCD site from subject ID"""
    if subject_id.startswith('100') or subject_id.startswith('300'):
        return 'HBCD_Site5_Arkansas_UNC'
    elif subject_id.startswith('200') or subject_id.startswith('672'):
        return 'HBCD_Site6_Cincinnati'
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
        
        id_col = None
        for col in df.columns:
            if any(kw in col.upper() for kw in ['ID', 'SUBJECT', 'PATIENT', 'MR']):
                id_col = col
                break
        
        if not id_col:
            return {}
        
        ga_col = None
        for col in df.columns:
            if any(kw in col.upper() for kw in ['GA', 'GESTATIONAL', 'AGE']) and \
               'MATERNAL' not in col.upper():
                ga_col = col
                break
        
        sex_col = None
        for col in df.columns:
            if any(kw in col.upper() for kw in ['SEX', 'GENDER']):
                sex_col = col
                break
        
        for _, row in df.iterrows():
            subject_id = str(row[id_col])
            # Extract subject ID from MR path if needed (for dHCP)
            if 'sub-' in subject_id:
                subject_id = Path(subject_id).stem.split('_brain')[0]
            
            demo_dict[subject_id] = {
                'GA': pd.to_numeric(row[ga_col], errors='coerce') if ga_col else None,
                'Sex': row[sex_col] if sex_col else None
            }
        
        return demo_dict
    except Exception as e:
        print(f"Error loading demographics from {excel_path}: {e}")
        return {}


def process_dhcp_dataset():
    """
    Special processing for dHCP dataset following Hyeokjin's instructions
    Uses preprocessed files from best_image_crop folders in nuc directory
    """
    print(f"\nProcessing dHCP (special path)...")
    
    if not DHCP_PATH.exists():
        print(f"  dHCP path not found: {DHCP_PATH}")
        return pd.DataFrame()
    
    # Load demographics CSV with quality info
    demo_dict = {}
    if DHCP_CSV.exists():
        try:
            df = pd.read_excel(DHCP_CSV)
            print(f"  Loaded dHCP demographics: {len(df)} records")
            
            for _, row in df.iterrows():
                # Parse MR filename from CSV
                mr_path = row.get('MR', '')
                if pd.isna(mr_path):
                    continue
                
                # Remove "data_dHCP_test/" and replace "_crop_rsl.nii.gz" with ".nii"
                # Example: data_dHCP_test/sub-CC00969XX21_ses-25531_run-13_T2w_brain_crop_rsl.nii.gz
                # becomes: sub-CC00969XX21_ses-25531_run-13_T2w_brain.nii
                mr_filename = mr_path.replace('data_dHCP_test/', '')
                mr_filename = mr_filename.replace('_crop_rsl.nii.gz', '.nii')
                
                # Extract subject ID
                subject_id = mr_filename.split('_ses-')[0]
                
                # Skip if marked as Poor quality
                if row.get('Poor', False) or str(row.get('Poor', '')).upper() == 'TRUE':
                    continue
                
                demo_dict[mr_filename] = {
                    'subject_id': subject_id,
                    'GA': pd.to_numeric(row.get('GA at scan', None), errors='coerce'),
                    'Sex': 1 if str(row.get('Gender', '')).upper() == 'MALE' else 0
                }
        except Exception as e:
            print(f"  Error loading dHCP demographics: {e}")
    
    # Find all best_image_crop directories
    best_img_dirs = []
    try:
        for subj_dir in DHCP_PATH.iterdir():
            if subj_dir.is_dir() and subj_dir.name.startswith('sub-'):
                best_img_crop = subj_dir / 'best_image_crop'
                if best_img_crop.exists():
                    best_img_dirs.append(best_img_crop)
    except (PermissionError, OSError) as e:
        print(f"  Error scanning dHCP directory: {e}")
        return pd.DataFrame()
    
    print(f"  Found {len(best_img_dirs)} best_image_crop directories")
    
    # Process each directory
    rows = []
    for best_img_dir in best_img_dirs:
        subject_id = best_img_dir.parent.name  # sub-CC00969XX21_ses-25531
        subject_base = subject_id.split('_ses-')[0]  # sub-CC00969XX21
        
        # Get all .nii files
        nii_files = sorted(best_img_dir.glob('*.nii'))
        
        for nii_file in nii_files:
            # Check if this file is in our demographics (not marked as Poor)
            demo = demo_dict.get(nii_file.name, {})
            
            if demo or not demo_dict:  # Include if no quality info or passes quality
                rows.append({
                    'ID': subject_base,
                    'MR': str(nii_file.absolute()),
                    'GA': float(demo.get('GA', 25.0)) if demo else 25.0,
                    'Sex': int(demo.get('Sex', 0)) if demo else 0,
                    'Site': 'dHCP',
                    'Quality': None,  # Quality already filtered in CSV
                    'Dataset': 'dHCP'
                })
    
    df = pd.DataFrame(rows)
    print(f"  Created {len(df)} stack entries from {len(df['ID'].unique())} subjects")
    
    return df


def create_dataset_csv(dataset_name, dataset_info):
    """Create CSV for a single dataset"""
    print(f"\nProcessing {dataset_name}...")
    
    # Try primary path
    base_path = BASE_PATH / dataset_info['path']
    
    # Try alternative path if primary doesn't exist
    if not base_path.exists() and 'path_alt' in dataset_info:
        base_path = BASE_PATH / dataset_info['path_alt']
        print(f"  Using alternative path")
    
    if not base_path.exists():
        print(f"  Path not found: {base_path}")
        return pd.DataFrame()
    
    best_img_dirs = find_best_images_dirs(base_path)
    print(f"  Found {len(best_img_dirs)} Best_Images_crop directories")
    
    demo_dict = {}
    if dataset_info.get('excel'):
        excel_path = BASE_PATH / dataset_info['excel']
        demo_dict = load_demographics(excel_path)
        print(f"  Loaded demographics for {len(demo_dict)} subjects")
    
    rows = []
    for best_img_dir in best_img_dirs:
        subject_id = extract_subject_id(best_img_dir)
        
        qa_file = best_img_dir.parent / 'quality_assessment.csv'
        quality_map = load_quality_scores(qa_file)
        
        nii_files = sorted(best_img_dir.glob('*.nii'))
        
        for nii_file in nii_files:
            quality = quality_map.get(nii_file.name, None)
            
            if quality is None or quality > 0.4:
                demo = demo_dict.get(subject_id, {})
                ga = demo.get('GA', 25.0)
                sex = demo.get('Sex', 0)
                
                if isinstance(sex, str):
                    sex = 1 if sex.upper() in ['M', 'MALE', '1'] else 0
                
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
    print("Creating Training CSVs for 4-Slice Approach (WITH dHCP)")
    print("="*80)
    
    all_dfs = []
    
    # Process regular datasets
    for dataset_name, dataset_info in DATASETS.items():
        df = create_dataset_csv(dataset_name, dataset_info)
        if not df.empty:
            all_dfs.append(df)
    
    # Process dHCP separately (special case)
    dhcp_df = process_dhcp_dataset()
    if not dhcp_df.empty:
        all_dfs.append(dhcp_df)
    
    # Combine all datasets
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined: {len(combined_df)} stacks from {len(combined_df['ID'].unique())} subjects")
    
    # Site distribution
    print("\nSite distribution:")
    print(combined_df['Site'].value_counts())
    
    # Split by site for held-out test sets
    test_sites = ['TMC', 'VGH_Site3_Siemens', 'VGH_Site4_GE']
    
    test_df = combined_df[combined_df['Site'].isin(test_sites)].copy()
    train_val_df = combined_df[~combined_df['Site'].isin(test_sites)].copy()
    
    print(f"\nHeld-out test set: {len(test_df)} stacks from sites {test_sites}")
    print(f"Train+Val set: {len(train_val_df)} stacks")
    
    # Stratified split for train/val
    from sklearn.model_selection import train_test_split
    
    subject_sites = train_val_df.groupby('ID')['Site'].first()
    subject_gas = train_val_df.groupby('ID')['GA'].first()
    
    ga_bins = pd.cut(subject_gas, bins=5, labels=False)
    strat_labels = [f"{site}_{ga_bin}" for site, ga_bin in zip(subject_sites, ga_bins)]
    
    train_subjects, val_subjects = train_test_split(
        subject_sites.index,
        test_size=0.1,
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