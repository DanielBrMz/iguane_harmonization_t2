import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path('/neuro/labs/grantlab/users/mri.team/fetal_mri/Data')

# Demographics file paths from email thread
DEMOGRAPHICS_FILES = {
    'CHD': BASE_PATH / 'CHD_protocol/CHD_updated_2021.12.30.xlsx',
    'Placenta': BASE_PATH / 'Placenta_protocol/Placenta_list_updated_2023.02.06.xlsx',
    'TMC': BASE_PATH / 'TMC_data/Data/TMC_TD_list.xlsx',
    'dHCP': Path('/neuro/labs/grantlab/research/MRI_processing/sungmin.you/Study/Brain_feat_Age/Data_analysis_dHCP/dHCP_fetal_t2_summary_BCH.xlsx')
}

def load_demographics_mapping():
    """Load all demographics into a unified mapping"""
    demo_map = {}
    
    print("Loading demographics from ground truth files...")
    
    # CHD demographics
    if DEMOGRAPHICS_FILES['CHD'].exists():
        try:
            df = pd.read_excel(DEMOGRAPHICS_FILES['CHD'])
            print(f"  CHD: {len(df)} records")
            
            # Find columns
            id_col = None
            for col in df.columns:
                if any(kw in col.upper() for kw in ['ID', 'SUBJECT', 'PATIENT']):
                    id_col = col
                    break
            
            ga_col = None
            for col in df.columns:
                if 'GA' in col.upper() and 'MATERNAL' not in col.upper():
                    ga_col = col
                    break
            
            sex_col = None
            for col in df.columns:
                if any(kw in col.upper() for kw in ['SEX', 'GENDER']):
                    sex_col = col
                    break
            
            if id_col and ga_col:
                for _, row in df.iterrows():
                    subj_id = str(row[id_col])
                    ga = pd.to_numeric(row[ga_col], errors='coerce')
                    sex = row[sex_col] if sex_col else None
                    
                    if pd.notna(ga):
                        demo_map[subj_id] = {
                            'GA': float(ga),
                            'Sex': 1 if str(sex).upper() in ['M', 'MALE', '1'] else 0
                        }
                print(f"    Loaded {len([k for k in demo_map.keys() if 'FCB' in k or 'FNB' in k])} CHD demographics")
        except Exception as e:
            print(f"  Error loading CHD demographics: {e}")
    
    # Placenta demographics
    if DEMOGRAPHICS_FILES['Placenta'].exists():
        try:
            df = pd.read_excel(DEMOGRAPHICS_FILES['Placenta'])
            print(f"  Placenta: {len(df)} records")
            
            id_col = None
            for col in df.columns:
                if any(kw in col.upper() for kw in ['ID', 'SUBJECT', 'PATIENT']):
                    id_col = col
                    break
            
            ga_col = None
            for col in df.columns:
                if 'GA' in col.upper() and 'MATERNAL' not in col.upper():
                    ga_col = col
                    break
            
            sex_col = None
            for col in df.columns:
                if any(kw in col.upper() for kw in ['SEX', 'GENDER']):
                    sex_col = col
                    break
            
            if id_col and ga_col:
                for _, row in df.iterrows():
                    subj_id = str(row[id_col])
                    ga = pd.to_numeric(row[ga_col], errors='coerce')
                    sex = row[sex_col] if sex_col else None
                    
                    if pd.notna(ga):
                        demo_map[subj_id] = {
                            'GA': float(ga),
                            'Sex': 1 if str(sex).upper() in ['M', 'MALE', '1'] else 0
                        }
                print(f"    Loaded {len([k for k in demo_map.keys() if k.isdigit()])} Placenta demographics")
        except Exception as e:
            print(f"  Error loading Placenta demographics: {e}")
    
    # TMC demographics
    if DEMOGRAPHICS_FILES['TMC'].exists():
        try:
            df = pd.read_excel(DEMOGRAPHICS_FILES['TMC'])
            print(f"  TMC: {len(df)} records")
            
            id_col = None
            for col in df.columns:
                if any(kw in col.upper() for kw in ['ID', 'SUBJECT', 'PATIENT']):
                    id_col = col
                    break
            
            ga_col = None
            for col in df.columns:
                if 'GA' in col.upper() and 'MATERNAL' not in col.upper():
                    ga_col = col
                    break
            
            sex_col = None
            for col in df.columns:
                if any(kw in col.upper() for kw in ['SEX', 'GENDER']):
                    sex_col = col
                    break
            
            if id_col and ga_col:
                for _, row in df.iterrows():
                    subj_id = str(row[id_col])
                    ga = pd.to_numeric(row[ga_col], errors='coerce')
                    sex = row[sex_col] if sex_col else None
                    
                    if pd.notna(ga):
                        demo_map[subj_id] = {
                            'GA': float(ga),
                            'Sex': 1 if str(sex).upper() in ['M', 'MALE', '1'] else 0
                        }
                print(f"    Loaded {len([k for k in demo_map.keys() if k.startswith('BM')])} TMC demographics")
        except Exception as e:
            print(f"  Error loading TMC demographics: {e}")
    
    # dHCP demographics
    if DEMOGRAPHICS_FILES['dHCP'].exists():
        try:
            df = pd.read_excel(DEMOGRAPHICS_FILES['dHCP'])
            print(f"  dHCP: {len(df)} records")
            
            for _, row in df.iterrows():
                # Extract subject ID from MR path
                mr_path = row.get('MR', '')
                if pd.isna(mr_path):
                    continue
                
                # Parse filename
                filename = mr_path.replace('data_dHCP_test/', '')
                subject_id = filename.split('_ses-')[0].replace('_brain_crop_rsl.nii.gz', '')
                
                # Skip if marked as Poor
                if row.get('Poor', False) or str(row.get('Poor', '')).upper() == 'TRUE':
                    continue
                
                ga = pd.to_numeric(row.get('GA at scan', None), errors='coerce')
                sex = row.get('Gender', None)
                
                if pd.notna(ga):
                    demo_map[subject_id] = {
                        'GA': float(ga),
                        'Sex': 1 if str(sex).upper() == 'MALE' else 0
                    }
            
            print(f"    Loaded {len([k for k in demo_map.keys() if k.startswith('sub-')])} dHCP demographics")
        except Exception as e:
            print(f"  Error loading dHCP demographics: {e}")
    
    print(f"\nTotal demographics loaded: {len(demo_map)}")
    return demo_map


def fix_csv_demographics(csv_path, demo_map):
    """Fix demographics in a CSV file"""
    print(f"\nFixing demographics in: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"  Original data: {len(df)} rows")
    print(f"  GA NaN count: {df['GA'].isna().sum()}")
    print(f"  GA range: {df['GA'].min():.1f} - {df['GA'].max():.1f} (excluding NaN)")
    
    # Fix demographics
    fixed_count = 0
    for idx, row in df.iterrows():
        subject_id = row['ID']
        
        # Check if demographics need fixing
        if pd.isna(row['GA']) or row['GA'] == 25.0:  # Default value
            if subject_id in demo_map:
                df.at[idx, 'GA'] = demo_map[subject_id]['GA']
                df.at[idx, 'Sex'] = demo_map[subject_id]['Sex']
                fixed_count += 1
    
    print(f"  Fixed {fixed_count} rows with ground truth demographics")
    
    # For remaining NaN values, use median imputation
    remaining_nan = df['GA'].isna().sum()
    if remaining_nan > 0:
        print(f"  {remaining_nan} rows still have NaN GA values")
        
        # Calculate median from non-NaN values
        ga_median = df['GA'].median()
        print(f"  Using median GA: {ga_median:.1f} weeks for remaining NaN values")
        
        df['GA'].fillna(ga_median, inplace=True)
        df['Sex'].fillna(0, inplace=True)
    
    print(f"  Final GA range: {df['GA'].min():.1f} - {df['GA'].max():.1f}")
    print(f"  Final GA NaN count: {df['GA'].isna().sum()}")
    
    # Save fixed CSV
    output_path = csv_path.replace('.csv', '_fixed.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return df


def main():
    print("="*80)
    print("FIXING DEMOGRAPHICS IN TRAINING CSVS")
    print("Using ground truth from Seungyoon's files")
    print("="*80)
    
    # Load demographics mapping
    demo_map = load_demographics_mapping()
    
    # Fix each CSV
    for csv_name in ['train_fetal_4slice.csv', 'val_fetal_4slice.csv', 'test_fetal_4slice.csv']:
        if Path(csv_name).exists():
            fix_csv_demographics(csv_name, demo_map)
    
    print("\n" + "="*80)
    print("DONE! Use the *_fixed.csv files for training")
    print("="*80)


if __name__ == "__main__":
    main()