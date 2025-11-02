"""
Non-interactive batch view labeling
Generates image sheets for manual review, then reads labels from a CSV
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys


def generate_labeling_sheet(data_path, output_dir='labeling_sheets', images_per_sheet=100):
    """
    Generate sheets of images for manual labeling
    Creates PNG files and a CSV template for labels
    """
    print("Loading data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    images = data['images']
    sites = data['site']
    n_images = len(images)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating labeling sheets for {n_images} images...")
    print(f"Output directory: {output_dir}")
    
    # Create CSV template for labels
    csv_data = []
    
    n_sheets = (n_images + images_per_sheet - 1) // images_per_sheet
    
    for sheet_idx in range(n_sheets):
        start_idx = sheet_idx * images_per_sheet
        end_idx = min(start_idx + images_per_sheet, n_images)
        n_in_sheet = end_idx - start_idx
        
        # Calculate grid
        cols = 10
        rows = (n_in_sheet + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 2*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i in range(len(axes)):
            ax = axes[i]
            img_idx = start_idx + i
            
            if i < n_in_sheet:
                # Show image
                img_2d = images[img_idx, :, :, 0]
                
                if img_2d.max() > 1.0:
                    img_2d = img_2d / 255.0
                
                ax.imshow(img_2d, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'{img_idx}\n{sites[img_idx]}', fontsize=8)
                ax.axis('off')
                
                # Add to CSV
                csv_data.append({
                    'image_index': img_idx,
                    'site': sites[img_idx],
                    'sheet': sheet_idx,
                    'view_label': ''  # To be filled manually
                })
            else:
                ax.axis('off')
        
        plt.suptitle(f'Labeling Sheet {sheet_idx+1}/{n_sheets} (Images {start_idx}-{end_idx-1})\n'
                     'Labels: S=sagittal, A=axial, C=coronal, U=unknown',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        sheet_path = output_dir / f'sheet_{sheet_idx:03d}.png'
        plt.savefig(sheet_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Generated sheet {sheet_idx+1}/{n_sheets}: {sheet_path}")
    
    # Save CSV template
    csv_path = output_dir / 'labels_template.csv'
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Generated {n_sheets} labeling sheets")
    print(f"✓ Created CSV template: {csv_path}")
    print("\nNEXT STEPS:")
    print(f"1. Open the PNG files in {output_dir}")
    print(f"2. Fill in the 'view_label' column in {csv_path}")
    print("   Use: S (sagittal), A (axial), C (coronal), U (unknown)")
    print(f"3. Run: python batch_label_views.py {data_path} --apply {csv_path}")


def apply_labels_from_csv(data_path, csv_path, output_path=None):
    """
    Apply labels from CSV to dataset
    """
    print("Loading data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Loading labels from CSV...")
    df = pd.read_csv(csv_path)
    
    # Filter out empty labels
    df_labeled = df[df['view_label'].notna() & (df['view_label'] != '')]
    
    print(f"Found {len(df_labeled)} labeled images")
    
    # Create label array
    view_labels = np.array(['unlabeled'] * len(data['images']), dtype=object)
    
    # Map short codes to full names
    label_map = {
        'S': 'sagittal', 's': 'sagittal',
        'A': 'axial', 'a': 'axial',
        'C': 'coronal', 'c': 'coronal',
        'U': 'unknown', 'u': 'unknown'
    }
    
    for _, row in df_labeled.iterrows():
        idx = int(row['image_index'])
        label_code = str(row['view_label']).strip()
        
        if label_code in label_map:
            view_labels[idx] = label_map[label_code]
        else:
            print(f"  Warning: Unknown label '{label_code}' at index {idx}")
    
    # Add to data
    data['view'] = view_labels
    
    # Statistics
    print("\nLabel distribution:")
    unique, counts = np.unique(view_labels, return_counts=True)
    for view, count in zip(unique, counts):
        pct = 100 * count / len(view_labels)
        print(f"  {view}: {count} ({pct:.1f}%)")
    
    # Save
    if output_path is None:
        output_path = data_path.replace('.pkl', '_labeled.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✓ Saved labeled data to: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch view labeling')
    parser.add_argument('data_path', help='Path to data pickle file')
    parser.add_argument('--apply', help='Apply labels from CSV file')
    parser.add_argument('--output', help='Output path for labeled data')
    parser.add_argument('--images_per_sheet', type=int, default=100,
                       help='Images per sheet (default: 100)')
    
    args = parser.parse_args()
    
    if args.apply:
        apply_labels_from_csv(args.data_path, args.apply, args.output)
    else:
        generate_labeling_sheet(args.data_path, 
                               images_per_sheet=args.images_per_sheet)
