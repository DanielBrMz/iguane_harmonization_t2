# IGUANe 2D - Fetal Brain T2-weighted Harmonization

This directory contains the IGUANe 2D implementation for harmonization of 2D fetal brain T2-weighted MR images using CycleGAN architecture with gestational age conditioning.

## Directory Structure

- **data/** - Dataset CSV files and metadata
  - `Andrea_total_list_230119.csv` - Main dataset listing
  - `stackQC_50stacks.csv` - Quality control data
  - `train_fetal_4slice.csv`, `val_fetal_4slice.csv`, `test_fetal_4slice.csv` - Train/val/test splits
  
- **training/** - Model training scripts
  - `train_fetal_2d_cyclegan.py` - Main CycleGAN training script
  - `fetal_brain_age_harmonization.py` - Age-conditioned harmonization training
  - `training.pid` - Process ID for running training
  
- **preprocessing/** - Data preprocessing scripts
  - `prepare_fetal_data.py` - General fetal data preparation
  - `prepare_fetal_4slice_data.py` - 4-slice extraction and preparation
  - `create_training_csvs_from_best_images.py` - Training set creation
  - `fix_demographics_in_csvs.py` - Demographics data cleaning
  - `normalize_dataset.py` - Dataset normalization by view
  - `split_by_view.py` - Split dataset by view type (sagittal/axial/coronal)
  - **labeling/** - View labeling tools
    - `auto_classify_views.py` - Semi-automated view classification
    - `batch_label_views.py` - Batch labeling workflow
    - `manual_label_views.py` - Interactive labeling interface
  
- **evaluation/** - Model evaluation and analysis
  - `evaluate_cyclegan_results.py` - CycleGAN evaluation
  - `evaluate_fetal_harmonization.py` - Harmonization quality metrics
  - `run_complete_evaluation.py` - Comprehensive evaluation pipeline
  - `create_visual_comparison.py` - Visual comparison generation
  - `debug_output.py` - Debugging utilities
  - `create_intensity_histogram_comparison.py` - Intensity distribution analysis
  - `create_training_contact_sheet.py` - Training data visualization
  - `plot_training_history.py` - Training loss/metric plots
  - `generate_kiho_results.py` - Generate presentation-ready results
  
- **harmonization/** - Harmonization inference
  - `harmonize_and_enhance.py` - Harmonization with enhancement
  
- **logs/** - Training logs
  - `cyclegan_2d/` - CycleGAN training logs
  
- **harmonized_results/** - Harmonization output
  - Results from harmonization runs
  
- **harmonized_monday/** - Additional harmonization results
  - Alternative output directory
  
- **harmonization_evaluation/** - Evaluation results
  - Quantitative and qualitative evaluation outputs

- **run_training.sh** - Training execution script

## Usage

### Training

Train the 2D CycleGAN model:
```bash
cd training
python train_fetal_2d_cyclegan.py [options]
```

Or use the convenience script:
```bash
./run_training.sh
```

### Data Preparation

Prepare 4-slice fetal brain data:
```bash
cd preprocessing
python prepare_fetal_4slice_data.py
```

### Evaluation

Run complete evaluation:
```bash
cd evaluation
python run_complete_evaluation.py
```

Evaluate specific checkpoint:
```bash
python evaluate_cyclegan_results.py
```

### Harmonization

Apply harmonization to new data:
```bash
cd harmonization
python harmonize_and_enhance.py
```

## Model Details

- **Architecture**: 2D CycleGAN with U-Net generators
- **Conditioning**: Gestational age embedding
- **Input**: 2D slices (138x176) from fetal brain T2-weighted images
- **Training**: Multi-site harmonization with collapse prevention
- **Sites**: BCH (CHD, Normative, Placenta), dHCP, CRL, UNC

## Requirements

- TensorFlow 2.x with GPU support
- See `../requirements.txt` for full dependencies
- See `../iguane.yml` for Anaconda environment

## Notes

This is an adaptation of the original IGUANe method for 2D fetal brain imaging. The model uses:
- 4 central slices from fetal brain stacks
- Gestational age as conditioning variable
- Site-specific harmonization (BCH ↔ Reference sites)
- Gradient accumulation for stable training
- Comprehensive collapse detection and prevention
