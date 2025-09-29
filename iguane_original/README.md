# IGUANe Original - 3D T1-weighted Brain Harmonization

This directory contains the original IGUANe implementation for harmonization of 3D T1-weighted brain MR images, as described in the [peer-reviewed publication](https://doi.org/10.1016/j.media.2024.103388).

## Directory Structure

- **all_in_one.py** - Full inference pipeline including preprocessing and harmonization
- **pipeline.py** - Core pipeline functions for preprocessing and harmonization
- **preprocessing/** - Preprocessing tools and scripts
  - `crop_mris.py` - Image cropping utilities
  - `median_norm.py` - Median normalization
  - `MNI152_T1_1mm_brain.nii.gz` - MNI152 template
- **harmonization/** - Harmonization model and training
  - `iguane_weights.h5` - Pre-trained model weights
  - `inference.py` - Harmonization inference script
  - `model_architectures.py` - Neural network architectures
  - `training/` - Training scripts and utilities
- **prediction/** - Prediction models
  - `inference.py` - Prediction inference
  - `model_architecture.py` - Model architecture
  - `training/` - Training utilities
- **metadata/** - Dataset metadata and CSV files

## Usage

### All-in-one Pipeline

Process a single MR image:
```bash
python all_in_one.py --in-mri <input.nii.gz> --out-mri <output.nii.gz>
```

Process multiple MR images from CSV:
```bash
python all_in_one.py --in-csv <data.csv>
```

### Harmonization Only

For preprocessed images, use the harmonization inference script:
```bash
cd harmonization
python inference.py
```

## Requirements

- See `../iguane.yml` for Anaconda environment setup
- GPU recommended but not required
- Additional tools: FSL, ANTs, HD-BET

## Reference

If you use this code, please cite:
```
[Add full citation here]
```
