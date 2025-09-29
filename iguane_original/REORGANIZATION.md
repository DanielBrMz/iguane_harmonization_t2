# Project Reorganization Summary

## Overview

The IGUANe harmonization project has been reorganized into two main directories:
- **iguane_original/** - Original 3D T1-weighted brain harmonization (published work)
- **iguane_2d/** - 2D fetal brain T2-weighted harmonization (in development)

## Complete File Mapping

### Root Level Files (Unchanged)
- `README.md` - Updated to reflect new structure
- `requirements.txt` - Shared Python dependencies
- `iguane.yml` - Shared Conda environment
- `__pycache__/` - Python cache (unchanged)

---

### IGUANe Original (3D T1-weighted)

#### Root Files
- `all_in_one.py` → **iguane_original/all_in_one.py**
- `pipeline.py` → **iguane_original/pipeline.py**

#### Preprocessing
- `preprocessing/crop_mris.py` → **iguane_original/preprocessing/crop_mris.py**
- `preprocessing/median_norm.py` → **iguane_original/preprocessing/median_norm.py**
- `preprocessing/MNI152_T1_1mm_brain.nii.gz` → **iguane_original/preprocessing/MNI152_T1_1mm_brain.nii.gz**

#### Harmonization
- `harmonization/iguane_weights.h5` → **iguane_original/harmonization/iguane_weights.h5**
- `harmonization/inference.py` → **iguane_original/harmonization/inference.py**
- `harmonization/model_architectures.py` → **iguane_original/harmonization/model_architectures.py**
- `harmonization/training/` → **iguane_original/harmonization/training/**
  - `main.py`
  - `trainers.py`
  - `input_pipeline/`

#### Prediction
- `prediction/inference.py` → **iguane_original/prediction/inference.py**
- `prediction/model_architecture.py` → **iguane_original/prediction/model_architecture.py**
- `prediction/training/` → **iguane_original/prediction/training/**

#### Metadata
- `metadata/ad_ge.csv` → **iguane_original/metadata/ad_ge.csv**
- `metadata/ad_test.csv` → **iguane_original/metadata/ad_test.csv**
- `metadata/ad_train.csv` → **iguane_original/metadata/ad_train.csv**
- `metadata/generalization_dataset.csv` → **iguane_original/metadata/generalization_dataset.csv**
- `metadata/miriad.csv` → **iguane_original/metadata/miriad.csv**
- `metadata/sample_synthseg.csv` → **iguane_original/metadata/sample_synthseg.csv**
- `metadata/training_dataset.csv` → **iguane_original/metadata/training_dataset.csv**
- `metadata/traveling_subject_dataset.csv` → **iguane_original/metadata/traveling_subject_dataset.csv**

---

### IGUANe 2D (Fetal Brain)

#### Data
- `Andrea_total_list_230119.csv` → **iguane_2d/data/Andrea_total_list_230119.csv**
- `stackQC_50stacks.csv` → **iguane_2d/data/stackQC_50stacks.csv**
- `train_fetal_4slice.csv` → **iguane_2d/data/train_fetal_4slice.csv**
- `val_fetal_4slice.csv` → **iguane_2d/data/val_fetal_4slice.csv**
- `test_fetal_4slice.csv` → **iguane_2d/data/test_fetal_4slice.csv**

#### Training
- `train_fetal_2d_cyclegan.py` → **iguane_2d/training/train_fetal_2d_cyclegan.py**
- `fetal_brain_age_harmonization.py` → **iguane_2d/training/fetal_brain_age_harmonization.py**
- `training.pid` → **iguane_2d/training/training.pid**

#### Preprocessing
- `prepare_fetal_data.py` → **iguane_2d/preprocessing/prepare_fetal_data.py**
- `prepare_fetal_4slice_data.py` → **iguane_2d/preprocessing/prepare_fetal_4slice_data.py**
- `create_training_csvs_from_best_images.py` → **iguane_2d/preprocessing/create_training_csvs_from_best_images.py**
- `fix_demographics_in_csvs.py` → **iguane_2d/preprocessing/fix_demographics_in_csvs.py**

#### Evaluation
- `evaluate_cyclegan_results.py` → **iguane_2d/evaluation/evaluate_cyclegan_results.py**
- `evaluate_fetal_harmonization.py` → **iguane_2d/evaluation/evaluate_fetal_harmonization.py**
- `run_complete_evaluation.py` → **iguane_2d/evaluation/run_complete_evaluation.py**
- `create_visual_comparison.py` → **iguane_2d/evaluation/create_visual_comparison.py**
- `debug_output.py` → **iguane_2d/evaluation/debug_output.py**

#### Harmonization
- `harmonize_and_enhance.py` → **iguane_2d/harmonization/harmonize_and_enhance.py**

#### Other
- `run_training.sh` → **iguane_2d/run_training.sh**
- `logs/` → **iguane_2d/logs/**
  - `cyclegan_2d/`
- `harmonized_results/` → **iguane_2d/harmonized_results/**
  - `enhancement_summary.csv`
  - `brain_masks/`
  - `comparison_figures/`
  - `enhanced_harmonized/`
  - `raw_harmonized/`
- `harmonized_monday/` → **iguane_2d/harmonized_monday/**
  - `enhancement_summary.csv`
  - `brain_masks/`
  - `comparison_figures/`
  - `enhanced_harmonized/`
  - `raw_harmonized/`
- `harmonization_evaluation/` → **iguane_2d/harmonization_evaluation/**
  - `20251027_151719/`
    - `quantitative_metrics/`
    - `visual_comparison/`

---

## New Files Created

- **iguane_original/README.md** - Documentation for original IGUANe
- **iguane_2d/README.md** - Documentation for IGUANe 2D
- **REORGANIZATION.md** - This file (file mapping documentation)

---

## Import Path Updates Required

Several Python files will need their import statements updated to reflect the new directory structure:

### Files in iguane_original/
- `all_in_one.py` - needs to update import of `pipeline`
- `pipeline.py` - needs to update imports from `harmonization.model_architectures`

### Files in iguane_2d/
- `evaluation/evaluate_cyclegan_results.py` - needs to update import from `train_fetal_2d_cyclegan`

These import paths should be updated based on how you plan to run the scripts (from root vs from subdirectories).

---

## Directory Structure

```
iguane_harmonization/
├── README.md (updated)
├── REORGANIZATION.md (new)
├── requirements.txt
├── iguane.yml
├── __pycache__/
│
├── iguane_original/
│   ├── README.md (new)
│   ├── all_in_one.py
│   ├── pipeline.py
│   ├── preprocessing/
│   ├── harmonization/
│   ├── prediction/
│   └── metadata/
│
└── iguane_2d/
    ├── README.md (new)
    ├── run_training.sh
    ├── data/
    ├── training/
    ├── preprocessing/
    ├── evaluation/
    ├── harmonization/
    ├── logs/
    ├── harmonized_results/
    ├── harmonized_monday/
    └── harmonization_evaluation/
```

---

## Benefits of New Organization

1. **Clear Separation** - Original published work separated from new development
2. **Better Navigation** - Easier to find files related to specific tasks
3. **Documentation** - Each subdirectory has its own README
4. **Maintainability** - Changes to one implementation don't affect the other
5. **Scalability** - Easy to add new implementations (e.g., iguane_3d_fetal)

---

## Next Steps

1. ✅ Reorganize directory structure
2. ✅ Create README files
3. ⏳ Update import statements in Python files
4. ⏳ Test that scripts still work in new locations
5. ⏳ Update any CI/CD pipelines or scripts that reference old paths
6. ⏳ Commit changes to git with clear message

---

Generated: November 1, 2025
