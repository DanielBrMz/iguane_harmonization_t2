#!/bin/bash
# Training script for fetal brain harmonization

# Set environment
export PYTHONPATH=/neuro/labs/grantlab/research/MRI_processing/daniel.barrerasmeraz/repos/iguane_harmonization:$PYTHONPATH

# Prepare data first
echo "Preparing dataset..."
python prepare_fetal_data.py

# Train slice-based model (recommended by Hyeokjin)
echo "Training slice-based model..."
python fetal_brain_age_harmonization.py \
    -train_csv processed_fetal_data.csv \
    -mode slice \
    -batch_size 128 \
    -n_slice 4 \
    -epochs 1000 \
    -gpu 0 \
    -output_dir ./results_slice_model

# Optional: Train stack-based model for comparison
echo "Training stack-based model..."
python fetal_brain_age_harmonization.py \
    -train_csv processed_fetal_data.csv \
    -mode stack \
    -batch_size 32 \
    -n_slice 4 \
    -epochs 1000 \
    -gpu 0 \
    -output_dir ./results_stack_model