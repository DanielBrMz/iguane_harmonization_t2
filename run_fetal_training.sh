#!/bin/bash
# Launch script for fetal brain harmonization training

echo "Starting Fetal Brain Harmonization Training"
echo "=========================================="

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the training
python fetal_brain_4slice_training.py \
    -train_csv Andrea_total_list_230119.csv \
    -batch_size 128 \
    -n_slice 4 \
    -gpu 0 \
    -epochs 1000 \
    -rl ./results/fetal_harmonization \
    -wl ./weights/fetal_harmonization \
    -hl ./history/fetal_harmonization \
    -output fetal_4slice_harmonization

echo "Training completed!"