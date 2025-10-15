#!/bin/bash
# Multi-GPU Training Script for IGUANe Harmonization
# Optimized for 3x NVIDIA RTX A5000 GPUs
# Author: Daniel Barreras Meraz

echo "IGUANe Multi-GPU Training Setup"
echo "=================================="

# Set environment variables for optimal GPU performance
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TF_CPP_MIN_LOG_LEVEL="1"
export TF_ENABLE_GPU_GARBAGE_COLLECTION="false"
export TF_GPU_THREAD_MODE="gpu_private"
export TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT="1"

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits

echo ""
echo "Available training modes:"
echo "1. Fetal Brain Harmonization (with age conditioning)"
echo "2. Standard IGUANe Harmonization (multi-site)"
echo "3. Test multi-GPU setup"

read -p "Select training mode (1-3): " mode

case $mode in
    1)
        echo "Starting Fetal Brain Harmonization Training..."
        python fetal_brain_harmonization_multigpu.py \
            --gpus 0,1,2 \
            -batch_size 96 \
            -epochs 1000 \
            -mode slice \
            -n_slice 4 \
            -output_dir ./results_fetal_multigpu \
            -train_csv processed_fetal_data.csv
        ;;
    2)
        echo "Starting IGUANe Harmonization Training..."
        python harmonization_multigpu_training.py \
            --gpus 0,1,2 \
            --batch-size 48 \
            --epochs 100 \
            --steps-per-epoch 200 \
            --output-dir ./results_harmonization_multigpu
        ;;
    3)
        echo "Testing multi-GPU setup..."
        python multi_gpu_config.py
        ;;
    *)
        echo "Invalid selection. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Training completed!"
echo "Final GPU status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits