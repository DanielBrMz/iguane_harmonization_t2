#!/bin/bash

# Training script with nohup and logging
# This will run in the background and save all output to a log file

# Create directories
mkdir -p weights/cyclegan_2d
mkdir -p results/cyclegan_2d
mkdir -p logs/cyclegan_2d

# Get timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/cyclegan_2d/training_${TIMESTAMP}.log"

echo "Starting IGUANe training..."
echo "Log file: $LOGFILE"
echo "PID will be saved to: training.pid"

# Run with nohup - IGUANe with 2D-appropriate batch sizes
nohup python3 train_fetal_2d_cyclegan_iguane.py \
    --train_data processed_data_4slice/train_4slice_data.pkl \
    --reference_site BCH_CHD \
    --epochs 100 \
    --batch_size_gen 8 \
    --batch_size_disc 16 \
    --initial_lr 0.0002 \
    --final_lr 0.00002 \
    --lambda_cycle 30.0 \
    --lambda_identity 15.0 \
    --save_freq 25 \
    --weight_dir ./weights/cyclegan_2d \
    --result_dir ./results/cyclegan_2d \
    --log_dir ./logs/cyclegan_2d \
    --ga_embedding_dim 16 \
    --gpu 0,1,2 \
    > "$LOGFILE" 2>&1 &

# Save PID
echo $! > training.pid

echo ""
echo "Training started successfully!"
echo "PID: $(cat training.pid)"
echo ""
echo "Monitor training with:"
echo "  tail -f $LOGFILE"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Stop training:"
echo "  kill \$(cat training.pid)"
echo ""