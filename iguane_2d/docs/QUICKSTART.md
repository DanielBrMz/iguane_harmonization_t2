# Quick Start Guide - Modular Training

This guide will help you quickly start training with the new modular implementation.

## Prerequisites

1. **Python Environment**: Python 3.8+
2. **Dependencies**: Install from requirements.txt
3. **Data**: Preprocessed 4-slice fetal brain data (pickle format)
4. **GPU**: CUDA-compatible GPU (optional but recommended)

## Installation

```bash
cd /home/brmz/Documents/Projects/BCH/iguane_harmonization

# Activate your environment
conda activate iguane  # or your environment name

# Verify dependencies
pip install -r requirements.txt
```

## Basic Training (5-Minute Start)

### Step 1: Navigate to Training Directory

```bash
cd iguane_2d/training
```

### Step 2: Run Training with Defaults

```bash
python train.py \
    --train_data ../../train_fetal_4slice.csv \
    --reference_site BCH \
    --epochs 200
```

That's it! Training will start with sensible defaults.

## Understanding the Output

### Console Output

```
================================================================================
FETAL BRAIN 2D CYCLEGAN - IGUANE TRAINING
================================================================================
TensorFlow version: 2.x.x
GPUs available: X
================================================================================

================================================================================
LOADING DATA
================================================================================
Loaded 450 samples from ../../train_fetal_4slice.csv
Reference site: BCH (150 samples)
Target sites: ['TMC', 'VGH'] (300 samples total)

================================================================================
CREATING TENSORFLOW DATASETS
================================================================================
Created dataset for BCH: 150 samples
Created dataset for TMC: 150 samples  
Created dataset for VGH: 150 samples

================================================================================
BUILDING MODEL
================================================================================
Building generators and discriminators...
Models compiled successfully

================================================================================
STARTING TRAINING
================================================================================
Epochs: 200
Batch size: 8
Generator LR: 0.0002
Discriminator LR: 0.0002
Lambda cycle: 10.0
Lambda identity: 1.0
================================================================================

Epoch 1/200
100%|████████████| 18/18 [00:15<00:00, 1.2it/s, G=3.45, D_BCH=0.85, Cyc=0.45, Collapse=0]

  Gen: 3.4500 | Disc BCH: 0.8500 | Cycle: 0.4500 | Identity: 0.1200
```

### Output Directories

```
iguane_2d/
├── weights/           # Model checkpoints (every 25 epochs)
├── results/           # Harmonized samples
├── logs/             # Training history CSV
└── training/         # Training scripts
```

## Common Training Scenarios

### Scenario 1: Quick Test Run

Test the pipeline with minimal epochs:

```bash
python train.py \
    --train_data ../../train_fetal_4slice.csv \
    --epochs 5 \
    --batch_size 4 \
    --save_freq 5
```

### Scenario 2: Production Training

Full training with all features:

```bash
python train.py \
    --train_data ../../train_fetal_4slice.csv \
    --reference_site BCH \
    --epochs 300 \
    --batch_size 8 \
    --lr_gen 2e-4 \
    --lr_disc 2e-4 \
    --lambda_cycle 10.0 \
    --lambda_identity 1.0 \
    --use_augmentation \
    --save_freq 25 \
    --memory_growth
```

### Scenario 3: Multi-GPU Training

Utilize multiple GPUs:

```bash
python train.py \
    --train_data ../../train_fetal_4slice.csv \
    --multi_gpu_strategy parallel \
    --gpu 0 1 \
    --batch_size 16 \
    --epochs 200
```

### Scenario 4: Resume from Checkpoint

(Future feature - manual weight loading for now)

```bash
# Edit train.py to load weights before training:
# cyclegan.gen_site2BCH.load_weights('weights/gen_site2BCH_epoch_100.weights.h5')
```

## Monitoring Training

### Loss Values

- **Generator Loss (G)**: Should stabilize around 2-5
- **Discriminator Loss (D_BCH)**: Should stabilize around 0.5-1.0
- **Cycle Loss (Cyc)**: Gradual decrease, typically 0.1-0.5
- **Identity Loss**: Should remain low < 0.1
- **Collapse Warning**: Should stay at 0-1

### GPU Memory

```bash
# In another terminal, monitor GPU usage:
watch -n 1 nvidia-smi
```

### Training History

```bash
# View loss curves:
cat logs/training_history.csv
```

## Troubleshooting

### Problem: Out of Memory

**Solution 1**: Reduce batch size
```bash
python train.py --train_data ../../train_fetal_4slice.csv --batch_size 4
```

**Solution 2**: Enable memory growth
```bash
python train.py --train_data ../../train_fetal_4slice.csv --memory_growth
```

### Problem: NaN Loss

**Solution 1**: Reduce learning rate
```bash
python train.py --train_data ../../train_fetal_4slice.csv --lr_gen 1e-4 --lr_disc 5e-5
```

**Solution 2**: Check data
```bash
python -c "import pickle; data = pickle.load(open('../../train_fetal_4slice.csv', 'rb')); print(data['images'].min(), data['images'].max())"
```

### Problem: Slow Training

**Solution 1**: Check data loading
```bash
# Add print statements in data_loader.py to check loading time
```

**Solution 2**: Enable XLA (automatic)
```bash
# XLA is enabled by default in gpu_utils.py
```

### Problem: Discriminator Collapse

**Signs**: Discriminator loss drops below 0.01 early in training

**Solution**: Increase label smoothing or adjust discriminator LR
```bash
# Edit losses.py: change label_smoothing to 0.2
# Or adjust LR ratio
python train.py --train_data ../../train_fetal_4slice.csv --lr_disc 1e-4
```

## Expected Timeline

| Milestone | Epoch | Time (1 GPU) |
|-----------|-------|--------------|
| Initial convergence | 25-50 | 30-60 min |
| Stable harmonization | 100 | 2 hours |
| High quality results | 200 | 4 hours |
| Optimal results | 300 | 6 hours |

*Times are approximate and depend on GPU, batch size, and data size*

## Checkpoints and Evaluation

### Checkpoint Files

```
weights/
├── gen_site2BCH_epoch_25.weights.h5
├── gen_BCH2TMC_epoch_25.weights.h5
├── gen_BCH2VGH_epoch_25.weights.h5
├── disc_BCH_epoch_25.weights.h5
├── disc_TMC_epoch_25.weights.h5
└── disc_VGH_epoch_25.weights.h5
```

### Evaluation Results

```
results/evaluation/
├── epoch_25_sample_0.png
├── epoch_25_sample_1.png
├── epoch_25_stats.csv
└── ...
```

## Next Steps

After training completes:

1. **Review training history**: `cat logs/training_history.csv`
2. **Inspect visual results**: Check `results/evaluation/` folder
3. **Use best checkpoint**: Typically around epoch 150-250
4. **Run inference**: Use trained generator for harmonization

## Getting Help

1. **Check README**: See `README.md` for detailed documentation
2. **Review module docs**: Each module has comprehensive docstrings
3. **Check logs**: Review console output and training history
4. **Inspect code**: Modular structure makes debugging easier

## Configuration Reference

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 200 | Number of training epochs |
| `--batch_size` | 8 | Batch size per GPU |
| `--lr_gen` | 2e-4 | Generator learning rate |
| `--lr_disc` | 2e-4 | Discriminator learning rate |
| `--lambda_cycle` | 10.0 | Cycle consistency weight |
| `--lambda_identity` | 1.0 | Identity loss weight |
| `--save_freq` | 25 | Checkpoint save frequency |
| `--use_augmentation` | False | Enable data augmentation |
| `--memory_growth` | False | Enable GPU memory growth |

### Full Parameter List

```bash
python train.py --help
```

## Example Training Session

```bash
# Navigate to training directory
cd iguane_2d/training

# Start training
python train.py \
    --train_data ../../train_fetal_4slice.csv \
    --reference_site BCH \
    --epochs 200 \
    --batch_size 8 \
    --use_augmentation \
    --memory_growth

# Training runs...
# Checkpoints saved every 25 epochs to weights/
# Evaluation results saved to results/evaluation/
# Training history saved to logs/training_history.csv

# After completion, view results
ls -lh weights/        # Model checkpoints
ls -lh results/        # Harmonized samples
cat logs/training_history.csv  # Loss curves
```

## Success Criteria

Your training is successful if:

✅ Losses converge (don't diverge or stay constant)  
✅ Generator loss stabilizes around 2-5  
✅ Discriminator loss stays around 0.5-1.0  
✅ No persistent collapse warnings  
✅ Visual results show harmonization effect  
✅ Statistics show reduced between-site variance  

## Ready to Train!

You're all set! Run the basic command to start:

```bash
cd iguane_2d/training
python train.py --train_data ../../train_fetal_4slice.csv --epochs 200
```

**Happy training! 🚀**
