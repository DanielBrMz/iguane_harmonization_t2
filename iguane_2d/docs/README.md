# IGUANe 2D Training - Modular Implementation

This directory contains the modularized implementation of the 2D CycleGAN training pipeline for fetal brain MRI harmonization.

## 📁 Module Overview

### Core Modules

#### `train.py` - Main Training Script
- **Purpose**: Entry point and training orchestration
- **Key Functions**:
  - `train()`: Main training loop
  - `train_epoch()`: Single epoch training
  - `save_checkpoint()`: Model weight persistence
  - `setup_environment()`: GPU and directory configuration
  - `load_and_prepare_data()`: Data loading pipeline
  - `build_model()`: Model instantiation

#### `config.py` - Configuration Management
- **Purpose**: Centralized configuration and argument parsing
- **Key Components**:
  - `TrainingConfig`: Dataclass with all hyperparameters
  - `parse_args()`: Command-line argument parser
  - Directory creation utilities
- **Configurable Parameters**:
  - Learning rates (generator, discriminator)
  - Loss weights (cycle, identity)
  - Training schedule (epochs, batch size)
  - GPU settings (multi-GPU, memory growth)
  - Data paths and output directories

#### `cyclegan.py` - CycleGAN Model
- **Purpose**: Multi-site CycleGAN implementation with IGUANe gradient accumulation
- **Key Class**: `CycleGAN2D_MultiSite`
- **Features**:
  - Multiple target sites (one-to-many harmonization)
  - Gradient accumulation for stable training
  - GPU device assignment (model parallelism)
  - Site-specific discriminators
  - Gestational age conditioning
  - Collapse detection and warnings

#### `models.py` - Neural Network Architectures
- **Purpose**: Generator and discriminator definitions
- **Key Functions**:
  - `build_2d_generator()`: U-Net generator with GA embedding
  - `build_2d_discriminator()`: PatchGAN discriminator
- **Key Classes**:
  - `SpectralNormalization`: Custom Keras layer for weight normalization

#### `losses.py` - Loss Functions
- **Purpose**: All loss computations
- **Key Functions**:
  - `cycle_consistency_loss()`: L1 cycle loss
  - `identity_loss()`: L1 identity loss
  - `discriminator_loss_smooth()`: Adversarial loss with label smoothing
  - `generator_loss()`: Adversarial loss for generator
  - `compute_generator_total_loss()`: Combined generator loss
  - `check_for_nan_loss()`: NaN detection

#### `data_loader.py` - Data Pipeline
- **Purpose**: Data loading, preprocessing, and augmentation
- **Key Functions**:
  - `load_preprocessed_data()`: Load pickle data files
  - `create_site_datasets()`: Split data by acquisition site
  - `create_all_datasets()`: Create TensorFlow datasets
- **Key Classes**:
  - `DataAugmenter`: Random augmentation (rotation, flip, brightness)

#### `evaluation.py` - Training Evaluation
- **Purpose**: Model evaluation and quality assessment
- **Key Functions**:
  - `evaluate_checkpoint()`: Comprehensive checkpoint evaluation
  - `detect_collapse()`: Multi-level collapse detection
  - `compute_harmonization_stats()`: Statistical metrics
  - `save_evaluation_figure()`: Visual comparisons
  - `generate_quality_report()`: Training summary

#### `gpu_utils.py` - GPU Management
- **Purpose**: GPU configuration and monitoring
- **Key Functions**:
  - `configure_gpu()`: GPU setup with memory growth
  - `assign_model_to_device()`: Model placement
  - `print_gpu_usage()`: Memory monitoring
  - `enable_xla_optimization()`: XLA JIT compilation

---

## 🚀 Usage

### Basic Training

```bash
cd iguane_2d/training
python train.py \
    --train_data ../../train_fetal_4slice.csv \
    --reference_site BCH \
    --epochs 200 \
    --batch_size 8 \
    --lr_gen 2e-4 \
    --lr_disc 2e-4
```

### Multi-GPU Training

```bash
python train.py \
    --train_data ../../train_fetal_4slice.csv \
    --multi_gpu_strategy parallel \
    --gpu 0 1 \
    --batch_size 16
```

### Custom Configuration

```bash
python train.py \
    --train_data ../../train_fetal_4slice.csv \
    --reference_site BCH \
    --epochs 300 \
    --batch_size 4 \
    --lr_gen 1e-4 \
    --lr_disc 5e-5 \
    --lambda_cycle 15.0 \
    --lambda_identity 2.0 \
    --use_augmentation \
    --save_freq 20 \
    --weight_dir ./custom_weights \
    --result_dir ./custom_results
```

---

## 📊 Training Monitoring

### Loss Tracking
Training losses are automatically saved to:
- `logs/training_history.csv` - Complete training history
- Console output - Real-time progress bars with loss values

### Checkpoint Evaluation
Every `save_freq` epochs (default: 25):
- Harmonized samples generated
- Collapse detection performed
- Statistics computed (mean, std differences)
- Visual comparisons saved to `results/evaluation/`

### GPU Monitoring
GPU memory usage printed every 10 epochs via `print_gpu_usage()`

---

## 🏗️ Architecture Details

### Generator (U-Net with GA Conditioning)
```
Input: [B, 138, 176, 1] + GA [B, 1]
├─ Encoder: 5 blocks (32→64→128→256→512 channels)
├─ GA Embedding: Dense(32) → Tile → Concat
├─ Bottleneck: 512 channels with GA features
├─ Decoder: 5 blocks (512→256→128→64→32)
└─ Output: [B, 138, 176, 1]

Features:
- Skip connections (U-Net style)
- Spectral normalization
- LeakyReLU activation
- Tanh output activation
```

### Discriminator (PatchGAN)
```
Input: [B, 138, 176, 1]
├─ Conv blocks: 4 layers (64→128→256→512)
├─ Patch output: [B, 17, 22, 1]
└─ Sigmoid activation

Features:
- Spectral normalization
- LeakyReLU activation
- 70x70 receptive field
```

---

## 🔧 Customization Guide

### Adding New Loss Functions
1. Add function to `losses.py`:
```python
def my_custom_loss(real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    """Your loss implementation"""
    return tf.reduce_mean(...)
```

2. Update `compute_generator_total_loss()` in `losses.py`
3. Update `train_step()` in `cyclegan.py`

### Modifying Generator Architecture
1. Edit `build_2d_generator()` in `models.py`
2. Adjust `img_shape` and `ga_embedding_dim` in config
3. Test with small dataset first

### Adding New Augmentations
1. Add method to `DataAugmenter` class in `data_loader.py`:
```python
def augment(self, image, ga):
    # Your augmentation
    return image, ga
```

2. Update `create_tf_dataset()` to use new augmentation

---

## 🐛 Debugging

### Common Issues

**1. NaN Loss Values**
- Check learning rates (try reducing by 10x)
- Check gradient clipping (default: 10.0)
- Inspect data for outliers

**2. Discriminator Collapse**
- Increase label smoothing (default: 0.1)
- Adjust discriminator LR relative to generator
- Monitor `collapse_warning` in logs

**3. Out of Memory**
- Reduce batch size
- Enable `memory_growth=True`
- Use gradient accumulation (increase `accumulation_steps`)

**4. Slow Training**
- Enable XLA: set `XLA_FLAGS` environment variable
- Use multi-GPU with `--multi_gpu_strategy parallel`
- Check data loading bottlenecks

### Debug Mode
Enable verbose logging:
```python
# In train.py, uncomment:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.debugging.set_log_device_placement(True)
```

---

## 📈 Expected Training Behavior

### Loss Trajectories
- **Generator Loss**: Should stabilize around 2-5
- **Discriminator Loss**: Should stabilize around 0.5-1.0
- **Cycle Loss**: Gradual decrease, typically 0.1-0.5
- **Identity Loss**: Should remain low < 0.1

### Convergence Time
- **Typical**: 100-200 epochs for stable harmonization
- **Early collapse detection**: Stops if discriminator loss < 0.01 before epoch 50
- **Memory cleanup**: Automatic every epoch

### Checkpoints
- Saved every `save_freq` epochs (default: 25)
- Format: `gen_site2BCH_epoch_X.weights.h5`
- Includes all generators and discriminators

---

## 📝 Module Dependencies

```
train.py
├── config.py
├── gpu_utils.py
├── data_loader.py
│   └── (TensorFlow, NumPy, Pickle)
├── cyclegan.py
│   ├── models.py
│   │   └── (Keras, SpectralNormalization)
│   └── losses.py
│       └── (TensorFlow ops)
└── evaluation.py
    └── (Matplotlib, Pandas)
```

**External Dependencies**:
- TensorFlow >= 2.10
- NumPy
- Pandas
- Matplotlib
- tqdm
- pathlib

---

## 🔄 Migration from Original Script

This modular implementation is a refactored version of `train_fetal_2d_cyclegan.py` (1123 lines) broken down into 8 maintainable modules.

**Preserved Features**:
✅ IGUANe gradient accumulation  
✅ Multi-site harmonization  
✅ Gestational age conditioning  
✅ Spectral normalization  
✅ Collapse detection  
✅ Same loss functions  
✅ Identical training logic  

**Improvements**:
✨ Clear separation of concerns  
✨ Easier debugging and testing  
✨ Modular architecture allows component swapping  
✨ Comprehensive documentation  
✨ Type hints throughout  
✨ Better error handling  

---

## 📚 Additional Resources

- **Original IGUANe Paper**: [Link to paper]
- **CycleGAN Paper**: Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
- **Spectral Normalization**: Miyato et al., "Spectral Normalization for Generative Adversarial Networks"

---

## 🙋 Support

For issues or questions:
1. Check this README
2. Review module docstrings
3. Inspect training logs
4. Check GPU memory usage

**Happy Training! 🎉**
