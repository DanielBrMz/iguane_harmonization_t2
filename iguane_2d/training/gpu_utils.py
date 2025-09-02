"""
GPU configuration utilities
Handles GPU detection, memory growth, and device assignment
"""

import os
import tensorflow as tf


def print_gpu_usage():
    """Print current GPU memory usage for all available GPUs"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("\n  GPU Memory Usage:")
        for i, gpu in enumerate(gpus):
            try:
                memory_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                current_mb = memory_info['current'] / 1024**2
                peak_mb = memory_info['peak'] / 1024**2
                print(f"    GPU {i}: Current={current_mb:.0f}MB, Peak={peak_mb:.0f}MB")
            except Exception as e:
                print(f"    GPU {i}: Unable to get memory info - {e}")
    else:
        print("\n  No GPUs available")


def configure_gpu(gpu_ids='0,1,2', memory_growth=True):
    """
    Configure GPU settings for training
    
    Args:
        gpu_ids: Comma-separated string of GPU IDs to use
        memory_growth: Whether to enable memory growth (prevents OOM)
    
    Returns:
        List of configured GPU devices
    """
    # Set GPU visibility
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    # Clear any existing session
    tf.keras.backend.clear_session()
    
    # Get available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth for all GPUs
            if memory_growth:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"\nConfigured {len(gpus)} GPU(s) with memory growth")
            else:
                print(f"\nConfigured {len(gpus)} GPU(s) without memory growth")
            
            # Print GPU details
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                
        except RuntimeError as e:
            print(f"\nGPU configuration error: {e}")
    else:
        print("\nNo GPUs available, using CPU")
    
    return gpus


def get_gpu_assignments(n_gpus, strategy='model_parallel'):
    """
    Determine GPU assignment strategy for models
    
    Args:
        n_gpus: Number of available GPUs
        strategy: 'model_parallel' or 'single'
    
    Returns:
        Dictionary mapping model components to GPU devices
    """
    if strategy == 'single' or n_gpus <= 1:
        # All on first GPU
        assignments = {
            'generators': '/GPU:0',
            'disc_BCH': '/GPU:0',
            'disc_sites': '/GPU:0'
        }
    elif n_gpus == 2:
        # Generators on GPU 0, discriminators on GPU 1
        assignments = {
            'generators': '/GPU:0',
            'disc_BCH': '/GPU:1',
            'disc_sites': '/GPU:1'
        }
    else:  # n_gpus >= 3
        # Spread across GPUs
        assignments = {
            'generators': '/GPU:0',
            'disc_BCH': '/GPU:1',
            'disc_sites': '/GPU:2'
        }
    
    print(f"\nGPU Assignment Strategy: {strategy}")
    for component, device in assignments.items():
        print(f"  {component}: {device}")
    
    return assignments


def enable_xla_optimization():
    """Enable XLA (Accelerated Linear Algebra) for faster training"""
    try:
        tf.config.optimizer.set_jit(True)
        print("\nXLA optimization enabled")
    except Exception as e:
        print(f"\nCould not enable XLA: {e}")


def setup_mixed_precision():
    """Setup mixed precision training for faster computation"""
    try:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("\nMixed precision training enabled (float16)")
    except Exception as e:
        print(f"\nCould not enable mixed precision: {e}")
