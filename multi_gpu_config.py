import os
import tensorflow as tf
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras import mixed_precision

def setup_multi_gpu_strategy(gpu_ids=[0, 1, 2], memory_growth=True):
    """
    Setup multi-GPU training strategy for 3 RTX A5000 GPUs
    
    Args:
        gpu_ids: List of GPU IDs to use (default: [0, 1, 2])
        memory_growth: Enable memory growth to avoid OOM errors
        
    Returns:
        strategy: TensorFlow distribution strategy
        batch_size_per_replica: Recommended batch size per GPU
    """
    
    # Set GPU visibility
    gpu_str = ','.join(map(str, gpu_ids))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    
    print(f"Configuring multi-GPU training on GPUs: {gpu_ids}")
    
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU {gpu} configured with memory growth: {memory_growth}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            
    # Enable mixed precision for better performance on RTX A5000
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision (float16) enabled for optimal performance")
    
    # Create MirroredStrategy for data parallelism
    strategy = MirroredStrategy()
    
    print(f"Number of replicas in sync: {strategy.num_replicas_in_sync}")
    print(f"Total GPUs configured: {len(gpus)}")
    
    # Calculate optimal batch size per GPU
    # RTX A5000 has 24GB VRAM, safe to use larger batches
    base_batch_size = 32  # Base batch size for harmonization models
    batch_size_per_replica = base_batch_size
    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    
    print(f"Recommended batch size per GPU: {batch_size_per_replica}")
    print(f"Global batch size: {global_batch_size}")
    
    return strategy, batch_size_per_replica, global_batch_size

def get_memory_info():
    """Print GPU memory information"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu}")
        # Get memory info if available
        try:
            memory_info = tf.config.experimental.get_memory_info(gpu)
            current_mb = memory_info['current'] / (1024**2)
            peak_mb = memory_info['peak'] / (1024**2)
            print(f"  Current memory: {current_mb:.1f} MB")
            print(f"  Peak memory: {peak_mb:.1f} MB")
        except:
            print("  Memory info not available")

def create_distributed_model(model_fn, strategy):
    """
    Create model within distribution strategy scope
    
    Args:
        model_fn: Function that returns the model
        strategy: Distribution strategy
        
    Returns:
        model: Distributed model
    """
    with strategy.scope():
        model = model_fn()
        print(f"Model created within distribution strategy scope")
        return model

def setup_distributed_optimizer(optimizer_fn, strategy):
    """
    Setup optimizer for distributed training
    
    Args:
        optimizer_fn: Function that returns the optimizer
        strategy: Distribution strategy
        
    Returns:
        optimizer: Distributed optimizer
    """
    with strategy.scope():
        optimizer = optimizer_fn()
        # Wrap with mixed precision optimizer for RTX A5000
        if hasattr(mixed_precision, 'LossScaleOptimizer'):
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
            print("LossScaleOptimizer applied for mixed precision")
        return optimizer

def print_gpu_utilization():
    """Print current GPU utilization"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print("\nCurrent GPU Status:")
            for i, line in enumerate(lines):
                util, mem_used, mem_total = line.split(', ')
                print(f"  GPU {i}: {util}% utilization, {mem_used}/{mem_total} MB memory")
        else:
            print("Could not get GPU utilization info")
    except Exception as e:
        print(f"❌ Error getting GPU info: {e}")

# Configuration presets for different training scenarios
TRAINING_CONFIGS = {
    'harmonization': {
        'description': 'IGUANe harmonization training',
        'batch_size_per_gpu': 16,  # Conservative for GAN training
        'learning_rate_scale': 1.0,  # Don't scale LR for small batch increase
        'mixed_precision': True,
    },
    'fetal_brain': {
        'description': 'Fetal brain age prediction',
        'batch_size_per_gpu': 32,  # Can be larger for classification
        'learning_rate_scale': 1.0,
        'mixed_precision': True,
    },
    'inference': {
        'description': 'Batch inference',
        'batch_size_per_gpu': 64,  # Large batches for inference
        'learning_rate_scale': 1.0,
        'mixed_precision': True,
    }
}

def get_training_config(config_name='harmonization'):
    """Get predefined training configuration"""
    config = TRAINING_CONFIGS.get(config_name, TRAINING_CONFIGS['harmonization'])
    print(f"📋 Using training config: {config['description']}")
    return config

if __name__ == "__main__":
    print("🚀 IGUANe Multi-GPU Configuration Test")
    print("=" * 50)
    
    # Test the setup
    strategy, batch_per_gpu, global_batch = setup_multi_gpu_strategy()
    
    print("\n📊 Memory Information:")
    get_memory_info()
    
    print("\n🖥️  GPU Utilization:")
    print_gpu_utilization()
    
    print("\n✅ Multi-GPU setup completed successfully!")
    print(f"🎯 Ready for training with {strategy.num_replicas_in_sync} GPUs")