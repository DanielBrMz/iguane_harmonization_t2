#!/usr/bin/env python3
"""
Multi-GPU IGUANe Harmonization Training
Optimized for 3x NVIDIA RTX A5000 GPUs
Author: Daniel Barreras Meraz
Date: October 2025
"""

# Multi-GPU setup - must be done before importing TensorFlow
from multi_gpu_config import setup_multi_gpu_strategy, create_distributed_model, setup_distributed_optimizer

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow import concat as tf_concat
from json import dump as json_dump
import sys
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser('Multi-GPU IGUANe Harmonization Training')
parser.add_argument('--gpus', default='0,1,2', type=str, help='Comma-separated GPU IDs to use')
parser.add_argument('--batch-size', default=48, type=int, help='Global batch size (distributed across GPUs)')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--steps-per-epoch', default=200, type=int, help='Steps per epoch')
parser.add_argument('--output-dir', default='./harmonization_multigpu_results', type=str, help='Output directory')
parser.add_argument('--single-gpu', action='store_true', help='Use single GPU instead of multi-GPU')
args = parser.parse_args()

# Setup multi-GPU strategy
gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
print(f"Configuring IGUANe harmonization for GPUs: {gpu_ids}")

if args.single_gpu or len(gpu_ids) == 1:
    print("Using single GPU training")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    strategy = tf.distribute.get_strategy()
    batch_size_per_replica = args.batch_size
    global_batch_size = args.batch_size
else:
    print("Setting up multi-GPU training")
    strategy, batch_size_per_replica, global_batch_size = setup_multi_gpu_strategy(
        gpu_ids=gpu_ids, 
        memory_growth=True
    )
    if args.batch_size != 48:
        global_batch_size = args.batch_size
        batch_size_per_replica = global_batch_size // strategy.num_replicas_in_sync

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

print('\n' + '='*80)
print('MULTI-GPU IGUANE HARMONIZATION TRAINING')
print('='*80)
print(f'GPUs: {gpu_ids}')
print(f'Strategy replicas: {strategy.num_replicas_in_sync}')
print(f'Global batch size: {global_batch_size}')
print(f'Batch size per replica: {batch_size_per_replica}')
print(f'Epochs: {args.epochs}')
print(f'Steps per epoch: {args.steps_per_epoch}')
print(f'Output directory: {os.path.realpath(args.output_dir)}')
print('='*80 + '\n')

#################INPUT DATASETS#########################
# TODO: Replace with your actual dataset loading
# from input_pipeline.tf_dataset import datasets_from_tfrecords, datasets_from_tfrecords_biasSampling
# dataset_pairs = # TO DEFINE

# For now, create dummy datasets for demonstration
def create_dummy_dataset_pairs(num_pairs=3):
    """Create dummy dataset pairs for testing multi-GPU setup"""
    dataset_pairs = []
    
    for i in range(num_pairs):
        # Create dummy reference and source datasets
        ref_data = tf.random.normal((batch_size_per_replica, 160, 192, 160, 1))
        src_data = tf.random.normal((batch_size_per_replica, 160, 192, 160, 1))
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((ref_data, src_data))
        dataset = dataset.batch(batch_size_per_replica)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Distribute dataset
        dist_dataset = strategy.experimental_distribute_dataset(dataset)
        dataset_pairs.append(dist_dataset)
    
    return dataset_pairs

dataset_pairs = create_dummy_dataset_pairs()

##########INPUT PARAMETERS################
DEST_DIR_PATH = args.output_dir
N_EPOCHS = args.epochs
STEPS_PER_EPOCH = args.steps_per_epoch

# Import model architectures
sys.path.append('harmonization')
try:
    from model_architectures import Generator, Discriminator
except ImportError:
    print("❌ Could not import model architectures. Make sure harmonization/model_architectures.py exists")
    sys.exit(1)

def create_models():
    """Create models within strategy scope"""
    def model_fn():
        gen_univ = Generator()
        gens_bwd = [Generator() for _ in range(len(dataset_pairs))]
        discs_ref = [Discriminator() for _ in range(len(dataset_pairs))]
        discs_bwd = [Discriminator() for _ in range(len(dataset_pairs))]
        return gen_univ, gens_bwd, discs_ref, discs_bwd
    
    return model_fn

# Create models within strategy scope
print("Creating models...")
with strategy.scope():
    gen_univ, gens_bwd, discs_ref, discs_bwd = create_models()()
    print("Models created successfully")

##################VALIDATION############################
def eval_model(): 
    """Evaluation function - replace with actual validation"""
    return np.random.uniform()

EVAL_FREQ = 5

#########################################################

# Initialization of the optimizers within strategy scope
print("Setting up optimizers...")

def create_optimizers():
    """Create optimizers within strategy scope"""
    INIT_LR = 0.0002
    END_LR = 0.00002
    
    # Scale learning rates by number of GPUs
    lr_scale = strategy.num_replicas_in_sync
    scaled_init_lr = INIT_LR * lr_scale
    scaled_end_lr = END_LR * lr_scale
    
    n_steps_gen_univ = N_EPOCHS * STEPS_PER_EPOCH * len(dataset_pairs)
    n_steps_gen_bwd = N_EPOCHS * STEPS_PER_EPOCH
    n_steps_discs = N_EPOCHS * STEPS_PER_EPOCH
    
    def optimizer(n_steps, init_lr, end_lr):
        opt = Adam(learning_rate=PolynomialDecay(init_lr, n_steps, end_lr))
        return tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic=True)
    
    genUnivOptimizer = optimizer(n_steps_gen_univ, scaled_init_lr, scaled_end_lr)
    genOptimizers_bwd = [optimizer(n_steps_gen_bwd, scaled_init_lr, scaled_end_lr) for _ in range(len(dataset_pairs))]
    discOptimizers_ref = [optimizer(n_steps_discs, scaled_init_lr, scaled_end_lr) for _ in range(len(dataset_pairs))]
    discOptimizers_bwd = [optimizer(n_steps_discs, scaled_init_lr, scaled_end_lr) for _ in range(len(dataset_pairs))]
    
    return genUnivOptimizer, genOptimizers_bwd, discOptimizers_ref, discOptimizers_bwd

with strategy.scope():
    genUnivOptimizer, genOptimizers_bwd, discOptimizers_ref, discOptimizers_bwd = create_optimizers()
    print("Optimizers created successfully")

# Import and create trainers within strategy scope
print("Setting up trainers...")
sys.path.append('harmonization/training')
try:
    from trainers import Discriminator_trainer, Generator_trainer
    
    with strategy.scope():
        genTrainers = [Generator_trainer(gen_univ, gens_bwd[i], discs_ref[i], discs_bwd[i], 
                                       genUnivOptimizer, genOptimizers_bwd[i]) 
                      for i in range(len(dataset_pairs))]
        discTrainers_ref = [Discriminator_trainer(discs_ref[i], gen_univ, discOptimizers_ref[i]) 
                           for i in range(len(dataset_pairs))]
        discTrainers_src = [Discriminator_trainer(discs_bwd[i], gens_bwd[i], discOptimizers_bwd[i]) 
                           for i in range(len(dataset_pairs))]
    print("Trainers created successfully")
    
except ImportError:
    print("Could not import trainers. Using dummy trainers for testing")
    
    class DummyTrainer:
        def train(self, *args):
            return tf.constant(0.5, dtype=tf.float32)
    
    with strategy.scope():
        genTrainers = [DummyTrainer() for _ in range(len(dataset_pairs))]
        discTrainers_ref = [DummyTrainer() for _ in range(len(dataset_pairs))]
        discTrainers_src = [DummyTrainer() for _ in range(len(dataset_pairs))]

# Distributed training step
@tf.function
def distributed_train_step(distributed_dataset_pair, site_id):
    """Distributed training step for a specific site"""
    
    def train_step(batch):
        images_ref, images_src = batch
        
        # Train discriminators
        disc_ref_loss = discTrainers_ref[site_id].train(images_ref, images_src)
        disc_src_loss = discTrainers_src[site_id].train(images_src, images_ref)
        
        # Train generators
        gen_losses = genTrainers[site_id].train(images_src, images_ref)
        
        return {
            f'disc_ref_{site_id}_loss': disc_ref_loss,
            f'disc_src_{site_id}_loss': disc_src_loss,
            f'gen_losses_{site_id}': gen_losses
        }
    
    # Run training step on each replica
    per_replica_losses = strategy.run(train_step, args=(next(iter(distributed_dataset_pair)),))
    
    # Reduce losses across replicas
    reduced_losses = {}
    for key, loss in per_replica_losses.items():
        reduced_losses[key] = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
    
    return reduced_losses

# Main training function with multi-GPU support
def multi_gpu_train_step():
    """Multi-GPU training step"""
    DISC_N_BATCHS = 2
    indices_sites = np.arange(len(dataset_pairs))
    np.random.shuffle(indices_sites)
    
    results = {'genFwd_idLoss': 0}
    
    for idSite in indices_sites:
        # Distributed training for this site
        site_losses = distributed_train_step(dataset_pairs[idSite], idSite)
        
        # Accumulate results
        for key, loss in site_losses.items():
            if isinstance(loss, tf.Tensor):
                results[key] = loss.numpy()
            else:
                results[key] = loss
    
    return results

# Training execution
print("Starting multi-GPU training...")

BEST_GEN_PATH = os.path.join(DEST_DIR_PATH, 'best_genUniv.h5')
record_dict = {}
best_score = None

try:
    for epoch in range(1, N_EPOCHS + 1):
        tmp_record = {}
        epoch_start_time = tf.timestamp()
        
        for step in range(1, STEPS_PER_EPOCH + 1):
            step_start_time = tf.timestamp()
            
            # Run distributed training step
            res = multi_gpu_train_step()
            
            step_time = tf.timestamp() - step_start_time
            
            if not tmp_record:
                for key in res.keys(): 
                    tmp_record[key] = float(res[key]) if hasattr(res[key], 'numpy') else res[key]
            else:
                for key in res.keys(): 
                    val = float(res[key]) if hasattr(res[key], 'numpy') else res[key]
                    tmp_record[key] += val
            
            # Progress logging
            log = f"Epoch {epoch}/{N_EPOCHS}, Step {step}/{STEPS_PER_EPOCH} | "
            for k in sorted(res.keys()): 
                val = float(res[k]) if hasattr(res[k], 'numpy') else res[k]
                log += f"{k} = {val:.4f}, "
            log += f"Step time: {float(step_time):.2f}s"
            print(log, end=f"{' '*20}\r")
        
        epoch_time = tf.timestamp() - epoch_start_time
        
        # Record epoch results
        if not record_dict:
            for key in sorted(tmp_record.keys()):
                record_dict[key] = [tmp_record[key] / STEPS_PER_EPOCH]
        else:
            for key in tmp_record:
                record_dict[key].append(tmp_record[key] / STEPS_PER_EPOCH)
        
        # Epoch logging
        log = f"Epoch {epoch} completed in {float(epoch_time):.2f}s -> "
        for key, value in record_dict.items():
            log += f"{key}: {value[-1]:.4f}, "
        print(log + ' ' * 20)
        
        # Validation and model saving
        if epoch % EVAL_FREQ == 0:
            score = eval_model()
            print(f"Validation score: {score:.3f}")
            
            if not best_score or score > best_score:
                best_score = score
                gen_univ.save_weights(BEST_GEN_PATH)
                print('New best model saved')
        
        print()
    
    # Save final models
    print("Saving final models...")
    gen_univ.save_weights(os.path.join(DEST_DIR_PATH, 'generator_univ.h5'))
    for i in range(len(dataset_pairs)):
        gens_bwd[i].save_weights(os.path.join(DEST_DIR_PATH, f'genBwd_{i+1}.h5'))
        discs_bwd[i].save_weights(os.path.join(DEST_DIR_PATH, f'discBwd_{i+1}.h5'))
        discs_ref[i].save_weights(os.path.join(DEST_DIR_PATH, f'discRef_{i+1}.h5'))
    
    # Save training statistics
    with open(os.path.join(DEST_DIR_PATH, 'stats.json'), 'w') as f: 
        json_dump(record_dict, f)
    
    print("Multi-GPU training completed successfully!")
    
    # Print final GPU status
    from multi_gpu_config import print_gpu_utilization
    print("\nFinal GPU Status:")
    print_gpu_utilization()

except Exception as e:
    print(f"Training failed with error: {e}")
    import traceback
    traceback.print_exc()

print(f"\nResults saved to: {DEST_DIR_PATH}")