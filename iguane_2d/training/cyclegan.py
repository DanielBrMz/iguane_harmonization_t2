"""
CycleGAN model for multi-site harmonization
Implements IGUANe-style training with gradient accumulation
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from typing import Dict, List

from models import build_2d_generator, build_2d_discriminator
from losses import (
    discriminator_loss_mse,
    generator_loss_mse,
    cycle_consistency_loss,
    identity_loss
)


class CycleGAN2D_MultiSite:
    """
    2D CycleGAN following IGUANe training procedure
    
    Architecture:
        - 1 universal forward generator (site -> BCH)
        - N site-specific backward generators (BCH -> site)
        - 1 BCH discriminator
        - N site discriminators
    
    Training strategy:
        - Forward generator accumulates gradients from all sites
        - Updated once per step with averaged gradients
        - Backward generators updated immediately per site
        - Discriminators updated per site
    """
    
    def __init__(
        self,
        img_shape=(138, 176, 1),
        ga_embedding_dim=16,
        target_sites=None,
        use_multi_gpu=True,
        lambda_cycle=30.0,
        lambda_identity=15.0,
        gradient_clip_norm=5.0
    ):
        """
        Initialize CycleGAN model
        
        Args:
            img_shape: Input image shape (H, W, C)
            ga_embedding_dim: Dimension for GA embedding
            target_sites: List of target site names (non-reference sites)
            use_multi_gpu: Whether to use multiple GPUs
            lambda_cycle: Weight for cycle consistency loss
            lambda_identity: Weight for identity loss
            gradient_clip_norm: Maximum gradient norm for clipping
        """
        self.img_shape = img_shape
        self.ga_embedding_dim = ga_embedding_dim
        self.target_sites = target_sites or []
        self.use_multi_gpu = use_multi_gpu
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.gradient_clip_norm = gradient_clip_norm
        
        # GPU setup
        self.gpus = tf.config.list_physical_devices('GPU')
        self.n_gpus = len(self.gpus)
        self._setup_gpu_assignments()
        
        # Build models
        self._build_generators()
        self._build_discriminators()
        
        # Collapse detection
        self.collapse_counter = 0
        
        self._print_model_info()
    
    def _setup_gpu_assignments(self):
        """Setup GPU device assignments for model parallel training"""
        if self.use_multi_gpu and self.n_gpus >= 2:
            self.gpu_assignments = {
                'generators': '/GPU:0',
                'disc_BCH': '/GPU:1' if self.n_gpus >= 2 else '/GPU:0',
                'disc_sites': '/GPU:2' if self.n_gpus >= 3 else '/GPU:1'
            }
        else:
            self.gpu_assignments = {
                'generators': '/GPU:0',
                'disc_BCH': '/GPU:0',
                'disc_sites': '/GPU:0'
            }
    
    def _build_generators(self):
        """Build all generator models"""
        # Universal forward generator (site -> BCH)
        with tf.device(self.gpu_assignments['generators']):
            self.gen_site2BCH = build_2d_generator(
                self.img_shape,
                self.ga_embedding_dim,
                name='gen_site2BCH'
            )
        
        # Site-specific backward generators (BCH -> site)
        self.gen_BCH2site = {}
        for site in self.target_sites:
            with tf.device(self.gpu_assignments['generators']):
                site_name = self._sanitize_name(site)
                self.gen_BCH2site[site] = build_2d_generator(
                    self.img_shape,
                    self.ga_embedding_dim,
                    name=f'gen_BCH2{site_name}'
                )
    
    def _build_discriminators(self):
        """Build all discriminator models"""
        # BCH discriminator
        with tf.device(self.gpu_assignments['disc_BCH']):
            self.disc_BCH = build_2d_discriminator(
                self.img_shape,
                self.ga_embedding_dim,
                name='disc_BCH'
            )
        
        # Site discriminators
        self.disc_sites = {}
        for i, site in enumerate(self.target_sites):
            # Distribute site discriminators across available GPUs
            if self.use_multi_gpu and self.n_gpus >= 3:
                gpu_id = i % self.n_gpus
                device = f'/GPU:{gpu_id}'
            else:
                device = self.gpu_assignments['disc_sites']
            
            with tf.device(device):
                site_name = self._sanitize_name(site)
                self.disc_sites[site] = build_2d_discriminator(
                    self.img_shape,
                    self.ga_embedding_dim,
                    name=f'disc_{site_name}'
                )
    
    @staticmethod
    def _sanitize_name(name: str, max_len: int = 20) -> str:
        """Sanitize site name for use in model names"""
        return name.replace('_', '').replace('-', '')[:max_len]
    
    def compile(self, lr_gen=0.0002, lr_disc=0.0001, beta_1=0.5):
        """
        Compile model with optimizers
        
        Args:
            lr_gen: Generator learning rate
            lr_disc: Discriminator learning rate
            beta_1: Adam optimizer beta_1 parameter
        """
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.beta_1 = beta_1
        
        # Forward generator optimizer
        self.gen_optimizer = Adam(learning_rate=lr_gen, beta_1=beta_1)
        self.gen_optimizer.build(self.gen_site2BCH.trainable_variables)
        
        # BCH discriminator optimizer
        self.disc_BCH_optimizer = Adam(learning_rate=lr_disc, beta_1=beta_1)
        self.disc_BCH_optimizer.build(self.disc_BCH.trainable_variables)
        
        # Site-specific optimizers
        self.disc_site_optimizers = {}
        self.gen_bwd_optimizers = {}
        
        for site in self.target_sites:
            # Discriminator optimizer
            disc_opt = Adam(learning_rate=lr_disc, beta_1=beta_1)
            disc_opt.build(self.disc_sites[site].trainable_variables)
            self.disc_site_optimizers[site] = disc_opt
            
            # Backward generator optimizer
            gen_opt = Adam(learning_rate=lr_gen, beta_1=beta_1)
            gen_opt.build(self.gen_BCH2site[site].trainable_variables)
            self.gen_bwd_optimizers[site] = gen_opt
        
        self._print_compile_info()
    
    def train_step(self, site_batches: Dict) -> Dict:
        """
        Single training step with IGUANe-style gradient accumulation
        
        Optimizations:
            1. Single forward pass per generator
            2. Batch loss accumulation (keeps tensors on GPU)
            3. Non-persistent gradient tapes
            4. Minimal CPU-GPU transfers
        
        Args:
            site_batches: Dict mapping site names to (images, ga) tuples
        
        Returns:
            Dictionary of loss values
        """
        # Shuffle site order
        sites = list(self.target_sites)
        np.random.shuffle(sites)
        
        # Initialize accumulators (keep as tensors on GPU)
        accumulated_gen_fwd_grads = None
        n_sites = tf.constant(0, dtype=tf.float32)
        
        # Loss accumulators
        total_gen_loss = tf.constant(0.0)
        total_disc_BCH_loss = tf.constant(0.0)
        total_cycle_loss = tf.constant(0.0)
        total_identity_loss = tf.constant(0.0)
        total_disc_site_losses = {}
        
        # Process each site
        for site_name in sites:
            if site_name not in site_batches or 'BCH_CHD' not in site_batches:
                continue
            
            real_site, ga_site = site_batches[site_name]
            real_BCH, ga_BCH = site_batches['BCH_CHD']
            disc_site = self.disc_sites[site_name]
            gen_BCH2site_i = self.gen_BCH2site[site_name]
            
            # Update discriminators
            disc_BCH_loss, disc_site_loss = self._update_discriminators(
                real_site, ga_site, real_BCH, ga_BCH,
                disc_site, gen_BCH2site_i, site_name
            )
            
            # Update generators and accumulate gradients
            gen_loss, cycle_loss, identity_loss, fwd_grads = self._update_generators(
                real_site, ga_site, real_BCH, ga_BCH,
                disc_site, gen_BCH2site_i, site_name
            )
            
            # Accumulate forward generator gradients
            if accumulated_gen_fwd_grads is None:
                accumulated_gen_fwd_grads = fwd_grads
            else:
                accumulated_gen_fwd_grads = [
                    ag + g for ag, g in zip(accumulated_gen_fwd_grads, fwd_grads)
                ]
            
            # Accumulate losses
            total_gen_loss += gen_loss
            total_disc_BCH_loss += disc_BCH_loss
            total_cycle_loss += cycle_loss
            total_identity_loss += identity_loss
            total_disc_site_losses[site_name] = disc_site_loss
            n_sites += 1.0
        
        # Apply accumulated forward generator gradients
        if n_sites > 0 and accumulated_gen_fwd_grads is not None:
            # Average gradients
            accumulated_gen_fwd_grads = [g / n_sites for g in accumulated_gen_fwd_grads]
            
            self.gen_optimizer.apply_gradients(
                zip(accumulated_gen_fwd_grads, self.gen_site2BCH.trainable_variables)
            )
            
            # Average losses
            total_gen_loss /= n_sites
            total_disc_BCH_loss /= n_sites
            total_cycle_loss /= n_sites
            total_identity_loss /= n_sites
        
        # Collapse detection
        self._update_collapse_counter(total_disc_BCH_loss)
        
        # Prepare loss dictionary
        losses = self._prepare_loss_dict(
            total_gen_loss, total_disc_BCH_loss,
            total_cycle_loss, total_identity_loss,
            total_disc_site_losses
        )
        
        return losses
    
    def _update_discriminators(self, real_site, ga_site, real_BCH, ga_BCH,
                               disc_site, gen_BCH2site_i, site_name):
        """Update discriminators for current site"""
        # Generate fake images (no gradient tracking for speed)
        fake_BCH = self.gen_site2BCH([real_site, ga_site], training=False)
        fake_site = gen_BCH2site_i([real_BCH, ga_BCH], training=False)
        
        # Update BCH discriminator
        with tf.GradientTape() as disc_BCH_tape:
            disc_real_BCH = self.disc_BCH([real_BCH, ga_BCH], training=True)
            disc_fake_BCH = self.disc_BCH([fake_BCH, ga_site], training=True)
            disc_BCH_loss = discriminator_loss_mse(disc_real_BCH, disc_fake_BCH)
        
        disc_BCH_grads = disc_BCH_tape.gradient(
            disc_BCH_loss, self.disc_BCH.trainable_variables
        )
        disc_BCH_grads, _ = tf.clip_by_global_norm(disc_BCH_grads, self.gradient_clip_norm)
        self.disc_BCH_optimizer.apply_gradients(
            zip(disc_BCH_grads, self.disc_BCH.trainable_variables)
        )
        
        # Update site discriminator
        with tf.GradientTape() as disc_site_tape:
            disc_real_site = disc_site([real_site, ga_site], training=True)
            disc_fake_site = disc_site([fake_site, ga_BCH], training=True)
            disc_site_loss = discriminator_loss_mse(disc_real_site, disc_fake_site)
        
        disc_site_grads = disc_site_tape.gradient(
            disc_site_loss, disc_site.trainable_variables
        )
        disc_site_grads, _ = tf.clip_by_global_norm(disc_site_grads, self.gradient_clip_norm)
        self.disc_site_optimizers[site_name].apply_gradients(
            zip(disc_site_grads, disc_site.trainable_variables)
        )
        
        return disc_BCH_loss, disc_site_loss
    
    def _update_generators(self, real_site, ga_site, real_BCH, ga_BCH,
                          disc_site, gen_BCH2site_i, site_name):
        """Update generators and return forward generator gradients"""
        # Use separate tapes for forward and backward generators
        with tf.GradientTape() as fwd_tape, tf.GradientTape() as bwd_tape:
            # Forward cycle: site -> BCH -> site
            fake_BCH = self.gen_site2BCH([real_site, ga_site], training=True)
            cycled_site = gen_BCH2site_i([fake_BCH, ga_site], training=True)
            
            # Backward cycle: BCH -> site -> BCH
            fake_site = gen_BCH2site_i([real_BCH, ga_BCH], training=True)
            cycled_BCH = self.gen_site2BCH([fake_site, ga_BCH], training=True)
            
            # Identity mappings
            identity_BCH = gen_BCH2site_i([real_BCH, ga_BCH], training=True)
            identity_site = self.gen_site2BCH([real_site, ga_site], training=True)
            
            # Discriminator predictions
            disc_fake_BCH = self.disc_BCH([fake_BCH, ga_site], training=False)
            disc_fake_site = disc_site([fake_site, ga_BCH], training=False)
            
            # Compute losses
            gen_site2BCH_loss = generator_loss_mse(disc_fake_BCH)
            gen_BCH2site_loss = generator_loss_mse(disc_fake_site)
            
            cycle_loss_fwd = cycle_consistency_loss(real_site, cycled_site)
            cycle_loss_bwd = cycle_consistency_loss(real_BCH, cycled_BCH)
            cycle_loss = cycle_loss_fwd + cycle_loss_bwd
            
            identity_loss_BCH = identity_loss(real_BCH, identity_BCH)
            identity_loss_site = identity_loss(real_site, identity_site)
            identity_loss_total = identity_loss_BCH + identity_loss_site
            
            gen_loss = (gen_site2BCH_loss + gen_BCH2site_loss +
                       self.lambda_cycle * cycle_loss +
                       self.lambda_identity * self.lambda_cycle * identity_loss_total)
        
        # Compute and clip forward generator gradients (don't apply yet)
        gen_fwd_vars = self.gen_site2BCH.trainable_variables
        gen_fwd_grads = fwd_tape.gradient(gen_loss, gen_fwd_vars)
        gen_fwd_grads, _ = tf.clip_by_global_norm(gen_fwd_grads, self.gradient_clip_norm)
        
        # Update backward generator immediately
        gen_bwd_vars = gen_BCH2site_i.trainable_variables
        gen_bwd_grads = bwd_tape.gradient(gen_loss, gen_bwd_vars)
        gen_bwd_grads, _ = tf.clip_by_global_norm(gen_bwd_grads, self.gradient_clip_norm)
        
        # Apply backward generator update using provided site_name
        self.gen_bwd_optimizers[site_name].apply_gradients(
            zip(gen_bwd_grads, gen_bwd_vars)
        )
        
        return gen_loss, cycle_loss, identity_loss_total, gen_fwd_grads
    
    def _update_collapse_counter(self, disc_loss):
        """Update collapse detection counter"""
        if disc_loss < 0.01:
            self.collapse_counter += 1
        else:
            self.collapse_counter = max(0, self.collapse_counter - 1)
    
    def _prepare_loss_dict(self, gen_loss, disc_BCH_loss, cycle_loss,
                          identity_loss, disc_site_losses):
        """Convert losses to Python dict with float values"""
        losses = {
            'gen_loss': float(gen_loss.numpy()),
            'disc_BCH_loss': float(disc_BCH_loss.numpy()),
            'cycle_loss': float(cycle_loss.numpy()),
            'identity_loss': float(identity_loss.numpy()),
            'collapse_warning': self.collapse_counter
        }
        
        for site_name, loss in disc_site_losses.items():
            losses[f'disc_{site_name}_loss'] = float(loss.numpy())
        
        return losses
    
    def _print_model_info(self):
        """Print model architecture information"""
        print(f"\nBuilding CycleGAN (IGUANe-style):")
        print(f"  - 1 universal forward generator (site->BCH)")
        print(f"  - {len(self.target_sites)} site-specific backward generators")
        print(f"  - {len(self.target_sites) + 1} discriminators")
        print(f"  - Lambda cycle: {self.lambda_cycle}")
        print(f"  - Lambda identity: {self.lambda_identity}")
        print(f"  - GPUs available: {self.n_gpus}")
    
    def _print_compile_info(self):
        """Print compilation information"""
        print("\nModel compiled:")
        print(f"  Gen site->BCH parameters: {self.gen_site2BCH.count_params():,}")
        if self.target_sites:
            print(f"  Gen BCH->site parameters: "
                  f"{list(self.gen_BCH2site.values())[0].count_params():,} "
                  f"x {len(self.target_sites)}")
        print(f"  Disc BCH parameters: {self.disc_BCH.count_params():,}")
        if self.target_sites:
            print(f"  Disc per-site parameters: "
                  f"{list(self.disc_sites.values())[0].count_params():,} "
                  f"x {len(self.target_sites)}")
        print(f"  Generator LR: {self.lr_gen}")
        print(f"  Discriminator LR: {self.lr_disc}")
        print(f"  Total optimizers: {1 + len(self.gen_bwd_optimizers) + 1 + len(self.disc_site_optimizers)}")
