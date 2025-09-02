"""
Loss functions for CycleGAN training
Includes adversarial, cycle consistency, and identity losses
"""

import tensorflow as tf


def cycle_consistency_loss(real_img: tf.Tensor, cycled_img: tf.Tensor) -> tf.Tensor:
    """
    Compute L1 cycle consistency loss
    
    Ensures that image->domain_B->domain_A recovers the original image
    
    Args:
        real_img: Original image
        cycled_img: Image after forward and backward transformation
    
    Returns:
        Scalar loss tensor
    """
    return tf.reduce_mean(tf.abs(real_img - cycled_img))


def identity_loss(real_img: tf.Tensor, same_img: tf.Tensor) -> tf.Tensor:
    """
    Compute L1 identity loss
    
    Ensures that generator preserves images from its target domain
    (e.g., BCH generator should keep BCH images unchanged)
    
    Args:
        real_img: Original image from target domain
        same_img: Image after passing through target domain generator
    
    Returns:
        Scalar loss tensor
    """
    return tf.reduce_mean(tf.abs(real_img - same_img))


def discriminator_loss_smooth(
    real_output: tf.Tensor,
    fake_output: tf.Tensor,
    label_smoothing: float = 0.1
) -> tf.Tensor:
    """
    Compute discriminator loss with label smoothing
    
    Label smoothing prevents discriminator from becoming overconfident,
    which helps stabilize training and prevent generator collapse
    
    Args:
        real_output: Discriminator output for real images
        fake_output: Discriminator output for fake images
        label_smoothing: Amount of label smoothing (0.0-1.0)
    
    Returns:
        Scalar loss tensor
    """
    # Real images should be classified as ~0.9 (not 1.0) due to smoothing
    real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(real_output) * (1.0 - label_smoothing),
            logits=real_output
        )
    )
    
    # Fake images should be classified as ~0.1 (not 0.0) due to smoothing
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(fake_output) + label_smoothing,
            logits=fake_output
        )
    )
    
    return real_loss + fake_loss


def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """
    Compute generator adversarial loss
    
    Generator tries to fool discriminator by making it classify
    fake images as real
    
    Args:
        fake_output: Discriminator output for generated images
    
    Returns:
        Scalar loss tensor
    """
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(fake_output),
            logits=fake_output
        )
    )


def compute_generator_total_loss(
    disc_fake_BCH: tf.Tensor,
    disc_fake_site: tf.Tensor,
    real_site: tf.Tensor,
    cycled_site: tf.Tensor,
    real_BCH: tf.Tensor,
    cycled_BCH: tf.Tensor,
    identity_BCH: tf.Tensor,
    identity_site: tf.Tensor,
    lambda_cycle: float = 30.0,
    lambda_identity: float = 15.0
) -> tuple:
    """
    Compute total generator loss with all components
    
    Args:
        disc_fake_BCH: Discriminator output for fake BCH images
        disc_fake_site: Discriminator output for fake site images
        real_site: Real site images
        cycled_site: Site images after site->BCH->site cycle
        real_BCH: Real BCH images
        cycled_BCH: BCH images after BCH->site->BCH cycle
        identity_BCH: BCH images after BCH->BCH identity mapping
        identity_site: Site images after site->site identity mapping
        lambda_cycle: Weight for cycle consistency loss
        lambda_identity: Weight for identity loss
    
    Returns:
        Tuple of (total_loss, adversarial_loss, cycle_loss, identity_loss)
    """
    # Adversarial losses
    gen_site2BCH_loss = generator_loss(disc_fake_BCH)
    gen_BCH2site_loss = generator_loss(disc_fake_site)
    adversarial_loss = gen_site2BCH_loss + gen_BCH2site_loss
    
    # Cycle consistency losses
    cycle_loss_forward = cycle_consistency_loss(real_site, cycled_site)
    cycle_loss_backward = cycle_consistency_loss(real_BCH, cycled_BCH)
    cycle_loss_total = cycle_loss_forward + cycle_loss_backward
    
    # Identity losses
    identity_loss_BCH = identity_loss(real_BCH, identity_BCH)
    identity_loss_site = identity_loss(real_site, identity_site)
    identity_loss_total = identity_loss_BCH + identity_loss_site
    
    # Total loss
    total_loss = (adversarial_loss +
                  lambda_cycle * cycle_loss_total +
                  lambda_identity * identity_loss_total)
    
    return total_loss, adversarial_loss, cycle_loss_total, identity_loss_total


def check_for_nan_loss(loss_dict: dict) -> bool:
    """
    Check if any loss values are NaN
    
    Args:
        loss_dict: Dictionary of loss names to loss values
    
    Returns:
        True if any NaN detected, False otherwise
    """
    import numpy as np
    for name, value in loss_dict.items():
        if isinstance(value, (int, float)):
            if np.isnan(value):
                print(f"  NaN detected in {name}")
                return True
        elif hasattr(value, 'numpy'):
            if np.isnan(value.numpy()):
                print(f"  NaN detected in {name}")
                return True
    return False
