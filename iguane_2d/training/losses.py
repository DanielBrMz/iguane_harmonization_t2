"""
Loss functions for CycleGAN training
"""

import tensorflow as tf


def discriminator_loss_mse(real_output, fake_output):
    """
    Discriminator loss using Mean Squared Error
    Matches IGUANe original implementation
    """
    real_loss = tf.reduce_mean(tf.square(real_output - 1.0))
    fake_loss = tf.reduce_mean(tf.square(fake_output))
    return real_loss + fake_loss


def generator_loss_mse(fake_output):
    """
    Generator adversarial loss using Mean Squared Error
    Matches IGUANe original implementation
    """
    return tf.reduce_mean(tf.square(fake_output - 1.0))


def cycle_consistency_loss(real_img, cycled_img):
    """L1 loss for cycle consistency"""
    return tf.reduce_mean(tf.abs(real_img - cycled_img))


def identity_loss(real_img, same_img):
    """L1 loss for identity mapping"""
    return tf.reduce_mean(tf.abs(real_img - same_img))


def compute_generator_total_loss(gen_fwd_adv_loss, gen_bwd_adv_loss, 
                                 cycle_loss, identity_loss_total,
                                 lambda_cycle=30.0, lambda_identity=0.5):
    """
    Compute total generator loss with proper weighting
    Matches IGUANe formula: genLoss = advLoss + λ_cyc*cycleLoss + 0.5*λ_cyc*idLoss
    """
    total_loss = (
        gen_fwd_adv_loss + 
        gen_bwd_adv_loss +
        lambda_cycle * cycle_loss +
        lambda_identity * lambda_cycle * identity_loss_total
    )
    return total_loss


def check_for_nan_loss(*losses):
    """Check if any loss is NaN"""
    # Handle both dictionary and individual loss values
    if len(losses) == 1 and isinstance(losses[0], dict):
        # Dictionary of losses passed
        loss_dict = losses[0]
        for name, value in loss_dict.items():
            if isinstance(value, (int, float)):
                if value != value:  # NaN check for Python numbers
                    return True
            else:
                # TensorFlow tensor
                if tf.math.reduce_any(tf.math.is_nan(value)):
                    return True
    else:
        # Individual loss tensors passed
        for loss in losses:
            if tf.math.is_nan(loss):
                return True
    return False
