"""
Neural network architectures for 2D CycleGAN
Includes Generator, Discriminator, and Spectral Normalization
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


class SpectralNormalization(layers.Wrapper):
    """
    Spectral normalization wrapper for layers
    Constrains discriminator Lipschitz constant to prevent mode collapse
    
    Reference: Miyato et al. "Spectral Normalization for Generative Adversarial Networks" (2018)
    """
    
    def __init__(self, layer, iteration=1, **kwargs):
        """
        Initialize spectral normalization
        
        Args:
            layer: Keras layer to wrap
            iteration: Number of power iterations
        """
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.iteration = iteration
    
    def build(self, input_shape):
        """Build the layer and create spectral norm weights"""
        self.layer.build(input_shape)
        
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        
        # Create u vector for power iteration
        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name='sn_u',
            dtype=tf.float32
        )
        
        super(SpectralNormalization, self).build()
    
    def call(self, inputs, training=None):
        """Apply spectral normalization and call wrapped layer"""
        self.update_weights()
        output = self.layer(inputs)
        return output
    
    def update_weights(self):
        """Update weights using power iteration method"""
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        
        u_hat = self.u
        v_hat = None
        
        # Power iteration
        for _ in range(self.iteration):
            # v = u^T * W
            v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
            v_hat = v_ / (tf.norm(v_) + 1e-12)
            
            # u = v * W
            u_ = tf.matmul(v_hat, w_reshaped)
            u_hat = u_ / (tf.norm(u_) + 1e-12)
        
        # Compute spectral norm (largest singular value)
        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        
        # Update u and normalize weights
        self.u.assign(u_hat)
        self.layer.kernel.assign(self.w / sigma)


def build_2d_generator(
    input_shape=(138, 176, 1),
    ga_embedding_dim=16,
    name='generator'
) -> Model:
    """
    Build 2D U-Net Generator with Gestational Age Conditioning
    
    Architecture:
        - Encoder: 3 downsampling blocks (32, 64, 128 filters)
        - Bottleneck: 256 filters with GA injection
        - Decoder: 3 upsampling blocks with skip connections
        - Residual learning with conservative scaling
    
    Anti-collapse features:
        - Residual connections scaled by 0.2
        - Batch normalization in decoder
        - Tanh activation for bidirectional corrections
        - Shape correction for odd dimensions
        - GA dropout for regularization
    
    Args:
        input_shape: Input image shape (H, W, C)
        ga_embedding_dim: Dimension of GA embedding
        name: Model name
    
    Returns:
        Keras Model with inputs [image, ga] and output [harmonized_image]
    """
    # Inputs
    img_input = layers.Input(shape=input_shape, name='image_input')
    ga_input = layers.Input(shape=(1,), name='ga_input')
    
    # GA embedding with dropout for regularization
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu', name='ga_dense1')(ga_input)
    ga_embedding = layers.Dropout(0.2, name='ga_dropout')(ga_embedding)
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu', name='ga_dense2')(ga_embedding)
    
    # ======== ENCODER ========
    # Block 1: 32 filters
    x = layers.Conv2D(32, 3, padding='same', name='enc1_conv1')(img_input)
    x = layers.LeakyReLU(0.2, name='enc1_act1')(x)
    x = layers.Conv2D(32, 3, padding='same', name='enc1_conv2')(x)
    x = layers.LeakyReLU(0.2, name='enc1_act2')(x)
    skip1 = x
    x = layers.MaxPooling2D(2, name='enc1_pool')(x)
    
    # Block 2: 64 filters
    x = layers.Conv2D(64, 3, padding='same', name='enc2_conv1')(x)
    x = layers.LeakyReLU(0.2, name='enc2_act1')(x)
    x = layers.Conv2D(64, 3, padding='same', name='enc2_conv2')(x)
    x = layers.LeakyReLU(0.2, name='enc2_act2')(x)
    skip2 = x
    x = layers.MaxPooling2D(2, name='enc2_pool')(x)
    
    # Block 3: 128 filters
    x = layers.Conv2D(128, 3, padding='same', name='enc3_conv1')(x)
    x = layers.LeakyReLU(0.2, name='enc3_act1')(x)
    x = layers.Conv2D(128, 3, padding='same', name='enc3_conv2')(x)
    x = layers.LeakyReLU(0.2, name='enc3_act2')(x)
    skip3 = x
    x = layers.MaxPooling2D(2, name='enc3_pool')(x)
    
    # ======== BOTTLENECK ========
    x = layers.Conv2D(256, 3, padding='same', name='bottleneck_conv1')(x)
    x = layers.LeakyReLU(0.2, name='bottleneck_act1')(x)
    x = layers.Conv2D(256, 3, padding='same', name='bottleneck_conv2')(x)
    x = layers.LeakyReLU(0.2, name='bottleneck_act2')(x)
    
    # Inject GA embedding spatially
    ga_spatial = layers.RepeatVector(x.shape[1] * x.shape[2], name='ga_repeat')(ga_embedding)
    ga_spatial = layers.Reshape((x.shape[1], x.shape[2], ga_embedding_dim), name='ga_reshape')(ga_spatial)
    x = layers.Concatenate(name='bottleneck_ga_concat')([x, ga_spatial])
    
    # ======== DECODER ========
    # Block 1: 128 filters
    x = layers.UpSampling2D(2, interpolation='bilinear', name='dec1_upsample')(x)
    # Shape correction for odd dimensions
    if x.shape[1] != skip3.shape[1] or x.shape[2] != skip3.shape[2]:
        x = layers.Resizing(skip3.shape[1], skip3.shape[2], name='dec1_resize')(x)
    x = layers.Concatenate(name='dec1_concat')([x, skip3])
    x = layers.Conv2D(128, 3, padding='same', name='dec1_conv1')(x)
    x = layers.BatchNormalization(name='dec1_bn1')(x)
    x = layers.LeakyReLU(0.2, name='dec1_act1')(x)
    x = layers.Conv2D(128, 3, padding='same', name='dec1_conv2')(x)
    x = layers.BatchNormalization(name='dec1_bn2')(x)
    x = layers.LeakyReLU(0.2, name='dec1_act2')(x)
    
    # Block 2: 64 filters
    x = layers.UpSampling2D(2, interpolation='bilinear', name='dec2_upsample')(x)
    if x.shape[1] != skip2.shape[1] or x.shape[2] != skip2.shape[2]:
        x = layers.Resizing(skip2.shape[1], skip2.shape[2], name='dec2_resize')(x)
    x = layers.Concatenate(name='dec2_concat')([x, skip2])
    x = layers.Conv2D(64, 3, padding='same', name='dec2_conv1')(x)
    x = layers.BatchNormalization(name='dec2_bn1')(x)
    x = layers.LeakyReLU(0.2, name='dec2_act1')(x)
    x = layers.Conv2D(64, 3, padding='same', name='dec2_conv2')(x)
    x = layers.BatchNormalization(name='dec2_bn2')(x)
    x = layers.LeakyReLU(0.2, name='dec2_act2')(x)
    
    # Block 3: 32 filters
    x = layers.UpSampling2D(2, interpolation='bilinear', name='dec3_upsample')(x)
    if x.shape[1] != skip1.shape[1] or x.shape[2] != skip1.shape[2]:
        x = layers.Resizing(skip1.shape[1], skip1.shape[2], name='dec3_resize')(x)
    x = layers.Concatenate(name='dec3_concat')([x, skip1])
    x = layers.Conv2D(32, 3, padding='same', name='dec3_conv1')(x)
    x = layers.BatchNormalization(name='dec3_bn1')(x)
    x = layers.LeakyReLU(0.2, name='dec3_act1')(x)
    x = layers.Conv2D(32, 3, padding='same', name='dec3_conv2')(x)
    x = layers.BatchNormalization(name='dec3_bn2')(x)
    x = layers.LeakyReLU(0.2, name='dec3_act2')(x)
    
    # Final shape guarantee
    if x.shape[1] != input_shape[0] or x.shape[2] != input_shape[1]:
        x = layers.Resizing(input_shape[0], input_shape[1], name='final_resize')(x)
    
    # ======== RESIDUAL OUTPUT ========
    # Predict residual correction instead of full image
    residual = layers.Conv2D(1, 1, padding='same', activation='tanh', name='residual_conv')(x)
    residual = layers.Lambda(lambda x: x * 0.2, name='residual_scale')(residual)  # Conservative scaling
    
    # Add residual to input
    output = layers.Add(name='residual_add')([img_input, residual])
    output = layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0), name='output_clip')(output)
    
    model = Model(inputs=[img_input, ga_input], outputs=output, name=name)
    
    return model


def build_2d_discriminator(
    input_shape=(138, 176, 1),
    ga_embedding_dim=16,
    use_spectral_norm=True,
    name='discriminator'
) -> Model:
    """
    Build 2D PatchGAN Discriminator with Gestational Age Conditioning
    
    Architecture:
        - 4 convolutional blocks with stride 2 or 1
        - Spectral normalization for stability
        - High dropout (0.4) for regularization
        - GA injection before final layer
    
    Anti-collapse features:
        - Spectral normalization on all conv layers
        - High dropout (0.4) prevents overfitting
        - No batch norm (conflicts with spectral norm)
        - LeakyReLU activation
    
    Args:
        input_shape: Input image shape (H, W, C)
        ga_embedding_dim: Dimension of GA embedding
        use_spectral_norm: Whether to apply spectral normalization
        name: Model name
    
    Returns:
        Keras Model with inputs [image, ga] and output [logits]
    """
    # Inputs
    img_input = layers.Input(shape=input_shape, name='image_input')
    ga_input = layers.Input(shape=(1,), name='ga_input')
    
    # GA embedding (no dropout in discriminator)
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu', name='ga_dense1')(ga_input)
    ga_embedding = layers.Dense(ga_embedding_dim, activation='relu', name='ga_dense2')(ga_embedding)
    
    # ======== DISCRIMINATOR PATH ========
    # Block 1: 64 filters, stride 2
    conv_layer = layers.Conv2D(64, 4, strides=2, padding='same', name='disc_conv1')
    if use_spectral_norm:
        conv_layer = SpectralNormalization(conv_layer, name='disc_sn1')
    x = conv_layer(img_input)
    x = layers.LeakyReLU(0.2, name='disc_act1')(x)
    x = layers.Dropout(0.4, name='disc_drop1')(x)
    
    # Block 2: 128 filters, stride 2
    conv_layer = layers.Conv2D(128, 4, strides=2, padding='same', name='disc_conv2')
    if use_spectral_norm:
        conv_layer = SpectralNormalization(conv_layer, name='disc_sn2')
    x = conv_layer(x)
    x = layers.LeakyReLU(0.2, name='disc_act2')(x)
    x = layers.Dropout(0.4, name='disc_drop2')(x)
    
    # Block 3: 256 filters, stride 2
    conv_layer = layers.Conv2D(256, 4, strides=2, padding='same', name='disc_conv3')
    if use_spectral_norm:
        conv_layer = SpectralNormalization(conv_layer, name='disc_sn3')
    x = conv_layer(x)
    x = layers.LeakyReLU(0.2, name='disc_act3')(x)
    x = layers.Dropout(0.4, name='disc_drop3')(x)
    
    # Block 4: 512 filters, stride 1
    conv_layer = layers.Conv2D(512, 4, strides=1, padding='same', name='disc_conv4')
    if use_spectral_norm:
        conv_layer = SpectralNormalization(conv_layer, name='disc_sn4')
    x = conv_layer(x)
    x = layers.LeakyReLU(0.2, name='disc_act4')(x)
    
    # ======== GA INJECTION ========
    ga_spatial = layers.RepeatVector(x.shape[1] * x.shape[2], name='ga_repeat')(ga_embedding)
    ga_spatial = layers.Reshape((x.shape[1], x.shape[2], ga_embedding_dim), name='ga_reshape')(ga_spatial)
    x = layers.Concatenate(name='ga_concat')([x, ga_spatial])
    
    # ======== OUTPUT ========
    # Output logits (no activation)
    output = layers.Conv2D(1, 4, strides=1, padding='same', name='output_conv')(x)
    
    model = Model(inputs=[img_input, ga_input], outputs=output, name=name)
    
    return model
