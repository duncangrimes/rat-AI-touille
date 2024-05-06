import json
import os
import cv2 
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt

# Suppress TensorFlow logging
 @@ -47,35 +48,64 @@ def load_images(folder):
batch_size = 256

# Generator with token input
def build_generator(latent_dim, num_tokens, image_height, image_width, image_channels):
    noise_input = layers.Input(shape=(latent_dim,))
    tokens_input = layers.Input(shape=(num_tokens,))
    x = layers.Concatenate()([noise_input, tokens_input])

    # Base dimensions
    base_units = 128
    base_height = image_height // 4
    base_width = image_width // 4

    # First dense layer
    x = layers.Dense(base_units * base_height * base_width, activation='relu')(x)
    x = layers.Reshape((base_height, base_width, base_units))(x)

    # Upsample to the target dimensions
    x = layers.Conv2DTranspose(base_units // 2, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(image_channels, (5, 5), strides=(2, 2), padding='same', activation='tanh')(x)

    return Model(inputs=[noise_input, tokens_input], outputs=x)

def build_discriminator(image_height, image_width, image_channels):
    inputs = layers.Input(shape=(image_height, image_width, image_channels))

    # Corrected usage of LeakyReLU with alpha parameter
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return Model(inputs=inputs, outputs=outputs)





# GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gen_input_noise = layers.Input(shape=(100,))
    gen_input_tokens = layers.Input(shape=(41,))
    gen_output = generator([gen_input_noise, gen_input_tokens])
    gan_output = discriminator(gen_output)
    gan = Model(inputs=[gen_input_noise, gen_input_tokens], outputs=gan_output)
    return gan


# Compile models
image_height = 368
image_width = 552
image_channels = 3

generator = build_generator(latent_dim=100, num_tokens=41, image_height=image_height, image_width=image_width, image_channels=image_channels)
discriminator = build_discriminator(image_height=image_height, image_width=image_width, image_channels=image_channels)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
gan = build_gan(generator, discriminator)
