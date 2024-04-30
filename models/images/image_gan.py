import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import json
import os
import cv2 
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load tokenized data from a JSON file
with open('/Users/blakelayman/rat-AI-touille/data/storage/stage_3/tokenized_dishnames.json', 'r') as f:
    tokenized_data = json.load(f)

# Define a function to load images from a folder
def load_images(folder):
    image_data = {}
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_id = os.path.splitext(filename)[0]
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                image_data[image_id] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image_data

# Load images from the specified folder
image_folder = '/Users/blakelayman/rat-AI-touille/data/processing/image_processing/images'
image_data = load_images(image_folder)

# Associate tokenized data with corresponding images
tokenized_with_images = {}
for recipe_id, tokenized_recipe in tokenized_data.items():
    if recipe_id in image_data:
        tokenized_with_images[recipe_id] = {
            'tokens': tokenized_recipe,
            'image': image_data[recipe_id]
        }

# Parameters
latent_dim = 100
num_tokens = 41
image_height = 370
image_width = 556
image_channels = 3
epochs = 15
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


# Define separate optimizers for the generator and the discriminator
optimizer_gen = tf.keras.optimizers.Adam(0.0004, beta_1=0.5)
optimizer_disc = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

# Compile models with these optimizers
generator.compile(optimizer=optimizer_gen, loss=cross_entropy)
discriminator.compile(optimizer=optimizer_disc, loss=cross_entropy)
gan.compile(optimizer=optimizer_gen, loss=cross_entropy)


# legacy_optimizer = tf.keras.optimizers.legacy.Adam(0.0002, beta_1=0.5)
# generator.compile(optimizer=legacy_optimizer, loss=cross_entropy)
# discriminator.compile(optimizer=legacy_optimizer, loss=cross_entropy)
# gan.compile(optimizer=legacy_optimizer, loss=cross_entropy)


# Prepare dataset
resized_dataset = [cv2.resize(img, (image_width, image_height)) * 2 / 255.0 - 1 for img in image_data.values()]  # Normalize images to [-1, 1]
tokens_list = [item['tokens'] for item in tokenized_with_images.values()]

# Ensure tokens are numpy arrays
tokens_list = np.array(tokens_list)

# Training function
def train_gan(generator, discriminator, gan, dataset, tokenized_inputs, epochs, batch_size):
    dataset = np.array(dataset)
    tokenized_inputs = np.array(tokenized_inputs)
    num_samples = len(dataset)
    steps_per_epoch = num_samples // batch_size

    for epoch in range(epochs):
        print(f'Starting Epoch {epoch+1}/{epochs}')
        indices = np.random.permutation(num_samples)
        dataset = dataset[indices]
        tokenized_inputs = tokenized_inputs[indices]

        for step in range(steps_per_epoch):
            start = step * batch_size
            end = start + batch_size
            image_batch = dataset[start:end]
            token_batch = tokenized_inputs[start:end]

            noise = np.random.randn(batch_size, latent_dim)
            generated_images = generator.predict([noise, token_batch])
            
            # Smoothing labels
            real_labels = np.ones((batch_size, 1)) * 0.9
            fake_labels = np.zeros((batch_size, 1)) + 0.1

            discriminator_loss_real = discriminator.train_on_batch(image_batch, real_labels)
            discriminator_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
            generator_loss = gan.train_on_batch([noise, token_batch], real_labels)

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Step {step+1}/{steps_per_epoch}, Discriminator Loss: {0.5 * (discriminator_loss_real + discriminator_loss_fake)}, Generator Loss: {generator_loss}")

# Start training
train_gan(generator, discriminator, gan, resized_dataset, tokens_list, epochs, batch_size)



# Check anchovy
def generate_image(generator, encoded_tokens, latent_dim):
    # Generate noise vector
    noise = np.random.randn(1, latent_dim)
    # Ensure tokens are in the correct shape for the generator input
    tokens_input = np.array([encoded_tokens])
    
    # Generate image using the generator
    generated_image = generator.predict([noise, tokens_input])
    
    return generated_image

# Define the tokenized data for "fried anchovies"
encoded_tokens_fried_anchovies = [37, 4588, 72584, 12831, 449, 54384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Use the revised function
generated_image = generate_image(generator, encoded_tokens_fried_anchovies, latent_dim)

print("Generated image shape:", generated_image.shape)

# Display the generated image
plt.imshow((generated_image[0] * 0.5 + 0.5))  # Scale image back to [0, 1] range if needed
plt.axis('off')
plt.show()