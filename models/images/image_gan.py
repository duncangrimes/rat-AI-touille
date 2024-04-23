import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import json
import os
import cv2 
import matplotlib.pyplot as plt

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
                image_data[image_id] = img
    return image_data

# Load images from the 'images' folder
image_folder = '/Users/blakelayman/rat-AI-touille/data/processing/image_processing/images'
image_data = load_images(image_folder)

# Associate tokenized data with corresponding images
tokenized_with_images = {}
for recipe_id, tokenized_recipe in tokenized_data.items():
    image_id = recipe_id  # Assuming image ID matches recipe ID
    if image_id in image_data:
        tokenized_with_images[recipe_id] = {
            'tokens': tokenized_recipe,
            'image': image_data[image_id]
        }

# Example usage
latent_dim = 100
image_height = 556  # Adjust according to your desired image size
image_width = 370   # Adjust according to your desired image size
image_channels = 3  # Assuming RGB images
epochs = 1
batch_size = 32

# Define Generator Network
def build_generator(latent_dim):
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(370 * 556 * 3, activation='relu')(inputs)  # Adjusted for the desired image size
    return Model(inputs, x)

# Define Discriminator Network
def build_discriminator():
    inputs = layers.Input(shape=(image_height, image_width, image_channels))
    x = layers.Flatten()(inputs)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

# Define GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    return Model(gan_input, gan_output)

# Define Losses
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Compile Models
generator = build_generator(latent_dim)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

generator.compile(optimizer=tf.keras.optimizers.Adam(), loss=cross_entropy)
discriminator.compile(optimizer=tf.keras.optimizers.Adam(), loss=cross_entropy)
gan.compile(optimizer=tf.keras.optimizers.Adam(), loss=cross_entropy)

# Training Loop
def train_gan(generator, discriminator, gan, dataset, epochs, batch_size):
    for epoch in range(epochs):
        for batch in dataset:
            # Train Discriminator
            noise = np.random.randn(batch_size, latent_dim)
            generated_data = generator.predict(noise)
            real_data = batch

            discriminator_loss_real = discriminator.train_on_batch(real_data, np.ones((len(real_data), 1)))
            discriminator_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))
            discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

            # Train Generator
            noise = np.random.randn(batch_size, latent_dim)
            generator_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

            print(f"Epoch {epoch+1}/{epochs}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")

# Prepare your dataset
# Assuming you have tokenized_with_images containing tokenized data and corresponding images

# Convert tokenized_with_images to a format suitable for training
dataset = [item['image'] for item in tokenized_with_images.values()]

# Convert the images to the desired size
resized_dataset = [cv2.resize(img, (image_height, image_width)) for img in dataset]

# Normalize images
resized_dataset = np.array(resized_dataset) / 255.0

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices(resized_dataset).batch(batch_size)

# Train GAN
train_gan(generator, discriminator, gan, dataset, epochs, batch_size)


# Define the tokenized data for "fried anchovies"
encoded_tokens_fried_anchovies = [37, 4588, 72584, 12831, 449, 54384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Example usage
# Define a function to generate an image based on a dish name
def generate_image(generator, encoded_tokens, latent_dim):
    # Generate noise vector
    noise = np.random.randn(1, latent_dim)
    
    # Fill the beginning of the noise vector with the encoded tokens
    noise[:, :len(encoded_tokens)] = encoded_tokens
    
    # Generate image using the generator
    generated_image = generator.predict(noise)
    
    return generated_image

# Example usage
generated_image = generate_image(generator, encoded_tokens_fried_anchovies, latent_dim)


print("Generated image shape:", generated_image.shape)


plt.imshow(generated_image[0])  # Assuming generated_image is a numpy array representing the generated image
plt.axis('off')
plt.show()
