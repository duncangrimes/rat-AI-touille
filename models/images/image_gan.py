import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import set_random_seed
import numpy as np
import json
import os
import cv2 
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Conv2DTranspose, Dense, Dropout, Reshape, Flatten, Conv2D, Concatenate, Input
from tensorflow.keras.initializers import RandomNormal, RandomUniform
import matplotlib.pyplot as plt
from PIL import Image


# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load tokenized data from a JSON file
with open('data/storage/stage_3/tokenized_dishnames.json', 'r') as f:
    tokenized_data = json.load(f)

# Define a function to load images from a folder

def load_images(folder, resize=False, resized_shape = (64,64)):
    image_data = {}
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_id = os.path.splitext(filename)[0]
            img = Image.open(os.path.join(folder, filename))
            # img.resize(resized_shape).show()
            if img is not None:
                image_data[image_id] = np.array(img.resize(resized_shape))
    return image_data

# Load images from the specified folder
image_folder = 'data/processing/image_processing/images'
image_data = load_images(image_folder, resize=True)
print("TESTING DATA")
idx = 12
print(list(image_data.values())[idx].shape)

# Display the image
# cv2.imshow("Image", )

# # Wait for the user to press a key
# cv2.waitKey(0)

# # Close all windows
# cv2.destroyAllWindows()

set_random_seed(42)

# Associate tokenized data with corresponding images
tokenized_with_images = {}
for recipe_id, tokenized_recipe in tokenized_data.items():
    if recipe_id in image_data.keys():
        tokenized_with_images[recipe_id] = {
            'tokens': tokenized_recipe,
            'image': image_data[recipe_id]
        }

# Parameters
latent_dim = 100
num_tokens = 41
image_height = list(image_data.values())[0].shape[0]
image_width = list(image_data.values())[0].shape[1]
image_channels = list(image_data.values())[0].shape[2]
epochs = 500
batch_size = 256

# Generator with token input
def build_generator(latent_dim, num_tokens, image_height, image_width, image_channels):
    # FROM https://github.com/soliao/DCGAN-food-image-generator/blob/master/DCGAN_food_colab.ipynb
    
    w_init = RandomNormal(mean=0.0, stddev=0.02)
    
    # Concatenated input layer
    noise_input = Input(shape=(latent_dim,))
    tokens_input = Input(shape=(num_tokens,))
    inputs = Concatenate()([noise_input, tokens_input])

    # Base dimensions
    base_units = 1024
    base_height = 4
    base_width = 4

    # First dense layer
    x = Dense(base_units * base_height * base_width, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Reshape((base_height, base_width, base_units))(x)

    ## Conv2D-T
    x = Conv2DTranspose(512, 3, 2, padding = 'same', kernel_initializer = w_init, use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)

    ## Conv2D-T
    x = Conv2DTranspose(256, 3, 2, padding = 'same', kernel_initializer = w_init, use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)
        
    ## Conv2D-T
    x = Conv2DTranspose(128, 3, 2, 'same', kernel_initializer = w_init, use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)
        
    ## last Conv2DT (no batch norm!)
    img = Conv2DTranspose(3, 3, 2, padding='same', activation = 'tanh', kernel_initializer = w_init, use_bias = False)(x)
    
    # generator model
    model = Model(inputs = inputs, outputs = img, name = 'generator')
    print(model.summary())

    return model

def build_discriminator(image_height, image_width, image_channels):
    # weight initialization
    w_init = RandomNormal(mean=0.0, stddev=0.02)
    
    inputs = layers.Input(shape=(image_height, image_width, image_channels))

    ## Conv
    x = Conv2D(128, 3, 2, 'same', kernel_initializer = w_init)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Dropout(0.2)(x)

    ## Conv
    x = Conv2D(256, 3, 2, 'same', kernel_initializer = w_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Dropout(0.2)(x)

    ## Conv
    x = Conv2D(512, 3, 2, 'same', kernel_initializer = w_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Dropout(0.2)(x)

    ## Conv
    x = Conv2D(1024, 3, 2, 'same', kernel_initializer = w_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Dropout(0.2)(x)

    ## final layer
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    y = Dense(1, activation = 'sigmoid')(x)

    # generator model
    model = Model(inputs = inputs, outputs = y, name = 'discriminator')
    print(model.summary())

    return model


# GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gen_input_noise = layers.Input(shape=(100,))
    gen_input_tokens = layers.Input(shape=(41,))
    concatenated = Concatenate()([gen_input_noise, gen_input_tokens])
    gen_output = generator(concatenated)
    gan_output = discriminator(gen_output)
    gan = Model(inputs=[gen_input_noise, gen_input_tokens], outputs=gan_output)
    return gan


# Compile models
image_height = list(image_data.values())[0].shape[0]
image_width = list(image_data.values())[0].shape[1]
image_channels = list(image_data.values())[0].shape[2]

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

def save_plot(g_loss_list, d_loss_list, filename):
    # After training, save final training loss plot
    plt.plot(g_loss_list)
    plt.plot(d_loss_list)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['generator loss', 'discriminator loss'], loc='upper left')
    with open(filename, 'w') as _:
        plt.savefig(filename)
    plt.show()

# Training function
def train_gan(generator, discriminator, gan, dataset, tokenized_inputs, epochs, batch_size, display_epochs=5):
    dataset = np.array(dataset)
    tokenized_inputs = np.array(tokenized_inputs)
    num_samples = len(dataset)
    steps_per_epoch = num_samples // batch_size
    
    d_loss_list = np.array([])
    g_loss_list = np.array([])

    for epoch in range(epochs):
        print(f'Starting Epoch {epoch+1}/{epochs}')
        indices = np.random.permutation(num_samples)
        dataset = dataset[indices]
        tokenized_inputs = tokenized_inputs[indices]

        for step in range(steps_per_epoch):
            start = step * batch_size
            end = start + batch_size
            image_batch = dataset[start:end]
            token_batch = np.float64(tokenized_inputs[start:end])

            noise = np.float64(np.random.randn(batch_size, latent_dim))
            concatenated = Concatenate()([noise, token_batch])
            generated_images = generator.predict(concatenated)
            
            # print("GEN:",generated_images[-1])
            # print("ACTUAL:", image_batch[-1])
            
            # Smoothing labels
            real_labels = np.ones((batch_size, 1)) * 0.9
            fake_labels = np.zeros((batch_size, 1)) + 0.1

            discriminator_loss_real = discriminator.train_on_batch(image_batch, real_labels)
            discriminator_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
            generator_loss = gan.train_on_batch([noise, token_batch], real_labels)
            
            d_loss_list = np.concatenate([d_loss_list, [0.5 * (discriminator_loss_real + discriminator_loss_fake)]])
            g_loss_list = np.concatenate([g_loss_list, [generator_loss]])

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Step {step+1}/{steps_per_epoch}, Discriminator Loss: {0.5 * (discriminator_loss_real + discriminator_loss_fake)}, Generator Loss: {generator_loss}")

        if epoch % display_epochs == 0:
            
            # plot fake images
            plt.figure(figsize = (20, 4))
            for i in range(10):
                fake_img = generated_images.copy()[i].reshape(64, 64, 3)
                fake_img = (127.5*(fake_img + 1)).astype(np.uint8)
                plt.subplot(1, 10, i+1)
                plt.axis('off')
                plt.imshow(fake_img)
                with open(f"models/images/samples/{epoch}_{i}.png", 'w') as f:
                    plt.imsave(f"models/images/samples/{epoch}_{i}.png", np.array(fake_img))
            # plt.show()
            
    save_plot(g_loss_list=g_loss_list, d_loss_list=d_loss_list, filename="models/images/training_plot.png")
            
# Start training
train_gan(generator, discriminator, gan, resized_dataset, tokens_list, epochs, batch_size)



# Check anchovy
def generate_image(generator, encoded_tokens, latent_dim):
    # Generate noise vector
    noise = np.float64(np.random.randn(1, latent_dim))
    # Ensure tokens are in the correct shape for the generator input
    tokens_input = np.float64(np.array([encoded_tokens]))
    
    concatenated = Concatenate()([noise, tokens_input])
    
    # Generate image using the generator
    generated_image = generator.predict(concatenated)
    
    return generated_image

# Define the tokenized data for "fried anchovies"
encoded_tokens_fried_anchovies = [37, 4588, 72584, 12831, 449, 54384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Use the revised function
generated_image = generate_image(generator, encoded_tokens_fried_anchovies, latent_dim)

num_samples = 10


print("Generated image shape:", generated_image.shape)

# Display the generated image
print((generated_image[0] * 0.5 + 0.5))
img = (generated_image[0] * 0.5 + 0.5)
imgplot = plt.imshow(np.dot(img[...,:3], [0.33, 0.33, 0.33]), cmap='gray')
plt.show()