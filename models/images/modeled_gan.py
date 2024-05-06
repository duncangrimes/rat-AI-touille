import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Activation, LeakyReLU, Reshape, Flatten
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def load_images(folder, resize=True, resized_shape=(64, 64)):
    image_data = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = Image.open(os.path.join(folder, filename))
            if resize:
                img = img.resize(resized_shape)
            if img is not None:
                img = np.array(img)
                img = (img / 127.5) - 1
                image_data.append(img)
    return np.array(image_data)

image_folder = '/content/drive/MyDrive/rat-AI-touille/data/processing/image_processing/images'
image_data = load_images(image_folder)
print("Loaded images:", len(image_data))

def smoothing_true_labels(labels):
    return 1 + np.random.uniform(low=-0.3, high=0.3, size=labels.shape)

def smoothing_fake_labels(labels):
    return np.random.uniform(low=0.0, high=0.3, size=labels.shape)

#GAN From https://github.com/soliao/DCGAN-food-image-generator/blob/master/DCGAN_food_colab.ipynb


def build_generator_DC(hidden_dim=100):
    w_init = RandomNormal(mean=0.0, stddev=0.02)
    z = Input(shape=(hidden_dim,))
    x = Dense(4*4*1024, kernel_initializer=w_init)(z)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((4, 4, 1024))(x)
    x = Conv2DTranspose(512, 3, strides=2, padding='same', kernel_initializer=w_init, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(256, 3, strides=2, padding='same', kernel_initializer=w_init, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(128, 3, strides=2, padding='same', kernel_initializer=w_init, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    img = Conv2DTranspose(3, 3, strides=2, padding='same', activation='tanh', kernel_initializer=w_init, use_bias=False)(x)
    model = Model(inputs=z, outputs=img, name='generator')
    return model

def build_discriminator_DC():
    w_init = RandomNormal(mean=0.0, stddev=0.02)
    img = Input(shape=(64, 64, 3))
    x = Conv2D(128, 3, strides=2, padding='same', kernel_initializer=w_init)(img)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, 3, strides=2, padding='same', kernel_initializer=w_init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(512, 3, strides=2, padding='same', kernel_initializer=w_init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(1024, 3, strides=2, padding='same', kernel_initializer=w_init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=img, outputs=y, name='discriminator')
    return model

generator = build_generator_DC()
discriminator = build_discriminator_DC()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

def build_DCGAN(generator, discriminator, hidden_dim):
    discriminator.trainable = False
    z = Input(shape=(hidden_dim,))
    img = generator(z)
    valid = discriminator(img)
    dcgan = Model(z, valid)
    dcgan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return dcgan

dcgan = build_DCGAN(generator, discriminator, 100)

def train_DCGAN(epochs, batch_size, display_epochs, data):
    epoch_losses = []
    m = data.shape[0]
    n_batch = m // batch_size
    n_remain = m % batch_size
    for epoch in range(epochs+1):
        d_loss_avg, g_loss_avg = 0, 0
        d_acc_avg = 0
        for batch_i in range(n_batch + 1):
            sample_size = batch_size if batch_i != n_batch else n_remain
            real_images = data[batch_i * batch_size: (batch_i + 1) * batch_size] if batch_i != n_batch else data[-n_remain:]

            real_labels = smoothing_true_labels(np.ones(sample_size))
            z = np.random.uniform(0, 1, (sample_size, 100))
            fake_images = generator.predict(z)
            fake_labels = smoothing_fake_labels(np.zeros(sample_size))

            d_loss_real, d_real_acc = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake, d_fake_acc = discriminator.train_on_batch(fake_images, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_acc = 0.5 * (d_real_acc + d_fake_acc)

            g_loss = dcgan.train_on_batch(z, np.ones(sample_size))

            d_loss_avg += d_loss
            g_loss_avg += g_loss
            d_acc_avg += d_acc

        d_loss_avg /= (n_batch + 1)
        g_loss_avg /= (n_batch + 1)
        d_acc_avg /= (n_batch + 1)

        epoch_losses.append({'discriminator': d_loss_avg, 'generator': g_loss_avg})

        if epoch % display_epochs == 0:
            print(f"Epoch {epoch}: d_loss: {d_loss_avg:.4f}, g_loss: {g_loss_avg:.4f}, d_acc: {d_acc_avg:.4f}")
            plt.figure(figsize=(20, 4))
            for i in range(min(10, sample_size)):
                fake_img = (127.5 * (fake_images[i] + 1)).astype(np.uint8)
                plt.subplot(1, 10, i + 1)
                plt.axis('off')
                plt.imshow(fake_img)
            plt.show()

    return epoch_losses

def plot_losses(epoch_losses):
    plt.figure(figsize=(10, 5))
    plt.plot([x['discriminator'] for x in epoch_losses], label='Discriminator Loss')
    plt.plot([x['generator'] for x in epoch_losses], label='Generator Loss')
    plt.title('Losses over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Parameters for training
epochs = 100
batch_size = 128
display_epochs = 10

# Train the DCGAN
epoch_losses = train_DCGAN(epochs, batch_size, display_epochs, image_data)

# Plot the training losses
plot_losses(epoch_losses)


