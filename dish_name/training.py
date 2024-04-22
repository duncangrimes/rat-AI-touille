import pandas as pd
import torch
# from transformers import BertModel, BertTokenizer
import tiktoken
import json
import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Input, LSTM, Dense, Concatenate
from sklearn.model_selection import train_test_split

# simple encoding function
def simple_encode(name):
    encoding = tiktoken.get_encoding("cl100k_base")
    return encoding.encode(name)

# Load and preprocess dataset
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    recipes = pd.DataFrame(data)
    # Tokenize titles
    embedded_titles = recipes['title'].map(lambda title : simple_encode(title))
    return embedded_titles

def preprocess_data(data):
    data = list(data)
    max_len = len(max(data, key = len))
    for vec in data:
        vec[:] = vec + [0] * (max_len - len(vec))
    return np.array(data)


# Create the GAN model by combining the generator and discriminator
def build_gan2(input_dim, output_dim):
    # Create the generator model
    input_data = Input(shape=(input_dim))
    generator_lstm = LSTM(256, return_sequences=True)(input_data)
    generator_output = Dense(output_dim, activation='sigmoid')(generator_lstm)
    generator = Model(inputs=input_data, outputs=generator_output)
    
    # Create the discriminator model
    discriminator_input = Input(shape=(output_dim))
    discriminator_lstm = LSTM(256, return_sequences=True)(discriminator_input)
    discriminator_output = Dense(1, activation='sigmoid')(discriminator_lstm)
    discriminator = Model(inputs=discriminator_input, outputs=discriminator_output)
    
    combined_input = Concatenate()([input_data, generator_output])
    combined_output = discriminator(combined_input)
    gan = Model(inputs=input_data,outputs=combined_output)
    return gan



# # Define the generator model
# def build_generator(latent_dim):
#     model = models.Sequential()
#     model.add(layers.Dense(64, input_dim=latent_dim))
#     model.add(layers.LeakyReLU(alpha=0.2))
#     model.add(layers.Dense(48))
#     model.add(layers.LeakyReLU(alpha=0.2))
#     model.add(layers.Dense(24))
#     model.add(layers.LeakyReLU(alpha=0.2))
#     model.add(layers.Dense(41, activation='softmax'))
#     print(model.summary())
#     return model

# # Define the discriminator model
# def build_discriminator():
#     model = models.Sequential()
#     model.add(layers.Dense(512, input_shape=(41,)))
#     model.add(layers.LeakyReLU(alpha=0.2))
#     model.add(layers.Dropout(0.3))
#     model.add(layers.Dense(256))
#     model.add(layers.LeakyReLU(alpha=0.2))
#     model.add(layers.Dropout(0.3))
#     model.add(layers.Dense(128))
#     model.add(layers.LeakyReLU(alpha=0.2))
#     model.add(layers.Dropout(0.3))
#     model.add(layers.Dense(1, activation='sigmoid'))
#     print(model.summary())
#     return model

# # Define the GAN model
# def build_gan(generator, discriminator):
#     discriminator.trainable = False
#     model = models.Sequential()
#     model.add(generator)
#     model.add(discriminator)
#     return model

# # Train GAN
# def train_gan(generator, discriminator, gan, data, latent_dim, epochs, batch_size):
#     for epoch in range(epochs):
#         idx = np.random.randint(0, data.shape[0], batch_size)
#         real_dishes = data[idx]

#         noise = np.random.normal(0, 1, (batch_size, latent_dim))
#         fake_dishes = generator.predict(noise)

#         real_labels = np.ones((batch_size, 1))
#         fake_labels = np.zeros((batch_size, 1))

#         d_loss_real = discriminator.train_on_batch(real_dishes, real_labels)
#         d_loss_fake = discriminator.train_on_batch(fake_dishes, fake_labels)

#         d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

#         noise = np.random.normal(0, 1, (batch_size, latent_dim))
#         gen_labels = np.ones((batch_size, 1))

#         g_loss = gan.train_on_batch(noise, gen_labels)

#         print(f"Epoch {epoch+1}, D Loss: {d_loss}, G Loss: {g_loss}")

# Main function
def main():
    file_path = '/Users/carlymiles/Desktop/EECS/DIS-NN/rat-AI-touille/data/processing/stage_1.5/recipes_df.json'
    
    # Parameters
    latent_dim = 100
    epochs = 10000
    batch_size = 64

    # Load and preprocess data
    data = load_data(file_path)
    data = preprocess_data(data)
    
    # OPTION 2: --------------
    input_dim = data.shape[-1]
    output_dim = input_dim
    # Compile the GAN model
    gan = build_gan2(input_dim=input_dim, output_dim=output_dim)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    # Train the GAN model on the dataset of real dish names
    gan.fit(data, np.ones((len(data), 1)), epochs=10)

    # Use the trained GAN model to generate 100 new dish names from noise
    noise = np.random.normal(0, 1, (100, latent_dim))
    generated = gan.predict(noise)
    print(generated)

    
    # OPTION 1: --------------
    # # Build and compile models
    # generator = build_generator(latent_dim)
    # discriminator = build_discriminator()
    # gan = build_gan(generator, discriminator)

    # discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # gan.compile(optimizer='adam', loss='binary_crossentropy')

    # # Train GAN
    # train_gan(generator, discriminator, gan, data, latent_dim, epochs, batch_size)

if __name__ == "__main__":
    main()
