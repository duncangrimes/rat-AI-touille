import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, losses
from keras.layers import Dense, Flatten, Reshape
from keras.models import Sequential, Model
from keras.optimizers import Adam

dishname_token_count = 41
ingredient_token_count = 100
equipment_token_count = 49
max_ingredient_token_size = 100163

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
            ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(100, activation='sigmoid'),
            ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded