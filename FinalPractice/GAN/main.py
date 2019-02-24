import numpy as np
from keras.datasets import mnist
from FinalPractice.GAN.model import GAN


# Load the dataset
(train_x, _), (_, _) = mnist.load_data()

# Rescale -1 to 1
train_x = train_x / 127.5 - 1.
train_x = np.expand_dims(train_x, axis=3)
input_shape = train_x.shape

model = GAN(input_shape=input_shape, latent_dim=100)
model.train(train_x, epochs=1000, batch_size=100)