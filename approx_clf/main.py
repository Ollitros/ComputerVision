import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, utils, backend, optimizers, losses, datasets
import tensorflow as tf
from keras.models import Sequential, model_from_json, Model, Input, load_model
from keras.layers import Dense, Flatten, Dropout
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_crossentropy


TRAIN = True
# Load and transform data for model
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

img_rows, img_cols = 28, 28
num_classes = 10

if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

model = models.Sequential([layers.Conv2D(32, input_shape=input_shape, kernel_size=(3, 3),
                                         padding='same', activation='relu'),
                           layers.MaxPool2D(),
                           layers.Dropout(0.25),
                           layers.Flatten(),
                           layers.Dense(128, activation='relu'),
                           layers.Dropout(0.25),
                           layers.Dense(10, activation='softmax')])
model.compile(optimizer=optimizers.SGD(lr=0.01),
              loss=losses.mean_squared_error,
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=250, epochs=50, verbose=1)

# Save model
path_to_model = 'data/model.h5'
model.save(path_to_model)
print("Saved model on disk")

path_to_model = 'data/model.h5'
loaded_model = load_model(path_to_model)
print("Loaded model from disk")

score_train = loaded_model.evaluate(x_train, y_train)
score_test = loaded_model.evaluate(x_test, y_test)
print("Test score for origin model - ", score_test, "Train score for origin model - ", score_train)











