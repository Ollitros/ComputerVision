import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, utils, backend, optimizers, losses, datasets
import tensorflow as tf
from keras.models import Sequential, model_from_json, Model, Input, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Nadam
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from keras.utils.np_utils import to_categorical


TRAIN = False
# Load and transform data for model
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()


img_rows, img_cols = 28, 28
num_classes = 10
input_shape = (1, img_rows, img_cols)

train_x = dict()
train_y = dict()
test_x = dict()
test_y = dict()

for i in range(10):
    indices_train = np.argwhere(y_train == i)
    train_x[i] = x_train[indices_train]
    train_y[i] = y_train[indices_train]

    indices_test = np.argwhere(y_test == i)
    test_x[i] = x_test[indices_test]
    test_y[i] = y_test[indices_test]


# define and fit the final model
inputs = Input(shape=input_shape)
dense1 = Dense(64)(inputs)
dropout = Dropout(0.5)(dense1)
dense2 = Dense(32)(dropout)
dropout1 = Dropout(0.5)(dense2)
flat = Flatten()(dropout1)
dense3 = Dense(1)(flat)
model = Model(inputs=inputs, outputs=dense3)
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(optimizer=nadam, loss='mean_squared_error', metrics=['accuracy'])

i = 0
# TRAINING
if TRAIN:
    model.fit(train_x[i], train_y[i], epochs=100, batch_size=500, verbose=1)

    # Save model
    path_to_model = '../data/approxes/model{i}.h5'.format(i=i)
    model.save(path_to_model)
    print("Saved model on disk")
# END OF TRAINING

path_to_model = '../data/approxes/model{i}.h5'.format(i=i)
loaded_model = load_model(path_to_model)
print("Loaded model from disk")

prediction = model.predict(train_x[i])
print(prediction[:10])
prediction = model.predict(test_x[i])
print(prediction[:10])
prediction = model.predict(test_x[i+1])
print(prediction[:10])
print(round(prediction[:10] * 1000))












