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
input_shape = (1, img_rows, img_cols)

x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# define and fit the final model
inputs = Input(shape=input_shape)
dense1 = Dense(48)(inputs)
dropout = Dropout(0.3)(dense1)
dense2 = Dense(24)(dropout)
dropout1 = Dropout(0.3)(dense2)
flat = Flatten()(dropout1)
dense3 = Dense(1)(flat)
regression = Model(inputs=inputs, outputs=dense3)
regression.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

rg0 = load_model('data/model0.h5')
rg1 = load_model('data/model1.h5')
rg2 = load_model('data/model2.h5')
rg3 = load_model('data/model3.h5')
rg4 = load_model('data/model4.h5')
rg5 = load_model('data/model5.h5')
rg6 = load_model('data/model6.h5')
rg7 = load_model('data/model7.h5')
rg8 = load_model('data/model8.h5')
rg9 = load_model('data/model9.h5')

pr0 = rg0.predict(x_train)
pr1 = rg1.predict(x_train)
pr2 = rg2.predict(x_train)
pr3 = rg3.predict(x_train)
pr4 = rg4.predict(x_train)
pr5 = rg5.predict(x_train)
pr6 = rg6.predict(x_train)
pr7 = rg7.predict(x_train)
pr8 = rg8.predict(x_train)
pr9 = rg9.predict(x_train)

array = []
for i in range(len(x_train)):
    example = np.array([pr0[i], pr1[i], pr2[i], pr3[i], pr4[i], pr5[i], pr6[i], pr7[i], pr8[i], pr9[i]])
    array.append([example])
x_train = np.array(array)
x_train = np.reshape(x_train, (len(x_train), 1, 10))
print(x_train.shape)

array = []
for i in range(len(x_test)):
    example = np.array([pr0[i], pr1[i], pr2[i], pr3[i], pr4[i], pr5[i], pr6[i], pr7[i], pr8[i], pr9[i]])
    array.append([example])
x_test = np.array(array)
x_test = np.reshape(x_test, (len(x_test), 1, 10))
print(x_test.shape)

input_shape = (1, 10)
# define and fit the final model
inputs = Input(shape=input_shape)
dense1 = Dense(256, activation='sigmoid')(inputs)
dropout = Dropout(0.25)(dense1)
dense2 = Dense(64, activation='sigmoid')(dropout)
dropout1 = Dropout(0.25)(dense2)
flat = Flatten()(dropout1)
dense3 = Dense(10, activation='softmax')(flat)

# TRAINING
if TRAIN:
    model = Model(inputs=inputs, outputs=dense3)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1000, batch_size=250, verbose=1)

    # Save model
    path_to_model = 'data/modelsss.h5'
    model.save(path_to_model)
    print("Saved model on disk")
# END OF TRAINING

path_to_model = 'data/modelsss.h5'
loaded_model = load_model(path_to_model)
print("Loaded model from disk")


score_train = loaded_model.evaluate(x_train, y_train)
score_test = loaded_model.evaluate(x_test, y_test)
print("Test score for origin model - ", score_test, "Train score for origin model - ", score_train)










