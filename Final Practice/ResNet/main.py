import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10

y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
input_shape = (32, 32, 3)

model = ResNet(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, batch_size=1000, epochs=100)

score = model.evaluate(x_test, y_test)
print(score)