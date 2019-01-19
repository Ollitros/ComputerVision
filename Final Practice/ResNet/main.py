import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10

y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
inut_shape = (32, 32, 3)


model = tf.keras.models.Sequential()
model.add(Conv2D(256, (3, 3), input_shape=inut_shape, activation='relu', padding='same'))
model.add(MaxPool2D())

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D())

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D())


model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=1000, epochs=10)

score = model.evaluate(x_test, y_test)
print(score)