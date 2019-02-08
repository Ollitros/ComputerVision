import tensorflow as tf
import time
from tensorflow.keras.datasets import cifar10
from FinalPractice.SENet.model import SEResNet

start_time = time.time()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10
batch_size = 1000
epochs = 5

y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
input_shape = (32, 32, 3)

model = SEResNet(input_shape=input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

score = model.evaluate(x_test, y_test)
print(score)

print("\n\n\nThe program has been finished for --- %s seconds ---" % (time.time() - start_time))