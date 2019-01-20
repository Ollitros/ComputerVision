import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from keras import models, layers, utils, backend, optimizers, losses, datasets


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


x_train = np.reshape(x_train, [-1, 784])
x_test = np.reshape(x_test, [-1, 784])

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

epochs = 100
batch_size = 30000


# Final prediction weights
# Output
W = tf.Variable(tf.random_normal([784, 10]))
B = tf.Variable(tf.random_normal([10]))


# Final prediction
out = tf.nn.softmax(tf.matmul(X, W) + B)

loss = tf.reduce_mean(tf.square(Y - out))

optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for epoch in range(epochs):
    for i in range(x_train.shape[0] // batch_size):
        x = x_train[(i * batch_size): (i + 1) * batch_size]
        y = y_train[(i * batch_size): (i + 1) * batch_size]

        sess.run(optimizer, feed_dict={X: x, Y: y})

        print(sess.run(loss, feed_dict={X: x, Y: y}))




