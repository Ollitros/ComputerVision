import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10


class ResNet():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Preliminary convolution weights
        self.w_conv = tf.Variable(tf.truncated_normal((3, 3, 3, 128), stddev=0.1))
        self.b_conv = tf.Variable(tf.constant(0.1, shape=[128]))

        # Final prediction weights
        # Full connect
        self.W_F = tf.Variable(tf.random_normal([128, 512]))
        self.B_F = tf.Variable(tf.random_normal([512]))

        # Output
        self.W = tf.Variable(tf.random_normal([512, self.output_shape]))
        self.B = tf.Variable(tf.random_normal([self.output_shape]))

    def convolve_block(self, inputs, filters):

        conv1 = tf.Variable(tf.truncated_normal((filters[0], filters[1], 128, 128), stddev=0.1))  # Conv1
        conv2 = tf.Variable(tf.truncated_normal((filters[0], filters[1], 128, 128), stddev=0.1))  # Conv2
        bias1 = tf.Variable(tf.constant(0.1, shape=[128]))  # Bias1
        bias2 = tf.Variable(tf.constant(0.1, shape=[128]))  # Bias2
        conv_shortcut = tf.Variable(tf.truncated_normal((1, 1, 128, 128), stddev=0.1))  # Shortcut conv
        bias_shortcut = tf.Variable(tf.constant(0.1, shape=[128]))  # Shortcut bias

        # Convolution
        norm = tf.nn.batch_normalization(inputs, 1, 1, 0, 1, 0.5)
        convolve_1 = tf.nn.conv2d(norm, conv1, strides=[1, 1, 1, 1], padding='SAME') + bias1
        dropout = tf.nn.dropout(convolve_1, 0.25)
        convolve_2 = tf.nn.conv2d(dropout, conv2, strides=[1, 1, 1, 1], padding='SAME') + bias2

        # Shortcut
        shortcut = tf.nn.conv2d(inputs, conv_shortcut, strides=[1, 1, 1, 1], padding='SAME') + bias_shortcut
        shortcut = tf.nn.batch_normalization(shortcut, 1, 1, 0, 1, 0.5)

        # Add
        norm = tf.nn.batch_normalization(convolve_2, 1, 1, 0, 1, 0.5)
        add = tf.add(norm, shortcut)
        residuals = tf.nn.leaky_relu(add)

        return residuals

    def straight_block(self, inputs, filters):

        conv1 = tf.Variable(tf.truncated_normal((filters[0], filters[1], 128, 128), stddev=0.1))  # Conv1
        conv2 = tf.Variable(tf.truncated_normal((filters[0], filters[1], 128, 128), stddev=0.1))  # Conv2
        bias1 = tf.Variable(tf.constant(0.1, shape=[128]))  # Bias1
        bias2 = tf.Variable(tf.constant(0.1, shape=[128]))  # Bias2

        # Convolution
        norm = tf.nn.batch_normalization(inputs, 1, 1, 0, 1, 0.5)
        convolve_1 = tf.nn.conv2d(norm, conv1, strides=[1, 1, 1, 1], padding='SAME') + bias1
        dropout = tf.nn.dropout(convolve_1, 0.25)
        convolve_2 = tf.nn.conv2d(dropout, conv2, strides=[1, 1, 1, 1], padding='SAME') + bias2

        # Add
        norm = tf.nn.batch_normalization(convolve_2, 1, 1, 0, 1, 0.5)
        add = tf.add(norm, inputs)
        residuals = tf.nn.leaky_relu(add)

        return residuals

    def predict(self, x_train):

        # Preliminary convolution
        convolve = tf.nn.conv2d(x_train, self.w_conv, strides=[1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(convolve)
        pooling = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Residual blocks
        # Block 1
        residuals = self.convolve_block(pooling, [3, 3])
        residuals = self.straight_block(residuals, [3, 3])
        residuals = tf.nn.max_pool(residuals, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Block 2
        residuals = self.convolve_block(residuals, [2, 2])
        residuals = self.straight_block(residuals, [2, 2])
        residuals = tf.nn.max_pool(residuals, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Block 3
        residuals = self.convolve_block(residuals, [1, 1])
        residuals = self.straight_block(residuals, [1, 1])
        residuals = tf.nn.max_pool(residuals, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Final prediction
        # final_pooling = tf.nn.avg_pool(residuals, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        final_pooling = tf.reduce_mean(residuals, axis=[1, 2])
        flatten = tf.layers.flatten(final_pooling)
        full_dense = tf.matmul(flatten, self.W_F) + self.B_F
        dropout = tf.nn.dropout(full_dense, 0.25)
        out = tf.nn.softmax(tf.matmul(tf.reshape(dropout, [-1, 512]), self.W) + self.B)

        return out

    def loss(self, predicted_y, desired_y):

        loss = tf.convert_to_tensor(tf.losses.softmax_cross_entropy(desired_y, predicted_y))
        return loss

    def fit(self, x: object, y: object, epochs: object, batch_size: object) -> object:

        pred = model.predict(X)
        loss = self.loss(pred, Y)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        writer = tf.summary.FileWriter('logs')
        writer.add_graph(tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                for i in range(x_train.shape[0] // batch_size):
                    x = x_train[(i * batch_size): (i + 1) * batch_size]
                    y = y_train[(i * batch_size): (i + 1) * batch_size]

                    sess.run(optimizer, feed_dict={X: x, Y: y})

                print("Loss at step {}: {}".format(epoch + 1, sess.run(loss, feed_dict={X: x, Y: y})))

    def evaluate(self, labels, prediction):
        isclose = np.isclose(np.argmax(prediction), np.argmax(labels))
        total_true = np.sum(isclose)
        accuracy = total_true / isclose.shape

        return accuracy


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10

y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
input_shape = (32, 32, 3)

x_train = np.reshape(x_train, [-1, 32, 32, 3])
x_test = np.reshape(x_test, [-1, 32, 32, 3])

X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
Y = tf.placeholder(tf.float32, shape=[None, 10])

model = ResNet(input_shape=input_shape, output_shape=num_classes)
model.fit(x_train, y_train, 10, 1000)

# Evaluation
predictions = model.predict(x_test[:500])
accuracy = model.evaluate(labels=y_test[:500], prediction=predictions)
print('Test accuracy: ', accuracy)




