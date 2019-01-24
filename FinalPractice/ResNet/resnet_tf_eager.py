import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from tensorflow.keras.datasets import cifar10


tf.enable_eager_execution()


class ResNet():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Preliminary convolution weights
        self.w_conv = tfe.Variable(tf.truncated_normal((3, 3, 3, 128), stddev=0.1))
        self.b_conv = tfe.Variable(tf.constant(0.1, shape=[128]))

        # Final prediction weights
        # Full connect
        self.W_F = tfe.Variable(tf.random_normal([128, 512]))
        self.B_F = tfe.Variable(tf.random_normal([512]))

        # Output
        self.W = tfe.Variable(tf.random_normal([512, self.output_shape]))
        self.B = tfe.Variable(tf.random_normal([self.output_shape]))

        self.variables = [self.W, self.B, self.w_conv, self.b_conv, self.W_F, self.B_F]
        self.weights = []

    def convolve_block(self, inputs, filters):

        self.weights.append(tfe.Variable(tf.truncated_normal((filters[0], filters[1], 128, 128), stddev=0.1))) # Conv1
        self.weights.append(tfe.Variable(tf.truncated_normal((filters[0], filters[1], 128, 128), stddev=0.1))) # Conv2
        self.weights.append(tfe.Variable(tf.constant(0.1, shape=[128])))                                       # Bias1
        self.weights.append(tfe.Variable(tf.constant(0.1, shape=[128])))                                       # Bias2
        self.weights.append(tfe.Variable(tf.truncated_normal((1, 1, 128, 128), stddev=0.1)))                   # Shortcut conv
        self.weights.append(tfe.Variable(tf.constant(0.1, shape=[128])))                                       # Shortcut bias

        # Convolution
        norm = tf.nn.batch_normalization(inputs, 1, 1, 0, 1, 0.5)
        convolve_1 = tf.nn.conv2d(norm, self.weights[0], strides=[1, 1, 1, 1], padding='SAME') + self.weights[2]
        dropout = tf.nn.dropout(convolve_1, 0.25)
        convolve_2 = tf.nn.conv2d(dropout, self.weights[1], strides=[1, 1, 1, 1], padding='SAME') + self.weights[3]

        # Shortcut
        shortcut = tf.nn.conv2d(inputs, self.weights[4], strides=[1, 1, 1, 1], padding='SAME') + self.weights[5]
        shortcut = tf.nn.batch_normalization(shortcut, 1, 1, 0, 1, 0.5)

        # Add
        norm = tf.nn.batch_normalization(convolve_2, 1, 1, 0, 1, 0.5)
        add = tf.add(norm, shortcut)
        residuals = tf.nn.leaky_relu(add)

        self.variables.append(self.weights[0])  # Conv1
        self.variables.append(self.weights[1])  # Conv2
        self.variables.append(self.weights[2])  # Bias1
        self.variables.append(self.weights[3])  # Bias2
        self.variables.append(self.weights[4])  # Shortcut conv
        self.variables.append(self.weights[5])  # Shortcut bias

        self.weights = []

        return residuals

    def straight_block(self, inputs, filters):

        self.weights.append(tfe.Variable(tf.truncated_normal((filters[0], filters[1], 128, 128), stddev=0.1)))  # Conv1
        self.weights.append(tfe.Variable(tf.truncated_normal((filters[0], filters[1], 128, 128), stddev=0.1)))  # Conv2
        self.weights.append(tfe.Variable(tf.constant(0.1, shape=[128])))  # Bias1
        self.weights.append(tfe.Variable(tf.constant(0.1, shape=[128])))  # Bias2

        # Convolution
        norm = tf.nn.batch_normalization(inputs, 1, 1, 0, 1, 0.5)
        convolve_1 = tf.nn.conv2d(norm, self.weights[0], strides=[1, 1, 1, 1], padding='SAME') + self.weights[2]
        dropout = tf.nn.dropout(convolve_1, 0.25)
        convolve_2 = tf.nn.conv2d(dropout, self.weights[1], strides=[1, 1, 1, 1], padding='SAME') + self.weights[3]

        # Add
        norm = tf.nn.batch_normalization(convolve_2, 1, 1, 0, 1, 0.5)
        add = tf.add(norm, inputs)
        residuals = tf.nn.leaky_relu(add)

        self.variables.append(self.weights[0])  # Conv1
        self.variables.append(self.weights[1])  # Conv2
        self.variables.append(self.weights[2])  # Bias1
        self.variables.append(self.weights[3])  # Bias2

        self.weights = []

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
        return tf.losses.softmax_cross_entropy(desired_y, predicted_y)

    def fit(self, x_train, y_train, epochs, batch_size):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        for epoch in range(epochs):
            for i in range(x_train.shape[0] // batch_size):
                with tf.GradientTape() as tape:
                    predicted = model.predict(x_train[(i * batch_size): (i+1) * batch_size])
                    curr_loss = self.loss(predicted, y_train[(i * batch_size): (i+1) * batch_size])
                grads = tape.gradient(curr_loss, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

                pred = model.predict(x_train[(i * batch_size): (i + 1) * batch_size])
                current_loss = self.loss(pred, y_train[(i * batch_size): (i + 1) * batch_size])
                train_accuracy = self.evaluate(y_train[(i * batch_size): (i + 1) * batch_size], pred)
                print("Loss at step {:d}: {:.3f} ||| accuracy - {}".format(epoch + 1, current_loss, train_accuracy))

        writer = tf.contrib.summary.create_file_writer('logs')

    def evaluate(self, labels, prediction):
        isclose = np.isclose(np.argmax(prediction, 1), np.argmax(labels, 1))
        total_true = np.sum(isclose)
        accuracy = total_true / isclose.shape

        return accuracy[0]


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10

y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
input_shape = (32, 32, 3)

x_train = tf.constant(np.reshape(x_train, [-1, 32, 32, 3]), dtype=tf.float32)
x_test = tf.constant(np.reshape(x_test, [-1, 32, 32, 3]), dtype=tf.float32)

model = ResNet(input_shape=input_shape, output_shape=num_classes)
model.fit(x_train, y_train, 1, 200)

# Evaluation
predictions = model.predict(x_test[:500])
accuracy = model.evaluate(labels=y_test[:500], prediction=predictions)
print('Test accuracy: ', accuracy)
