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
        self.b_conv = tfe.Variable(tf.constant(0.1, shape=[32]))

        # Convolve and Straight block weights
        # Convolve block 1 weights
        self.w_conv_1 = tfe.Variable(tf.truncated_normal((3, 3, 128, 128), stddev=0.1))
        self.w_conv_2 = tfe.Variable(tf.truncated_normal((3, 3, 128, 128), stddev=0.1))
        self.w_shortcut_1 = tfe.Variable(tf.truncated_normal((1, 1, 128, 128), stddev=0.1))

        # Convolve block 2 weights
        self.w_conv_3 = tfe.Variable(tf.truncated_normal((3, 3, 128, 128), stddev=0.1))
        self.w_conv_4 = tfe.Variable(tf.truncated_normal((3, 3, 128, 128), stddev=0.1))
        self.w_shortcut_2 = tfe.Variable(tf.truncated_normal((1, 1, 128, 128), stddev=0.1))

        # Convolve block 2 weights
        self.w_conv_5 = tfe.Variable(tf.truncated_normal((3, 3, 128, 128), stddev=0.1))
        self.w_conv_6 = tfe.Variable(tf.truncated_normal((3, 3, 128, 128), stddev=0.1))
        self.w_shortcut_3 = tfe.Variable(tf.truncated_normal((1, 1, 128, 128), stddev=0.1))

        # Straight block 1 weights
        self.w_straight_1 = tfe.Variable(tf.truncated_normal((3, 3, 128, 128), stddev=0.1))
        self.w_straight_2 = tfe.Variable(tf.truncated_normal((3, 3, 128, 128), stddev=0.1))

        # Straight block 2 weights
        self.w_straight_3 = tfe.Variable(tf.truncated_normal((3, 3, 128, 128), stddev=0.1))
        self.w_straight_4 = tfe.Variable(tf.truncated_normal((3, 3, 128, 128), stddev=0.1))

        # Straight block 2 weights
        self.w_straight_5 = tfe.Variable(tf.truncated_normal((3, 3, 128, 128), stddev=0.1))
        self.w_straight_6 = tfe.Variable(tf.truncated_normal((3, 3, 128, 128), stddev=0.1))

        # Final prediction weights
        # Full connect
        self.W_F = tfe.Variable(tf.random_normal([128, 512]))
        self.B_F = tfe.Variable(tf.random_normal([512]))

        # Output
        self.W = tfe.Variable(tf.random_normal([512, self.output_shape]))
        self.B = tfe.Variable(tf.random_normal([self.output_shape]))

        self.variables = [self.W, self.B, self.w_conv, self.b_conv, self.W_F, self.B_F,
                          self.w_conv_1, self.w_conv_2, self.w_shortcut_1, self.w_straight_1, self.w_straight_2,
                          self.w_conv_3, self.w_conv_4, self.w_shortcut_2, self.w_straight_3, self.w_straight_4,
                          self.w_conv_5, self.w_conv_6, self.w_shortcut_3, self.w_straight_5, self.w_straight_6]

    def convolve_block_1(self, inputs):
        # Convolution
        norm = tf.nn.batch_normalization(inputs, 1, 1, 0, 1, 0.5)
        convolve_1 = tf.nn.conv2d(norm, self.w_conv_1, strides=[1, 1, 1, 1], padding='SAME')
        dropout = tf.nn.dropout(convolve_1, 0.25)
        convolve_2 = tf.nn.conv2d(dropout, self.w_conv_2, strides=[1, 1, 1, 1], padding='SAME')

        # Shortcut
        shortcut = tf.nn.conv2d(inputs, self.w_shortcut_1, strides=[1, 1, 1, 1], padding='SAME')
        shortcut = tf.nn.batch_normalization(shortcut, 1, 1, 0, 1, 0.5)

        # Add
        norm = tf.nn.batch_normalization(convolve_2, 1, 1, 0, 1, 0.5)
        add = tf.add(norm, shortcut)
        residuals = tf.nn.leaky_relu(add)

        return residuals

    def straight_block_1(self, inputs):
        # Convolution
        norm = tf.nn.batch_normalization(inputs, 1, 1, 0, 1, 0.5)
        convolve_1 = tf.nn.conv2d(norm, self.w_straight_1, strides=[1, 1, 1, 1], padding='SAME')
        dropout = tf.nn.dropout(convolve_1, 0.25)
        convolve_2 = tf.nn.conv2d(dropout, self.w_straight_2, strides=[1, 1, 1, 1], padding='SAME')

        # Add
        norm = tf.nn.batch_normalization(convolve_2, 1, 1, 0, 1, 0.5)
        add = tf.add(norm, inputs)
        residuals = tf.nn.leaky_relu(add)

        return residuals

    def convolve_block_2(self, inputs):
        # Convolution
        norm = tf.nn.batch_normalization(inputs, 1, 1, 0, 1, 0.5)
        convolve_3 = tf.nn.conv2d(norm, self.w_conv_3, strides=[1, 1, 1, 1], padding='SAME')
        dropout = tf.nn.dropout(convolve_3, 0.25)
        convolve_4 = tf.nn.conv2d(dropout, self.w_conv_4, strides=[1, 1, 1, 1], padding='SAME')

        # Shortcut
        shortcut = tf.nn.conv2d(inputs, self.w_shortcut_2, strides=[1, 1, 1, 1], padding='SAME')
        shortcut = tf.nn.batch_normalization(shortcut, 1, 1, 0, 1, 0.5)

        # Add
        norm = tf.nn.batch_normalization(convolve_4, 1, 1, 0, 1, 0.5)
        add = tf.add(norm, shortcut)
        residuals = tf.nn.leaky_relu(add)

        return residuals

    def straight_block_2(self, inputs):
        # Convolution
        norm = tf.nn.batch_normalization(inputs, 1, 1, 0, 1, 0.5)
        convolve_3 = tf.nn.conv2d(norm, self.w_straight_3, strides=[1, 1, 1, 1], padding='SAME')
        dropout = tf.nn.dropout(convolve_3, 0.25)
        convolve_4 = tf.nn.conv2d(dropout, self.w_straight_4, strides=[1, 1, 1, 1], padding='SAME')

        # Add
        norm = tf.nn.batch_normalization(convolve_4, 1, 1, 0, 1, 0.5)
        add = tf.add(norm, inputs)
        residuals = tf.nn.leaky_relu(add)

        return residuals

    def convolve_block_3(self, inputs):
        # Convolution
        norm = tf.nn.batch_normalization(inputs, 1, 1, 0, 1, 0.5)
        convolve_5 = tf.nn.conv2d(norm, self.w_conv_5, strides=[1, 1, 1, 1], padding='SAME')
        dropout = tf.nn.dropout(convolve_5, 0.25)
        convolve_6 = tf.nn.conv2d(dropout, self.w_conv_6, strides=[1, 1, 1, 1], padding='SAME')

        # Shortcut
        shortcut = tf.nn.conv2d(inputs, self.w_shortcut_3, strides=[1, 1, 1, 1], padding='SAME')
        shortcut = tf.nn.batch_normalization(shortcut, 1, 1, 0, 1, 0.5)

        # Add
        norm = tf.nn.batch_normalization(convolve_6, 1, 1, 0, 1, 0.5)
        add = tf.add(norm, shortcut)
        residuals = tf.nn.leaky_relu(add)

        return residuals

    def straight_block_3(self, inputs):
        # Convolution
        norm = tf.nn.batch_normalization(inputs, 1, 1, 0, 1, 0.5)
        convolve_5 = tf.nn.conv2d(norm, self.w_straight_5, strides=[1, 1, 1, 1], padding='SAME')
        dropout = tf.nn.dropout(convolve_5, 0.25)
        convolve_6 = tf.nn.conv2d(dropout, self.w_straight_6, strides=[1, 1, 1, 1], padding='SAME')

        # Add
        norm = tf.nn.batch_normalization(convolve_6, 1, 1, 0, 1, 0.5)
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
        residuals = self.convolve_block_1(pooling)
        residuals = self.straight_block_1(residuals)
        residuals = tf.nn.max_pool(residuals, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Block 2
        residuals = self.convolve_block_2(residuals)
        residuals = self.straight_block_2(residuals)
        residuals = tf.nn.max_pool(residuals, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Block 3
        residuals = self.convolve_block_3(residuals)
        residuals = self.straight_block_3(residuals)
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
        return tf.reduce_mean(tf.square(predicted_y - desired_y))

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
model.fit(x_train, y_train, 10, 600)

# Evaluation
predictions = model.predict(x_test[:500])
accuracy = model.evaluate(labels=y_test[:500], prediction=predictions)
print('Test accuracy: ', accuracy)


