import tensorflow as tf
import time
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, \
    Input, BatchNormalization, LeakyReLU, GlobalAveragePooling2D

start_time = time.time()


# Convolution block
# Has convolution in shortcut
def convolve_block(inputs, filters):
    norm = BatchNormalization()(inputs)
    conv1 = Conv2D(128, filters, padding='same')(norm)
    drop = Dropout(0.25)(conv1)
    conv2 = Conv2D(128, filters, padding='same')(drop)

    # Sum
    norm = BatchNormalization()(conv2)

    shortcut = Conv2D(128, (1, 1))(inputs)
    shortcut = BatchNormalization()(shortcut)

    add = tf.keras.layers.add([norm, shortcut])
    residuals = LeakyReLU()(add)

    return residuals


# Straight block
# Has not convolution in shortcut
def straight_block(inputs, filters):
    norm = BatchNormalization()(inputs)
    convolve1 = Conv2D(128, filters, padding='same')(norm)
    drop = Dropout(0.25)(convolve1)
    convolve2 = Conv2D(128, filters, padding='same')(drop)

    # Sum
    norm = BatchNormalization()(convolve2)
    add = tf.keras.layers.add([norm, inputs])
    residuals = LeakyReLU()(add)

    return residuals


def ResNet(input_shape):

    # Neural network
    # Preliminary convolution
    inputs = Input(shape=input_shape)
    convolve = Conv2D(128, (3, 3),  activation='relu', padding='same')(inputs)
    residuals = MaxPool2D(padding='same')(convolve)

    # Residual blocks
    residuals = convolve_block(residuals, filters=(3, 3))
    residuals = straight_block(residuals, filters=(3, 3))
    residuals = MaxPool2D()(residuals)

    residuals = convolve_block(residuals, filters=(2, 2))
    residuals = straight_block(residuals, filters=(2, 2))
    residuals = MaxPool2D()(residuals)

    residuals = convolve_block(residuals, filters=(1, 1))
    residuals = straight_block(residuals, filters=(1, 1))
    residuals = MaxPool2D()(residuals)

    # Final part
    final_pooling = GlobalAveragePooling2D()(residuals)
    flatten = Flatten()(final_pooling)
    full_dense = Dense(512, activation='relu')(flatten)
    dropout = Dropout(0.25)(full_dense)
    dense = Dense(10, activation='softmax')(dropout)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[dense])

    return model


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10
batch_size = 1000
epochs = 5

y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
input_shape = (32, 32, 3)

model = ResNet(input_shape=input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

score = model.evaluate(x_test, y_test)
print(score)

print("\n\n\nThe program has been finished for --- %s seconds ---" % (time.time() - start_time))