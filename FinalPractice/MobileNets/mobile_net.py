import tensorflow as tf
import time
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, \
    Input, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, DepthwiseConv2D

start_time = time.time()


def conv_block(inputs, filters, filter_size,  alpha, strides=(1, 1)):
    x = Conv2D(int(filters*alpha), filter_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    return x


def depthwise_block(x, filters, alpha, channel_axis):

    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU()(x)

    x = Conv2D(int(alpha*filters), (1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU()(x)

    return x


def MobileNet(input_shape, alpha=1.0):
    if tf.keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # Preliminary convolution
    inputs = Input(shape=input_shape)
    x = conv_block(inputs, 32, (3, 3), alpha=alpha)

    # Depthwise blocks
    x = depthwise_block(x, 64,  alpha=alpha, channel_axis=channel_axis)
    x = depthwise_block(x, 128, alpha=alpha, channel_axis=channel_axis)
    x = Dropout(0.25)(x)
    x = MaxPool2D(strides=(2, 2))(x)

    x = depthwise_block(x, 128, alpha=alpha, channel_axis=channel_axis)
    x = depthwise_block(x, 256, alpha=alpha, channel_axis=channel_axis)
    x = Dropout(0.25)(x)
    x = MaxPool2D(strides=(2, 2))(x)

    x = depthwise_block(x, 256, alpha=alpha, channel_axis=channel_axis)
    x = depthwise_block(x, 512, alpha=alpha, channel_axis=channel_axis)
    x = Dropout(0.25)(x)
    x = MaxPool2D(strides=(2, 2))(x)

    x = depthwise_block(x, 512, alpha=alpha, channel_axis=channel_axis)
    x = depthwise_block(x, 512, alpha=alpha, channel_axis=channel_axis)
    x = Dropout(0.25)(x)
    x = MaxPool2D(strides=(2, 2))(x)

    # Final part
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)
    dense = Dense(10, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[dense])

    return model


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10
batch_size = 300
epochs = 5

y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
input_shape = (32, 32, 3)

model = MobileNet(input_shape=input_shape, alpha=1.0)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

score = model.evaluate(x_test, y_test)
print(score)

print("\n\n\nThe program has been finished for --- %s seconds ---" % (time.time() - start_time))