import tensorflow as tf
import time
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, \
    Input, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D, concatenate, Activation, ZeroPadding2D


start_time = time.time()


def conv_block(inputs, channel_axis, growth, residuals):
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(2 * growth, 1, use_bias=False)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(growth, (3, 3), padding='same', use_bias=False)(x)
    x = Dropout(0.25)(x)
    residuals.append(x)
    x = concatenate(residuals, axis=channel_axis)

    return x


def transition_block(inputs, reduction, channel_axis):

    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(int(K.int_shape(x)[channel_axis] * reduction), 1)(x)
    x = AveragePooling2D(2, strides=2)(x)
    return x


def dense_block(x, blocks, channel_axis):
    residuals = list()
    residuals.append(x)
    for i in range(blocks):
        x = conv_block(x, channel_axis, 16, residuals)

    return x


def DenseNet(input_shape, blocks):
    if tf.keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # Preliminary convolution
    inputs = Input(shape=input_shape)
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPool2D(strides=1, padding='same')(x)

    x = dense_block(x, blocks[0], channel_axis=channel_axis)
    x = transition_block(x, 0.5, channel_axis=channel_axis)
    x = dense_block(x, blocks[1], channel_axis=channel_axis)
    x = transition_block(x, 0.5, channel_axis=channel_axis)
    x = dense_block(x, blocks[2], channel_axis=channel_axis)
    x = transition_block(x, 0.5, channel_axis=channel_axis)
    x = dense_block(x, blocks[3], channel_axis=channel_axis)

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
batch_size = 200
epochs = 5
blocks = [3, 6, 8, 10]

y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
input_shape = (32, 32, 3)

model = DenseNet(input_shape=input_shape, blocks=blocks)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test[:5000], y_test[:5000]))

score = model.evaluate(x_test[5000:], y_test[5000:])
print(score)

print("\n\n\nThe program has been finished for --- %s seconds ---" % (time.time() - start_time))