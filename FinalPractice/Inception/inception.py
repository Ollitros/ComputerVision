import tensorflow as tf
import time
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, \
    Input, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, AveragePooling2D, concatenate

start_time = time.time()


def conv_block(inputs, filters, filter_size, strides=(1, 1)):
    x = Conv2D(filters, filter_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    return x


def InceptionNN(input_shape):
    if tf.keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # Preliminary convolution
    inputs = Input(shape=input_shape)
    x = conv_block(inputs, 32, (3, 3))
    x = conv_block(x, 32, (3, 3))
    x = conv_block(x, 64, (3, 3))
    x = MaxPool2D((3, 3), strides=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = conv_block(x, 80, (1, 1))
    x = conv_block(x, 192,(3, 3))
    x = MaxPool2D((3, 3), strides=(1, 1))(x)
    x = Dropout(0.25)(x)

    # Inception blocks
    # Inception block 1
    conv1x1 = conv_block(x, 64, (1, 1))
    conv5x5 = conv_block(x, 48, (1, 1))
    conv5x5 = conv_block(conv5x5, 64, (5, 5))

    conv3x3 = conv_block(x, 64, (1, 1))
    conv3x3 = conv_block(conv3x3, 96, (3, 3))
    conv3x3 = conv_block(conv3x3, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_block(branch_pool, 32, 1, 1)

    x = concatenate([conv1x1, conv5x5, conv3x3, branch_pool], axis=channel_axis, name='inception_block_1')
    x = MaxPool2D((3, 3), strides=(1, 1))(x)
    x = Dropout(0.25)(x)

    # Inception block 2
    conv1x1 = conv_block(x, 64, (1, 1))
    conv5x5 = conv_block(x, 48, (1, 1))
    conv5x5 = conv_block(conv5x5, 64, (5, 5))

    conv3x3 = conv_block(x, 64, (1, 1))
    conv3x3 = conv_block(conv3x3, 96, (3, 3))
    conv3x3 = conv_block(conv3x3, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_block(branch_pool, 32, 1, 1)

    x = concatenate([conv1x1, conv5x5, conv3x3, branch_pool], axis=channel_axis, name='inception_block_2')
    x = MaxPool2D((3, 3), strides=(1, 1))(x)
    x = Dropout(0.25)(x)

    # Inception block 3
    conv1x1 = conv_block(x, 64, (1, 1))
    conv5x5 = conv_block(x, 48, (1, 1))
    conv5x5 = conv_block(conv5x5, 64, (5, 5))

    conv3x3 = conv_block(x, 64, (1, 1))
    conv3x3 = conv_block(conv3x3, 96, (3, 3))
    conv3x3 = conv_block(conv3x3, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_block(branch_pool, 32, 1, 1)

    x = concatenate([conv1x1, conv5x5, conv3x3, branch_pool], axis=channel_axis, name='inception_block_3')
    x = MaxPool2D((3, 3), strides=(1, 1))(x)
    x = Dropout(0.25)(x)

    # Inception block 4
    conv1x1 = conv_block(x, 64, (1, 1))
    conv5x5 = conv_block(x, 48, (1, 1))
    conv5x5 = conv_block(conv5x5, 64, (5, 5))

    conv3x3 = conv_block(x, 64, (1, 1))
    conv3x3 = conv_block(conv3x3, 96, (3, 3))
    conv3x3 = conv_block(conv3x3, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_block(branch_pool, 32, 1, 1)

    x = concatenate([conv1x1, conv5x5, conv3x3, branch_pool], axis=channel_axis, name='inception_block_4')
    x = MaxPool2D((3, 3), strides=(1, 1))(x)
    x = Dropout(0.25)(x)

    # Inception block 5
    conv1x1 = conv_block(x, 64, (1, 1))
    conv5x5 = conv_block(x, 48, (1, 1))
    conv5x5 = conv_block(conv5x5, 64, (5, 5))

    conv3x3 = conv_block(x, 64, (1, 1))
    conv3x3 = conv_block(conv3x3, 96, (3, 3))
    conv3x3 = conv_block(conv3x3, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_block(branch_pool, 32, 1, 1)

    x = concatenate([conv1x1, conv5x5, conv3x3, branch_pool], axis=channel_axis, name='inception_block_5')
    x = MaxPool2D((3, 3), strides=(1, 1))(x)
    x = Dropout(0.25)(x)

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

model = InceptionNN(input_shape=input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

score = model.evaluate(x_test, y_test)
print(score)

print("\n\n\nThe program has been finished for --- %s seconds ---" % (time.time() - start_time))