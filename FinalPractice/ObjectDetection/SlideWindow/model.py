import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, \
     Input, BatchNormalization, concatenate, Dropout, Flatten, Dense, AveragePooling2D, LeakyReLU
from tensorflow.keras.models import Model


def conv_block(inputs, filters, filter_size,  alpha, strides=(1, 1)):
    x = Conv2D(int(filters*alpha), filter_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    return x


def inception_block(x, channel_axis, alpha):
    conv1x1 = conv_block(x, 64, (1, 1), alpha=alpha)
    conv5x5 = conv_block(x, 48, (1, 1), alpha=alpha)
    conv5x5 = conv_block(conv5x5, 64, (5, 5), alpha=alpha)

    conv3x3 = conv_block(x, 64, (1, 1), alpha=alpha)
    conv3x3 = conv_block(conv3x3, 96, (3, 3), alpha=alpha)
    conv3x3 = conv_block(conv3x3, 96, (3, 3), alpha=alpha)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_block(branch_pool, 32, (1, 1), alpha=alpha)

    x = concatenate([conv1x1, conv5x5, conv3x3, branch_pool], axis=channel_axis)
    x = MaxPool2D((3, 3), strides=(1, 1))(x)
    x = Dropout(0.25)(x)

    return x


def InceptionNN(input_shape, alpha=1.0):
    if tf.keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # Preliminary convolution
    inputs = Input(shape=input_shape)
    x = conv_block(inputs, 32, (3, 3), alpha=alpha)
    x = conv_block(x, 32, (3, 3), alpha=alpha)
    x = conv_block(x, 64, (3, 3), alpha=alpha)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = conv_block(x, 80, (1, 1), alpha=alpha)
    x = conv_block(x, 192, (3, 3), alpha=alpha)
    x = MaxPool2D((3, 3), strides=(1, 1))(x)
    x = Dropout(0.25)(x)

    # Inception blocks
    # Inception block 1
    x = inception_block(x, channel_axis=channel_axis, alpha=alpha)

    # Inception block 2
    x = inception_block(x, channel_axis=channel_axis, alpha=alpha)

    # Inception block 3
    x = inception_block(x, channel_axis=channel_axis, alpha=alpha)

    # Inception block 4
    x = inception_block(x, channel_axis=channel_axis, alpha=alpha)

    # Inception block 5
    x = inception_block(x, channel_axis=channel_axis, alpha=alpha)

    # Final part
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)
    classes = Dense(3, activation='softmax')(x)

    model = Model(inputs=[inputs], outputs=[classes])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model
