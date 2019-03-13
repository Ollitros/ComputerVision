import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, \
    Input, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, AveragePooling2D, concatenate


def Inception_resnet(input_shape, num_classes, alpha=1.0):

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

        x = inception_resnet_block(x, [conv1x1, conv5x5, conv3x3, branch_pool], channel_axis=channel_axis, alpha=alpha)
        x = MaxPool2D((3, 3), strides=(1, 1))(x)
        x = Dropout(0.25)(x)

        return x

    def inception_resnet_block(shortcut, concat, channel_axis, alpha):
        shortcut = Conv2D(int(192*alpha), (1, 1))(shortcut)
        shortcut = BatchNormalization()(shortcut)
        x = concatenate(concat, axis=channel_axis)
        x = Conv2D(int(192*alpha), (1, 1), padding='same')(x)
        x = tf.keras.layers.add([shortcut, x])

        return x

    def build(num_classes):
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
        x = inception_block(x, channel_axis, alpha)

        # Inception block 2
        x = inception_block(x, channel_axis, alpha)

        # Inception block 3
        x = inception_block(x, channel_axis, alpha)

        # Inception block 4
        x = inception_block(x, channel_axis, alpha)

        # Inception block 5
        x = inception_block(x, channel_axis, alpha)

        # Final part
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.25)(x)
        dense = Dense(num_classes, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=[inputs], outputs=[dense])

        return model

    model = build(num_classes=num_classes)

    return model
