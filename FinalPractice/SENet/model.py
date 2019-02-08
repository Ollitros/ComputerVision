import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, \
    Input, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Multiply, Reshape


# Convolution block
# Has convolution in shortcut
def convolve_block(inputs, filters):
    norm = BatchNormalization()(inputs)
    conv1 = Conv2D(128, filters, padding='same')(norm)
    drop = Dropout(0.25)(conv1)
    conv2 = Conv2D(128, filters, padding='same')(drop)

    # Sum
    norm = BatchNormalization()(conv2)

    # Start SE block
    se = GlobalAveragePooling2D()(norm)
    se = Dense(128 // 16, activation='relu')(se)
    se = Dense(128, activation='sigmoid')(se)
    se = Reshape([1, 1, 128])(se)
    x = Multiply()([norm, se])
    # End SE block

    shortcut = Conv2D(128, (1, 1))(inputs)
    shortcut = BatchNormalization()(shortcut)

    add = tf.keras.layers.add([x, shortcut])
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

    # Start SE block
    se = GlobalAveragePooling2D()(norm)
    se = Dense(128 // 16, activation='relu')(se)
    se = Dense(128, activation='sigmoid')(se)
    se = Reshape([1, 1, 128])(se)
    x = Multiply()([norm, se])
    # End SE block

    add = tf.keras.layers.add([x, inputs])
    residuals = LeakyReLU()(add)

    return residuals


def SEResNet(input_shape):

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


