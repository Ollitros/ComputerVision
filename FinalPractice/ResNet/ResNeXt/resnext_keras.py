import tensorflow as tf
import time
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, \
    Input, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Lambda, concatenate

start_time = time.time()


def exe_block(groups, channels):
    x = concatenate(groups)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

    return x


# Convolution block
# Has convolution in shortcut
def convolve_block(inputs,  cardinality, filters, channels):
    x = Conv2D(channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        return Conv2D(channels, filters, padding='same')(x)
    assert not channels % cardinality
    d = channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups
    # and convolutions are separately performed within each group
    groups_1 = []
    for j in range(cardinality):
        group_1 = Lambda(lambda z: z[:, :, :, j * d:j * d + d])(x)
        groups_1.append(Conv2D(d, filters, padding='same')(group_1))

    # the grouped convolutional layer concatenates them as the outputs of the layer
    x = exe_block(groups_1, channels)

    groups_2 = []
    for j in range(cardinality):
        group_2 = Lambda(lambda z: z[:, :, :, j * d:j * d + d])(x)
        groups_2.append(Conv2D(d, filters, padding='same')(group_2))

    x = exe_block(groups_2, channels)

    shortcut = Conv2D(channels, (1, 1))(inputs)
    shortcut = BatchNormalization()(shortcut)

    add = tf.keras.layers.add([x, shortcut])
    residuals = LeakyReLU()(add)

    return residuals


# Straight block
# Has not convolution in shortcut
def straight_block(inputs, cardinality, filters, channels):
    x = Conv2D(channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        return Conv2D(channels, filters, padding='same')(x)
    assert not channels % cardinality
    d = channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups
    # and convolutions are separately performed within each group
    groups_1 = []
    for j in range(cardinality):
        group_1 = Lambda(lambda z: z[:, :, :, j * d:j * d + d])(x)
        groups_1.append(Conv2D(d, filters, padding='same')(group_1))

    # the grouped convolutional layer concatenates them as the outputs of the layer
    x = exe_block(groups_1, channels)

    groups_2 = []
    for j in range(cardinality):
        group_2 = Lambda(lambda z: z[:, :, :, j * d:j * d + d])(x)
        groups_2.append(Conv2D(d, filters, padding='same')(group_2))

    x = exe_block(groups_2, channels)

    add = tf.keras.layers.add([x, inputs])
    residuals = LeakyReLU()(add)

    return residuals


def ResNet(input_shape, cardinality):

    # Neural network
    # Preliminary convolution
    inputs = Input(shape=input_shape)
    convolve = Conv2D(128, (3, 3),  activation='relu', padding='same')(inputs)
    residuals = MaxPool2D(padding='same')(convolve)

    # Residual blocks
    residuals = convolve_block(residuals, cardinality, filters=(3, 3), channels=64)
    residuals = straight_block(residuals, cardinality, filters=(3, 3), channels=64)
    residuals = MaxPool2D()(residuals)

    residuals = convolve_block(residuals, cardinality, filters=(2, 2), channels=64)
    residuals = straight_block(residuals, cardinality, filters=(2, 2), channels=64)
    residuals = MaxPool2D()(residuals)

    residuals = convolve_block(residuals, cardinality, filters=(1, 1), channels=64)
    residuals = straight_block(residuals, cardinality, filters=(1, 1), channels=64)
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
batch_size = 500
epochs = 30
cardinality = 32

y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
input_shape = (32, 32, 3)

model = ResNet(input_shape=input_shape, cardinality=cardinality)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

score = model.evaluate(x_test, y_test)
print(score)

print("\n\n\nThe program has been finished for --- %s seconds ---" % (time.time() - start_time))