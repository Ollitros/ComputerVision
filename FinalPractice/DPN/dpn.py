import tensorflow as tf
import time
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, \
     Input, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, AveragePooling2D, concatenate, Lambda, add, Activation

start_time = time.time()


def conv_block(inputs, filters, filter_size, strides=(1, 1)):
    x = Conv2D(filters, filter_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPool2D((3, 3), strides=(1, 1))(x)
    x = LeakyReLU()(x)

    return x


def relu_block(x, filters, kernel=(3, 3), stride=(1, 1), weight_decay=5e-4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay), strides=stride)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    return x


def grouped_block(x, grouped_channels, cardinality, strides, weight_decay=5e-4):
    if tf.keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    group_list = []
    init = x
    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        layer = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=strides,
                       kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(init)
        layer = BatchNormalization(axis=channel_axis)(layer)
        layer = Activation('relu')(layer)
        return layer

    for c in range(cardinality):
        layer = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels] if K.image_data_format() =='channels_last' else
        lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(x)

        layer = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=strides,
                       kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(layer)

        group_list.append(layer)

    group_merge = concatenate(group_list, axis=channel_axis)
    group_merge = BatchNormalization(axis=channel_axis)(group_merge)
    group_merge = Activation('relu')(group_merge)

    return group_merge


def dual_path_block(x, block_type, cardinality, filter_increment, pointwise_filters_a,
                    grouped_conv_filters_b, pointwise_filters_c):
    if tf.keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    grouped_channels = int(grouped_conv_filters_b / cardinality)
    init = concatenate(x, axis=channel_axis) if isinstance(x, list) else x

    if block_type == 'projection':
        stride = (1, 1)
        projection = True
    elif block_type == 'downsample':
        stride = (2, 2)
        projection = True
    elif block_type == 'normal':
        stride = (1, 1)
        projection = False
    else:
        raise ValueError('`block_type` must be one of ["projection", "downsample", "normal"]. Given %s' % block_type)

    if projection:
        projection_path = relu_block(init, filters=pointwise_filters_c + 2 * filter_increment, kernel=(1, 1), stride=stride)
        input_residual_path = Lambda(lambda z: z[:, :, :, :pointwise_filters_c]
        if K.image_data_format() == 'channels_last' else
        z[:, :pointwise_filters_c, :, :])(projection_path)
        input_dense_path = Lambda(lambda z: z[:, :, :, pointwise_filters_c:]
        if K.image_data_format() == 'channels_last' else
        z[:, pointwise_filters_c:, :, :])(projection_path)
    else:
        input_residual_path = x[0]
        input_dense_path = x[1]

    x = relu_block(init, filters=pointwise_filters_a, kernel=(1, 1))
    x = grouped_block(x, grouped_channels=grouped_channels, cardinality=cardinality, strides=stride)
    x = relu_block(x, filters=pointwise_filters_c + filter_increment, kernel=(1, 1))

    output_residual_path = Lambda(lambda z: z[:, :, :, :pointwise_filters_c]
    if K.image_data_format() == 'channels_last' else
    z[:, :pointwise_filters_c, :, :])(x)
    output_dense_path = Lambda(lambda z: z[:, :, :, pointwise_filters_c:]
    if K.image_data_format() == 'channels_last' else
    z[:, pointwise_filters_c:, :, :])(x)

    residual_path = add([input_residual_path, output_residual_path])
    dense_path = concatenate([input_dense_path, output_dense_path], axis=channel_axis)

    return [residual_path, dense_path]


def DPN(input_shape, base_filters, cardinality, filters_increment, depth, alpha=1.0):
    if tf.keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # Preliminary convolution
    inputs = Input(shape=input_shape)
    x = conv_block(inputs=inputs, filters=64, filter_size=(3, 3))

    filter_inc = filters_increment[0]
    filters = int(cardinality * alpha)
    base_filters = int(base_filters * alpha)
    N = list(depth)

    x = dual_path_block(x, pointwise_filters_a=filters, grouped_conv_filters_b=filters,
                        pointwise_filters_c=base_filters, filter_increment=filter_inc,
                        cardinality=cardinality, block_type='projection')

    for i in range(N[0] - 1):
        x = dual_path_block(x, pointwise_filters_a=filters, grouped_conv_filters_b=filters,
                            pointwise_filters_c=base_filters, filter_increment=filter_inc,
                            cardinality=cardinality, block_type='normal')

        # remaining blocks
    for k in range(1, len(N)):
        print("BLOCK %d" % (k + 1))
        filter_inc = filters_increment[k]
        filters *= 2
        base_filters *= 2

        x = dual_path_block(x, pointwise_filters_a=filters, grouped_conv_filters_b=filters,
                            pointwise_filters_c=base_filters, filter_increment=filter_inc,
                            cardinality=cardinality, block_type='downsample')

        for i in range(N[k] - 1):
            x = dual_path_block(x, pointwise_filters_a=filters, grouped_conv_filters_b=filters,
                                pointwise_filters_c=base_filters, filter_increment=filter_inc,
                                cardinality=cardinality, block_type='normal')

    x = concatenate(x, axis=channel_axis)

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
batch_size = 100
epochs = 5
alpha = 1.0
filters_increment = [4, 6, 8, 12]
base_filters = 64
depth = [2, 3, 8, 2]

y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
input_shape = (32, 32, 3)

model = DPN(input_shape=input_shape, alpha=alpha, base_filters=base_filters, filters_increment=filters_increment,
            cardinality=32, depth=depth)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

score = model.evaluate(x_test, y_test)
print(score)

print("\n\n\nThe program has been finished for --- %s seconds ---" % (time.time() - start_time))