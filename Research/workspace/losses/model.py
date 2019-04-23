from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model


def make_model(input_shape, num_classes):

    inputs = Input(input_shape)

    x = Conv2D(128, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model