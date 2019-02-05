from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, UpSampling2D, \
     Input, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, AveragePooling2D, concatenate, Lambda, add, Activation
from tensorflow.keras.models import Model


def model():
    inputs = Input(shape=(100, 100, 3))
    x = Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu')(inputs)
    x = MaxPool2D(padding='same')(x)
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(256, kernel_size=(4, 4), padding='same', activation='relu')(x)
    x = MaxPool2D(padding='same')(x)
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPool2D(padding='same')(x)
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(1, padding='same', kernel_size=(2, 2))(x)

    model = Model(inputs=[inputs], outputs=[x])

    return model
