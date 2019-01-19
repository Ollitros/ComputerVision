import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, Activation, Input, BatchNormalization, LeakyReLU, GlobalAveragePooling2D


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10

y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
input_shape = (32, 32, 3)


# Convolution block
def conv_block(x, stride):

    norm = BatchNormalization()(x)
    conv1 = Conv2D(128, stride, padding='same')(norm)
    conv2 = Conv2D(128, stride, padding='same')(conv1)

    # Sum
    norm = BatchNormalization()(conv2)
    add = tf.keras.layers.add([norm, x])
    residuals = LeakyReLU()(add)

    return residuals


# Neural network
# Preliminary convolution
inputs = Input(shape=input_shape)
conv = Conv2D(128, (3, 3),  activation='relu', padding='same')(inputs)
residuals = MaxPool2D(padding='same')(conv)

# Residual blocks
residuals = conv_block(residuals, stride=(3, 3))
residuals = conv_block(residuals, stride=(3, 3))
residuals = MaxPool2D()(residuals)

residuals = conv_block(residuals, stride=(2, 2))
residuals = conv_block(residuals, stride=(2, 2))
residuals = MaxPool2D()(residuals)

residuals = conv_block(residuals, stride=(1, 1))
residuals = MaxPool2D()(residuals)

# Final part
final_pooling = GlobalAveragePooling2D()(residuals)
flatten = Flatten()(final_pooling)
full_dense = Dense(512, activation='relu')(flatten)
dropout = (Dropout(0.25))(full_dense)
dense = Dense(10, activation='softmax')(dropout)

model = tf.keras.models.Model(inputs=[inputs], outputs=[dense])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, batch_size=1000, epochs=10)

score = model.evaluate(x_test, y_test)
print(score)