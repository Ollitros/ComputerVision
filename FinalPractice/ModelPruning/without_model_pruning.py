import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import *


batch_size = 128
num_classes = 10
epochs = 10


def build_model(input_shape):
    inp = Input(shape=input_shape)
    x = Conv2D(32, 5, padding='same', activation='relu')(inp)
    x = MaxPooling2D((2, 2), (2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 5, padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), (2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax')(x)

    return tf.keras.models.Model([inp], [out])


def train_and_save(model, x_train, y_train, x_test, y_test):

    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    # Print the model summary.
    model.summary()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Export and import the model. Check that accuracy persists.
    keras_file = 'data/model.h5'
    print('Saving model to: ', keras_file)

    keras.models.save_model(model, keras_file)
    loaded_model = keras.models.load_model(keras_file)
    score = loaded_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main():
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if tf.keras.backend.image_data_format() == 'channels_first':
      x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
      x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
      input_shape = (1, img_rows, img_cols)
    else:
      x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
      x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
      input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = build_model(input_shape)
    train_and_save(model, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()