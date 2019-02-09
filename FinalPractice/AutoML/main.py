from tensorflow.keras import models, layers, utils, backend, optimizers, losses, datasets
from FinalPractice.AutoML.model import AutoML


# Load and transform data for model
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

img_rows, img_cols = 28, 28
num_classes = 10

if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

model = AutoML(max_filters=512)
model.fit(x_train, y_train, x_test, y_test, train_epochs=3, search_epochs=10,
          input_shape=input_shape, num_classes=num_classes, batch_size=5000)