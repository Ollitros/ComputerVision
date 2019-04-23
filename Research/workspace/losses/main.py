import time
import tensorflow as tf
from keras import utils, backend
from Research.workspace.losses.model import make_model
from Research.workspace.losses.loss import custom_loss


start_time = time.time()

img_rows, img_cols = 28, 28
num_classes = 10
epochs = 10
batch_size = 256
loss_name = "Mean_squared_error"
test_version = 0.1

# Load and transform data for model
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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


model = make_model(input_shape, num_classes)
model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

score = model.evaluate(x_test, y_test)
print("Test evaluation\nLoss: ", score[0], "Accuracy:", score[1])

info = 'Test version: {test_version}\nLoss name: {loss_name}\n' \
       'Batch size: {batch_size}\nEpochs: {epochs}\n' \
       'Test accuracy: {accuracy}\n' \
       '"--- {time} seconds ---" '.format(test_version=test_version, loss_name=loss_name,
                                          batch_size=batch_size, epochs=epochs,
                                          accuracy=(score[1]*100), time=(time.time() - start_time))
dir_path = 'data/mse/'
with open(dir_path + "info.txt", "w") as file:
    file.write(info)

model.save(dir_path + 'model.h5')
print("--- %s seconds ---" % (time.time() - start_time))