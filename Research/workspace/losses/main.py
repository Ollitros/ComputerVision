import time
import statistics
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from Research.workspace.exp.model import make_model
from Research.workspace.exp.loss import custom_loss


start_time = time.time()

input_shape = (32, 32, 3)
num_classes = 10
epochs = 5
batch_size = 256
loss_name = "custom"
num_ensemble = 10

# Load training and testing data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)


total_score = []
for i in range(num_ensemble):
    print("||||||||||||||||| EPOCH {i} |||||||||||||||||".format(i=i))
    model = make_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    score = model.evaluate(x_test, y_test)
    print("Test evaluation\nLoss: ", score[0], "Accuracy:", score[1])
    total_score.append(score[1]*100)

mean_accuracy = statistics.mean(total_score)
info = 'Loss name: {loss_name}\n' \
       'Batch size: {batch_size}\nEpochs: {epochs}\n' \
       'Test mean accuracy: {accuracy}\n' \
       '--- {time} seconds --- \n' \
       'Total ensembles: {num_ensemble}'.format(loss_name=loss_name,
                                                batch_size=batch_size, epochs=epochs,
                                                accuracy=mean_accuracy, time=(time.time() - start_time),
                                                num_ensemble=num_ensemble)
dir_path = 'data/custom/'
with open(dir_path + "info.txt", "w") as file:
    file.write(info)

model.save(dir_path + 'model.h5')
print("--- %s seconds ---" % (time.time() - start_time))