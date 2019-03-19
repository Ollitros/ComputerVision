import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Research.workspace.models import Inception_resnet


# Load training and testing data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train, x_val, y_train, y_val = x_train[0:45000], x_train[45000:], y_train[0:45000], y_train[45000:]

# Model hyperparameters
input_shape = (32, 32, 3)
num_classes = 10
batch_size = 200
epochs = 100
alpha = 1.0

# One-hot encoding
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)
# y_val = to_categorical(y_val, num_classes=num_classes)

# checkpoint
filepath = "data/inception_resnet/models/checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


model = Inception_resnet(input_shape=input_shape, alpha=alpha, num_classes=num_classes)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint], validation_split=0.2)
model.save('data/inception_resnet/models/inception_resnet_cifar_10.h5')

eval = model.evaluate(x_test, y_test)
print("Error rate: %.2f%%" % (100 - eval[1]*100), "\nAccuracy: %.2f%%" % (eval[1]*100))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

info = "Name: {name}\n Alpha: {alpha}\n Epochs: {epochs}\n " \
       "Batchsize: {batch_size}\n Dataset: {dataset}\n Total params: 2 ml\n" \
       "Test Accuracy: {accuracy}".format(name='Inception_resnet', alpha=alpha, epochs=epochs,
                                          batch_size=batch_size, dataset='Cifar10', accuracy=(eval[1]*100))
with open("data/inception_resnet/info.txt", "w") as file:
    file.write(info)

warning = input("SAVE PLOTS MANUALY!!!")