import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model, model_from_json
from keras.layers import Conv2D, Dense, Dropout, MaxPool2D, Flatten, InputLayer, Reshape, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split


# Save Sequential in two files
def save(model, path_model='data/models/model.json', path_weights='data/models/model.h5'):
    model_json = model.to_json()
    with open(path_model, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(path_weights)
    print("Saved model to disk")


# Load model from json and weights
def load():
    path = 'data/models/model.json'
    json_file = open(path, 'r')
    # json_file = open(self.path_to_model_json, 'r')
    print('in preidcuit')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('data/models/model.h5')
    print("Loaded model from disk")

    return model


# Load data from local folder
train_dataset = pd.read_csv('data/train.csv')
test_dataset = pd.read_csv('data/test.csv')

# Split data on labels and features
train_y = train_dataset['label']
train_x = train_dataset.drop(['label'], axis=1)

# Get from DataFrames values and reshape them
train_y = train_y.values
train_x = train_x.values.reshape(-1, 28, 28, 1)
test_x = test_dataset.values.reshape(-1, 28, 28, 1)

# Transform targets into one-hot vector
train_y = to_categorical(train_y, num_classes=10)

# Create test data and make data normalization
train_x = train_x / 255.0
test_x = test_x / 255.0

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.01)

# Build model
model = Sequential()
model.add(Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
model.add(Dropout(0.25))

model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create generator for image augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,               # set input mean to 0 over the dataset
        samplewise_center=False,                # set each sample mean to 0
        featurewise_std_normalization=False,    # divide inputs by std of the dataset
        samplewise_std_normalization=False,     # divide each input by its std
        zca_whitening=False,                    # apply ZCA whitening
        rotation_range=10,                      # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,                         # Randomly zoom image
        width_shift_range=0.1,                  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,                 # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,                  # randomly flip images
        vertical_flip=False)                    # randomly flip images
datagen.fit(train_x)

# checkpoint
filepath = "data/models/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.99, min_lr=0.00001)

# Fit model
history = model.fit_generator(datagen.flow(train_x, train_y, 850), epochs=100, validation_data=(valid_x, valid_y),
                              callbacks=[learning_rate_reduction, checkpoint])

# Save model
save(model, path_model='data/models/model.json', path_weights='data/models/model_10.h5')


# Make predictions
predictions = model.predict(test_x)
pred = np.argmax(predictions, axis=1)

# Make submission file
my_submission = pd.DataFrame({'ImageId': range(1, len(test_x)+1), 'Label': pred})
my_submission.to_csv("data/submission.csv", index=False)

# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()