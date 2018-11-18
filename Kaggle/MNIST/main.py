import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, Dense, Dropout, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


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
train_y = to_categorical(train_y)


# Build model
model = Sequential()
model.add(Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Dropout(0.25))

model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(train_y.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create generator for image augmentation
datagen = ImageDataGenerator(rotation_range=0.5, zoom_range=0.5, width_shift_range=0.5,  height_shift_range=0.5)
datagen.fit(train_x)

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
history = model.fit_generator(datagen.flow(train_x, train_y, 250), epochs=10, callbacks=[learning_rate_reduction])
print(history.history['loss'])


# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()

# Save model
save(model, path_model='data/models/model.json', path_weights='data/models/model.h5')


# Make predictions
predictions = model.predict(test_x)
pred = np.argmax(predictions, axis=1)

# Make submission file
my_submission = pd.DataFrame({'ImageId': range(1, len(test_x)+1), 'Label': pred})
my_submission.to_csv("data/submission.csv", index=False)
