import os
import time
import matplotlib.pyplot as plt
import numpy as np
from FinalPractice.SemanticSegmentation.SimpleSegmentation.utils import images_to_binary, load_dataset, bce_dice_loss
from FinalPractice.SemanticSegmentation.SimpleSegmentation.model import model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop


start_time = time.time()

size = (128, 128)
# if files exists -> no loading
if os.path.exists('data/weizmann_horse_db/train') and os.path.exists('data/weizmann_horse_db/test'):
    x, y = load_dataset()
else:
    # Load images and annotations from directories and saves them into binary file
    images_to_binary(size=size)
    x, y = load_dataset()

x = np.resize(x, [328, 128, 128, 3])
y = np.resize(y, [328, 128, 128, 1])
x_train, x_test, y_train, y_test = train_test_split(x, y)
plt.imshow(x_test[0])
plt.show()

model = model()

model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=['accuracy'])


# checkpoint
filepath = "data/models/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=1, verbose=1, factor=0.99, min_lr=0.00001)

print(model.summary())

# Fit model
model.fit(x_train, y_train, epochs=50, batch_size=10, validation_data=(x_test, y_test), callbacks=[learning_rate_reduction, checkpoint])

# model.load_weights('data/model.h5')


model.save_weights('data/model.h5')
model.load_weights('data/model.h5')
prediction = model.predict(x_test)

for n in range(3):
    plt.subplot(3, 3, n+1)
    plt.imshow(x_test[n])

for n in range(3):
    plt.subplot(3, 3, n+4)
    plt.imshow(np.reshape(y_test[n], size))

for n in range(3):
    plt.subplot(3, 3, n+7)
    plt.imshow(np.reshape(prediction[n], size))

plt.show()

print("\n\n\nThe program has been finished for --- %s seconds ---" % (time.time() - start_time))