import os
import time
import matplotlib.pyplot as plt
import numpy as np
from FinalPractice.SemanticSegmentation.SimpleSegmentation.utils import images_to_binary, load_dataset, prediction_masking
from FinalPractice.SemanticSegmentation.SimpleSegmentation.model import model
from sklearn.model_selection import train_test_split


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

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.load_weights('data/model.h5')
print(model.summary())
model.fit(x_train, y_train, epochs=1, validation_data=[x_test, y_test], batch_size=10)

model.save_weights('data/model.h5')
model.load_weights('data/model.h5')
prediction = model.predict(x_test)

masked = prediction_masking(prediction[0], x_test[0])
plt.imshow(masked)
plt.show()
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