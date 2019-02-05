import os
import time
import matplotlib.pyplot as plt
import numpy as np
from FinalPractice.SemanticSegmentation.SimpleSegmentation.utils import images_to_binary, load_dataset
from FinalPractice.SemanticSegmentation.SimpleSegmentation.model import model
from sklearn.model_selection import train_test_split


start_time = time.time()


# if files exists -> no loading
if os.path.exists('data/weizmann_horse_db/train') and os.path.exists('data/weizmann_horse_db/test'):
    x, y = load_dataset()
else:
    # Load images and annotations from directories and saves them into binary file
    images_to_binary()
    x, y = load_dataset()

x = np.resize(x, [328, 100, 100, 3])
y = np.resize(y, [328, 100, 100, 1])
x_train, x_test, y_train, y_test = train_test_split(x, y)

model = model()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, epochs=30, validation_data=[x_test, y_test], batch_size=30)

model.save_weights('data/model.h5')
model.load_weights('data/model.h5')
prediction = model.predict(x_test)

for n in range(3):
    plt.subplot(3, 3, n+1)
    plt.imshow(x_test[n])

for n in range(3):
    plt.subplot(3, 3, n+4)
    plt.imshow(np.reshape(y_test[n], [100, 100]))

for n in range(3):
    plt.subplot(3, 3, n+7)
    plt.imshow(np.reshape(prediction[n], [100, 100]))

plt.show()

print("\n\n\nThe program has been finished for --- %s seconds ---" % (time.time() - start_time))