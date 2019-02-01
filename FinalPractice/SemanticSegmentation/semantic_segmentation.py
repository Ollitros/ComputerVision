import os
import time
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, \
     Input, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, AveragePooling2D, concatenate, Lambda, add, Activation
from FinalPractice.SemanticSegmentation.utils import images_to_binary, load_dataset


start_time = time.time()

x = np.array([])
y = np.array([])

# if files exists -> no loading
if os.path.exists('data/weizmann_horse_db/train') and os.path.exists('data/weizmann_horse_db/test'):
    x, y = load_dataset()
else:
    # Load images and annotations from directories and saves them into binary file
    images_to_binary()
    x, y = load_dataset()

plt.imshow(x[50])
plt.show()
plt.imshow(y[50])
plt.show()



print("\n\n\nThe program has been finished for --- %s seconds ---" % (time.time() - start_time))