import numpy as np
import pandas as pd
import tensorflow as tf
import cv2 as cv
from sklearn import preprocessing


def load_dataset():

    # Load features
    train_x = np.load("../data/labeled/train.npy")
    train_y = pd.read_csv("../data/labeled/train_labels.csv")

    # Load labels
    test_x = np.load("../data/labeled/test.npy")
    test_y = pd.read_csv("../data/labeled/test_labels.csv")

    # Reshape features
    train_x = np.reshape(train_x, (75, 256, 256, 1))
    test_x = np.reshape(test_x, (15, 256, 256, 1))

    return train_x, train_y, test_x, test_y


def label_encoding(train_y, test_y):

    # Saving values
    y1 = train_y[['xmin',  'ymin',  'xmax',  'ymax']]
    y2 = test_y[['xmin', 'ymin', 'xmax', 'ymax']]

    # Encode string labels into integers
    le = preprocessing.LabelEncoder()
    le.fit(['car', 'cat', 'planet'])
    train_y = le.transform(train_y['class'].values)
    test_y = le.transform(test_y['class'].values)

    # One hot encoding
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=3)
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=3)

    # Join together
    train_y = y1.join(pd.DataFrame(train_y, dtype='int'))
    train_y = train_y.values
    test_y = y2.join(pd.DataFrame(test_y, dtype='int'))
    test_y = test_y.values

    return train_y, test_y


def draw_rect(x, y):

    # This expression allows to draw color rectangles on greyscale image
    x = cv.cvtColor(x, cv.COLOR_GRAY2RGB)

    # Draw rectangle
    image = cv.rectangle(x, (y[0], y[1]), (y[2], y[3]), color=(255, 0, 0), thickness=4)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()