import numpy as np
import pandas as pd
import tensorflow as tf
import cv2 as cv
from sklearn import preprocessing
from sklearn.utils import shuffle


def load_dataset():

    # Load features
    train_x = np.load("../data/labeled/train.npy")
    test_x = np.load("../data/labeled/test.npy")

    # Load labels
    train_y = pd.read_csv("../data/labeled/train_labels.csv")
    test_y = pd.read_csv("../data/labeled/test_labels.csv")

    # Reshape features
    train_x = np.reshape(train_x, (75, 256, 256, 1))
    test_x = np.reshape(test_x, (15, 256, 256, 1))

    # Shuffle data
    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)

    return train_x, train_y, test_x, test_y


def label_encoding(train_y, test_y):

    # Encode string labels into integers
    le = preprocessing.LabelEncoder()
    le.fit(['car', 'cat', 'planet'])
    train_y = le.transform(train_y['class'].values)
    test_y = le.transform(test_y['class'].values)

    # One hot encoding
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=3)
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=3)

    return train_y, test_y


def draw_rect(x, y):

    # This expression allows to draw color rectangles on greyscale image
    image = cv.cvtColor(x, cv.COLOR_GRAY2RGB)

    # Draw rectangle
    # image = cv.rectangle(image, (y[0], y[1]), (y[2], y[3]), color=(255, 0, 0), thickness=4)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()