import numpy as np
import pandas as pd
import tensorflow as tf
import cv2 as cv
import time
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


def show_images(x):

    # This expression allows to draw color rectangles on greyscale image
    image = cv.cvtColor(x, cv.COLOR_GRAY2RGB)

    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def draw_rect(x, y):

    # This expression allows to draw color rectangles on greyscale image
    image = cv.cvtColor(x, cv.COLOR_GRAY2RGB)
    print(y.shape)
    # Draw rectangle
    for i in range(y.shape[0]):
        image = cv.rectangle(image, (y[i][0], y[i][1]), (y[i][2], y[i][3]), color=(255, 0, 0), thickness=1)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def predict_boxes(model, x, input_shape, strides, confidance):
    start_time = time.time()
    image_shape = x.shape
    x_min = 0
    y_min = 0
    predictions = []
    boxes = []
    width = image_shape[0]
    height = image_shape[1]

    while height - (y_min + input_shape[1]) > 0:
        print(height - (y_min + input_shape[1]))
        while width - (x_min + input_shape[0]) > 0:
            cut = x[x_min: (x_min + input_shape[0]), y_min: (y_min + input_shape[1])]
            prediction = model.predict(np.reshape(cut, [1, 256, 256, 1]))
            threshold = prediction > confidance

            if np.any(threshold):
                predictions.append(prediction)
                boxes.append([x_min, y_min,
                             (x_min + input_shape[0]), (y_min + input_shape[1])])

            x_min = x_min + strides

        x_min = 0
        y_min = y_min + strides

    print("\n\n\nThe program has been finished for --- %s seconds ---" % (time.time() - start_time))
    return boxes, predictions