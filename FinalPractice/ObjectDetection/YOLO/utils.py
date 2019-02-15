import numpy as np
import pandas as pd
import tensorflow as tf
import cv2 as cv
from sklearn import preprocessing
from sklearn.utils import shuffle
from functools import reduce


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


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


def preprocessing_boxes(train_y, test_y):
    # Box preprocessing.
    # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
    # Extracting targets
    train = train_y[['xmin', 'ymin', 'xmax', 'ymax']]
    test = test_y[['xmin', 'ymin', 'xmax', 'ymax']]

    # Encode string labels into integers
    le = preprocessing.LabelEncoder()
    le.fit(['car', 'cat', 'planet'])
    y1 = le.transform(train_y['class'].values)
    y2 = le.transform(test_y['class'].values)

    # Join together
    train_y = train.join(pd.DataFrame(y1, dtype='int'))
    train_y = train_y.values
    test_y = test.join(pd.DataFrame(y2, dtype='int'))
    test_y = test_y.values

    # Get extents as y_min, x_min, y_max, x_max, class for comparision with model output.
    train_extents = train_y[:, [2, 1, 4, 3, 0]]
    test_extents = test_y[:, [2, 1, 4, 3, 0]]

    # # Get box parameters as x_center, y_center, box_width, box_height, class.
    boxes = []
    for i in [train_y, test_y]:

        boxes_xy = 0.5 * (i[:, 3:5] + i[:, 1:3])
        boxes_wh = i[:, 3:5] - i[:, 1:3]
        boxes_xy = boxes_xy / 256
        boxes_wh = boxes_wh / 256
        boxes.append(np.concatenate((boxes_xy, boxes_wh, i[:, 0:1]), axis=1))

    train_boxes = boxes[0]
    test_boxes = boxes[1]

    return train_boxes, test_boxes, train_extents, test_extents





