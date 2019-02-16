import os
import glob
import colorsys
import random
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from functools import reduce
from PIL import Image, ImageDraw, ImageFont


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def load_train_dataset():
    # Load features
    train_x = np.load("data/train.npy")

    # Load labels
    train_y = pd.read_csv("../data/labeled/train_labels.csv")

    # Reshape features
    train_x = np.reshape(train_x, (75, 416, 416, 1))

    # Shuffle data
    train_x, train_y = shuffle(train_x, train_y)

    return train_x, train_y


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


# Convert original 256x256 images to 416x416 images for training
def convert_features_to_npy():

    features = []
    for directory in ['train_images']:
        image_path = os.path.join(os.getcwd(), 'data/images/{}'.format(directory))

        for file in glob.glob(image_path + '/*.jpg'):
            image = Image.open(file).convert('L')
            print(image)
            image = image.resize((416, 416))
            print(image)
            features.append(np.array(image))

        np.save('data/train.npy', features)
        features = []


def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors


def draw_boxes(image, boxes, box_classes, class_names, scores=None):
    """Draw bounding boxes on image.
    Draw bounding boxes with class name and optional box score on image.
    Args:
        image: An `array` of shape (width, height, 3) with values in [0, 1].
        boxes: An `array` of shape (num_boxes, 4) containing box corners as
            (y_min, x_min, y_max, x_max).
        box_classes: A `list` of indicies into `class_names`.
        class_names: A `list` of `string` class names.
        `scores`: A `list` of scores for each box.
    Returns:
        A copy of `image` modified with given bounding boxes.
    """
    # This fucking shit works only with such brainfucking canes
    image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    image = image.astype('uint8')
    image = Image.fromarray(image, mode='RGB')

    font = ImageFont.truetype(font='data/FiraMono-Medium.otf', size=10)
    thickness = 3

    colors = get_colors_for_classes(len(class_names))

    for i, c in list(enumerate(box_classes)):
        box_class = class_names[c]
        box = boxes[i]
        if isinstance(scores, np.ndarray):
            score = scores[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return np.array(image)