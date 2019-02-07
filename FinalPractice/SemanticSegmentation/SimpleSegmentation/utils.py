import os
import pickle
import numpy as np
import tensorflow.keras.backend as K
from PIL import Image
from tensorflow.keras.losses import binary_crossentropy


def images_to_binary(size):
    features_path = 'data/weizmann_horse_db/features'
    labels_path = 'data/weizmann_horse_db/labels'

    features = len([name for name in os.listdir(features_path) if os.path.isfile(os.path.join(features_path, name))])
    labels = len([name for name in os.listdir(labels_path) if os.path.isfile(os.path.join(labels_path, name))])

    x = list()
    y = list()

    for i in range(features):
        if i+1 < 10:
            img = Image.open("data/weizmann_horse_db/features/horse00{i}.jpg".format(i=i+1))
        elif 10 <= i+1 < 100:
            img = Image.open("data/weizmann_horse_db/features/horse0{i}.jpg".format(i=i+1))
        else:
            img = Image.open("data/weizmann_horse_db/features/horse{i}.jpg".format(i=i+1))
        print(i+1)
        img = img.resize(size)
        x.append(np.array(img))

    for i in range(labels):
        if i+1 < 10:
            img = Image.open("data/weizmann_horse_db/labels/horse00{i}.jpg".format(i=i+1))
        elif 10 <= i+1 < 100:
            img = Image.open("data/weizmann_horse_db/labels/horse0{i}.jpg".format(i=i+1))
        else:
            img = Image.open("data/weizmann_horse_db/labels/horse{i}.jpg".format(i=i+1))
        print(i+1)
        img = img.resize(size)
        y.append(np.array(img))

    with open('data/weizmann_horse_db/train', "wb") as file:
        pickle.dump(x, file)

    with open('data/weizmann_horse_db/test', "wb") as file:
        pickle.dump(y, file)


def load_dataset():
    with open('data/weizmann_horse_db/train', "rb") as file:
        x = pickle.load(file)

    with open('data/weizmann_horse_db/test', "rb") as file:
        y = pickle.load(file)

    return np.asarray(x), np.asarray(y)


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss