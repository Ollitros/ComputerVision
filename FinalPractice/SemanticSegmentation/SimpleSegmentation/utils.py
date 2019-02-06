import os
import pickle
import numpy as np
import cv2 as cv
from PIL import Image


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


def prediction_masking(prediction, original):

    # Load two images
    img1 = prediction
    img2 = original
    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv.bitwise_and(img2, img2, mask=mask)
    # Put logo in ROI and modify the main image
    dst = cv.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    return img1