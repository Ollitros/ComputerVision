import numpy as np
import cv2 as cv

image = cv.imread('data/src.jpg')
image = cv.resize(image, (64, 64))

x = []
for i in range(1000):
    x.append(image)
x = np.asarray(x)

np.save('data/x.npy', x)

