import cv2 as cv
import numpy as np
from FinalPractice.ObjectDetection.SlideWindow.model import InceptionNN
from FinalPractice.ObjectDetection.SlideWindow import utils


# Create model
input_shape = (256, 256, 1)
model = InceptionNN(input_shape=input_shape, alpha=1.0)
# Load pretrained weights
model.load_weights('models/weights-0.93.h5')

# Load and convert test image
x = cv.imread('image.jpg', 0)
# Make boxed prediction
boxes, predictions = utils.predict_boxes(model, x, input_shape=input_shape, strides=8, confidance=0.7)
utils.draw_rect(x, np.asarray(boxes))