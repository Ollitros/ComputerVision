#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import colorsys
import imghdr
import os
import random

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from FinalPractice.ObjectDetection.YOLO.yolo_model import yolo_eval, yolo_head


def main():
    model_path = 'data/models/model.h5'
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    classes_path = "data/classes.txt"
    anchors_path = "data/yolo_anchors.txt"
    output_path = 'data/images/output_images'
    test_path = 'data/images/test_images'

    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    print(yolo_model.output)
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

    scores, boxes, classes = yolo_eval(yolo_outputs, (416, 416))

    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={
        yolo_model.input: image_data,
        K.learning_phase(): 0
    })


if __name__ == '__main__':
    main()