"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse
import io
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import cv2 as cv
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import Model, load_model, save_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from FinalPractice.ObjectDetection.YOLO.yolo_model import (preprocess_true_boxes, yolo_body, yolo_eval, yolo_head, yolo_loss)
from FinalPractice.ObjectDetection.YOLO.yolo_utils import draw_boxes
from FinalPractice.ObjectDetection.YOLO import utils

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS


def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]

    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(np.asarray([box]), anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)


def create_model(anchors, class_names, load_pretrained=True, freeze_body=True):
    '''
    returns the body of the model and the model
    # Params:
    load_pretrained: whether or not to load the pretrained model or initialize all weights
    freeze_body: whether or not to freeze all weights except for the last layer's
    # Returns:
    model_body: YOLOv2 with new output layer
    model: YOLOv2 with custom loss Lambda layer
    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 1))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('data/models', 'for_pre_load.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('data/model', 'for_pre_load_recreated.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model


def train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes, batch_size, epochs,
          validation_split=0.1):
    '''
    retrain/fine-tune the model
    logs training with tensorboard
    saves training weights in current directory
    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    print(model.summary())

    logging = TensorBoard()
    checkpoint = ModelCheckpoint("data/models/trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=validation_split,
              batch_size=batch_size[0],
              epochs=epochs[0],
              callbacks=[logging])
    model.save_weights('data/models/trained_stage_1.h5')

    model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False)

    model.load_weights('data/models/trained_stage_1.h5')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=batch_size[1],
              epochs=epochs[1],
              callbacks=[logging])

    model.save_weights('data/models/trained_stage_2.h5')

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=batch_size[2],
              epochs=epochs[2],
              callbacks=[logging, checkpoint, early_stopping])

    model.save_weights('data/models/trained_stage_3.h5')

    # Save model body not model to use in testing
    model_body.save('data/models/model.h5')


def draw(model_body, class_names, anchors, image_data, image_set='val',
            weights_name='data/models/trained_stage_3_best.h5', out_path="data/output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
    if image_set == 'train':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[:int(len(image_data)*.9)]])
    elif image_set == 'val':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[int(len(image_data)*.9):]])
    elif image_set == 'all':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data])
    else:
        ValueError("draw argument image_set must be 'train', 'val', or 'all'")
    # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0, iou_threshold=0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)
        print(out_scores)
        print(out_classes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                      class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path, str(i)+'.png'))

        # To display (pauses the program):
        cv.imshow('image', image_with_boxes)
        cv.waitKey(0)
        cv.destroyAllWindows()


def main():
    classes_path = "data/classes.txt"
    anchors_path = "data/yolo_anchors.txt"

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)

    # Load dataset
    train_x, train_y, test_x, test_y = utils.load_dataset()

    # Box preprocessing
    train_boxes, test_boxes, train_extents, test_extents = utils.preprocessing_boxes(train_y, test_y)

    anchors = YOLO_ANCHORS

    detectors_mask, matching_true_boxes = get_detector_mask(train_boxes, anchors)

    model_body, model = create_model(anchors, class_names, load_pretrained=False)

    train_boxes = np.reshape(train_boxes, [75, 1, 5])
    train(model, class_names, anchors, train_x, train_boxes, detectors_mask, matching_true_boxes,
          batch_size=[2, 2, 2], epochs=[10, 100, 100])

    draw(model_body, class_names, anchors, train_x, image_set='val',
         weights_name='data/models/trained_stage_3_best.h5', save_all=False)


if __name__ == '__main__':
    main()