import numpy as np
import cv2 as cv
import random
from keras.models import Sequential
from FinalPractice.DyingNN.network.model import Gan


def killing(input_shape, x):

    model = Gan(input_shape=input_shape, num_classes=10, batch_size=5, filter_coeff=1.0)
    model.load_weights()
    print(model.g.summary())

    outputs = [layer.output for layer in model.g.layers]  # all layer outputs
    print(outputs)

    layers = []
    layers.append(model.g.get_layer('input_3'))
    layers.append(model.g.get_layer('reshape_1'))
    layers.append(model.g.get_layer('conv2d_1'))
    layers.append(model.g.get_layer('conv2d_2'))
    layers.append(model.g.get_layer('instance_normalization_1'))
    layers.append(model.g.get_layer('leaky_re_lu_1'))
    layers.append(model.g.get_layer('conv2d_3'))
    layers.append(model.g.get_layer('instance_normalization_2'))
    layers.append(model.g.get_layer('leaky_re_lu_2'))
    layers.append(model.g.get_layer('conv2d_4'))
    layers.append(model.g.get_layer('instance_normalization_3'))
    layers.append(model.g.get_layer('leaky_re_lu_3'))
    layers.append(model.g.get_layer('conv2d_5'))
    layers.append(model.g.get_layer('conv2d_6'))
    layers.append(model.g.get_layer('instance_normalization_4'))
    layers.append(model.g.get_layer('leaky_re_lu_4'))
    layers.append(model.g.get_layer('pixel_shuffler_1'))
    layers.append(model.g.get_layer('conv2d_7'))
    layers.append(model.g.get_layer('instance_normalization_5'))
    layers.append(model.g.get_layer('leaky_re_lu_5'))
    layers.append(model.g.get_layer('pixel_shuffler_2'))
    layers.append(model.g.get_layer('conv2d_8'))
    layers.append(model.g.get_layer('instance_normalization_6'))
    layers.append(model.g.get_layer('leaky_re_lu_6'))
    layers.append(model.g.get_layer('pixel_shuffler_3'))
    layers.append(model.g.get_layer('conv2d_9'))

    weights_list = []
    layers_index = []
    weights_zeros = []

    # conv2d_1
    weights = np.asarray(layers[2].get_weights())
    ones = np.ones_like(weights)
    weights_zeros.append(ones)
    weights_list.append(weights)
    layers_index.append(2)

    # conv2d_2
    weights = np.asarray(layers[3].get_weights())
    ones_w = np.ones_like(weights[0])
    ones_b = np.ones_like(weights[1])
    ones = [ones_w, ones_b]
    ones = np.asarray(ones)

    weights_zeros.append(ones)
    weights_list.append(weights)
    layers_index.append(3)

    # conv2d_3
    weights = np.asarray(layers[6].get_weights())
    ones_w = np.ones_like(weights[0])
    ones_b = np.ones_like(weights[1])
    ones = [ones_w, ones_b]
    ones = np.asarray(ones)

    weights_zeros.append(ones)
    weights_list.append(weights)
    layers_index.append(6)

    # conv2d_4
    weights = np.asarray(layers[9].get_weights())
    ones_w = np.ones_like(weights[0])
    ones_b = np.ones_like(weights[1])
    ones = [ones_w, ones_b]
    ones = np.asarray(ones)

    weights_zeros.append(ones)
    weights_list.append(weights)
    layers_index.append(9)

    # conv2d_5
    weights = np.asarray(layers[12].get_weights())
    ones_w = np.ones_like(weights[0])
    ones_b = np.ones_like(weights[1])
    ones = [ones_w, ones_b]
    ones = np.asarray(ones)

    weights_zeros.append(ones)
    weights_list.append(weights)
    layers_index.append(12)

    # conv2d_6
    weights = np.asarray(layers[13].get_weights())
    ones_w = np.ones_like(weights[0])
    ones_b = np.ones_like(weights[1])
    ones = [ones_w, ones_b]
    ones = np.asarray(ones)

    weights_zeros.append(ones)
    weights_list.append(weights)
    layers_index.append(13)

    # conv2d_7
    weights = np.asarray(layers[17].get_weights())
    ones_w = np.ones_like(weights[0])
    ones_b = np.ones_like(weights[1])
    ones = [ones_w, ones_b]
    ones = np.asarray(ones)

    weights_zeros.append(ones)
    weights_list.append(weights)
    layers_index.append(17)

    # conv2d_8
    weights = np.asarray(layers[21].get_weights())
    ones_w = np.ones_like(weights[0])
    ones_b = np.ones_like(weights[1])
    ones = [ones_w, ones_b]
    ones = np.asarray(ones)

    weights_zeros.append(ones)
    weights_list.append(weights)
    layers_index.append(21)

    # for i in range(512):

        # for n in range():
        # shape = np.asarray(weights_zeros[0]).shape
        # print(shape)

        # weights = np.asarray(weights_zeros[1])
        # print(weights[0].shape)
        # print(weights[1].shape)

        # weights = np.asarray(weights_zeros[2])
        # print(weights[0].shape)
        # print(weights[1].shape)

        # weights = np.asarray(weights_zeros[3])
        # print(weights[0].shape)
        # print(weights[1].shape)

        # weights = np.asarray(weights_zeros[4])
        # print(weights[0].shape)
        # print(weights[1].shape)

        # weights = np.asarray(weights_zeros[5])
        # print(weights[0].shape)
        # print(weights[1].shape)

        # weights = np.asarray(weights_zeros[6])
        # print(weights[0].shape)
        # print(weights[1].shape)

        # weights = np.asarray(weights_zeros[7])
        # print(weights[0].shape)
        # print(weights[1].shape)



        # layers[layers_index[0]].set_weights(zeros)

        # conv2d_9
        # weights = np.asarray(layers[25].get_weights())
        # print(weights)
        # print(weights[0].shape)
        # print(weights[1].shape)
        # zeros_w = np.zeros_like(weights[0])
        # zeros_b = np.zeros_like(weights[1])
        # zeros = [zeros_w, zeros_b]
        # zeros = np.asarray(zeros)
        # print(zeros.shape)
        # layers[25].set_weights(zeros)

    weights = []
    for i in range(256):
        for n in range(3):
            for f in range(3):
                for b in range(64):
                    weights = np.asarray(layers[21].get_weights())
                    weights = weights[0]
                    weights[n, f, b, i] = 0
                    print(weights)

        weight = np.asarray(layers[21].get_weights())
        layers[21].set_weights([weights, weight[1]])
        generator = Sequential(layers)

        if i % 16 == 0:
            noise = np.random.normal(0, 1, (10, input_shape[0] * input_shape[1] * input_shape[2]))
            prediction = generator.predict(noise)
            prediction = np.float32(prediction[0] * 255)
            cv.imwrite('data/dying/{i}.jpg'.format(i=i), np.reshape(prediction, input_shape))


def main():

    # Load the dataset
    x = np.load('data/x.npy')
    input_shape = (64, 64, 3)
    print(x.shape)

    x = np.asarray(x).astype('float32')
    x /= 255

    killing(input_shape=input_shape, x=x)


if __name__ == '__main__':
    main()