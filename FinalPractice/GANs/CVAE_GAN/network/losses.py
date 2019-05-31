from keras.layers import Lambda, concatenate
from tensorflow.contrib.distributions import Beta
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy, binary_crossentropy
import keras.backend as K
import tensorflow as tf
import numpy as np

""" 
    Loss implementations 
    Code has been politely stolen from https://github.com/shaoanlu/faceswap-GAN/blob/master/networks/losses.py
"""


def custom_loss(generator, noise, fake, real):
    def loss(y_true, y_pred):

        # Reconstruction loss
        loss_R = reconstruction_loss(generator, real, fake)

        # Edge loss
        loss_E = edge_loss(real, fake)

        # Binary crossentropy loss
        bc = binary_crossentropy(y_true, y_pred)

        # Total loss
        total_loss = K.mean(loss_E + loss_R + bc)

        return total_loss

    return loss


def first_order(x, axis=1):

    img_nrows = x.shape[1]
    img_ncols = x.shape[2]
    if axis == 1:
        return K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    elif axis == 2:
        return K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    else:
        return None


def calc_loss(pred, target, loss='l2'):
    if loss.lower() == "l2":
        return K.mean(K.square(pred - target))
    elif loss.lower() == "l1":
        return K.mean(K.abs(pred - target))
    elif loss.lower() == "cross_entropy":
        return -K.mean(K.log(pred + K.epsilon()) * target + K.log(1 - pred + K.epsilon()) * (1 - target))
    else:
        raise ValueError(f'Recieve an unknown loss type: {loss}.')


def reconstruction_loss(generator, real, fake):

    loss_G = 0
    loss_G += 1.0 * calc_loss(fake, real, "l1")

    for out in generator.outputs[:-1]:
        out_size = out.get_shape().as_list()
        resized_real = tf.image.resize_images(real, out_size[1:3])
        loss_G += 1.0 * calc_loss(out, resized_real, "l1")

    return loss_G


def edge_loss(real, fake):

    # Edge loss
    gen_inputs = tf.reshape(real, (128, 32, 32, 1))
    gen_outputs = tf.reshape(fake, (128, 32, 32, 1))

    loss_G = 0
    loss_G += 0.1 * calc_loss(first_order(gen_outputs, axis=1), first_order(gen_inputs, axis=1), "l1")
    loss_G += 0.1 * calc_loss(first_order(gen_outputs, axis=2), first_order(gen_inputs, axis=2), "l1")

    return loss_G