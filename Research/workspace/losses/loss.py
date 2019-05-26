from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
import tensorflow.keras.backend as K


def custom_loss(y_true, y_pred):

    loss = K.abs(K.pow(y_true, 2) - K.pow(y_pred, 2))

    return loss