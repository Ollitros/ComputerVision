from tensorflow.keras.losses import categorical_crossentropy, mean_squared_error


def custom_loss(y_true, y_pred):
    loss = mean_squared_error(y_true[0:4], y_pred[0:4]) + categorical_crossentropy(y_true[4:7], y_pred[4:7])
    return loss