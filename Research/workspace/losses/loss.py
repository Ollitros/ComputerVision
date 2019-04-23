from tensorflow.keras.losses import mean_squared_error


def custom_loss(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)