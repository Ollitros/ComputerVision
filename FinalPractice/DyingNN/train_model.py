import numpy as np
import cv2 as cv
import time
from FinalPractice.DyingNN.network.model import Gan


def train_model(input_shape, x, epochs, batch_size, filter_coeff):
    model = Gan(input_shape=input_shape, num_classes=10, batch_size=batch_size, filter_coeff=filter_coeff)
    # model.load_weights()

    t0 = time.time()
    iters = np.asarray(x).shape[0] // batch_size
    if np.asarray(x).shape[0] - iters * batch_size == 0:
        train_x = x.tolist()
        train_x.append(train_x[-1])
    train_x = np.asarray(x)
    train_x = np.reshape(train_x, (train_x.shape[0], input_shape[0], input_shape[1], input_shape[2]))

    valid = np.ones((batch_size, 1))

    for epoch in range(epochs):

        step = 0
        for iter in range(iters):

            # ---------------------
            #  Train Generator
            # ---------------------

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, input_shape[0] * input_shape[1] * input_shape[2]))
            g_loss = model.generator.train_on_batch([noise, train_x[step:(step + batch_size)]],
                                                     valid)
            step = step + batch_size

            if iter % 100 == 0:
                print("Interior step", iter)

        # Plot the progress
        print("%d  ---- [G loss: %f] --- time: %f" %
             (epoch, g_loss, time.time() - t0))

        noise = np.random.normal(0, 1, (batch_size, input_shape[0] * input_shape[1] * input_shape[2]))
        prediction = model.generator.predict([noise, train_x])
        cv.imwrite('data/temp/image{i}.jpg'.format(i=epoch + 0), prediction[0] * 255)
        model.save_weights()


def main():

    # Load the dataset
    x = np.load('data/x.npy')
    input_shape = (64, 64, 3)
    print(x.shape)

    x = np.asarray(x).astype('float32')
    x /= 255
    epochs = 50
    batch_size = 5
    filter_coeff = 1.0

    train_model(input_shape=input_shape, x=x, epochs=epochs, batch_size=batch_size,  filter_coeff=filter_coeff)


if __name__ == '__main__':
    main()


