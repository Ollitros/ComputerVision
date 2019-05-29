import numpy as np
import cv2 as cv
import time
from keras.datasets import cifar10
from FinalPractice.GANs.CVAE_GAN.network.model import Gan


def sample_images(model, epoch):

    sampled_labels = np.arange(0, 10).reshape(-1, 1)

    batch_noise = np.random.normal(0, 1, (10, 32 * 32 * 3))
    prediction = model.generator.predict([batch_noise, sampled_labels])
    prediction = np.float32(prediction * 255)[:, :, :, 1:4]

    image = cv.hconcat([prediction[0], prediction[1], prediction[2], prediction[3], prediction[4],
                        prediction[5],  prediction[6], prediction[7], prediction[8], prediction[9]])

    cv.imwrite('data/images/{step}.jpg'.format(step=epoch), image)


def train_model(input_shape, x, y, epochs, batch_size):

    model = Gan()
    model.build_train_functions()

    errG_sum = errD_sum = 0
    display_iters = 1

    t0 = time.time()
    model.load_weights()

    iters = np.asarray(x).shape[0] // batch_size
    if np.asarray(x).shape[0] - iters * batch_size == 0:
        train_x = x.tolist()
        train_x.append(train_x[-1])
    train_x = np.asarray(x)
    for epoch in range(epochs):

        # Train generator predict first frame form noise
        # Train discriminator
        batch_y = []
        for f in range(batch_size):
            batch_y.append(train_x[0])

        # Train discriminator
        step = 0
        for iter in range(iters):
            batch_noise = np.random.normal(0, 1, (batch_size, input_shape[0] * input_shape[1] * input_shape[2]))
            fake = model.generator.predict([batch_noise, y])
            fake = np.reshape(fake[:, :, :, 1:4], (batch_size, input_shape[0] * input_shape[1] * input_shape[2]))
            errD = model.train_discriminator(X=fake, Y=train_x[step: (step + batch_size)], cond=y[step: (step + batch_size)])
            step = step + batch_size

            if iter % 100 == 0:
                print("Discriminator interior step", iter)

        errD_sum += errD[0]

        # Train generator
        step = 0
        for iter in range(iters):
            batch_noise = np.random.normal(0, 1, (batch_size, input_shape[0] * input_shape[1] * input_shape[2]))
            errG = model.train_generator(X=batch_noise,
                                         Y=train_x[step:(step + batch_size)],
                                         cond=y[step: (step + batch_size)])
            step = step + batch_size

            if iter % 100 == 0:
                print("Generator interior step", iter)

        errG_sum += errG[0]

        # Visualization
        if epoch % display_iters == 0:
            print("----------")
            print('[iter %d] Loss_D: %f Loss_G: %f  time: %f' % (epoch, errD_sum / display_iters,
                                                                 errG_sum / display_iters, time.time() - t0))
            print("----------")
            display_iters = display_iters + 1

        # Makes predictions after each epoch and save into temp folder.
        if epoch % 1 == 0:
            sample_images(model, epoch + 0)
            model.save_weights()


def main():

    # Load the dataset
    (x_train, y_train), (_, _) = cifar10.load_data()
    print(x_train.shape)

    x_train = x_train.astype('float32')
    x_train /= 255

    epochs = 10
    batch_size = 128
    input_shape = (32, 32, 3)
    train_model(input_shape=input_shape, x=x_train, y=y_train, epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    main()


