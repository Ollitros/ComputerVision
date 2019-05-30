import numpy as np
import cv2 as cv
import time
from keras.datasets import cifar10, mnist
from FinalPractice.GANs.CVAE_GAN.network.model import Gan


def sample_images(model, epoch):

    # sampled_labels = np.arange(0, 10).reshape(-1, 1)
    #
    # batch_noise = np.random.normal(0, 1, (10, 32 * 32 * 3))
    # prediction = model.generator.predict([batch_noise, sampled_labels])
    # prediction = np.float32(prediction * 255)[:, :, :, 1:4]
    #
    # image = cv.hconcat([prediction[0], prediction[1], prediction[2], prediction[3], prediction[4],
    #                     prediction[5],  prediction[6], prediction[7], prediction[8], prediction[9]])
    #
    # cv.imwrite('data/images/{step}.jpg'.format(step=epoch), image)

    batch_noise = np.random.rand(10, 4 * 4 * 128)
    prediction = model.decoder.predict(batch_noise)
    prediction = np.float32(prediction * 255)[:, :, :, 1:4]

    image = cv.hconcat([prediction[0], prediction[1], prediction[2], prediction[3], prediction[4],
                        prediction[5],  prediction[6], prediction[7], prediction[8], prediction[9]])

    cv.imwrite('data/images/{step}.jpg'.format(step=epoch), image)


def test(x, y):

    model = Gan()
    model.build_train_functions()
    model.load_weights()

    for i in range(100):

        batch_noise = np.random.normal(0, 1, (1, 32 * 32 * 3))
        # prediction = model.generator.predict([batch_noise, np.asarray([y[i]])])
        prediction = model.generator.predict([batch_noise, np.asarray([5])])
        prediction = np.float32(prediction * 255)[:, :, :, 1:4]

        print(y[i])
        print(prediction.shape)
        cv.imwrite('data/{i}.jpg'.format(i=i), np.reshape(prediction, (32, 32, 3)))
        cv.imshow('asdasd', np.reshape(prediction, (32, 32, 3)))
        cv.waitKey(0)
        cv.destroyAllWindows()


def train_model(input_shape, x, y, epochs, batch_size):

    model = Gan()
    model.build_train_functions()

    errG_sum = errD_sum = 0
    display_iters = 1

    t0 = time.time()
    # model.load_weights()

    iters = np.asarray(x).shape[0] // batch_size
    if np.asarray(x).shape[0] - iters * batch_size == 0:
        train_x = x.tolist()
        train_x.append(train_x[-1])
    train_x = np.asarray(x)
    for epoch in range(epochs):

        # Train discriminator
        step = 0
        for iter in range(iters):
            batch_noise = np.random.normal(0, 1, (batch_size, input_shape[0] * input_shape[1] * input_shape[2]))
            fake = model.generator.predict([batch_noise, y[step: (step + batch_size)]])
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
    (x_train, y_train), (_, _) = mnist.load_data()
    input_shape = (32, 32, 3)
    print(x_train.shape)

    x_resized = []
    for i in x_train:
        x_resized.append(cv.resize(i, (32, 32)))

    print(np.asarray(x_resized).shape)

    x = np.random.normal(0, 1, (60000, 32, 32, 3))
    x[:, :, :, 0] = x_resized
    x[:, :, :, 1] = x_resized
    x[:, :, :, 2] = x_resized
    print(x.shape)

    x = x.astype('float32')
    x /= 255

    epochs = 10
    batch_size = 128

    # test(x, y_train)
    train_model(input_shape=input_shape, x=x, y=y_train, epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    main()


