import numpy as np
import cv2 as cv
import time
import keras.backend as K
from keras.datasets import mnist
from FinalPractice.GANs.CVAE_GAN.network.model import Gan


def sample_images(model, epoch):

    sampled_labels = np.arange(0, 10).reshape(-1, 1)

    batch_noise = np.random.normal(0, 1, (10, 32 * 32 * 1))
    useless = np.random.normal(0, 1, (10, 32, 32, 1))
    prediction = model.generator.predict([batch_noise, sampled_labels, useless])

    # Rescale images
    # prediction = np.float32(prediction) * 127.5 + 127.5
    prediction = np.float32(prediction) * 255

    image = cv.hconcat([prediction[0], prediction[1], prediction[2], prediction[3], prediction[4],
                        prediction[5],  prediction[6], prediction[7], prediction[8], prediction[9]])

    cv.imwrite('data/images/{step}.jpg'.format(step=epoch), image)


def test(input_shape, x, y):

    model = Gan(input_shape=input_shape, num_classes=10)
    model.load_weights()

    for i in range(100):
        # useless = np.random.normal(0, 1, (10, 32, 32, 1))
        # batch_noise = np.random.normal(0, 1, (1, 32 * 32 * 1))
        # prediction = model.generator.predict([batch_noise, np.asarray([5]), useless])
        # prediction = np.float32(prediction * 255)
        #
        # print(y[i])
        # print(prediction.shape)
        # cv.imwrite('data/{i}.jpg'.format(i=i), np.reshape(prediction, (32, 32, 1)))
        # cv.imshow('asdasd', np.reshape(prediction, (32, 32, 1)))
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        useless = np.random.normal(0, 1, (1, 4 * 4 * 128))
        prediction = model.decoder.predict(useless)
        prediction = np.float32(prediction * 255)

        print(y[i])
        print(prediction.shape)
        cv.imwrite('data/{i}.jpg'.format(i=i), np.reshape(prediction, (32, 32, 1)))
        cv.imshow('asdasd', np.reshape(prediction, (32, 32, 1)))
        cv.waitKey(0)
        cv.destroyAllWindows()


def train_model(input_shape, x, y, epochs, batch_size):

    model = Gan(input_shape=input_shape, num_classes=10)
    # model.load_weights()

    sample_interval = 1
    t0 = time.time()
    iters = np.asarray(x).shape[0] // batch_size
    if np.asarray(x).shape[0] - iters * batch_size == 0:
        train_x = x.tolist()
        train_x.append(train_x[-1])
    train_x = np.asarray(x)
    train_x = np.reshape(train_x, (train_x.shape[0], 32, 32, 1))

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        step = 0
        for iter in range(iters):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 32 * 32 * 1))

            # Generate a half batch of new images
            gen_fake = model.generator.predict([noise, y[step: (step + batch_size)], train_x[step:(step + batch_size)]])
            gen_fake = np.reshape(gen_fake, (batch_size, 32, 32, 1))

            # Train the discriminator
            d_loss_real = model.discriminator.train_on_batch([train_x[step:(step + batch_size)], y[step:(step + batch_size)]], valid)
            d_loss_fake = model.discriminator.train_on_batch([gen_fake, y[step:(step + batch_size)]], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 32 * 32 * 1))
            g_loss = model.generator.train_on_batch([noise, y[step:(step + batch_size)], train_x[step:(step + batch_size)]], valid)

            # ---------------------
            #  Train Combined
            # ---------------------

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 32 * 32 * 1))
            c_loss = model.combined.train_on_batch([noise, y[step:(step + batch_size)], train_x[step:(step + batch_size)]], valid)

            step = step + batch_size

            if iter % 100 == 0:
                print("Interior step", iter)

        # Plot the progress
        print("%d  ---- [D loss: %f, acc.: %.2f%%] [G loss: %f] [C loss: %f] --- time: %f" %
             (epoch, d_loss[0], 100*d_loss[1], g_loss, c_loss, time.time() - t0))

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            sample_images(model, epoch + 0)

        model.save_weights()


def main():

    # Load the dataset
    (x_train, y_train), (_, _) = mnist.load_data()
    input_shape = (32, 32, 1)
    print(x_train.shape)

    x_resized = []
    for i in x_train:
        x_resized.append(cv.resize(i, (32, 32)))

    print(np.asarray(x_resized).shape)

    x = np.asarray(x_resized).astype('float32')
    x /= 255
    epochs = 10
    batch_size = 128

    # test(input_shape, x, y_train)
    train_model(input_shape=input_shape, x=x, y=y_train, epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    main()


