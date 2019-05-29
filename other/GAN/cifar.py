from __future__ import print_function, division
from keras.datasets import  cifar10
from keras.models import Model
from FinalPractice.GANs.CVAE_GAN.network.custom_blocks import *
from FinalPractice.GANs.CVAE_GAN.network.custom_layers import *
from FinalPractice.GANs.CVAE_GAN.network.losses import *
import matplotlib.pyplot as plt
import numpy as np


class CGAN():

    def __init__(self):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 256

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.discriminator.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=custom_loss, optimizer='Adam')

    def build_generator(self):

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])

        x = Reshape((16, 16, 1))(model_input)
        x = conv_block(64, strides=(1, 1))(x)
        x = conv_block(128, strides=(1, 1))(x)
        x = conv_block(256, strides=(1, 1))(x)
        x = upscale(256)(x)
        x = conv_block(128, strides=(1, 1))(x)
        x = conv_block(64, strides=(1, 1))(x)
        x = conv_block(32, strides=(1, 1))(x)
        x = Conv2D(3, kernel_size=3, use_bias=False, padding="same", activation='tanh')(x)

        model = Model([noise, label], x)
        model.summary()

        return model

    def build_discriminator(self):

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)
        model_input = multiply([flat_img, label_embedding])

        x = Reshape((32, 32, 3))(model_input)
        x = dis_layer(x, 128)
        x = dis_layer(x, 256)
        x = dis_layer(x, 512)
        x = self_attn_block(x, 512)

        activ_map_size = self.img_shape[0] // 8
        while activ_map_size > 8:
            x = dis_layer(x, 512)
            x = self_attn_block(x, 512)
            activ_map_size = activ_map_size // 2

        x = Conv2D(128, kernel_size=3, padding="same")(x)
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model([img, label], x)
        model.summary()

        return model

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (x_train, y_train), (_, _) = cifar10.load_data()

        # Configure input
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs, labels = x_train[idx], y_train[idx]

            # Sample noise as generator input
            # noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            noise = np.random.normal(0, 1, (batch_size, 256))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

        self.generator.save('data/models/generator.h5')
        self.combined.save('data/models/combined.h5')

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 256))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, : , 0], cmap='gray')
                axs[i, j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("data/images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=10000, batch_size=128, sample_interval=100)