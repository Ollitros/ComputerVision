from keras.models import Model
from keras.optimizers import Adam
from FinalPractice.DyingNN.network.custom_blocks import *
from FinalPractice.DyingNN.network.losses import *


class Gan:

    def __init__(self, input_shape, num_classes, batch_size, filter_coeff=1.0):
        self.lrD = 2e-4
        self.lrG = 1e-4
        self.img_shape = input_shape
        self.num_classes = num_classes
        self.latent_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.batch_size = batch_size
        self.filter_coeff = filter_coeff
        self.latent = 8 * 8 * 3
        adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        # Inputs
        noise = Input(shape=(self.latent_dim,))
        real = Input(shape=self.img_shape)
        generator_input = [noise, real]

        # Build generator
        self.g = self.build_generator()

        # Create model outputs tensors
        fake = self.g(generator_input)
        self.generator = Model(generator_input, fake)

        # Create custom loss for generator model
        generator_loss = custom_loss(self.generator, fake, real, self.batch_size, self.img_shape)
        self.generator.compile(loss=generator_loss, optimizer=adam)

        # Print model`s summaries
        print(self.generator.summary())

    def build_generator(self):

        # # #######################
        # # ## Build encoder
        # # #######################

        noise = Input(shape=(self.latent_dim,))
        real = Input(shape=self.img_shape)

        x = Reshape(self.img_shape)(noise)
        x = Conv2D(int(32 * self.filter_coeff), kernel_size=5, use_bias=False, padding="same")(x)
        x = conv_block(int(64 * self.filter_coeff))(x)
        x = conv_block(int(128 * self.filter_coeff))(x)
        x = conv_block(int(128 * self.filter_coeff))(x)
        encoder_output = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)

        # # #######################
        # # ## Build decoder
        # # #######################

        x = upscale(int(128 * self.filter_coeff))(encoder_output)
        x = upscale(int(64 * self.filter_coeff))(x)
        x = upscale(int(64 * self.filter_coeff))(x)
        x = Conv2D(self.img_shape[2], kernel_size=3, padding='same', activation="tanh")(x)

        generator = Model(inputs=[noise, real], outputs=x)

        return generator

    def load_weights(self, path="data/models"):
        self.generator.load_weights("{path}/generator.h5".format(path=path))
        print("Model weights files are successfully loaded.")

    def save_weights(self, path="data/models"):
        self.generator.save("{path}/generator.h5".format(path=path))
        print("Model weights files have been saved to {path}.".format(path=path))



