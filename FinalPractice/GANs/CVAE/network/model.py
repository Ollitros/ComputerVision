from keras.models import Model
from keras.optimizers import Adam
from FinalPractice.GANs.CVAE.network.custom_blocks import *
from FinalPractice.GANs.CVAE.network.losses import *


class Gan:

    def __init__(self, input_shape, num_classes, batch_size, latent):
        self.lrD = 2e-4
        self.lrG = 1e-4
        self.img_shape = input_shape
        self.num_classes = num_classes
        self.latent_dim = input_shape[0] * input_shape[1] * input_shape[2]
        assert (latent / (2 * 2)) >= 1
        self.latent = latent
        self.latent_channels = int(self.latent / (2 * 2))
        self.batch_size = batch_size

        adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        # Inputs
        noise = Input(shape=(self.latent_dim,))
        real = Input(shape=self.img_shape)
        label = Input(shape=(1,))
        encoder_input = [noise, label, real]

        # Build encoder and decoder
        self.encoder, self.decoder = self.build_generator()
        z_mean, z_log_sigma, encode = self.encoder(encoder_input)
        decoder_output = self.decoder(encode)

        # Create generator model
        self.generator = Model(encoder_input, decoder_output)

        # Create model outputs tensors
        fake = self.generator([noise, label, real])

        # Create custom loss for generator model
        generator_loss = custom_loss(self.generator, fake, real, z_mean, z_log_sigma, self.batch_size, self.img_shape)
        self.generator.compile(loss=generator_loss, optimizer=adam)

        # Print model`s summaries
        print(self.encoder.summary())
        print(self.decoder.summary())
        print(self.generator.summary())

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_sigma / 2) * epsilon

    def build_generator(self):

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        real = Input(shape=self.img_shape)
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])

        # # #######################
        # # ## Build encoder
        # # #######################
        x = Reshape(self.img_shape)(model_input)
        x = Conv2D(64, kernel_size=5, use_bias=False, padding="same")(x)
        x = conv_block(128)(x)
        x = self_attn_block(x, 128)
        x = conv_block(256)(x)
        x = self_attn_block(x, 256)
        x = conv_block(256)(x)

        activ_map_size = self.img_shape[0] // 16
        while activ_map_size > 4:
            x = conv_block(256)(x)
            activ_map_size = activ_map_size // 2

        # Latent Variable Calculation
        flatten_mean = Flatten()(x)
        dense_mean = Dense(self.latent)(flatten_mean)
        z_mean = BatchNormalization()(dense_mean)

        flatten_sigma = Flatten()(x)
        dense_sigma = Dense(self.latent)(flatten_sigma)
        z_log_sigma = BatchNormalization()(dense_sigma)

        encoder_output = Lambda(self.sampling, output_shape=(self.latent,))([z_mean, z_log_sigma])

        encoder = Model(inputs=[noise, label, real], outputs=[z_mean, z_log_sigma, encoder_output])

        # # #######################
        # # ## Build decoder
        # # #######################

        decoder_input = Input(shape=(self.latent, ))
        x = Reshape((2, 2, self.latent_channels))(decoder_input)
        x = upscale(128)(x)
        x = upscale(256)(x)
        x = self_attn_block(x, 256)
        x = upscale(256)(x)
        x = self_attn_block(x, 256)
        x = upscale(128)(x)
        x = res_block(x, 128)
        x = self_attn_block(x, 128)

        outputs = []
        activ_map_size = self.img_shape[0] * 8
        while activ_map_size < 128:
            outputs.append(Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x))
            x = upscale(128)(x)
            x = conv_block(128, strides=1)(x)
            activ_map_size *= 2

        x = Conv2D(self.img_shape[2], kernel_size=3, padding='same', activation="tanh")(x)

        decoder = Model(inputs=decoder_input, outputs=x)

        return encoder, decoder

    def load_weights(self, path="data/models"):
        self.generator.load_weights("{path}/generator.h5".format(path=path))
        self.decoder.save_weights("{path}/decoder.h5".format(path=path))
        self.encoder.save_weights("{path}/encoder.h5".format(path=path))
        print("Model weights files are successfully loaded.")

    def save_weights(self, path="data/models"):
        self.generator.save_weights("{path}/generator.h5".format(path=path))
        self.decoder.save_weights("{path}/decoder.h5".format(path=path))
        self.encoder.save_weights("{path}/encoder.h5".format(path=path))
        print("Model weights files have been saved to {path}.".format(path=path))


