from keras.models import Model
from FinalPractice.GANs.CVAE_GAN.network.custom_blocks import *
from FinalPractice.GANs.CVAE_GAN.network.losses import *


class Gan:

    def __init__(self, input_shape, num_classes):
        self.lrD = 2e-4
        self.lrG = 1e-4
        self.img_shape = input_shape
        self.num_classes = num_classes
        self.latent_dim = input_shape[0] * input_shape[1] * input_shape[2]

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

        # Inputs
        noise = Input(shape=(self.latent_dim,))
        real = Input(shape=(32, 32, 1))
        label = Input(shape=(1,))
        encoder_input = [noise, label, real]

        # Build encoder and decoder
        self.encoder, self.decoder = self.build_generator()
        z_mean, z_log_sigma, encode = self.encoder(encoder_input)
        decoder_output = self.decoder(encode)

        # Create generator model
        self.generator = Model(encoder_input, decoder_output)
        self.discriminator.trainable = False
        self.discriminator.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

        # Create model outputs tensors
        fake = self.generator([noise, label, real])
        valid = self.discriminator([fake, label])

        # Create custom loss for combined model
        combined_loss = custom_loss(self.generator, fake, real, z_mean, z_log_sigma)

        # Create combined model
        self.combined = Model([noise, label, real], valid)
        self.combined.compile(loss=combined_loss, optimizer='Adam')

        # Print model`s summaries
        print(self.encoder.summary())
        print(self.decoder.summary())
        print(self.generator.summary())
        print(self.discriminator.summary())
        print(self.combined.summary())

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(4 * 4 * 128,), mean=0, stddev=1)
        return z_mean + K.exp(z_log_sigma / 2) * epsilon

    def build_generator(self):

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        real = Input(shape=(32, 32, 1))
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])

        # # #######################
        # # ## Build encoder
        # # #######################
        x = Reshape((32, 32, 1))(model_input)
        x = Conv2D(32, kernel_size=5, use_bias=False, padding="same")(x)
        x = conv_block(64)(x)
        x = self_attn_block(x, 64)
        x = conv_block(64)(x)
        x = self_attn_block(x, 64)
        x = conv_block(64)(x)

        activ_map_size = self.img_shape[0] // 16
        while activ_map_size > 4:
            x = conv_block(64)(x)
            activ_map_size = activ_map_size // 2

        x = Dense(128)(Flatten()(x))
        x = Dense(4 * 4 * 128)(x)

        # Latent Variable Calculation
        z_mean = BatchNormalization()(x)
        dense_2 = Dense(4 * 4 * 128, name='z_log_sigma', activation='tanh')(x)
        z_log_sigma = BatchNormalization()(dense_2)
        encoder_output = Lambda(self.sampling)([z_mean, z_log_sigma])

        encoder = Model(inputs=[noise, label, real], outputs=[z_mean, z_log_sigma, encoder_output])

        # # #######################
        # # ## Build decoder
        # # #######################

        decoder_input = Input(shape=(4 * 4 * 128, ))
        x = Reshape((4, 4, 128))(decoder_input)
        x = upscale(64)(x)
        x = upscale(64)(x)
        x = self_attn_block(x, 64)
        x = upscale(64)(x)
        x = res_block(x, 64)
        x = self_attn_block(x, 64)

        outputs = []
        activ_map_size = self.img_shape[0] * 8
        while activ_map_size < 128:
            outputs.append(Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x))
            x = upscale(64)(x)
            x = conv_block(64, strides=1)(x)
            activ_map_size *= 2

        x = Conv2D(1, kernel_size=3, padding='same', activation="tanh")(x)

        decoder = Model(inputs=decoder_input, outputs=x)

        return encoder, decoder

    def build_discriminator(self):

        img = Input(shape=(32, 32, 1))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod((32, 32, 1)))(label))
        flat_img = Flatten()(img)
        model_input = multiply([flat_img, label_embedding])

        x = Reshape((32, 32, 1))(model_input)
        x = dis_layer(x, 64)
        x = dis_layer(x, 128)
        x = dis_layer(x, 256)
        x = self_attn_block(x, 256)

        activ_map_size = self.img_shape[0] // 8
        while activ_map_size > 8:
            x = dis_layer(x, 256)
            x = self_attn_block(x, 256)
            activ_map_size = activ_map_size // 2

        x = Conv2D(32, kernel_size=3, padding="same")(x)
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model([img, label], x)

        return model

    def load_weights(self, path="data/models"):
        self.generator.load_weights("{path}/generator.h5".format(path=path))
        self.discriminator.load_weights("{path}/discriminator.h5".format(path=path))
        # self.combined.load_weights("{path}/combined.h5".format(path=path))
        print("Model weights files are successfully loaded.")

    def save_weights(self, path="data/models"):
        self.generator.save_weights("{path}/generator.h5".format(path=path))
        self.discriminator.save_weights("{path}/discriminator.h5".format(path=path))
        # self.combined.save_weights("{path}/combined.h5".format(path=path))
        print("Model weights files have been saved to {path}.".format(path=path))


