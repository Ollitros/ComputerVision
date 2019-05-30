from keras.models import Model
from FinalPractice.GANs.CVAE_GAN.network.custom_blocks import *
from keras.optimizers import Adam
from FinalPractice.GANs.CVAE_GAN.network.losses import *


class Gan:

    def __init__(self):
        self.lrD = 2e-4
        self.lrG = 1e-4
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 32 * 32 * 3

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build the generator
        self.encoder, self.decoder = self.build_generator()

        self.encoder_input = self.encoder.inputs
        self.encode = self.encoder(self.encoder_input)

        # Create generators
        self.decode = self.decoder(self.encode)
        self.generator = Model(self.encoder_input, self.decode)
        self.encoder.summary()
        self.decoder.summary()
        self.generator.summary()

        # Define variables
        self.distorted, self.fake, self.condition = self.define_variables(generator=self.generator)

        self.real = Input(shape=self.img_shape)

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(4 * 4 * 128,), mean=0, stddev=1)
        return z_mean + K.exp(z_log_sigma / 2) * epsilon

    def build_generator(self):

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])

        # # #######################
        # # ## Build encoder
        # # #######################
        x = Reshape((32, 32, 3))(model_input)
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
        dense_2 = Dense(4 * 4 * 128, name='z_log_sigma')(z_mean)
        z_log_sigma = BatchNormalization()(dense_2)
        encoder_output = Lambda(self.sampling)([z_mean, z_log_sigma])

        encoder = Model(inputs=[noise, label], outputs=encoder_output)

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

        alpha = Conv2D(1, kernel_size=3, padding='same', activation="sigmoid")(x)
        bgr = Conv2D(3, kernel_size=3, padding='same', activation="tanh")(x)
        x = concatenate([alpha, bgr])

        decoder = Model(inputs=decoder_input, outputs=x)

        return encoder, decoder

    def build_discriminator(self):

        img = Input(shape=(32, 32, 6))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod((32, 32, 6)))(label))
        flat_img = Flatten()(img)
        model_input = multiply([flat_img, label_embedding])

        x = Reshape((32, 32, 6))(model_input)
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
        model.summary()

        return model

    @staticmethod
    def define_variables(generator):
        distorted_input = generator.inputs[0]
        condition = generator.inputs[1]
        fake_output = generator.outputs[-1]

        return distorted_input, fake_output, condition

    def build_train_functions(self):

        weights = {}
        weights['w_D'] = 0.1  # Discriminator
        weights['w_recon'] = 1.  # L1 reconstruction loss
        weights['w_edge'] = 0.1  # edge loss

        # Adversarial loss
        loss_dis, loss_adv_gen = adversarial_loss(self.discriminator, self.real,  self.fake, self.distorted, self.condition)

        # Reconstruction loss
        loss_recon_gen = reconstruction_loss(self.real, self.fake, self.generator.outputs, weights=weights)

        # Edge loss
        loss_edge_gen = edge_loss(self.real, self.fake, weights=weights)

        # Losses
        loss_gen = loss_adv_gen + loss_recon_gen + loss_edge_gen

        # Alpha mask total variation loss
        # loss_gen += 0.1 * K.mean(first_order(self.mask, axis=1))
        # loss_gen += 0.1 * K.mean(first_order(self.mask, axis=2))

        # L2 weight decay
        # https://github.com/keras-team/keras/issues/2662
        for loss_tensor in self.generator.losses:
            loss_gen += loss_tensor
        for loss_tensor in self.discriminator.losses:
            loss_dis += loss_tensor

        weights_dis = self.discriminator.trainable_weights
        weights_gen = self.generator.trainable_weights

        # Define training functions
        lr_factor = 1
        training_updates = Adam(lr=self.lrD * lr_factor, beta_1=0.5).get_updates(weights_dis, [], loss_dis)
        self.net_dis_train = K.function([self.distorted, self.real, self.condition], [loss_dis], training_updates)

        training_updates = Adam(lr=self.lrG * lr_factor, beta_1=0.5).get_updates(weights_gen, [], loss_gen)
        self.net_gen_train = K.function([self.distorted, self.real, self.condition],
                                        [loss_gen, loss_adv_gen, loss_recon_gen + loss_edge_gen], training_updates)

    def load_weights(self, path="data/models"):
        self.generator.load_weights("{path}/generator.h5".format(path=path))
        self.discriminator.load_weights("{path}/discriminator.h5".format(path=path))
        print("Model weights files are successfully loaded.")

    def save_weights(self, path="data/models"):
        self.generator.save_weights("{path}/generator.h5".format(path=path))
        self.discriminator.save_weights("{path}/discriminator.h5".format(path=path))
        print("Model weights files have been saved to {path}.".format(path=path))

    def train_generator(self, X, Y, cond):

        err_gen = self.net_gen_train([X, Y, cond])

        return err_gen

    def train_discriminator(self, X, Y, cond):

        err_dis = self.net_dis_train([X, Y, cond])

        return err_dis

