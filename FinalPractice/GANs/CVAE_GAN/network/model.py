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
        self.generator = self.build_generator()

        # Define variables
        self.distorted, self.fake, self.condition = self.define_variables(generator=self.generator)

        self.real = Input(shape=self.img_shape)

    def build_generator(self):

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])

        x = Reshape((32, 32, 3))(model_input)
        x = conv_block(128, strides=(1, 1))(x)
        x = conv_block(256, strides=(1, 1))(x)
        x = conv_block(256, strides=(1, 1))(x)
        x = conv_block(128, strides=(1, 1))(x)
        x = conv_block(64, strides=(1, 1))(x)
        x = conv_block(32, strides=(1, 1))(x)

        alpha = Conv2D(1, kernel_size=3, padding='same', activation="sigmoid")(x)
        bgr = Conv2D(3, kernel_size=3, padding='same', activation="tanh")(x)
        x = concatenate([alpha, bgr])

        model = Model([noise, label], x)
        model.summary()

        return model

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
        weights['w_eyes'] = 30.  # reconstruction and edge loss on eyes area
        weights['w_pl'] = (0.01, 0.1, 0.3, 0.1)  # perceptual loss (0.003, 0.03, 0.3, 0.3)

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

