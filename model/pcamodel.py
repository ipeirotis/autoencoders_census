import tensorflow as tf
from keras.layers import Input
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import Orthogonal


class PCAAutoencoderWithOrthogonality(tf.keras.Model):
    def __init__(self, encoder, decoder, lambda_ortho=1e-2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lambda_ortho = lambda_ortho

    def call(self, x):
        return self.decoder(self.encoder(x))

    def orthogonality_penalty(self, W):
        WT_W = tf.matmul(tf.transpose(W), W)
        I = tf.eye(W.shape[1], dtype=WT_W.dtype)
        return tf.reduce_sum(tf.square(WT_W - I))

    def train_step(self, data):
        x = data[0]

        with tf.GradientTape() as tape:
            z = self.encoder(x)
            x_hat = self.decoder(z)
            recon_loss = tf.reduce_mean(tf.keras.losses.mse(x, x_hat))

            W = self.encoder.trainable_weights[0]
            ortho_loss = self.orthogonality_penalty(W)
            total_loss = recon_loss + self.lambda_ortho * ortho_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "orthogonality_loss": ortho_loss,
        }


class LinearAutoencoder(tf.keras.Model):
    def __init__(self, cardinalities: list):
        super().__init__()
        self.INPUT_SHAPE = None
        self.attribute_cardinalities = cardinalities

    def split_train_test(self, df, test_size=0.2):
        """
        Split the dataset into train and test

        :param df: Dataframe to split
        :return:
            X_train: train data
            X_test: test data
        """
        # X_train, X_test = train_test_split(df.copy(), test_size=0)
        self.INPUT_SHAPE = df.shape[1:]
        return df.dropna(), df.dropna()

    def build_encoder(self, config):
        inputs = Input(shape=self.INPUT_SHAPE)
        x = inputs
        x = layers.Dense(config.get("latent_space_dim", 2), use_bias=False, activation=None, kernel_initializer=Orthogonal())(x)

        return Model(inputs, x)

    def build_decoder(self, config):
        decoder_inputs = Input(shape=(config.get("latent_space_dim", 2),))
        x = decoder_inputs
        x = layers.Dense(self.INPUT_SHAPE[0], use_bias=False, activation=None, kernel_initializer=Orthogonal())(x)

        return Model(decoder_inputs, x)

    def build_autoencoder(self, config):
        learning_rate = config.get("learning_rate", 1e-3)

        encoder = self.build_encoder(config)
        decoder = self.build_decoder(config)

        autoencoder = PCAAutoencoderWithOrthogonality(encoder, decoder, lambda_ortho=1e-2)
        autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9))

        return autoencoder