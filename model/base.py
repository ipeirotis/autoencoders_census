import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.config.run_functions_eagerly(True)


class VAE(keras.Model):
    def __init__(
        self, encoder, decoder, attribute_cardinalities, kl_loss_weight, **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.attribute_cardinalities = attribute_cardinalities
        log_cardinalities = [
            np.log(cardinality) for cardinality in self.attribute_cardinalities
        ]
        log_cardinalities_tensor = tf.constant(log_cardinalities, dtype=tf.float32)
        self.log_cardinalities_expanded = tf.expand_dims(
            log_cardinalities_tensor, axis=-1
        )
        self.kl_loss_weight = kl_loss_weight

    def summary(self):
        print("Encoder Summary:")
        self.encoder.summary()
        print("\nDecoder Summary:")
        self.decoder.summary()

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @staticmethod
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def reconstruction_loss(self, y_true, y_pred):
        y_true_splits = tf.split(y_true, self.attribute_cardinalities, axis=1)
        y_pred_splits = tf.split(y_pred, self.attribute_cardinalities, axis=1)

        max_size = max(self.attribute_cardinalities)

        y_true_splits = [
            tf.pad(split, [[0, 0], [0, max_size - tf.shape(split)[1]]])
            for split in y_true_splits
        ]
        y_pred_splits = [
            tf.pad(split, [[0, 0], [0, max_size - tf.shape(split)[1]]])
            for split in y_pred_splits
        ]

        xent_losses = tf.keras.losses.categorical_crossentropy(
            y_true_splits, y_pred_splits
        )

        normalized_xent_losses = xent_losses / self.log_cardinalities_expanded

        reconstruction_loss = tf.reduce_mean(
            tf.keras.backend.sum(normalized_xent_losses, axis=0)
        )

        return reconstruction_loss

    def kl_loss(self, z_mean, z_log_var):
        kl_loss = -0.5 * tf.keras.backend.sum(
            1
            + z_log_var
            - tf.keras.backend.square(z_mean)
            - tf.keras.backend.exp(z_log_var),
            axis=1,
        )
        return tf.reduce_mean(kl_loss)

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling((z_mean, z_log_var))

        reconstruction = self.decoder(z)

        return reconstruction, z_mean, z_log_var

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            reconstruction, z_mean, z_log_var = self(inputs=x)

            reconstruction_loss = self.reconstruction_loss(y, reconstruction)
            kl_loss = self.kl_loss(z_mean, z_log_var) * self.kl_loss_weight

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x, y = data
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling((z_mean, z_log_var))
        reconstruction = self.decoder(z)

        reconstruction_loss = self.reconstruction_loss(y, reconstruction)
        kl_loss = self.kl_loss(z_mean, z_log_var) * self.kl_loss_weight

        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
