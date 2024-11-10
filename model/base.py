import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Lambda

from model.layers import build_encoder, build_decoder 

tf.config.run_functions_eagerly(True)


class VAE(keras.Model):
    def __init__(
        self, encoder, decoder, attribute_cardinalities, kl_loss_weight, config, input_shape, **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = input_shape
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.attribute_cardinalities = attribute_cardinalities
        self.kl_loss_weight = kl_loss_weight
        self.config = config

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

    @staticmethod
    def reconstruction_loss(
        attribute_cardinalities,
        y_true,
        y_pred,
    ):
        xent_loss = []
        start_idx = 0

        for categories in attribute_cardinalities:
            x_attr = y_true[:, start_idx : start_idx + categories]
            y_attr = y_pred[:, start_idx : start_idx + categories]

            x_attr = tf.keras.backend.cast(x_attr, "float32")
            y_attr = tf.keras.backend.cast(y_attr, "float32")

            xent_loss.append(
                tf.keras.backend.categorical_crossentropy(x_attr, y_attr)
                / np.log(categories)
            )

            start_idx += categories

        xent_loss_tensor = tf.stack(xent_loss, axis=0)

        return tf.reduce_mean(xent_loss_tensor, axis=0)

    @staticmethod
    def kl_loss(z_mean, z_log_var):
        kl_loss = -0.5 * tf.keras.backend.sum(
            1
            + z_log_var
            - tf.keras.backend.square(z_mean)
            - tf.keras.backend.exp(z_log_var),
            axis=1,
        )

        return kl_loss

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling((z_mean, z_log_var))

        reconstruction = self.decoder(z)

        return reconstruction, z_mean, z_log_var

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            reconstruction, z_mean, z_log_var = self(inputs=x)

            reconstruction_loss = self.reconstruction_loss(
                self.attribute_cardinalities,
                y,
                reconstruction,
            )
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
            "temperature": self.current_temperature,
        }

    def test_step(self, data):
        x, y = data
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling((z_mean, z_log_var))
        reconstruction = self.decoder(z)

        reconstruction_loss = self.reconstruction_loss(
            self.attribute_cardinalities,
            y,
            reconstruction,
        )
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

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            'attribute_cardinalities': self.attribute_cardinalities,
            'kl_loss_weight': self.kl_loss_weight,
            'config': self.config,
            'input_shape': self.input_shape,
            'temperature': self.current_temperature,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config_ = config.pop('config')
        input_shape = config.pop('input_shape')
        attribute_cardinalities = config.pop('attribute_cardinalities')
        temperature = config.pop('temperature')
        encoder = build_encoder(input_shape, config_, len(attribute_cardinalities))
        decoder = build_decoder(attribute_cardinalities, config_)
        kl_loss_weight = config.pop('kl_loss_weight')
        class_instance =  cls(encoder=encoder, decoder=decoder,
                   attribute_cardinalities=attribute_cardinalities,
                   kl_loss_weight=kl_loss_weight, config=config_, input_shape=input_shape)
       
        class_instance.current_temperature = temperature

        return class_instance
