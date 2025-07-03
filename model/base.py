import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Lambda

from model.layers import build_encoder, build_decoder

tf.config.run_functions_eagerly(True)


class VAE(keras.Model):
    def __init__(
        self,
        encoder,
        decoder,
        attribute_cardinalities,
        kl_loss_weight,
        config,
        input_shape,
        initial_temperature,
        prior,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.input_sh = input_shape
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.attribute_cardinalities = attribute_cardinalities
        self.num_classes = len(attribute_cardinalities)
        self.kl_loss_weight = kl_loss_weight
        self.config = config
        self.temperature = initial_temperature
        self.base_kl_loss_weight = kl_loss_weight
        self.kl_loss_weight = 0.0
        self.prior = prior
        self.anneal_period = self.config.get("anneal_period", 20000)
        self.cycles = self.config.get("cycles", 4)
        self.min_temperature = self.config.get("min_temperature", 0.2)
        self.temperature_decay = self.config.get("temperature_decay", 0.0005)
        self.warmup_period = self.config.get("warmup_period", 500)

    def summary(self):
        print("Encoder Summary:")
        self.encoder.summary()
        print("\nDecoder Summary:")
        self.decoder.summary()

    def update_kl_weight(self, step):
        if step < self.warmup_period:
            kl_weight = 0.0
        else:
            cycle_length = self.anneal_period // self.cycles
            cycle_position = (step - self.warmup_period) % cycle_length
            kl_weight = (
                0.5
                * self.base_kl_loss_weight
                * (1 - tf.cos(np.pi * cycle_position / cycle_length))
            )
        self.kl_loss_weight = tf.cast(kl_weight, tf.float32)

    def update_temperature(self):
        new_temp = self.temperature * tf.exp(-self.temperature_decay)
        self.temperature = tf.maximum(new_temp, self.min_temperature)

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
    def sample_gumbel(shape, eps=1e-20):
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits):
        logits = tf.reshape(
            logits, (-1, self.num_classes, int(logits.shape[1] / self.num_classes))
        )

        y = logits + self.sample_gumbel(tf.shape(logits))
        return tf.nn.softmax(y / self.temperature, axis=-1)

    def gumbel_softmax(self, logits):
        return Lambda(lambda x: self.gumbel_softmax_sample(x))(logits)

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
    def kl_loss_gaussian(z_mean, z_log_var):
        kl_loss = -0.5 * tf.keras.backend.sum(
            1
            + z_log_var
            - tf.keras.backend.square(z_mean)
            - tf.keras.backend.exp(z_log_var),
            axis=1,
        )

        return kl_loss

    @staticmethod
    def kl_loss_gumbel(logits, temperature, num_classes):
        logits = tf.reshape(logits, (logits.shape[0], num_classes, -1))

        q_y = tf.nn.softmax(logits / temperature, axis=-1)

        p_y = tf.ones_like(q_y) / logits.shape[2]

        kl_loss = tf.reduce_sum(
            q_y * (tf.math.log(q_y + 1e-20) - tf.math.log(p_y + 1e-20)), axis=-1
        )

        return tf.reduce_sum(kl_loss, axis=-1)

    def call(self, inputs):
        if self.prior == "gaussian":
            z_mean, z_log_var = self.encoder(inputs)
            z = self.sampling((z_mean, z_log_var))
            reconstruction = self.decoder(z)

            return reconstruction, z_mean, z_log_var

        elif self.prior == "gumbel":
            z = self.encoder(inputs)
            z_out = self.gumbel_softmax(z)

            noise = tf.random.normal(shape=tf.shape(z_out), mean=0.0, stddev=0.1)
            z_out += noise

            reconstruction = self.decoder(
                tf.reshape(z_out, (-1, z_out.shape[1] * z_out.shape[2]))
            )

            return reconstruction, z, z_out

        else:
            raise ValueError("Invalid prior")

    def train_step(self, data):
        x, y = data
        step = tf.keras.backend.get_value(self.optimizer.iterations)

        self.update_kl_weight(step)

        self.update_temperature()

        with tf.GradientTape() as tape:
            reconstruction, z1, z2 = self(inputs=x)

            reconstruction_loss = self.reconstruction_loss(
                self.attribute_cardinalities,
                y,
                reconstruction,
            )

            if self.prior == "gaussian":
                z_mean, z_log_var = z1, z2
                kl_loss = self.kl_loss_gaussian(z_mean, z_log_var) * self.kl_loss_weight
                variance_penalty = 0

            elif self.prior == "gumbel":
                kl_loss = (
                    self.kl_loss_gumbel(z1, self.temperature, self.num_classes)
                    * self.kl_loss_weight
                )
                variance_penalty = tf.reduce_mean(tf.math.reduce_std(z2, axis=0))

            else:
                raise ValueError("Invalid prior")

            total_loss = reconstruction_loss + kl_loss - 0.01 * variance_penalty

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "temperature": self.temperature,
        }

    def test_step(self, data):
        x, y = data

        reconstruction, z1, z2 = self(inputs=x)

        reconstruction_loss = self.reconstruction_loss(
            self.attribute_cardinalities,
            y,
            reconstruction,
        )

        if self.prior == "gaussian":
            z_mean, z_log_var = z1, z2
            kl_loss = self.kl_loss_gaussian(z_mean, z_log_var) * self.kl_loss_weight

        elif self.prior == "gumbel":
            kl_loss = (
                self.kl_loss_gumbel(z1, self.temperature, self.num_classes)
                * self.kl_loss_weight
            )

        else:
            raise ValueError("Invalid prior")

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
        config.update(
            {
                "attribute_cardinalities": self.attribute_cardinalities,
                "kl_loss_weight": float(self.kl_loss_weight.numpy()),
                "config": self.config,
                "input_shape": self.input_sh,
                "temperature": float(self.temperature.numpy()),
                "prior": self.prior,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config_ = config.pop("config")
        input_shape = config.pop("input_shape")
        attribute_cardinalities = config.pop("attribute_cardinalities")
        temperature = config.pop("temperature")
        prior = config.pop("prior")
        encoder = build_encoder(
            input_shape, config_, prior, len(attribute_cardinalities)
        )
        decoder = build_decoder(
            attribute_cardinalities, config_, prior, len(attribute_cardinalities)
        )
        kl_loss_weight = config.pop("kl_loss_weight")
        class_instance = cls(
            encoder=encoder,
            decoder=decoder,
            attribute_cardinalities=attribute_cardinalities,
            kl_loss_weight=kl_loss_weight,
            config=config_,
            input_shape=input_shape,
            initial_temperature=temperature,
            prior=prior,
        )

        return class_instance
