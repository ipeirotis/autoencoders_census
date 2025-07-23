import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.experimental.numpy.experimental_enable_numpy_behavior()


@keras.utils.register_keras_serializable()
class CustomCategoricalCrossentropyAE(tf.keras.losses.Loss):
    def __init__(self, attribute_cardinalities, name="custom_categorical_crossentropy"):
        super(CustomCategoricalCrossentropyAE, self).__init__(name=name)
        self.attribute_cardinalities = attribute_cardinalities
        log_cardinalities = [
            np.log(cardinality) for cardinality in self.attribute_cardinalities
        ]
        log_cardinalities_tensor = tf.constant(log_cardinalities, dtype=tf.float32)
        self.log_cardinalities_expanded = tf.expand_dims(
            log_cardinalities_tensor, axis=-1
        )

    def call(self, y_true, y_pred):

        xent_loss = 0
        start_idx = 0

        for categories in self.attribute_cardinalities:
            x_attr = y_true[:, start_idx : start_idx + categories]
            y_attr = y_pred[:, start_idx : start_idx + categories]

            x_attr = tf.keras.backend.cast(x_attr, "float32")
            y_attr = tf.keras.backend.cast(y_attr, "float32")

            xent_loss += tf.keras.backend.mean(
                tf.keras.backend.categorical_crossentropy(x_attr, y_attr)
            ) / np.log(categories)

            start_idx += categories

        return xent_loss / len(self.attribute_cardinalities)

    def get_config(self):
        return {"attribute_cardinalities": self.attribute_cardinalities}


@keras.utils.register_keras_serializable()
class CustomCategoricalCrossentropyVAE(tf.keras.losses.Loss):
    """Categorical crossentropy with KL penalty used for simple tests."""

    def __init__(self, attribute_cardinalities, kl_loss_weight=1.0, name="custom_categorical_crossentropy_vae"):
        super().__init__(name=name)
        self.attribute_cardinalities = attribute_cardinalities
        self.kl_loss_weight = kl_loss_weight

    def call(self, y_true, y_pred):
        # reconstruction part identical to the AE loss
        xent_loss = 0
        start_idx = 0
        for categories in self.attribute_cardinalities:
            x_attr = tf.cast(y_true[:, start_idx : start_idx + categories], tf.float32)
            y_attr = tf.cast(y_pred[:, start_idx : start_idx + categories], tf.float32)
            xent_loss += tf.keras.backend.mean(
                tf.keras.backend.categorical_crossentropy(x_attr, y_attr)
            ) / np.log(categories)
            start_idx += categories
        recon_loss = xent_loss / len(self.attribute_cardinalities)

        # simple KL term assuming y_pred also encodes Gaussian parameters
        epsilon = 1e-10
        z_mean, z_log_var = tf.split(tf.clip_by_value(y_pred, epsilon, 1 - epsilon), num_or_size_splits=2, axis=1)
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        )
        kl_loss = tf.reduce_mean(kl_loss)

        return recon_loss + self.kl_loss_weight * kl_loss

    def get_config(self):
        return {
            "attribute_cardinalities": self.attribute_cardinalities,
            "kl_loss_weight": self.kl_loss_weight,
        }

