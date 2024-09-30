import numpy as np
import tensorflow as tf
from tensorflow import keras


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
        # Your custom loss logic here
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

        return tf.reduce_mean(normalized_xent_losses, axis=0)

    def get_config(self):
        return {"attribute_cardinalities": self.attribute_cardinalities}


@keras.utils.register_keras_serializable()
class CustomCategoricalCrossentropyVAE(tf.keras.losses.Loss):
    def __init__(self, attribute_cardinalities, name="custom_categorical_crossentropy"):
        super(CustomCategoricalCrossentropyVAE, self).__init__(name=name)
        self.attribute_cardinalities = attribute_cardinalities
        log_cardinalities = [
            np.log(cardinality) for cardinality in self.attribute_cardinalities
        ]
        log_cardinalities_tensor = tf.constant(log_cardinalities, dtype=tf.float32)
        self.log_cardinalities_expanded = tf.expand_dims(
            log_cardinalities_tensor, axis=-1
        )

    def call(self, y_true, y_pred):
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

        reconstruction_loss = tf.reduce_mean(normalized_xent_losses)

        split_size = tf.shape(y_pred)[1] // 2
        z_mean = y_pred[:, :split_size]
        z_log_var = y_pred[:, split_size:-1]

        kl_divergence = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        )

        kl_divergence = tf.reduce_mean(kl_divergence)

        total_loss = reconstruction_loss + kl_divergence

        return total_loss

    def get_config(self):
        return {"attribute_cardinalities": self.attribute_cardinalities}
