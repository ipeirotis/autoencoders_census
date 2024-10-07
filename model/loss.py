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
