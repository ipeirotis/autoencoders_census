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
