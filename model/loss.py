import numpy as np
import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable()
class CustomCategoricalCrossentropyAE(tf.keras.losses.Loss):
    def __init__(self, attribute_cardinalities, name="custom_categorical_crossentropy"):
        super(CustomCategoricalCrossentropyAE, self).__init__(name=name)
        self.attribute_cardinalities = attribute_cardinalities
        self.percentile = percentile  # Exclude top X% highest-loss samples

    # def call(self, y_true, y_pred):
    #
    #     xent_loss = 0
    #     start_idx = 0
    #
    #     for categories in self.attribute_cardinalities:
    #         x_attr = y_true[:, start_idx : start_idx + categories]
    #         y_attr = y_pred[:, start_idx : start_idx + categories]
    #
    #         x_attr = tf.keras.backend.cast(x_attr, "float32")
    #         y_attr = tf.keras.backend.cast(y_attr, "float32")
    #
    #         xent_loss += tf.keras.backend.mean(
    #             tf.keras.backend.categorical_crossentropy(x_attr, y_attr)
    #         ) / np.log(categories)
    #
    #         start_idx += categories
    #
    #     return xent_loss / len(self.attribute_cardinalities)


    def call(self, y_true, y_pred):
        """
        Computes the categorical crossentropy loss and applies percentile loss adjustment.

        Args:
            y_true (tensor): True one-hot encoded categorical values.
            y_pred (tensor): Predicted probability distributions.

        Returns:
            Adjusted loss based on percentile threshold.
        """
        xent_losses = []
        start_idx = 0

        for categories in self.attribute_cardinalities:
            x_attr = y_true[:, start_idx : start_idx + categories]
            y_attr = y_pred[:, start_idx : start_idx + categories]

            x_attr = tf.keras.backend.cast(x_attr, "float32")
            y_attr = tf.keras.backend.cast(y_attr, "float32")

            loss_per_sample = tf.keras.backend.categorical_crossentropy(x_attr, y_attr) / np.log(categories)
            xent_losses.append(loss_per_sample)

            start_idx += categories

        total_loss_per_sample = tf.reduce_mean(tf.stack(xent_losses, axis=0), axis=0)

        # sorted_losses = tf.sort(total_loss_per_sample)  # Sort losses
        # num_samples = tf.shape(sorted_losses)[0]  # Get number of samples
        #
        # threshold_index = tf.cast((self.percentile / 100.0) * tf.cast(num_samples, tf.float32), tf.int32)  # Safe casting
        # if self.percentile == 100:
        #     threshold_index = num_samples - 1  # Include all samples if percentile is 100
        # threshold_value = sorted_losses[threshold_index]  # Get threshold loss value
        #
        # mask = tf.cast(total_loss_per_sample <= threshold_value, tf.float32)
        #
        # loss = tf.reduce_sum(total_loss_per_sample * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)

        # 1. Determine how many samples to keep based on the percentile
        num_samples = tf.shape(total_loss_per_sample)[0]
        k = tf.cast((self.percentile / 100.0) * tf.cast(num_samples, tf.float32), dtype=tf.int32)
        if self.percentile == 100:
            k = num_samples
        # Ensure k is at least 1 to avoid errors on small batches
        k = tf.maximum(k, 1)

        # 2. Find the k smallest losses using top_k on the negative losses
        # tf.nn.top_k finds the largest values, so we flip the sign.
        # The result will be the k smallest losses, but with a negative sign.
        smallest_losses, _ = tf.nn.top_k(-total_loss_per_sample, k=k)

        # 3. Calculate the mean of these losses
        # Flip the sign back to positive before taking the mean.
        final_loss = tf.reduce_mean(-smallest_losses)

        return final_loss

        # return loss
# class CustomCategoricalCrossentropyAE(tf.keras.losses.Loss):
#     def __init__(self, attribute_cardinalities, k=3.0, name="custom_categorical_crossentropy"):
#         """
#         Custom loss that applies sigma-based thresholding to filter high-loss samples.
#         Args:
#             attribute_cardinalities (list): List of categorical attribute cardinalities.
#             k (float): Multiplier for standard deviation in sigma rule (e.g., 2.0 = mean + 2σ).
#         """
#         super(CustomCategoricalCrossentropyAE, self).__init__(name=name)
#         self.attribute_cardinalities = attribute_cardinalities
#         self.k = k
#
#     def call(self, y_true, y_pred):
#         xent_losses = []
#         start_idx = 0
#
#         for categories in self.attribute_cardinalities:
#             x_attr = y_true[:, start_idx : start_idx + categories]
#             y_attr = y_pred[:, start_idx : start_idx + categories]
#
#             x_attr = tf.cast(x_attr, "float32")
#             y_attr = tf.cast(y_attr, "float32")
#
#             loss_per_sample = tf.keras.backend.categorical_crossentropy(x_attr, y_attr) / tf.math.log(tf.cast(categories, tf.float32))
#             xent_losses.append(loss_per_sample)
#             start_idx += categories
#
#         total_loss_per_sample = tf.reduce_mean(tf.stack(xent_losses, axis=0), axis=0)
#
#         # Sigma threshold: keep samples with loss <= mean + k * std
#         mean_loss = tf.reduce_mean(total_loss_per_sample)
#         std_loss = tf.math.reduce_std(total_loss_per_sample)
#         threshold = mean_loss + self.k * std_loss
#
#         mask = tf.cast(total_loss_per_sample <= threshold, tf.float32)
#
#         print(tf.reduce_sum(mask)/512)
#
#         loss = tf.reduce_sum(total_loss_per_sample * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)
#
#         return loss

    def get_config(self):
        return {"attribute_cardinalities": self.attribute_cardinalities}
