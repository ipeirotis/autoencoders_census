import unittest
import numpy as np
import tensorflow as tf

from model.loss import CustomCategoricalCrossentropyAE, CustomCategoricalCrossentropyVAE


class TestCustomLossAE(unittest.TestCase):

    def setUp(self):
        self.attribute_cardinalities = [
            3,
            2,
            3,
        ]  # 3 unique categories in each attribute

    def test_custom_categorical_crossentropyAE(self):
        y_true = np.array([[1, 0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0, 1]])
        y_true = y_true.astype(np.float32)
        y_pred = np.array(
            [
                [0.8, 0.1, 0.1, 0.7, 0.3, 0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1, 0.3, 0.7, 0.05, 0.05, 0.9],
            ]
        )
        y_pred = y_pred.astype(np.float32)
        result = CustomCategoricalCrossentropyAE(
            attribute_cardinalities=self.attribute_cardinalities
        )(y_true, y_pred)
        # p=[1,0,0],q=[0.8,0.1,0.1]: 0.2231 / np.log(3)
        # p=[1,0],q=[0.7,0.3]: 0.3567 / np.log(2)
        # p=[1,0,0],q=[0.9,0.05,0.05]: 0.1054 / np.log(3)
        # p=[0,1,0],q=[0.1,0.8,0.1]: 0.2231 / np.log(3)
        # p=[0,1,0],q=[0.3,0.7]: 0.3567 / np.log(2)
        # p=[0,0,1],q=[0.05,0.05,0.9]: 0.1054 / np.log(3)
        print(result)
        self.assertIsNotNone(result)
        # We divide by 2 because we have two examples
        # We divide by 3 because we have 3 attributes per example
        # We divide by np.log(3) for normalization (all attributes have cardinality 3)
        self.assertAlmostEqual(
            result.numpy(),
            (0.2231 / np.log(3) + 0.3567 / np.log(2) + 0.1054 / np.log(3)) / 3,
            places=2,
        )

    def test_custom_categorical_crossentropyVAE(self):
        y_true = np.array([[1, 0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0, 1]])
        y_pred = np.array(
            [
                [0.8, 0.1, 0.1, 0.7, 0.3, 0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1, 0.3, 0.7, 0.05, 0.05, 0.9],
            ]
        )
        y_pred = y_pred.astype(np.float32)

        result = CustomCategoricalCrossentropyVAE(
            attribute_cardinalities=self.attribute_cardinalities
        )(y_true, y_pred)
        print(result)

        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        z_mean_tf, z_log_var_tf = tf.split(
            tf.convert_to_tensor(y_pred, dtype=tf.float32), num_or_size_splits=2, axis=1
        )

        kl_divergence_tf = -0.5 * tf.reduce_sum(
            1 + z_log_var_tf - tf.square(z_mean_tf) - tf.exp(z_log_var_tf), axis=1
        )

        kl_divergence_tf = tf.reduce_mean(kl_divergence_tf)

        self.assertAlmostEqual(
            result,
            np.sum(
                (0.2231 / np.log(3) + 0.3567 / np.log(2) + 0.1054 / np.log(3)) / 3
                + kl_divergence_tf
            ),
            places=2,
        )
