import unittest
import numpy as np

from model.loss import CustomCategoricalCrossentropyAE


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
        self.assertIsNotNone(result)
        # We divide by 2 because we have two examples
        # We divide by 3 because we have 3 attributes per example
        # We divide by np.log(3) for normalization (all attributes have cardinality 3)
        self.assertAlmostEqual(
            result.numpy(),
            (0.2231 / np.log(3) + 0.3567 / np.log(2) + 0.1054 / np.log(3)) / 3,
            places=2,
        )

    def test_get_config_includes_percentile(self):
        loss = CustomCategoricalCrossentropyAE(
            attribute_cardinalities=self.attribute_cardinalities, percentile=90
        )
        config = loss.get_config()
        self.assertIn("percentile", config)
        self.assertEqual(config["percentile"], 90)
        self.assertIn("attribute_cardinalities", config)
        self.assertEqual(config["attribute_cardinalities"], self.attribute_cardinalities)
