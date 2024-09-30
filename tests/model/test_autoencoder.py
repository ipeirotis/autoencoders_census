import unittest
import pandas as pd
import numpy as np

from model.autoencoder import AutoencoderModel


class TestAutoencoderModel(unittest.TestCase):

    def setUp(self):
        # Create a DataFrame with synthetic data
        self.df = pd.DataFrame(
            {
                "col1": ["A", "B", "A", "C", "B"],
                "col2": [1, 2, 1, 2, 2],
                "col3": ["X", "Y", "X", "Y", "Z"],
            }
        )
        self.attribute_cardinalities = [
            3,
            2,
            3,
        ]  # 3 unique categories in each attribute
        self.autoencoder = AutoencoderModel(self.attribute_cardinalities)

    def test_split_train_test(self):
        X_train, X_test = self.autoencoder.split_train_test(self.df)
        self.assertEqual(len(X_train) + len(X_test), len(self.df))

    def test_masked_mse(self):
        y_true = np.array([[1.0, np.nan], [2.0, 2.0]])
        y_pred = np.array([[1.0, 1.0], [3.0, 2.0]])
        result = self.autoencoder.masked_mse(y_true, y_pred)
        self.assertIsNotNone(result)
