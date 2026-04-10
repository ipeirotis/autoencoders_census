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

    def test_log_cardinalities_guards_cardinality_one(self):
        """Cardinality-1 attributes must not produce log(1)=0 in log_cardinalities."""
        model = AutoencoderModel([1, 3, 1])
        log_vals = model.log_cardinalities_expanded.numpy().flatten()
        for v in log_vals:
            self.assertGreater(v, 0, "log_cardinalities contains zero (log(1)=0)")

    def test_single_attribute_decoder_builds(self):
        """Decoder with a single attribute must not crash on Concatenate."""
        model = AutoencoderModel([4])
        df = pd.DataFrame({"c": ["A", "B", "C", "D"] * 5})
        model.split_train_test(df, test_size=0.2)
        config = {"encoder_layers": 1, "decoder_layers": 1, "latent_space_dim": 2,
                  "encoder_units_1": 8, "decoder_units_1": 8,
                  "encoder_batch_norm_1": False, "decoder_batch_norm_1": False}
        autoencoder = model.build_autoencoder(config)
        self.assertIsNotNone(autoencoder)
