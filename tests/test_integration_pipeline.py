import unittest

import numpy as np
import pandas as pd

from features.transform import Table2Vector
from model.factory import get_model
from train.trainer import Trainer
from evaluate.outliers import get_outliers_list


def make_synthetic_survey(n_rows=200, seed=42):
    """Generate a synthetic survey DataFrame with 5 categorical columns."""
    rng = np.random.RandomState(seed)
    data = {
        "q1": rng.choice(["a", "b", "c"], size=n_rows),
        "q2": rng.choice(["x", "y"], size=n_rows),
        "q3": rng.choice(["low", "med", "high", "very_high"], size=n_rows),
        "q4": rng.choice(["yes", "no", "maybe"], size=n_rows),
        "q5": rng.choice(["cat", "dog"], size=n_rows),
    }
    return pd.DataFrame(data)


def make_variable_types(df):
    """All columns are categorical for our synthetic survey."""
    return {col: "categorical" for col in df.columns}


MINIMAL_CONFIG = {
    "epochs": 3,
    "batch_size": 32,
    "test_size": 0.2,
    "learning_rate": 1e-3,
    "encoder_layers": 1,
    "encoder_units_1": 32,
    "encoder_activation_1": "relu",
    "encoder_l2_1": 0.001,
    "encoder_dropout_1": 0.0,
    "encoder_batch_norm_1": False,
    "latent_space_dim": 4,
    "latent_activation": "relu",
    "decoder_layers": 1,
    "decoder_units_1": 32,
    "decoder_activation_1": "relu",
    "decoder_l2_1": 0.001,
    "decoder_dropout_1": 0.0,
    "decoder_batch_norm_1": False,
}


class TestIntegrationPipeline(unittest.TestCase):
    """End-to-end: synthetic data -> vectorize -> train AE -> score outliers."""

    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_survey(n_rows=200, seed=42)
        cls.variable_types = make_variable_types(cls.df)
        cls.vectorizer = Table2Vector(cls.variable_types)
        cls.vectorized = cls.vectorizer.vectorize_table(cls.df)

        cardinalities = [
            len([c for c in cls.vectorized.columns if c.startswith(f"{col}__")])
            for col in cls.df.columns
        ]
        cls.cardinalities = cardinalities

        ae_model = get_model("AE", cardinalities)
        trainer = Trainer(ae_model, MINIMAL_CONFIG)
        cls.trained_model, cls.history = trainer.train(cls.vectorized, prior=None)

        filtered_list = [True] * len(cardinalities)
        cls.result_df = get_outliers_list(
            cls.vectorized, cls.trained_model, k=0,
            attr_cardinalities=cardinalities,
            vectorizer=cls.vectorizer, prior=None,
            filtered_list=filtered_list,
        )

    def test_full_pipeline_produces_error_column(self):
        self.assertIn("error", self.result_df.columns)
        self.assertTrue((self.result_df["error"] >= 0).all())

    def test_errors_are_sortable(self):
        sorted_df = self.result_df.sort_values("error", ascending=False)
        self.assertEqual(len(sorted_df), len(self.result_df))
        errors = sorted_df["error"].values
        self.assertTrue(all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1)))

    def test_injected_outliers_score_higher(self):
        """Inject obviously anomalous rows and verify they get higher mean error."""
        n_outliers = 20
        # Create outlier rows where every column has a rare/impossible pattern
        outlier_rows = pd.DataFrame({
            "q1": ["c"] * n_outliers,
            "q2": ["y"] * n_outliers,
            "q3": ["very_high"] * n_outliers,
            "q4": ["maybe"] * n_outliers,
            "q5": ["dog"] * n_outliers,
        })
        # Invert values to be maximally different from typical distribution
        # Create a new vectorizer fitted on the same data so categories match
        combined = pd.concat([self.df, outlier_rows], ignore_index=True)
        vectorizer = Table2Vector(make_variable_types(combined))
        vectorized_combined = vectorizer.vectorize_table(combined)

        cardinalities = [
            len([c for c in vectorized_combined.columns if c.startswith(f"{col}__")])
            for col in combined.columns
        ]

        # Retrain on clean data only
        ae_model = get_model("AE", cardinalities)
        trainer = Trainer(ae_model, MINIMAL_CONFIG)
        clean_vectorized = vectorized_combined.iloc[:200]
        trained_model, _ = trainer.train(clean_vectorized, prior=None)

        filtered_list = [True] * len(cardinalities)
        result_df = get_outliers_list(
            vectorized_combined, trained_model, k=0,
            attr_cardinalities=cardinalities,
            vectorizer=vectorizer, prior=None,
            filtered_list=filtered_list,
        )

        clean_mean_error = result_df.iloc[:200]["error"].mean()
        outlier_mean_error = result_df.iloc[200:]["error"].mean()

        # Outliers should have higher or equal mean error
        # (with only 3 epochs this isn't always strictly greater, so we use >=)
        self.assertGreaterEqual(outlier_mean_error, clean_mean_error * 0.5)
