import unittest

import numpy as np
import pandas as pd

from features.transform import Table2Vector


class TestDataTransformer(unittest.TestCase):
    def setUp(self):
        self.variable_types = {
            "age": "numeric",
            "gender": "categorical",
            "income": "numeric",
            "gender_at_birth": "categorical",
        }
        self.vectorizer = Table2Vector(self.variable_types)
        self.SEP = self.vectorizer.SEP
        self.MISSING = self.vectorizer.MISSING

        self.data = pd.DataFrame(
            {
                "age": [25, 30, 35, np.nan],
                "gender": ["male", "female", np.nan, "female"],
                "income": [50000.0, np.nan, 70000.0, 80000.0],
                "gender_at_birth": ["female", "female", np.nan, "male"],
            }
        )

    def test_transform_dataframe(self):

        vectorized_df = self.vectorizer.vectorize_table(self.data)

        # Check that original DataFrame has been transformed properly
        self.assertNotIn("gender", vectorized_df.columns)
        self.assertIn("gender__male", vectorized_df.columns)
        self.assertIn("gender__female", vectorized_df.columns)

        # Check that missing values have been handled correctly
        # self.assertEqual(vectorized_df.loc[3, 'MISSING__age'], 1)
        # self.assertEqual(vectorized_df.loc[0, 'MISSING__age'], 0)

        # Check that numeric columns have been scaled correctly
        self.assertEqual(vectorized_df.loc[0, "age"], 0)
        self.assertEqual(vectorized_df.loc[1, "age"], 0.5)
        self.assertEqual(vectorized_df.loc[2, "age"], 1)
        self.assertTrue(np.isnan(vectorized_df.loc[3, "age"]))

    def test_proba_to_onehot(self):
        proba = np.array([[0.1, 0.9], [0.7, 0.3]])
        expected_onehot = np.array([[0, 1], [1, 0]])

        np.testing.assert_array_equal(
            self.vectorizer.proba_to_onehot(proba), expected_onehot
        )

    def test_reverse_transform_dataframe(self):
        vector_df = self.vectorizer.vectorize_table(self.data)
        reversed_df = self.vectorizer.tabularize_vector(vector_df)

        # Check that DataFrame has been reversed correctly
        pd.testing.assert_frame_equal(reversed_df, self.data, check_like=True)

        # Check that the missing data has been reversed correctly
        self.assertTrue(pd.isnull(reversed_df.loc[3, "age"]))

        # Check that the numeric scaling has been reversed correctly
        self.assertTrue("age" in reversed_df.columns)
        self.assertTrue("income" in reversed_df.columns)
        self.assertListEqual(
            list(self.data["age"].dropna()), list(reversed_df["age"].dropna())
        )
        self.assertTrue(
            np.array_equal(
                np.isnan(self.data["income"]), np.isnan(reversed_df["income"])
            )
        )

        # Check that the categorical encoding has been reversed correctly
        self.assertTrue("gender" in reversed_df.columns)
        self.assertListEqual(list(self.data["gender"]), list(reversed_df["gender"]))
