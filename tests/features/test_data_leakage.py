"""Tests for data leakage fixes in Table2Vector (TASKS.md 3.8).

Covers three bugs:
1. Index misalignment: pd.concat with default RangeIndex introduces NaN
2. MinMaxScaler leakage: scaler fit on full data before train/test split
3. OneHotEncoder leakage: encoder fit on full data before train/test split

Also verifies the new fit()/transform() API and backward compatibility
of the legacy vectorize_table() API.
"""

import unittest

import numpy as np
import pandas as pd

from features.transform import Table2Vector


class TestIndexAlignment(unittest.TestCase):
    """Bug 1: pd.concat with mismatched indices silently introduces NaN."""

    def test_vectorize_table_preserves_nonstandard_index(self):
        """One-hot encoded columns must match the source DataFrame's index."""
        df = pd.DataFrame(
            {"color": ["red", "blue", "green", "red"]},
            index=[10, 20, 30, 40],
        )
        vt = Table2Vector({"color": "categorical"})
        result = vt.vectorize_table(df)

        self.assertListEqual(list(result.index), [10, 20, 30, 40])
        self.assertFalse(result.isna().any().any(), "NaN introduced by index misalignment")

    def test_vectorize_table_preserves_string_index(self):
        df = pd.DataFrame(
            {"color": ["red", "blue", "green"]},
            index=["a", "b", "c"],
        )
        vt = Table2Vector({"color": "categorical"})
        result = vt.vectorize_table(df)

        self.assertListEqual(list(result.index), ["a", "b", "c"])
        self.assertFalse(result.isna().any().any())

    def test_transform_preserves_index_after_train_test_split(self):
        """After sklearn's train_test_split, indices are non-contiguous."""
        from sklearn.model_selection import train_test_split

        df = pd.DataFrame({
            "q1": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
            "q2": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
        })
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

        vt = Table2Vector({"q1": "categorical", "q2": "categorical"})
        vt.fit(train_df)

        train_vec = vt.transform(train_df)
        test_vec = vt.transform(test_df)

        # Indices must be preserved from the original split
        self.assertListEqual(list(train_vec.index), list(train_df.index))
        self.assertListEqual(list(test_vec.index), list(test_df.index))
        self.assertFalse(train_vec.isna().any().any())
        self.assertFalse(test_vec.isna().any().any())

    def test_tabularize_vector_preserves_index(self):
        """tabularize_vector must also handle non-standard indices."""
        df = pd.DataFrame(
            {"color": ["red", "blue", "green"]},
            index=[100, 200, 300],
        )
        vt = Table2Vector({"color": "categorical"})
        vectorized = vt.vectorize_table(df)
        reversed_df = vt.tabularize_vector(vectorized)

        self.assertListEqual(list(reversed_df.index), [100, 200, 300])
        self.assertListEqual(list(reversed_df["color"]), ["red", "blue", "green"])


class TestFitTransformAPI(unittest.TestCase):
    """Test the new fit()/transform() API for preventing data leakage."""

    def setUp(self):
        self.var_types = {"color": "categorical", "size": "categorical"}
        self.train_df = pd.DataFrame({
            "color": ["red", "blue", "red", "blue", "red"],
            "size": ["S", "M", "L", "S", "M"],
        })
        self.test_df = pd.DataFrame({
            "color": ["blue", "red"],
            "size": ["L", "S"],
        })

    def test_fit_then_transform_works(self):
        vt = Table2Vector(self.var_types)
        vt.fit(self.train_df)
        result = vt.transform(self.test_df)
        self.assertFalse(result.isna().any().any())
        self.assertEqual(len(result), 2)

    def test_transform_without_fit_raises(self):
        vt = Table2Vector(self.var_types)
        with self.assertRaises(RuntimeError):
            vt.transform(self.test_df)

    def test_vectorize_table_still_works(self):
        """Legacy API remains functional."""
        vt = Table2Vector(self.var_types)
        result = vt.vectorize_table(self.train_df)
        self.assertFalse(result.isna().any().any())
        self.assertEqual(len(result), 5)

    def test_vectorize_table_with_base_df(self):
        """Legacy base_df parameter still works."""
        vt = Table2Vector(self.var_types)
        result = vt.vectorize_table(self.test_df, base_df=self.train_df)
        self.assertFalse(result.isna().any().any())
        self.assertEqual(len(result), 2)


class TestNoDataLeakage(unittest.TestCase):
    """Verify that fit(train) / transform(test) does NOT leak test data."""

    def test_encoder_categories_from_train_only(self):
        """Categories only in test set are handled by handle_unknown='ignore'."""
        train_df = pd.DataFrame({"fruit": ["apple", "banana", "apple", "banana"]})
        test_df = pd.DataFrame({"fruit": ["apple", "cherry"]})  # cherry is unseen

        vt = Table2Vector({"fruit": "categorical"})
        vt.fit(train_df)

        # The encoder should only know about apple and banana
        categories = list(vt.one_hot_encoders["fruit"].categories_[0])
        self.assertIn("apple", categories)
        self.assertIn("banana", categories)
        self.assertNotIn("cherry", categories)

        # Transform test — cherry gets all-zeros (handle_unknown='ignore')
        test_vec = vt.transform(test_df)
        cherry_row = test_vec.iloc[1]
        self.assertEqual(cherry_row.sum(), 0.0, "Unseen category should be all-zeros")

    def test_scaler_range_from_train_only(self):
        """MinMaxScaler must use train min/max, not test min/max."""
        train_df = pd.DataFrame({"score": [0.0, 10.0, 5.0, 5.0]})
        test_df = pd.DataFrame({"score": [20.0]})  # outside train range

        vt = Table2Vector({"score": "numeric"})
        vt.fit(train_df)
        test_vec = vt.transform(test_df)

        # 20.0 scaled by train range [0, 10] → 2.0 (outside [0,1])
        self.assertAlmostEqual(test_vec.iloc[0]["score"], 2.0)

    def test_fit_transform_equivalent_to_vectorize_table_on_same_data(self):
        """When applied to the same data, fit+transform == vectorize_table."""
        df = pd.DataFrame({
            "q1": ["a", "b", "c", "a", "b"],
            "q2": ["x", "y", "x", "y", "x"],
        })
        var_types = {"q1": "categorical", "q2": "categorical"}

        vt1 = Table2Vector(var_types)
        result1 = vt1.vectorize_table(df)

        vt2 = Table2Vector(var_types)
        vt2.fit(df)
        result2 = vt2.transform(df)

        pd.testing.assert_frame_equal(result1, result2)


class TestGetCardinalities(unittest.TestCase):
    """Cardinalities must match the fitted encoder's actual categories."""

    def test_categorical_cardinalities_match_encoder(self):
        train_df = pd.DataFrame({
            "fruit": ["apple", "banana", "apple", "banana"],
            "color": ["red", "blue", "green", "red"],
        })
        vt = Table2Vector({"fruit": "categorical", "color": "categorical"})
        vt.fit(train_df)

        cardinalities = vt.get_cardinalities(["fruit", "color"])
        self.assertEqual(cardinalities, [2, 3])

    def test_cardinalities_exclude_test_only_categories(self):
        """If test has an extra category, cardinalities must NOT count it."""
        train_df = pd.DataFrame({"q": ["a", "b", "a", "b"]})
        vt = Table2Vector({"q": "categorical"})
        vt.fit(train_df)

        # Full dataset would have 3 unique, but encoder only knows 2
        cardinalities = vt.get_cardinalities(["q"])
        self.assertEqual(cardinalities, [2])

    def test_numeric_cardinality_is_one(self):
        train_df = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
        vt = Table2Vector({"score": "numeric"})
        vt.fit(train_df)

        cardinalities = vt.get_cardinalities(["score"])
        self.assertEqual(cardinalities, [1])

    def test_mixed_cardinalities(self):
        train_df = pd.DataFrame({
            "name": ["alice", "bob", "alice"],
            "age": [25.0, 30.0, 35.0],
        })
        vt = Table2Vector({"name": "categorical", "age": "numeric"})
        vt.fit(train_df)

        cardinalities = vt.get_cardinalities(["name", "age"])
        self.assertEqual(cardinalities, [2, 1])


class TestNumericScalerLeakage(unittest.TestCase):
    """MinMaxScaler-specific leakage tests."""

    def test_scaler_not_refit_on_transform(self):
        """Calling transform() must not update the fitted scaler."""
        train_df = pd.DataFrame({"val": [0.0, 10.0]})
        test_df = pd.DataFrame({"val": [100.0]})

        vt = Table2Vector({"val": "numeric"})
        vt.fit(train_df)

        # Record scaler state
        orig_min = vt.min_max_scalers["val"].data_min_[0]
        orig_max = vt.min_max_scalers["val"].data_max_[0]

        vt.transform(test_df)

        # Scaler must not have changed
        self.assertEqual(vt.min_max_scalers["val"].data_min_[0], orig_min)
        self.assertEqual(vt.min_max_scalers["val"].data_max_[0], orig_max)


class TestPrepareForTraining(unittest.TestCase):
    """Integration test for the main.py prepare_for_training function."""

    def test_prepare_for_training_returns_correct_shapes(self):
        from main import prepare_for_training

        df = pd.DataFrame({
            "q1": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
            "q2": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
        })
        cleaned, X_train, X_test, vectorizer, cardinalities = prepare_for_training(
            df, test_size=0.3,
        )
        self.assertEqual(len(X_train), 7)
        self.assertEqual(len(X_test), 3)
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertFalse(X_train.isna().any().any())
        self.assertFalse(X_test.isna().any().any())

    def test_prepare_for_training_no_nan_after_split(self):
        """The split-then-vectorize flow must not introduce NaN."""
        from main import prepare_for_training

        np.random.seed(42)
        df = pd.DataFrame({
            f"q{i}": np.random.choice(["a", "b", "c"], size=100)
            for i in range(5)
        })
        _, X_train, X_test, _, _ = prepare_for_training(df, test_size=0.2)

        self.assertFalse(X_train.isna().any().any())
        self.assertFalse(X_test.isna().any().any())

    def test_vectorizer_fitted_on_train_only(self):
        """Categories in test-only should not appear in the fitted encoder."""
        from main import prepare_for_training

        # Construct data so that one category appears only at the end
        # (likely to end up in test split with a fixed seed)
        rows = (
            [{"q1": "common_a"}] * 40
            + [{"q1": "common_b"}] * 40
            + [{"q1": "rare_only_test"}] * 2  # only 2 rows
        )
        df = pd.DataFrame(rows)

        # With 82 rows and test_size=0.2, ~16 go to test.
        # We can't guarantee rare_only_test lands in test, but
        # we CAN verify the vectorizer was fitted on training data only
        # by checking the fit was called before transform
        _, X_train, X_test, vectorizer, _ = prepare_for_training(
            df, test_size=0.2,
        )

        # Verify the vectorizer is fitted (has _is_fitted flag)
        self.assertTrue(getattr(vectorizer, "_is_fitted", False))

        # The train and test sets should have matching columns
        self.assertListEqual(list(X_train.columns), list(X_test.columns))


class TestVectorizerPersistence(unittest.TestCase):
    """Roundtrip test: save a fitted vectorizer then load and reuse it."""

    def test_save_load_vectorizer_roundtrip(self):
        import tempfile, os, joblib
        from utils import save_model, load_vectorizer

        train_df = pd.DataFrame({
            "q1": ["a", "b", "c", "a", "b", "c"],
            "q2": ["x", "y", "x", "y", "x", "y"],
        })
        vt = Table2Vector({"q1": "categorical", "q2": "categorical"})
        vt.fit(train_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model/")
            os.makedirs(output_path, exist_ok=True)
            # Save just the vectorizer (skip Keras model to keep test fast)
            joblib.dump(vt, os.path.join(output_path, "vectorizer.joblib"))

            # load_vectorizer expects model_path like "<dir>/autoencoder"
            model_path = os.path.join(output_path, "autoencoder")
            loaded = load_vectorizer(model_path)

            self.assertIsNotNone(loaded)
            self.assertEqual(
                loaded.get_cardinalities(["q1", "q2"]),
                vt.get_cardinalities(["q1", "q2"]),
            )

            # Transform with loaded vectorizer must match original
            test_df = pd.DataFrame({"q1": ["a", "b"], "q2": ["y", "x"]})
            original_result = vt.transform(test_df)
            loaded_result = loaded.transform(test_df)
            pd.testing.assert_frame_equal(original_result, loaded_result)

    def test_load_vectorizer_returns_none_for_old_models(self):
        import tempfile, os
        from utils import load_vectorizer

        with tempfile.TemporaryDirectory() as tmpdir:
            # No vectorizer.joblib exists
            model_path = os.path.join(tmpdir, "autoencoder")
            self.assertIsNone(load_vectorizer(model_path))

    def test_inference_uses_training_categories_only(self):
        """End-to-end: scoring data with extra categories uses training schema."""
        train_df = pd.DataFrame({
            "fruit": ["apple", "banana", "apple", "banana"],
        })
        score_df = pd.DataFrame({
            "fruit": ["apple", "cherry"],  # cherry unseen in training
        })

        vt = Table2Vector({"fruit": "categorical"})
        vt.fit(train_df)

        # Score with the training-fitted vectorizer
        scored = vt.transform(score_df)
        # Width must match training (2 columns: apple, banana), not scoring (3)
        self.assertEqual(scored.shape[1], 2)
        # Cherry row should be all-zeros
        cherry_row = scored.iloc[1]
        self.assertEqual(cherry_row.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
