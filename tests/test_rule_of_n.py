"""Tests for the configurable "Rule of N" cardinality threshold
(TASKS.md 3.1).

Previously the max-unique-values threshold for dropping columns was
hardcoded to 9 in several places:

- ``DataLoader.prepare_original_dataset`` in ``dataset/loader.py``
- ``prepare_for_categorical`` in ``main.py`` (plus its transitive
  callers ``prepare_for_model`` and ``prepare_for_training``)
- Inline Rule-of-9 loops in ``worker.py`` and ``train/task.py``

These tests verify that the threshold is now configurable end-to-end:

- ``DataLoader`` honours the ``max_unique_values`` constructor argument
  on every loader entry point that calls ``prepare_original_dataset``
  (including the uploaded-CSV path via ``load_uploaded_csv``).
- ``DataLoader.prepare_original_dataset`` accepts a per-call override.
- ``main.prepare_for_categorical`` / ``prepare_for_model`` /
  ``prepare_for_training`` all plumb the parameter through correctly.
- ``worker._resolve_max_unique_values`` reads the ``MAX_UNIQUE_VALUES``
  environment variable with sensible fallbacks for garbage input.
- Values below 2 are rejected (a threshold below 2 would drop every
  column, since Rule-of-N also rejects columns with <= 1 unique
  values).
"""

import io
import os
import unittest
from unittest import mock

import pandas as pd

from dataset.loader import DEFAULT_MAX_UNIQUE_VALUES, DataLoader
from main import (
    prepare_for_categorical,
    prepare_for_model,
    prepare_for_training,
)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _make_loader(max_unique_values=None) -> DataLoader:
    return DataLoader(
        drop_columns=[],
        rename_columns={},
        columns_of_interest=[],
        max_unique_values=max_unique_values,
    )


def _sample_df_with_varied_cardinalities() -> pd.DataFrame:
    """Fixture: each column has a known distinct cardinality from 2 to 15."""
    return pd.DataFrame({
        "card_2": ["a", "b"] * 30,
        "card_5": ["a", "b", "c", "d", "e"] * 12,
        "card_9": [f"v{i % 9}" for i in range(60)],
        "card_12": [f"v{i % 12}" for i in range(60)],
        "card_15": [f"v{i % 15}" for i in range(60)],
    })


class TestDefaultThresholdUnchanged(unittest.TestCase):
    """The default behaviour must match the historical Rule of 9."""

    def test_default_constant_is_9(self):
        self.assertEqual(DEFAULT_MAX_UNIQUE_VALUES, 9)

    def test_default_loader_keeps_up_to_9(self):
        loader = _make_loader()  # default threshold
        self.assertEqual(loader.max_unique_values, 9)

        df = _sample_df_with_varied_cardinalities()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        # Columns with 2, 5, 9 unique values survive; 12 and 15 are dropped.
        self.assertIn("card_2", clean_df.columns)
        self.assertIn("card_5", clean_df.columns)
        self.assertIn("card_9", clean_df.columns)
        self.assertNotIn("card_12", clean_df.columns)
        self.assertNotIn("card_15", clean_df.columns)

        ignored_names = {c["name"] for c in meta["ignored_columns"]}
        self.assertEqual(ignored_names, {"card_12", "card_15"})


class TestDataLoaderThreshold(unittest.TestCase):
    """DataLoader must honour the configured threshold."""

    def test_loader_with_high_threshold_keeps_more_columns(self):
        loader = _make_loader(max_unique_values=15)
        self.assertEqual(loader.max_unique_values, 15)

        df = _sample_df_with_varied_cardinalities()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        # All five columns have <= 15 unique values, so all survive.
        for col in ["card_2", "card_5", "card_9", "card_12", "card_15"]:
            self.assertIn(col, clean_df.columns)

        ignored_names = {c["name"] for c in meta["ignored_columns"]}
        self.assertEqual(ignored_names, set())

    def test_loader_with_low_threshold_drops_more_columns(self):
        loader = _make_loader(max_unique_values=4)
        self.assertEqual(loader.max_unique_values, 4)

        df = _sample_df_with_varied_cardinalities()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        # Only card_2 has <= 4 unique values.
        self.assertEqual(list(clean_df.columns), ["card_2"])

        ignored_names = {c["name"] for c in meta["ignored_columns"]}
        self.assertEqual(
            ignored_names, {"card_5", "card_9", "card_12", "card_15"}
        )

        # Ignored metadata should report each column's actual cardinality
        # alongside a reason that mentions the configured threshold.
        for entry in meta["ignored_columns"]:
            self.assertGreater(entry["unique_values"], 4)
            self.assertIn("4", entry["reason"])

    def test_loader_threshold_still_drops_single_value_columns(self):
        """Dropping constant columns is independent of N — they must
        still be filtered even with a very high threshold."""
        loader = _make_loader(max_unique_values=50)
        df = pd.DataFrame({
            "constant": ["x"] * 20,
            "varied": ["a", "b", "a", "b"] * 5,
        })
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        self.assertNotIn("constant", clean_df.columns)
        self.assertIn("varied", clean_df.columns)

    def test_per_call_override_beats_instance_default(self):
        """``prepare_original_dataset`` accepts a per-call override."""
        loader = _make_loader(max_unique_values=9)  # historical default
        df = _sample_df_with_varied_cardinalities()

        # Per-call override widens the window to 15.
        clean_df, meta = loader.prepare_original_dataset(
            df.copy(), replacements={}, max_unique_values=15
        )
        self.assertIn("card_12", clean_df.columns)
        self.assertIn("card_15", clean_df.columns)

        # Instance default is unchanged.
        self.assertEqual(loader.max_unique_values, 9)

    def test_loader_rejects_threshold_below_two(self):
        with self.assertRaises(ValueError):
            _make_loader(max_unique_values=1)
        with self.assertRaises(ValueError):
            _make_loader(max_unique_values=0)

    def test_prepare_original_dataset_rejects_per_call_threshold_below_two(self):
        loader = _make_loader()
        with self.assertRaises(ValueError):
            loader.prepare_original_dataset(
                pd.DataFrame({"a": ["x", "y"]}),
                replacements={},
                max_unique_values=1,
            )


class TestMainHelpers(unittest.TestCase):
    """The main.py cleaning / vectorization helpers must thread the
    threshold through correctly."""

    def test_prepare_for_categorical_default(self):
        df = _sample_df_with_varied_cardinalities()
        clean_df = prepare_for_categorical(df)
        self.assertEqual(
            set(clean_df.columns), {"card_2", "card_5", "card_9"}
        )

    def test_prepare_for_categorical_with_high_threshold(self):
        df = _sample_df_with_varied_cardinalities()
        clean_df = prepare_for_categorical(df, max_unique_values=15)
        self.assertEqual(
            set(clean_df.columns),
            {"card_2", "card_5", "card_9", "card_12", "card_15"},
        )

    def test_prepare_for_categorical_with_low_threshold(self):
        df = _sample_df_with_varied_cardinalities()
        clean_df = prepare_for_categorical(df, max_unique_values=4)
        self.assertEqual(set(clean_df.columns), {"card_2"})

    def test_prepare_for_categorical_rejects_threshold_below_two(self):
        with self.assertRaises(ValueError):
            prepare_for_categorical(
                pd.DataFrame({"a": ["x", "y"]}), max_unique_values=1
            )

    def test_prepare_for_model_threads_threshold(self):
        df = _sample_df_with_varied_cardinalities()
        cleaned_df, vectorized_df, vectorizer, cardinalities = prepare_for_model(
            df, variable_types=None, max_unique_values=12
        )

        self.assertEqual(
            set(cleaned_df.columns),
            {"card_2", "card_5", "card_9", "card_12"},
        )
        # Vectorized width should match the sum of per-column cardinalities
        # of the *kept* columns (2 + 5 + 9 + 12 = 28).
        self.assertEqual(vectorized_df.shape[1], sum(cardinalities))
        self.assertEqual(sum(cardinalities), 2 + 5 + 9 + 12)

    def test_prepare_for_training_threads_threshold(self):
        # Use a larger dataset so the train/test split has rows on both
        # sides even for columns with up to 15 unique values.
        df = pd.concat(
            [_sample_df_with_varied_cardinalities()] * 5, ignore_index=True
        )
        cleaned_df, X_train, X_test, vectorizer, cardinalities = prepare_for_training(
            df, variable_types=None, test_size=0.2, max_unique_values=15
        )

        self.assertEqual(
            set(cleaned_df.columns),
            {"card_2", "card_5", "card_9", "card_12", "card_15"},
        )
        # Full one-hot width: 2 + 5 + 9 + 12 + 15 = 43
        self.assertEqual(sum(cardinalities), 2 + 5 + 9 + 12 + 15)
        self.assertEqual(X_train.shape[1], sum(cardinalities))
        self.assertEqual(X_test.shape[1], sum(cardinalities))


class TestWorkerEnvVarResolution(unittest.TestCase):
    """``worker._resolve_max_unique_values`` reads MAX_UNIQUE_VALUES
    with graceful fallback for invalid values."""

    def _resolve(self):
        # Lazy import so this test file does not load worker.py (and its
        # GCP client init) unless this class actually runs.
        from worker import _resolve_max_unique_values
        return _resolve_max_unique_values()

    def test_resolve_default_when_env_var_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAX_UNIQUE_VALUES", None)
            self.assertEqual(self._resolve(), DEFAULT_MAX_UNIQUE_VALUES)

    def test_resolve_default_when_env_var_empty(self):
        with mock.patch.dict(os.environ, {"MAX_UNIQUE_VALUES": ""}):
            self.assertEqual(self._resolve(), DEFAULT_MAX_UNIQUE_VALUES)

    def test_resolve_reads_valid_int(self):
        with mock.patch.dict(os.environ, {"MAX_UNIQUE_VALUES": "20"}):
            self.assertEqual(self._resolve(), 20)

    def test_resolve_ignores_non_integer(self):
        with mock.patch.dict(os.environ, {"MAX_UNIQUE_VALUES": "abc"}):
            self.assertEqual(self._resolve(), DEFAULT_MAX_UNIQUE_VALUES)

    def test_resolve_ignores_below_two(self):
        with mock.patch.dict(os.environ, {"MAX_UNIQUE_VALUES": "1"}):
            self.assertEqual(self._resolve(), DEFAULT_MAX_UNIQUE_VALUES)
        with mock.patch.dict(os.environ, {"MAX_UNIQUE_VALUES": "-5"}):
            self.assertEqual(self._resolve(), DEFAULT_MAX_UNIQUE_VALUES)


class TestTrainTaskEnvVarResolution(unittest.TestCase):
    """``train.task._resolve_max_unique_values`` mirrors the worker's
    env var reader for Vertex AI containers."""

    def _resolve(self):
        from train.task import _resolve_max_unique_values
        return _resolve_max_unique_values()

    def test_resolve_default_when_env_var_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAX_UNIQUE_VALUES", None)
            self.assertEqual(self._resolve(), DEFAULT_MAX_UNIQUE_VALUES)

    def test_resolve_reads_valid_int(self):
        with mock.patch.dict(os.environ, {"MAX_UNIQUE_VALUES": "7"}):
            self.assertEqual(self._resolve(), 7)

    def test_resolve_ignores_invalid_values(self):
        for raw in ("0", "1", "-3", "not-a-number", "9.5"):
            with mock.patch.dict(os.environ, {"MAX_UNIQUE_VALUES": raw}):
                self.assertEqual(self._resolve(), DEFAULT_MAX_UNIQUE_VALUES)


if __name__ == "__main__":
    unittest.main()
