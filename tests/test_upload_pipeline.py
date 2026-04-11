"""Integration tests for ``DataLoader.load_uploaded_csv`` (TASKS.md 2.2).

Covers the edge cases called out in task 2.2:

- Mixed numeric and categorical columns
- Column names with special characters, spaces, or unicode
- Very wide datasets (100+ columns)
- Datasets with mostly missing values
- Completely numeric datasets (no categorical columns to keep before binning)

Plus a few regression cases that expose behavioural gaps between
``DataLoader.prepare_original_dataset`` and ``worker.py``'s inline
cleaning / ``main.prepare_for_categorical``:

- All-NaN columns should be dropped (currently become a useless
  single-value ``"NA"`` column)
- Single-value categorical columns should be dropped per the
  Rule-of-9 spec in ``CLAUDE.md``
- NaN in categorical columns should become ``"missing"`` not the
  string literal ``"nan"``
"""

import io
import unittest

import numpy as np
import pandas as pd

from dataset.loader import DataLoader
from features.transform import Table2Vector


def _csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to CSV bytes the way a browser upload would."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _make_loader() -> DataLoader:
    """Build the loader the way ``worker.py`` does for uploads — no
    drop/rename/select config, since the user's CSV is fully unknown."""
    return DataLoader(drop_columns=[], rename_columns={}, columns_of_interest=[])


class TestUploadedCsvEdgeCases(unittest.TestCase):
    """End-to-end tests for ``DataLoader.load_uploaded_csv``."""

    def test_mixed_numeric_and_categorical(self):
        """Mixed numeric + categorical columns: numerics are binned,
        categoricals pass through, and both land in the clean frame."""
        np.random.seed(0)
        df = pd.DataFrame({
            "category": np.random.choice(["A", "B", "C"], size=50),
            "score": np.random.randn(50),
            "bucket": np.random.choice(["low", "medium", "high"], size=50),
            "value": np.random.randint(0, 100, size=50),
        })
        loader = _make_loader()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        self.assertEqual(len(clean_df), 50)
        self.assertIn("category", clean_df.columns)
        self.assertIn("bucket", clean_df.columns)
        # Numeric columns are binned into *_cat columns
        self.assertIn("score_cat", clean_df.columns)
        self.assertIn("value_cat", clean_df.columns)
        # All kept columns are string-typed (ready for one-hot encoding)
        for col in clean_df.columns:
            self.assertTrue(
                pd.api.types.is_string_dtype(clean_df[col])
                or clean_df[col].dtype == object,
                f"column {col!r} should be string-like, got {clean_df[col].dtype}",
            )
        # Metadata is well-formed and matches clean_df
        self.assertEqual(set(meta["variable_types"]), set(clean_df.columns))
        self.assertTrue(
            all(t == "categorical" for t in meta["variable_types"].values())
        )

    def test_special_characters_in_column_names(self):
        """Column names with spaces, hyphens, dots, slashes must survive
        through read → clean → vectorize unchanged."""
        df = pd.DataFrame({
            "col 1": ["a", "b", "a", "b", "a"],
            "col-2": ["x", "y", "z", "x", "y"],
            "col.3": ["p", "p", "q", "q", "p"],
            "col/4": ["A", "B", "A", "B", "A"],
        })
        loader = _make_loader()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        for col in ["col 1", "col-2", "col.3", "col/4"]:
            self.assertIn(col, clean_df.columns, f"lost column {col!r}")

        # Vectorizer should accept these column names too (this is what the
        # worker/trainer actually feeds the model).
        vec = Table2Vector(meta["variable_types"])
        vec.fit(clean_df)
        vectorized = vec.transform(clean_df)
        self.assertEqual(len(vectorized), len(clean_df))
        self.assertFalse(
            bool(np.isnan(vectorized.values).any()),
            "vectorized output should not contain NaN",
        )

    def test_unicode_column_names_and_values(self):
        """UTF-8 column names and values must round-trip through the upload
        path without corruption."""
        df = pd.DataFrame({
            "pays départ": ["France", "日本", "Canada", "日本", "France"],
            "价格": ["low", "high", "medium", "low", "high"],
            "emoji🙂": ["a", "b", "a", "b", "a"],
        })
        loader = _make_loader()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        self.assertIn("pays départ", clean_df.columns)
        self.assertIn("价格", clean_df.columns)
        self.assertIn("emoji🙂", clean_df.columns)
        # Values should still be the original unicode strings
        self.assertIn("日本", clean_df["pays départ"].tolist())

        vec = Table2Vector(meta["variable_types"])
        vec.fit(clean_df)
        vectorized = vec.transform(clean_df)
        self.assertEqual(len(vectorized), len(clean_df))

    def test_wide_dataset_100_plus_columns(self):
        """Very wide datasets (120 columns, mix of categorical + numeric)
        should process without error and vectorize cleanly."""
        np.random.seed(7)
        n_rows = 100
        n_cat = 60
        n_num = 60
        data = {}
        for i in range(n_cat):
            data[f"cat_{i}"] = np.random.choice(["A", "B", "C", "D"], size=n_rows)
        for i in range(n_num):
            data[f"num_{i}"] = np.random.randn(n_rows)
        df = pd.DataFrame(data)

        loader = _make_loader()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        # All 120 columns should survive (numerics become *_cat)
        self.assertEqual(len(clean_df.columns), n_cat + n_num)
        self.assertEqual(len(clean_df), n_rows)
        self.assertEqual(len(meta["ignored_columns"]), 0)

        # Vectorizer should build a fixed-width matrix
        vec = Table2Vector(meta["variable_types"])
        vec.fit(clean_df)
        X = vec.transform(clean_df).astype("float32")
        self.assertEqual(len(X), n_rows)
        self.assertGreater(X.shape[1], len(clean_df.columns))  # one-hot expansion
        self.assertFalse(bool(np.isnan(X.values).any()))

    def test_mostly_missing_values(self):
        """A dataset with >90% missing values should not crash the loader
        and should produce a frame that can be fed to the vectorizer."""
        np.random.seed(42)
        n_rows = 200
        cat_vals = [
            np.random.choice(["X", "Y", "Z"]) if np.random.rand() > 0.9 else None
            for _ in range(n_rows)
        ]
        num_vals = [
            np.random.randint(0, 10) if np.random.rand() > 0.9 else None
            for _ in range(n_rows)
        ]
        df = pd.DataFrame({"cat": cat_vals, "num": num_vals})

        loader = _make_loader()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        # Both columns should survive — they each have multiple observed values
        self.assertEqual(len(clean_df), n_rows)
        self.assertIn("cat", clean_df.columns)
        self.assertIn("num_cat", clean_df.columns)

        # Missing categorical values must be encoded as the literal string
        # "missing", NOT the float-to-string artifact "nan". This matches
        # worker.py's inline cleaning and main.prepare_for_categorical.
        self.assertIn("missing", clean_df["cat"].unique())
        self.assertNotIn("nan", clean_df["cat"].unique())

        # Missing numeric values should land in the "missing" bin from
        # convert_to_categorical.
        self.assertIn("missing", clean_df["num_cat"].unique())

        # Vectorizer should produce a NaN-free matrix.
        vec = Table2Vector(meta["variable_types"])
        vec.fit(clean_df)
        X = vec.transform(clean_df).astype("float32")
        self.assertFalse(bool(np.isnan(X.values).any()))

    def test_completely_numeric_dataset(self):
        """All-numeric uploads should end up as binned categorical columns
        (no columns silently dropped just because nothing was string-typed)."""
        np.random.seed(1)
        df = pd.DataFrame({
            "x1": np.random.randn(80),
            "x2": np.random.randint(0, 50, 80),
            "x3": np.random.rand(80),
            "x4": np.random.randn(80) * 10,
        })

        loader = _make_loader()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        # Every numeric column should survive as a *_cat binning
        self.assertEqual(len(clean_df), 80)
        for orig in ["x1", "x2", "x3", "x4"]:
            self.assertIn(f"{orig}_cat", clean_df.columns)

        # Bins live in the documented vocabulary
        valid_bins = {
            "top-extreme", "high", "bottom-extreme", "low",
            "normal", "zero", "missing", "unknown", "NA",
        }
        for col in clean_df.columns:
            self.assertTrue(
                set(clean_df[col].unique()).issubset(valid_bins),
                f"{col} has unexpected bins: {set(clean_df[col].unique())}",
            )

        # End-to-end vectorizer fit+transform should succeed.
        vec = Table2Vector(meta["variable_types"])
        vec.fit(clean_df)
        X = vec.transform(clean_df).astype("float32")
        self.assertEqual(len(X), 80)
        self.assertFalse(bool(np.isnan(X.values).any()))


class TestUploadedCsvRuleOfNine(unittest.TestCase):
    """Regression tests for the Rule-of-9 contract documented in CLAUDE.md:

        "Columns with more than 9 unique values or only 1 unique value
         are dropped before training."

    These used to pass silently in ``prepare_original_dataset`` (which only
    filtered the high-cardinality half of the rule), producing clean_df
    values that ``worker.py``'s inline cleaning would have rejected.
    """

    def test_single_value_categorical_column_is_dropped(self):
        df = pd.DataFrame({
            "constant": ["yes"] * 10,
            "varied": ["a", "b", "a", "c", "a", "b", "c", "a", "b", "c"],
        })
        loader = _make_loader()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        self.assertNotIn("constant", clean_df.columns)
        self.assertIn("varied", clean_df.columns)
        ignored_names = {c["name"] for c in meta["ignored_columns"]}
        self.assertIn("constant", ignored_names)

    def test_all_nan_column_is_dropped(self):
        df = pd.DataFrame({
            "missing_col": [None] * 10,
            "good_col": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
        })
        loader = _make_loader()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        # The all-NaN column used to survive as a constant "NA" column
        # with n_unique=1 — worker.py would drop it but this path did not.
        surviving_cols = set(clean_df.columns)
        self.assertNotIn("missing_col", surviving_cols)
        self.assertNotIn("missing_col_cat", surviving_cols)
        self.assertIn("good_col", surviving_cols)

    def test_high_cardinality_column_is_dropped(self):
        df = pd.DataFrame({
            "low_card": ["A", "B", "A", "B"] * 15,
            "high_card": [f"x{i}" for i in range(60)],
        })
        loader = _make_loader()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        self.assertIn("low_card", clean_df.columns)
        self.assertNotIn("high_card", clean_df.columns)

        ignored_names = {c["name"] for c in meta["ignored_columns"]}
        self.assertIn("high_card", ignored_names)

    def test_metadata_is_consistent_with_clean_df(self):
        df = pd.DataFrame({
            "a": ["x", "y", "x", "y"] * 5,
            "b": ["p", "p", "p", "p"] * 5,  # single-value -> dropped
            "c": [f"u{i}" for i in range(20)],  # high-cardinality -> dropped
        })
        loader = _make_loader()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        self.assertEqual(set(meta["variable_types"]), set(clean_df.columns))
        ignored_names = {c["name"] for c in meta["ignored_columns"]}
        self.assertTrue(ignored_names.isdisjoint(set(clean_df.columns)))


class TestUploadedCsvMinimalInputs(unittest.TestCase):
    """Smoke tests for small/unusual CSV shapes that should not crash."""

    def test_single_column_csv(self):
        df = pd.DataFrame({"color": ["red", "blue", "green", "red", "blue", "green"]})
        loader = _make_loader()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        self.assertEqual(list(clean_df.columns), ["color"])
        self.assertEqual(len(clean_df), 6)

    def test_single_row_csv(self):
        df = pd.DataFrame({"a": ["x"], "b": [1]})
        loader = _make_loader()
        # Should not raise; result may be empty after Rule-of-9 (both cols
        # have nunique == 1) — either behaviour is acceptable, we just want
        # to assert the call does not crash.
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))
        self.assertIsInstance(clean_df, pd.DataFrame)
        self.assertIsInstance(meta, dict)
        self.assertIn("variable_types", meta)
        self.assertIn("ignored_columns", meta)

    def test_bytes_vs_str_csv_produce_same_clean_df(self):
        """load_uploaded_csv takes bytes; the older path also accepts str
        file paths via load_original_data. Both should produce identical
        cleaned frames."""
        import os
        import tempfile

        df = pd.DataFrame({
            "a": ["x", "y", "x", "y", "x"] * 4,
            "b": ["p", "q", "r", "p", "q"] * 4,
        })
        loader_bytes = _make_loader()
        clean_a, _ = loader_bytes.load_uploaded_csv(_csv_bytes(df))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            df.to_csv(f, index=False)
            path = f.name
        try:
            loader_path = _make_loader()
            clean_b, _ = loader_path.prepare_original_dataset(
                loader_path.load_original_data(path), replacements={}
            )
        finally:
            os.unlink(path)

        pd.testing.assert_frame_equal(
            clean_a.reset_index(drop=True),
            clean_b.reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()
