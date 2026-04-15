"""
Tests for per-column reconstruction error (TASKS.md 3.3).

Task 3.3 adds per-column error scores so users can see *which* survey
questions a flagged respondent answered anomalously. The scores are exposed
via two public APIs:

* ``compute_per_column_errors`` -- low-level helper returning shape
  ``(N, num_attrs)`` ndarray.
* ``get_outliers_list(..., attr_names=...)`` -- adds ``col_error__{name}``
  columns to the output DataFrame when ``attr_names`` is supplied.
"""

import numpy as np
import pandas as pd
import pytest

from evaluate.outliers import (
    compute_per_column_errors,
    compute_reconstruction_error,
    get_outliers_list,
)
from features.transform import Table2Vector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_data():
    """Two categorical attributes with cardinalities [3, 2]."""
    data = np.array([
        [1.0, 0.0, 0.0, 1.0, 0.0],  # attr1=cat0, attr2=cat0
        [0.0, 1.0, 0.0, 0.0, 1.0],  # attr1=cat1, attr2=cat1
        [0.0, 0.0, 1.0, 1.0, 0.0],  # attr1=cat2, attr2=cat0
    ])
    cardinalities = [3, 2]
    return data, cardinalities


class TestComputePerColumnErrors:
    """Unit tests for compute_per_column_errors."""

    def test_output_shape(self):
        data, cardinalities = _simple_data()
        result = compute_per_column_errors(data, data, cardinalities)
        assert result.shape == (3, 2), f"Expected (3, 2), got {result.shape}"

    def test_dtype_is_float(self):
        data, cardinalities = _simple_data()
        result = compute_per_column_errors(data, data, cardinalities)
        assert result.dtype.kind == "f"

    def test_perfect_reconstruction_yields_near_zero(self):
        data, cardinalities = _simple_data()
        result = compute_per_column_errors(data, data, cardinalities)
        assert np.allclose(result, 0.0, atol=1e-5), f"Expected ~0, got {result}"

    def test_mean_over_attrs_equals_compute_reconstruction_error(self):
        """aggregate = mean of per-column errors."""
        data, cardinalities = _simple_data()
        pred = np.array([
            [0.7, 0.2, 0.1, 0.9, 0.1],
            [0.3, 0.5, 0.2, 0.2, 0.8],
            [0.1, 0.2, 0.7, 0.6, 0.4],
        ])

        per_col = compute_per_column_errors(data, pred, cardinalities)
        aggregate_from_per_col = per_col.mean(axis=1)
        aggregate_direct = compute_reconstruction_error(data, pred, cardinalities)

        assert np.allclose(aggregate_from_per_col, aggregate_direct, atol=1e-6), (
            f"Mean of per-column errors must equal compute_reconstruction_error.\n"
            f"  per_col.mean(axis=1) = {aggregate_from_per_col}\n"
            f"  compute_reconstruction_error = {aggregate_direct}"
        )

    def test_worse_attr_has_higher_column_error(self):
        """A deliberately wrong prediction on attr2 must raise col_error for attr2."""
        data = np.array([[1.0, 0.0, 0.0, 1.0, 0.0]])
        cardinalities = [3, 2]

        # Good prediction on both attributes
        good_pred = np.array([[0.9, 0.05, 0.05, 0.9, 0.1]])
        # Bad prediction on attr2 only
        bad_pred_attr2 = np.array([[0.9, 0.05, 0.05, 0.1, 0.9]])

        good = compute_per_column_errors(data, good_pred, cardinalities)
        bad = compute_per_column_errors(data, bad_pred_attr2, cardinalities)

        # attr1 (col 0) unchanged
        assert np.isclose(good[0, 0], bad[0, 0], atol=1e-5), (
            "attr1 error should be the same across both predictions"
        )
        # attr2 (col 1) is worse
        assert bad[0, 1] > good[0, 1], (
            f"attr2 error should be higher for the bad prediction: "
            f"good={good[0,1]}, bad={bad[0,1]}"
        )

    def test_independent_column_errors_per_row(self):
        """Each row's column errors are independent — row 0's attr2 doesn't
        affect row 1's attr2."""
        data = np.array([
            [1.0, 0.0, 0.0, 1.0, 0.0],  # row0
            [1.0, 0.0, 0.0, 1.0, 0.0],  # row1 (same ground truth)
        ])
        cardinalities = [3, 2]

        # row0 has a bad prediction on attr2; row1 has a good prediction on attr2
        pred = np.array([
            [0.9, 0.05, 0.05, 0.1, 0.9],  # bad attr2
            [0.9, 0.05, 0.05, 0.9, 0.1],  # good attr2
        ])

        result = compute_per_column_errors(data, pred, cardinalities)

        assert result[0, 1] > result[1, 1], (
            "Row 0 should have higher attr2 error than row 1"
        )
        assert np.isclose(result[0, 0], result[1, 0], atol=1e-5), (
            "attr1 error should be equal for rows with the same prediction"
        )

    def test_accepts_dataframe_and_ndarray(self):
        data_np, cardinalities = _simple_data()
        pred_np = data_np.copy()
        pred_np[0, 0] = 0.8
        pred_np[0, 1] = 0.1
        pred_np[0, 2] = 0.1

        data_df = pd.DataFrame(data_np, columns=["a", "b", "c", "d", "e"])
        pred_df = pd.DataFrame(pred_np, columns=["a", "b", "c", "d", "e"])

        r1 = compute_per_column_errors(data_np, pred_np, cardinalities)
        r2 = compute_per_column_errors(data_df, pred_df, cardinalities)
        r3 = compute_per_column_errors(data_df, pred_np, cardinalities)

        assert np.allclose(r1, r2)
        assert np.allclose(r1, r3)

    def test_unseen_category_penalized_in_per_column(self):
        """All-zero block for a categorical attribute → column error = 1.0."""
        cardinalities = [3, 2]
        attr_is_categorical = [True, True]

        data = np.array([
            [1.0, 0.0, 0.0, 1.0, 0.0],  # normal
            [0.0, 0.0, 0.0, 1.0, 0.0],  # attr1 unseen
        ])
        pred = np.array([
            [0.7, 0.2, 0.1, 0.9, 0.1],
            [0.7, 0.2, 0.1, 0.9, 0.1],
        ])

        result = compute_per_column_errors(
            data, pred, cardinalities, attr_is_categorical=attr_is_categorical
        )

        # Unseen attr1 → col 0 capped at 1.0
        assert result[1, 0] == pytest.approx(1.0, abs=1e-5), (
            f"Unseen attr1 column error should be 1.0, got {result[1, 0]}"
        )
        # attr2 observed for both rows → similar error
        assert result[0, 1] == pytest.approx(result[1, 1], abs=1e-5)

    def test_numeric_attr_not_clamped(self):
        """Numeric (non-categorical) attributes must not fire the unseen penalty."""
        cardinalities = [1, 3]
        attr_is_categorical = [False, True]

        # Numeric col = 0.3 (a valid MinMax value below 0.5)
        data = np.array([[0.3, 1.0, 0.0, 0.0]])
        pred = np.array([[0.3, 0.99, 0.005, 0.005]])

        result = compute_per_column_errors(
            data, pred, cardinalities, attr_is_categorical=attr_is_categorical
        )

        # col 0 (numeric) should NOT be clamped to 1.0
        assert result[0, 0] < 0.1, (
            f"Numeric column with value 0.3 must not be clamped; got {result[0, 0]}"
        )

    def test_attr_is_categorical_mismatch_raises(self):
        data = np.array([[1.0, 0.0, 0.0, 1.0, 0.0]])
        with pytest.raises(ValueError, match="attr_is_categorical length"):
            compute_per_column_errors(
                data, data, [3, 2], attr_is_categorical=[True]  # wrong length
            )


class TestGetOutliersListPerColumnColumns:
    """Tests for per-column error columns added to get_outliers_list output."""

    def _make_vectorizer_and_data(self):
        df = pd.DataFrame({
            "q1": ["a", "b", "c", "a", "b"],
            "q2": ["x", "y", "x", "y", "x"],
            "q3": ["lo", "hi", "lo", "hi", "lo"],
        })
        variable_types = {col: "categorical" for col in df.columns}
        vectorizer = Table2Vector(variable_types)
        vectorized = vectorizer.vectorize_table(df)
        cardinalities = [
            len([c for c in vectorized.columns if c.startswith(f"{col}__")])
            for col in df.columns
        ]
        return df, vectorized, cardinalities, vectorizer

    class _StubModel:
        def __init__(self, recon):
            self._recon = recon

        def predict(self, data):
            return self._recon

    def test_col_error_columns_present_when_attr_names_given(self):
        df, vectorized, cardinalities, vectorizer = self._make_vectorizer_and_data()
        attr_names = ["q1", "q2", "q3"]
        recon = vectorized.to_numpy().copy()  # perfect reconstruction

        result = get_outliers_list(
            vectorized,
            self._StubModel(recon),
            k=0,
            attr_cardinalities=cardinalities,
            vectorizer=vectorizer,
            prior=None,
            attr_names=attr_names,
        )

        for name in attr_names:
            col = f"col_error__{name}"
            assert col in result.columns, (
                f"Expected column '{col}' in result; got {list(result.columns)}"
            )

    def test_col_error_columns_absent_when_attr_names_not_given(self):
        """Backward compat: no attr_names → no col_error__ columns."""
        df, vectorized, cardinalities, vectorizer = self._make_vectorizer_and_data()
        recon = vectorized.to_numpy().copy()

        result = get_outliers_list(
            vectorized,
            self._StubModel(recon),
            k=0,
            attr_cardinalities=cardinalities,
            vectorizer=vectorizer,
            prior=None,
        )

        assert not any(c.startswith("col_error__") for c in result.columns), (
            "No col_error__ columns should appear when attr_names is not supplied"
        )

    def test_col_error_aggregate_equals_mean_of_per_col(self):
        """The ``error`` column must equal the mean of all ``col_error__*`` columns."""
        df, vectorized, cardinalities, vectorizer = self._make_vectorizer_and_data()
        attr_names = ["q1", "q2", "q3"]

        # Create a noisy reconstruction so errors are non-trivial
        rng = np.random.RandomState(42)
        raw = vectorized.to_numpy().copy()
        raw += rng.randn(*raw.shape) * 0.3
        raw = np.clip(raw, 1e-6, None)
        recon = raw / raw.sum(axis=1, keepdims=True)

        result = get_outliers_list(
            vectorized,
            self._StubModel(recon),
            k=0,
            attr_cardinalities=cardinalities,
            vectorizer=vectorizer,
            prior=None,
            attr_names=attr_names,
        )

        col_error_cols = [f"col_error__{n}" for n in attr_names]
        per_col_matrix = result[col_error_cols].to_numpy()
        expected_aggregate = per_col_matrix.mean(axis=1)

        assert np.allclose(result["error"].to_numpy(), expected_aggregate, atol=1e-6), (
            "``error`` column must equal the mean of all per-column error scores"
        )

    def test_col_error_values_are_non_negative(self):
        df, vectorized, cardinalities, vectorizer = self._make_vectorizer_and_data()
        attr_names = ["q1", "q2", "q3"]

        rng = np.random.RandomState(7)
        raw = vectorized.to_numpy().copy()
        raw += rng.randn(*raw.shape) * 0.5
        raw = np.clip(raw, 1e-6, None)
        recon = raw / raw.sum(axis=1, keepdims=True)

        result = get_outliers_list(
            vectorized,
            self._StubModel(recon),
            k=0,
            attr_cardinalities=cardinalities,
            vectorizer=vectorizer,
            prior=None,
            attr_names=attr_names,
        )

        for name in attr_names:
            col = f"col_error__{name}"
            assert (result[col] >= 0).all(), (
                f"Column {col} has negative values: {result[col].to_numpy()}"
            )

    def test_high_error_attr_visible_in_per_col_column(self):
        """Injecting a bad prediction on q2 only should raise col_error__q2."""
        df, vectorized, cardinalities, vectorizer = self._make_vectorizer_and_data()
        attr_names = ["q1", "q2", "q3"]

        # Start with a near-perfect reconstruction
        recon = vectorized.to_numpy().astype(float).copy()
        # Find the q2 block and scramble just those columns for row 0
        q2_start = cardinalities[0]
        q2_end = q2_start + cardinalities[1]
        # Flip the probability mass so it's all on the wrong category
        orig_q2 = recon[0, q2_start:q2_end].copy()
        recon[0, q2_start:q2_end] = orig_q2[::-1]  # reverse to put mass elsewhere

        result = get_outliers_list(
            vectorized,
            self._StubModel(recon),
            k=0,
            attr_cardinalities=cardinalities,
            vectorizer=vectorizer,
            prior=None,
            attr_names=attr_names,
        )

        # Row 0 should have higher q2 error than q1 error
        assert result.iloc[0]["col_error__q2"] > result.iloc[0]["col_error__q1"], (
            "Row 0 had q2 deliberately scrambled; col_error__q2 should dominate"
        )

    def test_correct_number_of_col_error_columns(self):
        """Exactly one col_error__ column per attribute."""
        df, vectorized, cardinalities, vectorizer = self._make_vectorizer_and_data()
        attr_names = ["q1", "q2", "q3"]
        recon = vectorized.to_numpy().copy()

        result = get_outliers_list(
            vectorized,
            self._StubModel(recon),
            k=0,
            attr_cardinalities=cardinalities,
            vectorizer=vectorizer,
            prior=None,
            attr_names=attr_names,
        )

        col_error_cols = [c for c in result.columns if c.startswith("col_error__")]
        assert len(col_error_cols) == len(attr_names), (
            f"Expected {len(attr_names)} col_error__ columns, got {len(col_error_cols)}"
        )


class TestComputeAttrLayoutReturnsNames:
    """_compute_attr_layout now returns a third element: attr_names."""

    def test_attr_names_returned_and_in_slice_order(self):
        from main import _compute_attr_layout

        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0],
            "cat1": ["a", "b", "a"],
            "num2": [0.1, 0.2, 0.3],
            "cat2": ["x", "y", "x"],
        })
        variable_types = {
            "num1": "numeric",
            "cat1": "categorical",
            "num2": "numeric",
            "cat2": "categorical",
        }
        vectorizer = Table2Vector(variable_types)
        vectorizer.fit(df)

        attr_cardinalities, attr_is_categorical, attr_names = _compute_attr_layout(
            vectorizer, df.columns
        )

        # Numerics come first in slice order
        assert attr_names == ["num1", "num2", "cat1", "cat2"], (
            f"Expected ['num1', 'num2', 'cat1', 'cat2'], got {attr_names}"
        )
        assert len(attr_names) == len(attr_cardinalities)
        assert len(attr_names) == len(attr_is_categorical)

    def test_all_categorical_attr_names_match_input_order(self):
        """All-categorical dataset: slice order equals input column order."""
        from main import _compute_attr_layout

        df = pd.DataFrame({
            "q1": ["a", "b", "c"],
            "q2": ["x", "y", "x"],
            "q3": ["lo", "hi", "lo"],
        })
        variable_types = {col: "categorical" for col in df.columns}
        vectorizer = Table2Vector(variable_types)
        vectorizer.fit(df)

        _, _, attr_names = _compute_attr_layout(vectorizer, df.columns)

        assert attr_names == ["q1", "q2", "q3"], (
            f"All-categorical names should match input order, got {attr_names}"
        )
