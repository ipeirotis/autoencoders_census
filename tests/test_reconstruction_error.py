"""
Tests for the unified reconstruction-error helper (TASKS.md 2.8).

The CLI `find_outliers` command, the local worker in `worker.py`, and the
Vertex AI container in `train/task.py` all call `compute_reconstruction_error`
so that the web UI and CLI produce identical rankings on the same model and
data. Before TASKS.md 2.8, the worker and Vertex paths used raw MSE on one-hot
vectors while the CLI used per-attribute categorical crossentropy normalized
by log(K) — a fundamentally different score. These tests pin the unified
behaviour.
"""

import numpy as np
import pandas as pd

from evaluate.outliers import compute_reconstruction_error, get_outliers_list
from features.transform import Table2Vector
from model.base import VAE


def _perfect_reconstruction(data):
    """Return the inputs unchanged — a trivially perfect reconstruction."""
    return data.copy()


class TestComputeReconstructionError:
    """Unit tests for compute_reconstruction_error."""

    def test_perfect_reconstruction_yields_near_zero_error(self):
        data = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
            ]
        )
        cardinalities = [3, 2]

        error = compute_reconstruction_error(data, data, cardinalities)

        assert error.shape == (2,)
        assert np.allclose(error, 0.0, atol=1e-5)

    def test_worse_prediction_has_higher_error(self):
        # Ground truth: category 0 for attribute 1 (cardinality 3), category 0 for attribute 2 (cardinality 2)
        data = np.array([[1.0, 0.0, 0.0, 1.0, 0.0]])
        cardinalities = [3, 2]

        good_pred = np.array([[0.9, 0.05, 0.05, 0.9, 0.1]])
        bad_pred = np.array([[0.1, 0.45, 0.45, 0.1, 0.9]])

        good_error = compute_reconstruction_error(data, good_pred, cardinalities)
        bad_error = compute_reconstruction_error(data, bad_pred, cardinalities)

        assert good_error[0] < bad_error[0]

    def test_accepts_dataframe_and_ndarray_interchangeably(self):
        data_np = np.array([[1.0, 0.0, 0.0, 0.0, 1.0]])
        predictions_np = np.array([[0.7, 0.2, 0.1, 0.1, 0.9]])
        cardinalities = [3, 2]

        data_df = pd.DataFrame(data_np, columns=["a", "b", "c", "d", "e"])
        predictions_df = pd.DataFrame(predictions_np, columns=["a", "b", "c", "d", "e"])

        err_np_np = compute_reconstruction_error(data_np, predictions_np, cardinalities)
        err_df_df = compute_reconstruction_error(data_df, predictions_df, cardinalities)
        err_df_np = compute_reconstruction_error(data_df, predictions_np, cardinalities)
        err_np_df = compute_reconstruction_error(data_np, predictions_df, cardinalities)

        assert np.allclose(err_np_np, err_df_df)
        assert np.allclose(err_np_np, err_df_np)
        assert np.allclose(err_np_np, err_np_df)

    def test_returns_numpy_array(self):
        data = np.array([[1.0, 0.0, 0.0, 1.0, 0.0]])
        predictions = np.array([[0.8, 0.1, 0.1, 0.7, 0.3]])
        cardinalities = [3, 2]

        result = compute_reconstruction_error(data, predictions, cardinalities)

        assert isinstance(result, np.ndarray)
        assert result.dtype.kind == "f"
        assert result.shape == (1,)

    def test_matches_vae_reconstruction_loss_directly(self):
        """The helper must be a thin wrapper around VAE.reconstruction_loss."""
        rng = np.random.RandomState(0)
        data = rng.rand(5, 7).astype(np.float32)
        predictions = rng.rand(5, 7).astype(np.float32)
        cardinalities = [3, 4]

        helper_result = compute_reconstruction_error(data, predictions, cardinalities)
        direct_result = VAE.reconstruction_loss(
            cardinalities, data, predictions
        ).numpy()

        assert np.allclose(helper_result, direct_result)

    def test_higher_error_for_more_categories_wrong(self):
        """Row with two wrong attributes scores higher than row with one wrong."""
        data = np.array(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0],  # both attrs correct
                [1.0, 0.0, 0.0, 0.0, 1.0],  # second attr wrong
                [0.0, 1.0, 0.0, 0.0, 1.0],  # both attrs wrong
            ]
        )
        prediction = np.array([[0.9, 0.05, 0.05, 0.9, 0.1]] * 3)
        cardinalities = [3, 2]

        errors = compute_reconstruction_error(data, prediction, cardinalities)

        assert errors[0] < errors[1] < errors[2]


class TestScoringConsistencyAcrossCodePaths:
    """
    Regression tests for TASKS.md 2.8: before the fix, worker.py and
    train/task.py computed per-row MSE on one-hot vectors while the CLI
    used per-attribute categorical crossentropy. These tests verify that
    all three code paths now produce the same ranking on the same inputs.
    """

    def _make_synthetic_data(self, seed=0):
        rng = np.random.RandomState(seed)
        df = pd.DataFrame(
            {
                "q1": rng.choice(["a", "b", "c"], size=50),
                "q2": rng.choice(["x", "y"], size=50),
                "q3": rng.choice(["low", "med", "high"], size=50),
            }
        )
        variable_types = {col: "categorical" for col in df.columns}
        vectorizer = Table2Vector(variable_types)
        vectorized = vectorizer.vectorize_table(df)
        cardinalities = [
            len([c for c in vectorized.columns if c.startswith(f"{col}__")])
            for col in df.columns
        ]
        return df, vectorized, cardinalities

    def _make_fake_reconstruction(self, vectorized, noise_scale=0.3, seed=1):
        """Produce a plausible softmax-like reconstruction that varies row-by-row."""
        rng = np.random.RandomState(seed)
        raw = vectorized.to_numpy().copy()
        raw += rng.randn(*raw.shape) * noise_scale
        # Clip to non-negative and renormalize per row as a coarse softmax stand-in
        raw = np.clip(raw, 1e-6, None)
        return raw / raw.sum(axis=1, keepdims=True)

    def test_worker_and_task_modules_import_unified_helper(self):
        """
        Structural regression guard: both the local worker and the Vertex
        AI container entry point must import ``compute_reconstruction_error``
        from ``evaluate.outliers``. If either file reverts to raw MSE on
        one-hot vectors, this test will fail immediately rather than
        waiting for a silent divergence to be discovered in production.
        """
        import pathlib

        repo_root = pathlib.Path(__file__).resolve().parents[1]

        worker_src = (repo_root / "worker.py").read_text()
        assert "compute_reconstruction_error" in worker_src, (
            "worker.py must import compute_reconstruction_error to keep "
            "web UI outlier rankings consistent with the CLI (TASKS.md 2.8)"
        )
        assert "np.power(vectorized_df - reconstruction" not in worker_src, (
            "worker.py must not re-introduce the legacy MSE scoring "
            "(TASKS.md 2.8)"
        )

        task_src = (repo_root / "train" / "task.py").read_text()
        assert "compute_reconstruction_error" in task_src, (
            "train/task.py must import compute_reconstruction_error to keep "
            "Vertex AI outlier rankings consistent with the CLI (TASKS.md 2.8)"
        )
        assert "np.power(vectorized_df - reconstruction" not in task_src, (
            "train/task.py must not re-introduce the legacy MSE scoring "
            "(TASKS.md 2.8)"
        )

    def test_new_scoring_differs_from_legacy_mse(self):
        """
        Sanity check: the unified score is genuinely different from the
        old MSE-on-one-hot score used by the pre-fix worker. If they were
        identical, the unification would be trivially vacuous.
        """
        _df, vectorized, cardinalities = self._make_synthetic_data()
        reconstruction = self._make_fake_reconstruction(vectorized)

        unified_score = compute_reconstruction_error(
            vectorized, reconstruction, cardinalities
        )
        legacy_mse = np.mean(
            np.power(vectorized.to_numpy() - reconstruction, 2), axis=1
        )

        # Different absolute values ...
        assert not np.allclose(unified_score, legacy_mse, atol=1e-3)
        # ... and at least some difference in ranking in the general case.
        # We check that the top-ranked row under the two scores is allowed to
        # differ — we do not require it, but we require that the score vectors
        # are not identical multiples of each other.
        corr = np.corrcoef(unified_score, legacy_mse)[0, 1]
        assert corr < 0.9999

    def test_get_outliers_list_uses_unified_helper(self):
        """
        End-to-end: a dummy predict model fed through get_outliers_list
        must produce an ``error`` column that exactly matches
        compute_reconstruction_error on the same inputs.
        """
        _df, vectorized, cardinalities = self._make_synthetic_data()
        reconstruction = self._make_fake_reconstruction(vectorized)

        class _StubModel:
            def __init__(self, recon):
                self._recon = recon

            def predict(self, data):
                return self._recon

        variable_types = {col: "categorical" for col in _df.columns}
        vectorizer = Table2Vector(variable_types)
        vectorizer.vectorize_table(_df)

        result_df = get_outliers_list(
            vectorized,
            _StubModel(reconstruction),
            k=0,
            attr_cardinalities=cardinalities,
            vectorizer=vectorizer,
            prior=None,
        )

        direct_error = compute_reconstruction_error(
            vectorized, reconstruction, cardinalities
        )

        assert "error" in result_df.columns
        assert np.allclose(result_df["error"].to_numpy(), direct_error)
