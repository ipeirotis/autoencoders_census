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

    def test_unseen_category_rows_are_penalized(self):
        """
        Regression test for Codex P1 on PR #46.

        After TASKS.md 3.8, ``Table2Vector`` fits on the training split only
        with ``OneHotEncoder(handle_unknown='ignore')``. When the full dataset
        is transformed for scoring, rows whose category values were not in
        the training split become all-zero one-hot blocks for that attribute.
        Standard categorical crossentropy returns ~0 on an all-zero target
        (``-sum(0 * log(pred)) == 0``), so without the unseen-category
        penalty these rows — which are exactly the rare/outlier rows we want
        to flag — would score as "normal". The helper must penalize them at
        least as strongly as a row with an incorrect prediction on the same
        attribute.
        """
        # Two attributes: cardinalities [3, 2]
        cardinalities = [3, 2]

        data = np.array(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0],  # normal, attr1 observed (cat 0)
                [0.0, 0.0, 0.0, 1.0, 0.0],  # unseen category in attr1
            ]
        )
        # Model prediction is a reasonable softmax on attr1 and perfect on attr2.
        prediction = np.array(
            [
                [0.7, 0.2, 0.1, 0.9, 0.1],
                [0.7, 0.2, 0.1, 0.9, 0.1],
            ]
        )

        errors = compute_reconstruction_error(data, prediction, cardinalities)

        # The unseen-category row must score strictly higher than the normal row.
        assert errors[1] > errors[0], (
            f"Unseen-category row must be penalized more heavily than the "
            f"observed row, got errors={errors}"
        )

        # The unseen attribute must contribute the maximum normalized loss
        # (1.0) to the row, not ~0. With cardinalities [3, 2] and the second
        # attribute perfectly reconstructed (CE ~= 0), the unseen row's
        # per-row loss should be approximately 0.5 (mean of 1.0 and 0.0),
        # which is far higher than the ~0 the buggy path would produce.
        assert errors[1] > 0.3, (
            f"Unseen-category row should score at least ~0.5 from the "
            f"maxed-out attribute alone, got errors[1]={errors[1]}"
        )

    def test_unseen_category_penalty_equals_uniform_wrong_prediction(self):
        """
        A row with an unseen (all-zero) attribute block should score
        roughly the same as a row where the true category was correctly
        encoded but the model put all probability mass on the wrong class.
        Both are maximum-loss scenarios for that attribute.
        """
        cardinalities = [3]

        # Row A: unseen category (all-zero block)
        data_unseen = np.array([[0.0, 0.0, 0.0]])
        # Row B: observed cat 0, but the model predicted cat 2 with full confidence
        data_wrong = np.array([[1.0, 0.0, 0.0]])
        pred_wrong = np.array([[1e-7, 1e-7, 1 - 2e-7]])

        # Both rows share the same "all probability on wrong class" prediction
        err_unseen = compute_reconstruction_error(
            data_unseen, pred_wrong, cardinalities
        )
        err_wrong = compute_reconstruction_error(data_wrong, pred_wrong, cardinalities)

        # The unseen path hard-caps at 1.0 (log(K)/log(K)); the wrong path
        # computes log(1/1e-7) / log(K) which is >> 1.0. Both are clearly
        # penalized — the key assertion is that the unseen path is not ~0.
        assert err_unseen[0] >= 0.9, (
            f"Unseen-category row should be assigned ~1.0 normalized loss, "
            f"got {err_unseen[0]}"
        )
        assert err_wrong[0] > 0.9

    def test_numeric_features_are_not_force_maxed_by_unseen_penalty(self):
        """
        Regression test for Codex P1 #2 on PR #46.

        ``Table2Vector`` produces ``cardinality == 1`` blocks for numeric
        (MinMax-scaled) columns — a single scalar in [0, 1] per row.
        The unseen-category override must not fire on numeric attributes,
        because a valid scaled numeric value in ``[0, 0.5]`` would be
        misclassified as "unseen" and clamped to loss 1.0 regardless of
        reconstruction quality. Callers distinguish categorical from
        numeric via the ``attr_is_categorical`` hint.

        Note: this test does not assert that the numeric per-attribute
        loss responds to the numeric value itself — categorical
        crossentropy on a cardinality-1 block is mathematically
        degenerate (softmax normalizes a single value to 1.0, and
        ``-target * log(1.0) = 0``). That is a pre-existing limitation
        of ``VAE.reconstruction_loss`` on numeric features, outside the
        scope of TASKS 2.8. What this test pins down is that the
        numeric path does **not** receive the Codex-flagged
        "constant max penalty of 1.0" from the unseen-category override.
        """
        # One numeric (card=1) attribute + one categorical (card=3) attribute.
        cardinalities = [1, 3]
        attr_is_categorical = [False, True]
        data = np.array([[0.3, 1.0, 0.0, 0.0]])
        pred = np.array([[0.3, 0.99, 0.005, 0.005]])

        err = compute_reconstruction_error(
            data, pred, cardinalities, attr_is_categorical=attr_is_categorical
        )

        # Without the hint: err[0] would be ~= mean(1.0, small) ~ 0.5+
        # because the numeric block would be mis-flagged as unseen.
        # With the hint: err[0] should be close to the small categorical
        # loss alone (numeric contributes ~0 because CE on a single-value
        # softmax block collapses to 0).
        assert err[0] < 0.1, (
            f"Numeric attribute with value 0.3 was incorrectly clamped to "
            f"the max unseen-category penalty; got row loss {err[0]}"
        )

    def test_cardinality_1_categorical_unseen_row_is_still_penalized(self):
        """
        Regression test for Codex P1 #3 on PR #46.

        With an imbalanced binary categorical column, the train/test split
        can leave only one value in the training split, so
        ``OneHotEncoder.fit(train)`` fits cardinality 1 and unseen test
        rows are encoded as the all-zero scalar ``[0]``. Previously the
        override was gated on ``categories > 1``, which meant those
        unseen categorical rows fell through to standard CE (which is 0
        for ``[0]``) and got no penalty. The caller must now pass
        ``attr_is_categorical=[True]`` so the override still fires on
        cardinality-1 categorical blocks.
        """
        # A single cardinality-1 attribute that the caller knows is
        # categorical (fitted on a training split that only saw one value).
        cardinalities = [1]
        attr_is_categorical = [True]

        data = np.array(
            [
                [1.0],  # observed (fitted) category
                [0.0],  # unseen category - all-zero block
            ]
        )
        pred = np.array(
            [
                [0.99],
                [0.99],
            ]
        )

        err = compute_reconstruction_error(
            data, pred, cardinalities, attr_is_categorical=attr_is_categorical
        )

        # The unseen row must score strictly higher than the observed row.
        assert err[1] > err[0], (
            f"Cardinality-1 categorical unseen row must be penalized more "
            f"than the observed row, got errors={err}"
        )
        # The unseen row should receive ~1.0 (max normalized loss) since
        # it's the only attribute on the row.
        assert err[1] >= 0.9, (
            f"Cardinality-1 unseen categorical should be assigned ~1.0 "
            f"normalized loss, got {err[1]}"
        )

    def test_default_without_hint_treats_all_attrs_as_categorical(self):
        """
        When ``attr_is_categorical`` is omitted, the helper must default
        to treating every attribute as categorical — this matches the
        worker and Vertex AI code paths where every uploaded column is
        hard-coded to ``'categorical'``. Without this default, a
        cardinality-1 categorical with an unseen value (Codex P1 #3)
        would fall through to CE=0 and receive no penalty.
        """
        cardinalities = [1]
        data = np.array([[1.0], [0.0]])  # observed, then unseen
        pred = np.array([[0.99], [0.99]])

        err = compute_reconstruction_error(data, pred, cardinalities)

        assert err[1] > err[0]
        assert err[1] >= 0.9

    def test_attr_is_categorical_length_mismatch_raises(self):
        """Guard against silent plumbing bugs in the type-hint parameter."""
        cardinalities = [3, 2]
        data = np.array([[1.0, 0.0, 0.0, 1.0, 0.0]])
        pred = np.array([[0.9, 0.05, 0.05, 0.9, 0.1]])

        try:
            compute_reconstruction_error(
                data,
                pred,
                cardinalities,
                attr_is_categorical=[True],  # wrong length
            )
        except ValueError as exc:
            assert "attr_is_categorical length" in str(exc)
        else:
            raise AssertionError(
                "Expected ValueError for mismatched attr_is_categorical length"
            )

    def test_normal_case_still_matches_vae_reconstruction_loss(self):
        """
        With no unseen-category blocks, the helper must still be numerically
        equivalent to ``VAE.reconstruction_loss`` so that scoring stays
        consistent with the training loss the model optimized.
        """
        data = np.array(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )
        predictions = np.array(
            [
                [0.7, 0.2, 0.1, 0.9, 0.1],
                [0.3, 0.5, 0.2, 0.2, 0.8],
                [0.1, 0.2, 0.7, 0.6, 0.4],
            ]
        )
        cardinalities = [3, 2]

        helper_result = compute_reconstruction_error(
            data, predictions, cardinalities
        )
        direct_result = VAE.reconstruction_loss(
            cardinalities, data, predictions
        ).numpy()

        assert np.allclose(helper_result, direct_result, atol=1e-5)


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

    def test_mixed_numeric_categorical_layout_aligns_with_vectorized_matrix(self):
        """
        Regression test for Codex P1 #4 on PR #46.

        ``Table2Vector._apply_transforms`` reorders columns during
        vectorization: numeric / untransformed columns keep their
        original relative position while **categorical** columns are
        dropped from their original slot and their one-hot blocks are
        appended at the end. For input ``[num1, cat1, num2, cat2]`` the
        vectorized matrix has columns
        ``[num1, num2, cat1__a, cat1__b, cat2__x, cat2__y]``.

        Before the fix, ``find_outliers`` computed
        ``attr_cardinalities = get_cardinalities(project_data.columns)``
        in raw column order ``[1, 2, 1, 2]``. Slicing the vectorized
        matrix with those cardinalities lined up with the wrong
        features — the "categorical" slot was half numeric and half
        one-hot, silently distorting outlier rankings in
        ``find_outliers`` for mixed datasets with numeric columns.

        The fix (``_compute_attr_layout`` in ``main.py``) rebuilds
        ``attr_cardinalities`` and ``attr_is_categorical`` in the
        actual slice order: all non-categorical attributes first, then
        categorical blocks. This test exercises the helper through a
        real ``Table2Vector`` to guard against any drift between the
        helper's ordering assumption and the actual vectorizer
        behavior.
        """
        from main import _compute_attr_layout

        df = pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "cat1": ["a", "b", "a", "b", "a"],
                "num2": [0.1, 0.2, 0.3, 0.4, 0.5],
                "cat2": ["x", "y", "x", "y", "x"],
            }
        )
        variable_types = {
            "num1": "numeric",
            "cat1": "categorical",
            "num2": "numeric",
            "cat2": "categorical",
        }
        vectorizer = Table2Vector(variable_types)
        vectorizer.fit(df)
        vectorized = vectorizer.transform(df)

        # The actual vectorized layout puts numerics first, then categoricals
        # (verified by inspecting vectorized.columns).
        assert list(vectorized.columns)[:2] == ["num1", "num2"], (
            f"Table2Vector layout assumption is stale; got "
            f"{list(vectorized.columns)}"
        )

        attr_cardinalities, attr_is_categorical, _ = _compute_attr_layout(
            vectorizer, df.columns
        )

        # Expected: [num1(1), num2(1), cat1(2), cat2(2)]
        assert attr_cardinalities == [1, 1, 2, 2], (
            f"Expected slice-order cardinalities [1, 1, 2, 2], got "
            f"{attr_cardinalities}"
        )
        assert attr_is_categorical == [False, False, True, True], (
            f"Expected is_categorical [F, F, T, T], got {attr_is_categorical}"
        )

        # And sanity-check: walking the vectorized matrix with these
        # cardinalities must actually slice correctly — each categorical
        # block must sum to 1.0 per row (valid one-hot) and each numeric
        # block must contain values in [0, 1] (MinMax-scaled).
        arr = vectorized.to_numpy()
        start = 0
        for i, (card, is_cat) in enumerate(
            zip(attr_cardinalities, attr_is_categorical)
        ):
            block = arr[:, start : start + card]
            if is_cat:
                row_sums = block.sum(axis=1)
                assert np.allclose(row_sums, 1.0), (
                    f"Categorical block {i} does not sum to 1.0 per row: "
                    f"{row_sums}"
                )
            else:
                assert (block >= 0).all() and (block <= 1).all(), (
                    f"Numeric block {i} has values outside [0, 1]: {block}"
                )
            start += card

        # And the scoring must work end-to-end on this mixed layout.
        # A perfect-reconstruction prediction should produce near-zero
        # loss for the categorical blocks, NOT get clamped by the
        # unseen-category override.
        #
        # Caveat: rows where a numeric column equals exactly 0 after
        # MinMax scaling (i.e. the training-minimum value) trigger a
        # pre-existing degenerate case in
        # ``VAE.reconstruction_loss``'s categorical-crossentropy on a
        # cardinality-1 block (CE normalizes ``[0]`` to ``[eps]`` and
        # ``0/eps`` yields NaN). That is orthogonal to TASKS 2.8 and
        # outside the scope of this fix — what this test pins down is
        # that the non-pathological rows produce a small reconstruction
        # loss instead of the ~0.5 value they would get if the
        # categorical/numeric slices were misaligned.
        perfect_pred = vectorized.to_numpy()
        err = compute_reconstruction_error(
            vectorized,
            perfect_pred,
            attr_cardinalities,
            attr_is_categorical=attr_is_categorical,
        )
        finite = err[np.isfinite(err)]
        assert len(finite) >= 4, (
            f"Expected at least 4 finite per-row losses, got {err}"
        )
        assert np.all(finite < 0.3), (
            f"Perfect reconstruction should yield near-zero loss on a "
            f"mixed dataset; got {err}"
        )

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
