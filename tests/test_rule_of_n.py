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
- ``apply_rule_of_n=False`` disables the filter entirely so scoring
  paths with a saved vectorizer do not silently drop high-cardinality
  columns that the vectorizer was trained on (Codex P1 PR#49).
"""

import io
import os
import unittest
from unittest import mock

import pandas as pd

from dataset.loader import DEFAULT_MAX_UNIQUE_VALUES, DataLoader
from features.transform import Table2Vector
from main import (
    _clean_for_saved_vectorizer,
    prepare_for_categorical,
    prepare_for_model,
    prepare_for_training,
)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _make_loader(max_unique_values=None, apply_rule_of_n=True) -> DataLoader:
    return DataLoader(
        drop_columns=[],
        rename_columns={},
        columns_of_interest=[],
        max_unique_values=max_unique_values,
        apply_rule_of_n=apply_rule_of_n,
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


class TestApplyRuleOfNFlag(unittest.TestCase):
    """``apply_rule_of_n=False`` bypasses the Rule-of-N filter entirely
    so scoring paths with a saved vectorizer can let the vectorizer be
    the authoritative source of truth on kept columns (Codex P1 PR#49).
    """

    def test_default_applies_filter(self):
        loader = _make_loader()
        self.assertTrue(loader.apply_rule_of_n)

    def test_explicit_false_disables_filter(self):
        loader = _make_loader(apply_rule_of_n=False)
        self.assertFalse(loader.apply_rule_of_n)

    def test_filter_disabled_keeps_high_cardinality_columns(self):
        """With ``apply_rule_of_n=False``, all columns survive even if
        their cardinality exceeds ``max_unique_values``."""
        loader = _make_loader(
            max_unique_values=9, apply_rule_of_n=False
        )
        df = _sample_df_with_varied_cardinalities()
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        # Every column survives regardless of its cardinality.
        for col in ["card_2", "card_5", "card_9", "card_12", "card_15"]:
            self.assertIn(col, clean_df.columns)
        self.assertEqual(meta["ignored_columns"], [])

    def test_filter_disabled_keeps_constant_columns(self):
        """``apply_rule_of_n=False`` also skips the lower bound —
        single-value columns survive so that the vectorizer can align
        them to trained columns."""
        loader = _make_loader(apply_rule_of_n=False)
        df = pd.DataFrame({
            "constant": ["yes"] * 20,
            "varied": ["a", "b", "a", "b"] * 5,
        })
        clean_df, meta = loader.load_uploaded_csv(_csv_bytes(df))

        self.assertIn("constant", clean_df.columns)
        self.assertIn("varied", clean_df.columns)
        self.assertEqual(meta["ignored_columns"], [])

    def test_per_call_apply_rule_of_n_override(self):
        """``prepare_original_dataset`` accepts a per-call
        ``apply_rule_of_n`` override that beats the instance default."""
        loader = _make_loader(apply_rule_of_n=True)
        df = _sample_df_with_varied_cardinalities()

        # Per-call override disables the filter.
        clean_df, meta = loader.prepare_original_dataset(
            df.copy(),
            replacements={},
            apply_rule_of_n=False,
        )
        for col in ["card_2", "card_5", "card_9", "card_12", "card_15"]:
            self.assertIn(col, clean_df.columns)
        self.assertEqual(meta["ignored_columns"], [])

        # Instance default is unchanged — a follow-up call without the
        # override re-applies the default threshold.
        self.assertTrue(loader.apply_rule_of_n)
        clean_df2, meta2 = loader.prepare_original_dataset(
            df.copy(), replacements={}
        )
        self.assertNotIn("card_12", clean_df2.columns)
        self.assertNotIn("card_15", clean_df2.columns)

    def test_filter_disabled_still_casts_to_string(self):
        """``apply_rule_of_n=False`` still converts columns to
        strings so downstream one-hot encoding receives consistent
        dtypes."""
        loader = _make_loader(apply_rule_of_n=False)
        df = pd.DataFrame({
            "cat": ["a", "b", "a", "b"] * 5,
            "many": [f"v{i}" for i in range(20)],
        })
        clean_df, _ = loader.load_uploaded_csv(_csv_bytes(df))

        # Accept either object dtype (pandas <= 2.x) or StringDtype
        # (pandas 3.x default). What matters is that the values are
        # Python ``str`` instances that Table2Vector will accept.
        for col in clean_df.columns:
            self.assertTrue(
                all(isinstance(v, str) for v in clean_df[col]),
                f"column {col!r} contains non-string values",
            )


class TestVertexContainerArgPropagation(unittest.TestCase):
    """The Rule-of-N threshold must flow from the worker dispatcher
    into the Vertex AI training container via a CLI arg, because
    ``CustomContainerTrainingJob.run()`` does not forward the
    dispatcher's env vars to the container (Codex P1 PR#49).
    """

    def test_build_vertex_args_includes_max_unique_values(self):
        from worker import _build_vertex_training_args

        args = _build_vertex_training_args(
            "job-123",
            "my-bucket",
            "uploads/job-123/data.csv",
            max_unique_values=15,
        )
        self.assertIn("--job-id=job-123", args)
        self.assertIn("--bucket-name=my-bucket", args)
        self.assertIn("--file-path=uploads/job-123/data.csv", args)
        self.assertIn("--max-unique-values=15", args)

    def test_build_vertex_args_omits_max_unique_values_when_none(self):
        """Backwards compat: when ``max_unique_values=None`` (not
        supplied), the arg is omitted so a Vertex container running an
        older ``train/task.py`` that doesn't recognise the flag still
        accepts the args."""
        from worker import _build_vertex_training_args

        args = _build_vertex_training_args(
            "job-123",
            "my-bucket",
            "uploads/job-123/data.csv",
            max_unique_values=None,
        )
        self.assertEqual(
            args,
            [
                "--job-id=job-123",
                "--bucket-name=my-bucket",
                "--file-path=uploads/job-123/data.csv",
            ],
        )

    def test_build_vertex_args_coerces_to_int_string(self):
        """Any numeric value is emitted as a base-10 int string so
        ``argparse`` on the container side parses it cleanly."""
        from worker import _build_vertex_training_args

        args = _build_vertex_training_args(
            "j",
            "b",
            "f",
            max_unique_values=9.0,  # e.g. float from a YAML config
        )
        self.assertIn("--max-unique-values=9", args)

    def test_worker_forwards_env_var_to_vertex(self):
        """End-to-end: when the operator sets MAX_UNIQUE_VALUES=20 on
        the worker, the helper that builds the Vertex CLI args picks
        it up via ``_resolve_explicit_max_unique_values`` (the same
        function ``process_upload_vertex`` now uses) so both modes
        stay in sync."""
        from worker import (
            _build_vertex_training_args,
            _resolve_explicit_max_unique_values,
        )

        with mock.patch.dict(os.environ, {"MAX_UNIQUE_VALUES": "20"}):
            args = _build_vertex_training_args(
                "j",
                "b",
                "f",
                max_unique_values=_resolve_explicit_max_unique_values(),
            )
        self.assertIn("--max-unique-values=20", args)

    def test_worker_omits_flag_when_env_var_unset(self):
        """Rolling-deploy safety: when the operator has NOT set
        ``MAX_UNIQUE_VALUES``, the Vertex dispatcher must omit the
        ``--max-unique-values`` flag entirely so a container running an
        older ``train/task.py`` (which doesn't recognise the flag)
        still parses the arg list cleanly (Codex P1 PR#49).

        This is the scenario the first fix attempt missed:
        ``_resolve_max_unique_values()`` always returns 9 when the env
        var is unset, which would always append the flag and defeat
        the backwards-compat path in ``_build_vertex_training_args``.
        ``_resolve_explicit_max_unique_values()`` returns ``None`` in
        that case, so the flag is correctly omitted.
        """
        from worker import (
            _build_vertex_training_args,
            _resolve_explicit_max_unique_values,
        )

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAX_UNIQUE_VALUES", None)
            resolved = _resolve_explicit_max_unique_values()
            args = _build_vertex_training_args(
                "j",
                "b",
                "f",
                max_unique_values=resolved,
            )
        self.assertIsNone(resolved)
        self.assertNotIn(
            "--max-unique-values",
            " ".join(args),
            f"expected flag to be omitted; got {args}",
        )
        self.assertEqual(
            args,
            [
                "--job-id=j",
                "--bucket-name=b",
                "--file-path=f",
            ],
        )

    def test_worker_omits_flag_when_env_var_invalid(self):
        """Same as ``test_worker_omits_flag_when_env_var_unset`` for
        invalid values (non-integer, below 2): the resolver returns
        ``None`` and the dispatcher omits the flag, preserving rolling
        deploy compatibility."""
        from worker import _resolve_explicit_max_unique_values

        for bad in ("", "abc", "9.5", "0", "1", "-3"):
            with mock.patch.dict(os.environ, {"MAX_UNIQUE_VALUES": bad}):
                self.assertIsNone(
                    _resolve_explicit_max_unique_values(),
                    f"bad value {bad!r} should resolve to None",
                )

    def test_explicit_resolver_returns_valid_int(self):
        """Happy path: valid env var → parsed int, not ``None``."""
        from worker import _resolve_explicit_max_unique_values

        with mock.patch.dict(os.environ, {"MAX_UNIQUE_VALUES": "15"}):
            self.assertEqual(_resolve_explicit_max_unique_values(), 15)

    def test_process_upload_vertex_uses_explicit_resolver(self):
        """Drift guard: read ``worker.py`` as source and verify
        ``process_upload_vertex`` references
        ``_resolve_explicit_max_unique_values`` (not
        ``_resolve_max_unique_values``) when building Vertex args.
        If a future change regresses back to the non-explicit
        resolver, the flag will always be appended and rolling
        deploys break — this test catches that before CI / prod."""
        import pathlib

        source_path = pathlib.Path(__file__).resolve().parents[1] / "worker.py"
        source = source_path.read_text()

        # Locate the process_upload_vertex function body and check
        # which resolver it passes into _build_vertex_training_args.
        start = source.index("def process_upload_vertex(")
        # Rough body end: the next top-level ``def `` declaration or
        # end-of-file — good enough for a drift guard.
        next_def = source.find("\ndef ", start + 1)
        body = source[start:next_def] if next_def != -1 else source[start:]

        self.assertIn("_build_vertex_training_args", body)
        self.assertIn("_resolve_explicit_max_unique_values", body)
        # Make sure the non-explicit resolver is NOT what we pass into
        # _build_vertex_training_args. It may still appear in other
        # contexts (e.g. logging), so look at the
        # ``max_unique_values=`` keyword argument line specifically.
        for line in body.splitlines():
            if "max_unique_values=" in line and "_resolve_max_unique_values" in line:
                self.fail(
                    f"process_upload_vertex uses the non-explicit "
                    f"resolver, which always returns an int and "
                    f"defeats the rolling-deploy compat path: "
                    f"{line.strip()!r}"
                )

    def test_train_task_cli_accepts_max_unique_values(self):
        """``train/task.py``'s argparse must accept the new flag so
        the dispatcher's CLI arg lands somewhere on the container
        side."""
        import argparse

        # Mirror the argparse config in train/task.py's __main__.
        parser = argparse.ArgumentParser()
        parser.add_argument("--job-id", type=str, required=True)
        parser.add_argument("--bucket-name", type=str, required=True)
        parser.add_argument("--file-path", type=str, required=True)
        parser.add_argument(
            "--max-unique-values",
            type=int,
            default=None,
            dest="max_unique_values",
        )
        parsed = parser.parse_args([
            "--job-id=j",
            "--bucket-name=b",
            "--file-path=f",
            "--max-unique-values=15",
        ])
        self.assertEqual(parsed.max_unique_values, 15)

        # And it's optional, defaulting to None.
        parsed2 = parser.parse_args([
            "--job-id=j",
            "--bucket-name=b",
            "--file-path=f",
        ])
        self.assertIsNone(parsed2.max_unique_values)

    def test_train_task_module_registers_same_cli_contract(self):
        """Drift guard: if someone renames/removes ``--max-unique-values``
        on ``train/task.py``, this test fails so the worker→container
        contract stays aligned."""
        import pathlib
        source_path = pathlib.Path(__file__).resolve().parents[1] / "train" / "task.py"
        source = source_path.read_text()
        self.assertIn("--max-unique-values", source)
        self.assertIn("max_unique_values", source)

    def test_train_task_train_and_predict_accepts_max_unique_values(self):
        """``train_and_predict`` must accept ``max_unique_values`` as a
        keyword argument so the CLI layer can forward the parsed flag
        to it without the function ignoring or error-ing."""
        import inspect
        from train.task import train_and_predict

        sig = inspect.signature(train_and_predict)
        self.assertIn("max_unique_values", sig.parameters)
        self.assertEqual(
            sig.parameters["max_unique_values"].default, None
        )


class TestSavedVectorizerScoringRegression(unittest.TestCase):
    """End-to-end regression for Codex P1 PR#49: a model trained with
    ``max_unique_values=15`` must continue to score correctly on the
    same data even when the scoring-time CLI default is the historical
    9. The bug was that the default threshold silently dropped the
    high-cardinality columns at load time, and
    ``_clean_for_saved_vectorizer`` re-inserted them as constant
    ``"missing"`` values, corrupting the one-hot matrix the model sees.

    With the fix, scoring paths detect the saved vectorizer and
    disable the Rule-of-N filter on the DataLoader, so the kept
    columns match the vectorizer's ``var_types`` and the transformed
    matrix is identical to the one produced at training time.
    """

    def test_training_threshold_preserved_at_scoring_time(self):
        # Training data with a column whose cardinality exceeds the
        # historical Rule-of-9 default.
        df = pd.DataFrame({
            "card_4": ["a", "b", "c", "d"] * 15,
            "card_12": [f"v{i % 12}" for i in range(60)],
        })

        # --- Training path: user explicitly widens the threshold to 15
        train_loader = _make_loader(max_unique_values=15)
        clean_train, meta_train = train_loader.load_uploaded_csv(
            _csv_bytes(df)
        )
        self.assertIn("card_4", clean_train.columns)
        self.assertIn("card_12", clean_train.columns)

        variable_types = meta_train["variable_types"]
        vectorizer = Table2Vector(variable_types)
        vectorizer.fit(clean_train)

        trained_cols = list(
            vectorizer.var_types.get("categorical", [])
        ) + list(vectorizer.var_types.get("numeric", []))
        self.assertIn("card_12", trained_cols)

        expected_transformed = vectorizer.transform(clean_train).astype(
            "float32"
        )

        # --- Scoring path: no `--max_unique_values` passed, so the
        # CLI default of 9 applies. Before the fix this silently drops
        # ``card_12``. With the fix, we detect the saved vectorizer and
        # construct the DataLoader with ``apply_rule_of_n=False``.
        score_loader = _make_loader(
            max_unique_values=DEFAULT_MAX_UNIQUE_VALUES,
            apply_rule_of_n=False,  # what evaluate/find_outliers/generate do
        )
        clean_score, _ = score_loader.load_uploaded_csv(_csv_bytes(df))

        # Critical assertion: ``card_12`` survives load_data instead of
        # being dropped and then silently backfilled.
        self.assertIn("card_12", clean_score.columns)
        self.assertIn("card_4", clean_score.columns)

        aligned = _clean_for_saved_vectorizer(clean_score, vectorizer)
        scored = vectorizer.transform(aligned).astype("float32")

        # The scoring-time transformed matrix must match the
        # training-time matrix exactly (same shape, same values).
        self.assertEqual(scored.shape, expected_transformed.shape)
        self.assertTrue(
            (scored.values == expected_transformed.values).all(),
            "scoring-time transform differs from training-time transform",
        )

    def test_regression_default_threshold_would_corrupt_inputs(self):
        """Proof that the old code path *was* broken: if the DataLoader
        still applies the default Rule-of-N=9 filter, the high-card
        column is dropped and backfilled as constant ``"missing"``,
        producing a transformed matrix that differs from training. This
        test pins the faulty behaviour so a future regression that
        silently re-enables the filter will fail loudly.
        """
        df = pd.DataFrame({
            "card_4": ["a", "b", "c", "d"] * 15,
            "card_12": [f"v{i % 12}" for i in range(60)],
        })

        train_loader = _make_loader(max_unique_values=15)
        clean_train, meta_train = train_loader.load_uploaded_csv(
            _csv_bytes(df)
        )
        variable_types = meta_train["variable_types"]
        vectorizer = Table2Vector(variable_types)
        vectorizer.fit(clean_train)
        expected = vectorizer.transform(clean_train).astype("float32")

        # Simulate the broken pre-fix path: scoring loader uses the
        # default threshold (9) *and* apply_rule_of_n is left True.
        broken_loader = _make_loader(
            max_unique_values=DEFAULT_MAX_UNIQUE_VALUES,
            apply_rule_of_n=True,
        )
        clean_broken, meta_broken = broken_loader.load_uploaded_csv(
            _csv_bytes(df)
        )
        # card_12 is dropped by Rule-of-9.
        self.assertNotIn("card_12", clean_broken.columns)

        aligned_broken = _clean_for_saved_vectorizer(
            clean_broken, vectorizer
        )
        scored_broken = vectorizer.transform(aligned_broken).astype(
            "float32"
        )

        # Shapes match (alignment adds the column back), but values
        # diverge because card_12 became a constant "missing" column.
        self.assertEqual(scored_broken.shape, expected.shape)
        self.assertFalse(
            (scored_broken.values == expected.values).all(),
            "broken path happens to match training transform — the "
            "regression guard below is therefore meaningless",
        )


if __name__ == "__main__":
    unittest.main()
