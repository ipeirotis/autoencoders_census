"""
Tests for structured error reporting (TASKS.md 2.3).

Covers the two new pieces introduced in task 2.3:

1. :class:`worker.UploadValidationError` — a ``ValueError`` subclass that
   carries a stable :class:`ErrorCode` so callers can distinguish validation
   failures by category (empty file, too large, encoding, too few rows, ...)
   instead of parsing error message strings.

2. :func:`worker.mark_job_error` — the helper that writes a structured
   error payload ``{error, errorCode, errorType}`` to Firestore and
   tolerates invalid transitions gracefully.

Together these replace the old "write ``str(e)`` to Firestore" path that
leaked raw Python error messages to the frontend. The frontend consumes
the new fields via ``frontend/client/utils/jobErrors.ts``
(``resolveJobError``).
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from worker import (
    ErrorCode,
    ErrorType,
    JobStatus,
    UploadValidationError,
    mark_job_error,
    validate_csv,
)


class TestUploadValidationError:
    """UploadValidationError behavior + ValueError compatibility."""

    def test_is_a_value_error(self):
        """Existing `except ValueError:` / `pytest.raises(ValueError)` call
        sites must keep working without modification."""
        err = UploadValidationError("bad", error_code=ErrorCode.CSV_EMPTY)
        assert isinstance(err, ValueError)

    def test_carries_error_code(self):
        err = UploadValidationError(
            "bad", error_code=ErrorCode.CSV_TOO_LARGE
        )
        assert err.error_code == ErrorCode.CSV_TOO_LARGE

    def test_default_error_type_is_validation(self):
        err = UploadValidationError("bad", error_code=ErrorCode.CSV_EMPTY)
        assert err.error_type == ErrorType.VALIDATION

    def test_error_type_can_be_overridden(self):
        err = UploadValidationError(
            "no usable columns",
            error_code=ErrorCode.NO_USABLE_COLUMNS,
            error_type=ErrorType.PROCESSING,
        )
        assert err.error_type == ErrorType.PROCESSING


class TestValidateCsvErrorCodes:
    """validate_csv raises UploadValidationError with the right codes."""

    def test_empty_file_raises_csv_empty(self):
        with pytest.raises(UploadValidationError) as excinfo:
            validate_csv(b"")
        assert excinfo.value.error_code == ErrorCode.CSV_EMPTY
        assert excinfo.value.error_type == ErrorType.VALIDATION

    def test_too_large_raises_csv_too_large(self):
        # 2 MB > 1 MB limit
        csv_content = b"a,b,c\n" + b"1,2,3\n" * 400000
        assert len(csv_content) > 1 * 1024 * 1024

        with pytest.raises(UploadValidationError) as excinfo:
            validate_csv(csv_content, max_size_mb=1)
        assert excinfo.value.error_code == ErrorCode.CSV_TOO_LARGE

    def test_too_few_rows_raises_csv_too_few_rows(self):
        csv_content = b"name,age\nAlice,25\nBob,30\n"  # 2 data rows
        with pytest.raises(UploadValidationError) as excinfo:
            validate_csv(csv_content)
        assert excinfo.value.error_code == ErrorCode.CSV_TOO_FEW_ROWS

    def test_too_few_columns_raises_csv_too_few_columns(self):
        csv_content = b"only_col\n" + b"value\n" * 15
        with pytest.raises(UploadValidationError) as excinfo:
            validate_csv(csv_content)
        assert excinfo.value.error_code == ErrorCode.CSV_TOO_FEW_COLUMNS

    def test_header_only_raises_too_few_rows(self):
        with pytest.raises(UploadValidationError) as excinfo:
            validate_csv(b"col1,col2,col3\n")
        assert excinfo.value.error_code == ErrorCode.CSV_TOO_FEW_ROWS

    def test_error_is_catchable_as_value_error(self):
        """Sanity check: existing tests in test_csv_validation.py rely on
        `pytest.raises(ValueError, match=...)` and must keep working."""
        with pytest.raises(ValueError, match="CSV file is empty"):
            validate_csv(b"")


def _make_mock_job_ref(current_status: str):
    """Build a job_ref that reports a specific current_status in the Firestore
    snapshot. Used to simulate a real Firestore transaction enough for
    update_job_status / mark_job_error to run end-to-end."""
    mock_snapshot = Mock()
    mock_snapshot.exists = True
    mock_snapshot.get.return_value = current_status

    mock_job_ref = Mock()
    mock_job_ref.id = "test-job-id"
    mock_job_ref.get.return_value = mock_snapshot
    return mock_job_ref


class TestMarkJobError:
    """mark_job_error writes structured error state to Firestore."""

    def _make_transaction(self):
        mock_transaction = MagicMock()
        mock_transaction._read_only = False
        mock_transaction._id = b"test-transaction-id"
        mock_transaction._max_attempts = 5
        mock_transaction.in_progress = True
        mock_transaction._write_pbs = []
        mock_transaction._clean_up = Mock()
        return mock_transaction

    def test_writes_error_code_and_type_to_firestore(self):
        """Happy path: mark_job_error transitions the doc to ERROR and
        writes error/errorCode/errorType fields."""
        import worker

        mock_transaction = self._make_transaction()
        mock_job_ref = _make_mock_job_ref(current_status="processing")

        with patch.object(worker.db, "transaction", return_value=mock_transaction):
            mark_job_error(
                mock_job_ref,
                "test-job-id",
                "Could not load CSV",
                error_code=ErrorCode.LOAD_FAILURE,
                error_type=ErrorType.PROCESSING,
            )

        # One transaction.update call was made with the classified payload
        mock_transaction.update.assert_called_once()
        call_args = mock_transaction.update.call_args
        update_payload = call_args[0][1]
        assert update_payload["status"] == JobStatus.ERROR
        assert update_payload["error"] == "Could not load CSV"
        assert update_payload["errorCode"] == "load_failure"
        assert update_payload["errorType"] == "processing"

    def test_error_code_enum_is_serialized_as_string(self):
        """Firestore stores strings, not Python enum objects."""
        import worker

        mock_transaction = self._make_transaction()
        mock_job_ref = _make_mock_job_ref(current_status="queued")

        with patch.object(worker.db, "transaction", return_value=mock_transaction):
            mark_job_error(
                mock_job_ref,
                "test-job-id",
                "Empty CSV",
                error_code=ErrorCode.CSV_EMPTY,
                error_type=ErrorType.VALIDATION,
            )

        call_args = mock_transaction.update.call_args
        update_payload = call_args[0][1]
        assert update_payload["errorCode"] == "csv_empty"
        assert isinstance(update_payload["errorCode"], str)
        assert update_payload["errorType"] == "validation"
        assert isinstance(update_payload["errorType"], str)

    def test_accepts_plain_string_error_code(self):
        """Plain strings are accepted for both error_code and error_type
        (used when the code comes from a dynamic source)."""
        import worker

        mock_transaction = self._make_transaction()
        mock_job_ref = _make_mock_job_ref(current_status="processing")

        with patch.object(worker.db, "transaction", return_value=mock_transaction):
            mark_job_error(
                mock_job_ref,
                "test-job-id",
                "custom failure",
                error_code="custom_code",
                error_type="custom_type",
            )

        call_args = mock_transaction.update.call_args
        update_payload = call_args[0][1]
        assert update_payload["errorCode"] == "custom_code"
        assert update_payload["errorType"] == "custom_type"

    def test_swallows_invalid_transition_silently(self):
        """If the job is already in a terminal state the transition check
        raises ValueError; mark_job_error must log and return instead of
        propagating the failure (the caller is already in an exception
        handler)."""
        import worker

        mock_transaction = self._make_transaction()
        # Simulate terminal state (CANCELED) so queued/processing -> ERROR
        # is not the issue; ERROR -> ERROR is rejected by the state machine.
        mock_job_ref = _make_mock_job_ref(current_status="canceled")

        with patch.object(worker.db, "transaction", return_value=mock_transaction):
            # Should NOT raise
            mark_job_error(
                mock_job_ref,
                "test-job-id",
                "something bad",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
            )

        # transaction.update was never called because is_valid_transition
        # rejected the transition before reaching it
        mock_transaction.update.assert_not_called()


class TestProcessUploadLocalStageClassification:
    """process_upload_local reports stage-specific error codes.

    Regression for Codex P2 review r3067501311: scoring-stage failures
    (keras_model.predict, compute_per_column_contributions, the final
    Firestore COMPLETE write) used to fall through to the
    ``except Exception`` catch-all and get written as INTERNAL_ERROR,
    losing the stage information even though SCORING_FAILURE already
    existed in both the worker ErrorCode enum and the frontend's
    resolveJobError mapping.

    These tests drive process_upload_local with just enough mocking to
    reach the specific stage boundary we care about and assert that the
    Firestore payload carries the expected {errorCode, errorType}.
    """

    def _run_with_scoring_failure(self, predict_side_effect):
        """Drive process_upload_local end-to-end up to the scoring stage
        and raise ``predict_side_effect`` inside keras_model.predict.
        Returns the final update_job_status call payload (the one where
        the ERROR transition is written via mark_job_error)."""
        import pandas as pd
        import worker

        # A tiny synthetic dataset with 2 low-cardinality categorical
        # columns and 15 rows. validate_csv accepts ≥10 rows and ≥2 cols.
        df = pd.DataFrame({
            "cat_a": (["x", "y", "z"] * 5),
            "cat_b": (["p", "q", "p", "q", "p"] * 3),
        })
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        captured_error_updates = []

        # A mock transaction that records every transaction.update() call
        # so we can inspect what mark_job_error wrote.
        def make_transaction():
            txn = MagicMock()
            txn._read_only = False
            txn._id = b"test-txn"
            txn._max_attempts = 5
            txn.in_progress = True
            txn._write_pbs = []
            txn._clean_up = Mock()

            def _update(ref, payload):
                if payload.get("status") == worker.JobStatus.ERROR:
                    captured_error_updates.append(payload)
                # Mirror the state transition so subsequent reads see the
                # new status.
                ref._current_status = payload.get("status")

            txn.update.side_effect = _update
            return txn

        # Build a job_ref whose .get(...) snapshot always reflects the
        # last status written by any transaction.
        class _FakeJobRef:
            id = "scoring-failure-job"
            _current_status = "queued"

            def get(self, transaction=None):
                snap = Mock()
                snap.exists = True
                snap.get = Mock(return_value=self._current_status)
                return snap

            def update(self, payload):
                self._current_status = payload.get(
                    "status", self._current_status
                )

        job_ref = _FakeJobRef()

        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value = job_ref
        mock_db.transaction.side_effect = make_transaction

        # Mock GCS download so the CSV bytes come from our in-memory df.
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = csv_bytes
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client = MagicMock()
        mock_storage_client.bucket.return_value = mock_bucket

        # Mock keras model: .fit() succeeds, .predict() raises the
        # caller-supplied exception to simulate a scoring-stage failure.
        mock_keras_model = MagicMock()
        mock_keras_model.fit = MagicMock()
        mock_keras_model.predict.side_effect = predict_side_effect
        mock_ae_wrapper = MagicMock()
        mock_ae_wrapper.build_autoencoder.return_value = mock_keras_model

        mock_autoencoder_module = MagicMock()
        mock_autoencoder_module.AutoencoderModel = MagicMock(
            return_value=mock_ae_wrapper
        )

        mock_outliers_module = MagicMock()
        mock_outliers_module.compute_per_column_contributions = MagicMock(
            return_value=[]
        )

        mock_tf = MagicMock()
        # CancellationCallback inherits from this class; provide a no-op
        # base so the subclass can be instantiated.
        mock_tf.keras.callbacks.Callback = type("Callback", (), {})

        import sys
        saved_modules = {}
        for modname in (
            "model.autoencoder",
            "evaluate.outliers",
            "tensorflow",
        ):
            if modname in sys.modules:
                saved_modules[modname] = sys.modules[modname]
        sys.modules["model.autoencoder"] = mock_autoencoder_module
        sys.modules["evaluate.outliers"] = mock_outliers_module
        sys.modules["tensorflow"] = mock_tf

        try:
            with patch.object(worker, "db", mock_db), \
                 patch.object(worker.storage, "Client", return_value=mock_storage_client), \
                 patch.object(worker, "is_job_canceled", return_value=False):
                message = MagicMock()
                message.message_id = "msg-1"
                worker.process_upload_local(
                    "scoring-failure-job", "test-bucket", "test.csv", message
                )
        finally:
            for modname in ("model.autoencoder", "evaluate.outliers", "tensorflow"):
                if modname in saved_modules:
                    sys.modules[modname] = saved_modules[modname]
                else:
                    sys.modules.pop(modname, None)

        return captured_error_updates

    def test_scoring_exception_reported_as_scoring_failure(self):
        """A runtime error inside keras_model.predict should be written
        as SCORING_FAILURE / scoring, not INTERNAL_ERROR / internal."""
        captured = self._run_with_scoring_failure(
            predict_side_effect=RuntimeError("numeric instability")
        )

        assert len(captured) >= 1, (
            "expected at least one ERROR transition to be written to "
            "Firestore, got none"
        )
        last_error = captured[-1]
        assert last_error["errorCode"] == "scoring_failure", (
            f"expected errorCode='scoring_failure', got "
            f"{last_error.get('errorCode')!r}"
        )
        assert last_error["errorType"] == "scoring", (
            f"expected errorType='scoring', got "
            f"{last_error.get('errorType')!r}"
        )
        # The user-facing message should reference scoring, not "an
        # unexpected error occurred" (which is the INTERNAL_ERROR copy).
        assert "scoring" in last_error["error"].lower()


class TestErrorCodesCoverAllPaths:
    """Every pipeline-stage error code has both a worker enum value and a
    matching ErrorCode. Guard against typos drifting the two apart."""

    def test_all_expected_error_codes_exist(self):
        expected = {
            "csv_too_large",
            "csv_empty",
            "csv_encoding",
            "csv_parse",
            "csv_inconsistent_columns",
            "csv_too_few_rows",
            "csv_too_few_columns",
            "load_failure",
            "no_usable_columns",
            "training_failure",
            "scoring_failure",
            "internal_error",
        }
        actual = {code.value for code in ErrorCode}
        assert expected.issubset(actual), (
            f"Missing ErrorCode values: {expected - actual}"
        )

    def test_all_expected_error_types_exist(self):
        expected = {"validation", "processing", "training", "scoring", "internal"}
        actual = {etype.value for etype in ErrorType}
        assert expected == actual
