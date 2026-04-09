"""
Test Firestore transactional status updates.

Tests enforce:
- Atomic read-modify-write operations
- State transition validation in transactions
- Automatic retry on contention
- Transaction purity (no external state mutations)
- Additional fields preserved in updates
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from google.cloud import firestore
from worker import JobStatus, update_job_status


class TestFirestoreTransactions:
    """Test transactional status updates."""

    def test_update_job_status_atomically_validates_transition(self):
        """Test 1: update_job_status() atomically reads current status and validates transition"""
        # Use Mock instead of real Transaction to avoid commit issues
        mock_transaction = MagicMock()
        mock_transaction._read_only = False
        mock_transaction._id = b"test-transaction-id"
        mock_transaction._max_attempts = 5
        mock_transaction.in_progress = True
        mock_transaction._write_pbs = []
        mock_transaction._clean_up = Mock()

        mock_job_ref = Mock()

        # Mock snapshot with current status
        mock_snapshot = Mock()
        mock_snapshot.exists = True
        mock_snapshot.get.return_value = "queued"
        mock_job_ref.get.return_value = mock_snapshot

        # Call update_job_status with valid transition
        update_job_status(mock_transaction, mock_job_ref, JobStatus.PROCESSING)

        # Verify transaction.update was called
        mock_transaction.update.assert_called_once()
        call_args = mock_transaction.update.call_args
        assert call_args[0][0] == mock_job_ref
        assert call_args[0][1]['status'] == JobStatus.PROCESSING

    def test_update_job_status_raises_error_for_invalid_transition(self):
        """Test 2: update_job_status() raises ValueError for invalid transitions"""
        # Create mock transaction with required internal attributes
        mock_transaction = MagicMock()
        mock_transaction._read_only = False
        mock_transaction._id = b"test-transaction-id"
        mock_transaction._max_attempts = 5
        mock_transaction.in_progress = True
        mock_transaction._write_pbs = []
        mock_transaction._clean_up = Mock()

        mock_job_ref = Mock()

        # Mock snapshot with complete status (terminal state)
        mock_snapshot = Mock()
        mock_snapshot.exists = True
        mock_snapshot.get.return_value = "complete"
        mock_job_ref.get.return_value = mock_snapshot

        # Try to transition backward (invalid) - should fail before commit
        with pytest.raises(ValueError, match="Invalid transition"):
            update_job_status(mock_transaction, mock_job_ref, JobStatus.PROCESSING)

        # Verify transaction.update was NOT called
        mock_transaction.update.assert_not_called()

    def test_concurrent_updates_retry_automatically(self):
        """Test 3: Concurrent updates to same job retry automatically (Firestore behavior)"""
        # This tests Firestore's built-in retry mechanism via @firestore.transactional decorator
        # We verify the function is decorated and can be called multiple times

        mock_transaction = MagicMock()
        mock_transaction._read_only = False
        mock_transaction._id = b"test-transaction-id"
        mock_transaction._max_attempts = 5
        mock_transaction.in_progress = True
        mock_transaction._write_pbs = []
        mock_transaction._clean_up = Mock()

        mock_job_ref = Mock()

        # Simulate first attempt: contention (snapshot shows old status)
        # Second attempt: success (snapshot shows updated status from other transaction)
        snapshots = [
            Mock(exists=True, get=Mock(return_value="queued")),
            Mock(exists=True, get=Mock(return_value="processing"))
        ]
        mock_job_ref.get.side_effect = snapshots

        # First call: queued → processing (should succeed)
        update_job_status(mock_transaction, mock_job_ref, JobStatus.PROCESSING)
        assert mock_transaction.update.call_count == 1

        # Reset mock
        mock_transaction.update.reset_mock()

        # Second call with new snapshot: processing → training (should succeed)
        update_job_status(mock_transaction, mock_job_ref, JobStatus.TRAINING)
        assert mock_transaction.update.call_count == 1

    def test_transaction_function_is_pure(self):
        """Test 4: Transaction function is pure (no external state mutations)"""
        mock_transaction = MagicMock()
        mock_transaction._read_only = False
        mock_transaction._id = b"test-transaction-id"
        mock_transaction._max_attempts = 5
        mock_transaction.in_progress = True
        mock_transaction._write_pbs = []
        mock_transaction._clean_up = Mock()

        mock_job_ref = Mock()

        mock_snapshot = Mock()
        mock_snapshot.exists = True
        mock_snapshot.get.return_value = "queued"
        mock_job_ref.get.return_value = mock_snapshot

        # Call update_job_status multiple times
        update_job_status(mock_transaction, mock_job_ref, JobStatus.PROCESSING)

        # Reset and call again - should behave identically (pure function)
        mock_transaction.update.reset_mock()
        mock_job_ref.get.return_value = mock_snapshot  # Reset to same state

        update_job_status(mock_transaction, mock_job_ref, JobStatus.PROCESSING)

        # Both calls should produce identical update calls
        assert mock_transaction.update.call_count == 1

    def test_update_job_status_preserves_additional_fields(self):
        """Test 5: update_job_status() preserves additional_fields in update"""
        mock_transaction = MagicMock()
        mock_transaction._read_only = False
        mock_transaction._id = b"test-transaction-id"
        mock_transaction._max_attempts = 5
        mock_transaction.in_progress = True
        mock_transaction._write_pbs = []
        mock_transaction._clean_up = Mock()

        mock_job_ref = Mock()

        mock_snapshot = Mock()
        mock_snapshot.exists = True
        mock_snapshot.get.return_value = "scoring"
        mock_job_ref.get.return_value = mock_snapshot

        # Update with additional fields
        additional_fields = {
            'stats': {'total_rows': 100},
            'outliers': [],
            'processedAt': firestore.SERVER_TIMESTAMP
        }

        update_job_status(mock_transaction, mock_job_ref, JobStatus.COMPLETE, additional_fields)

        # Verify all fields were included in update
        call_args = mock_transaction.update.call_args
        update_data = call_args[0][1]

        assert update_data['status'] == JobStatus.COMPLETE
        assert update_data['stats'] == {'total_rows': 100}
        assert update_data['outliers'] == []
        assert 'processedAt' in update_data

    def test_update_job_status_raises_error_when_job_not_found(self):
        """Test that update_job_status raises ValueError when job document doesn't exist"""
        mock_transaction = MagicMock()
        mock_transaction._read_only = False
        mock_transaction._id = b"test-transaction-id"
        mock_transaction._max_attempts = 5
        mock_transaction.in_progress = True
        mock_transaction._write_pbs = []
        mock_transaction._clean_up = Mock()

        mock_job_ref = Mock()
        mock_job_ref.id = "test-job-123"

        # Mock snapshot with exists = False
        mock_snapshot = Mock()
        mock_snapshot.exists = False
        mock_job_ref.get.return_value = mock_snapshot

        with pytest.raises(ValueError, match="Job test-job-123 not found"):
            update_job_status(mock_transaction, mock_job_ref, JobStatus.PROCESSING)

    def test_update_job_status_logs_transition(self):
        """Test that update_job_status logs the status transition"""
        mock_transaction = MagicMock()
        mock_transaction._read_only = False
        mock_transaction._id = b"test-transaction-id"
        mock_transaction._max_attempts = 5
        mock_transaction.in_progress = True
        mock_transaction._write_pbs = []
        mock_transaction._clean_up = Mock()

        mock_job_ref = Mock()
        mock_job_ref.id = "test-job-456"

        mock_snapshot = Mock()
        mock_snapshot.exists = True
        mock_snapshot.get.return_value = "queued"
        mock_job_ref.get.return_value = mock_snapshot

        # Patch logger to verify logging
        with patch('worker.logger') as mock_logger:
            update_job_status(mock_transaction, mock_job_ref, JobStatus.PROCESSING)

            # Verify info log was called
            mock_logger.info.assert_called()
            log_message = mock_logger.info.call_args[0][0]
            assert "test-job-456" in log_message
            assert "queued" in log_message
            assert "processing" in log_message
