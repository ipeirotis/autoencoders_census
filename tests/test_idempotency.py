"""
Tests for idempotent message processing using Firestore.

Tests that duplicate Pub/Sub messages do not trigger duplicate processing.
Uses mocks to avoid requiring Firestore emulator.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json


def test_check_idempotency_returns_true_for_duplicate():
    """Test that check_idempotency returns True when message ID already processed."""
    from worker import check_idempotency

    # Mock Firestore document that exists (already processed)
    mock_snapshot = Mock()
    mock_snapshot.exists = True

    mock_ref = Mock()
    mock_ref.get = Mock(return_value=mock_snapshot)

    mock_transaction = Mock()

    mock_db = Mock()
    mock_db.collection = Mock(return_value=Mock(document=Mock(return_value=mock_ref)))
    mock_db.transaction = Mock(return_value=mock_transaction)

    with patch('worker.db', mock_db):
        with patch('worker.firestore') as mock_firestore:
            # Set up the transactional decorator to execute the function
            def transactional_decorator(func):
                def wrapper(transaction, ref):
                    return func(transaction, ref)
                return wrapper

            mock_firestore.transactional = transactional_decorator
            mock_firestore.SERVER_TIMESTAMP = "TIMESTAMP"

            result = check_idempotency("msg-123", "job-456")

    assert result is True, "Should return True for already processed message"


def test_check_idempotency_returns_false_for_first_time():
    """Test that check_idempotency returns False and marks processed on first occurrence."""
    from worker import check_idempotency

    # Mock Firestore document that doesn't exist (first time)
    mock_snapshot = Mock()
    mock_snapshot.exists = False

    mock_ref = Mock()
    mock_ref.get = Mock(return_value=mock_snapshot)

    mock_transaction = Mock()
    mock_transaction.set = Mock()

    mock_db = Mock()
    mock_db.collection = Mock(return_value=Mock(document=Mock(return_value=mock_ref)))
    mock_db.transaction = Mock(return_value=mock_transaction)

    with patch('worker.db', mock_db):
        with patch('worker.firestore') as mock_firestore:
            mock_firestore.transactional = lambda func: func
            mock_firestore.SERVER_TIMESTAMP = "TIMESTAMP"
            result = check_idempotency("msg-123", "job-456")

    assert result is False, "Should return False for first time processing"


def test_duplicate_messages_skip_processing():
    """Test that duplicate messages skip processing and ack immediately."""
    from worker import callback

    # Create mock message
    message = Mock()
    message.data = json.dumps({
        "jobId": "job-123",
        "bucket": "test-bucket",
        "file": "test.csv"
    }).encode("utf-8")
    message.message_id = "msg-duplicate"
    message.ack = Mock()
    message.nack = Mock()

    # Mock check_idempotency to return True (already processed)
    with patch('worker.check_idempotency', return_value=True):
        callback(message)

    # Should ack, not nack
    message.ack.assert_called_once()
    message.nack.assert_not_called()


def test_firestore_transaction_prevents_race_condition():
    """Test that Firestore transaction prevents race when two workers check same message."""
    from worker import check_idempotency

    # Simulate race condition: document exists when checked in transaction
    # (another worker created it between our initial check and transaction)
    mock_snapshot_exists = Mock()
    mock_snapshot_exists.exists = True

    mock_ref = Mock()
    mock_ref.get = Mock(return_value=mock_snapshot_exists)

    mock_transaction = Mock()
    mock_transaction.set = Mock()

    mock_db = Mock()
    mock_db.collection = Mock(return_value=Mock(document=Mock(return_value=mock_ref)))
    mock_db.transaction = Mock(return_value=mock_transaction)

    with patch('worker.db', mock_db):
        with patch('worker.firestore') as mock_firestore:
            # Set up the transactional decorator to execute the function
            def transactional_decorator(func):
                def wrapper(transaction, ref):
                    return func(transaction, ref)
                return wrapper

            mock_firestore.transactional = transactional_decorator
            mock_firestore.SERVER_TIMESTAMP = "TIMESTAMP"

            result = check_idempotency("msg-race", "job-race")

    # Should detect document was created by other worker and return True
    assert result is True, "Should handle race condition correctly"
    # Should NOT have called transaction.set (document already exists)
    mock_transaction.set.assert_not_called()
