"""
Tests for idempotent message processing using Firestore.

Tests that duplicate Pub/Sub messages do not trigger duplicate processing,
and that the idempotency marker is only written AFTER processing completes
so that a crash mid-processing does not cause Pub/Sub redeliveries to be
silently dropped.

Uses mocks to avoid requiring the Firestore emulator.
"""

import json
from unittest.mock import Mock, patch


def test_check_idempotency_returns_true_for_duplicate():
    """check_idempotency() returns True when the marker already exists."""
    from worker import check_idempotency

    mock_snapshot = Mock()
    mock_snapshot.exists = True

    mock_ref = Mock()
    mock_ref.get = Mock(return_value=mock_snapshot)

    mock_db = Mock()
    mock_db.collection = Mock(
        return_value=Mock(document=Mock(return_value=mock_ref))
    )

    with patch('worker.db', mock_db):
        result = check_idempotency("msg-123")

    assert result is True, "Should return True for already processed message"
    # Read-only: should NOT open a transaction, just a plain get().
    mock_ref.get.assert_called_once_with()


def test_check_idempotency_returns_false_for_first_time():
    """check_idempotency() returns False when no marker exists yet and does NOT write."""
    from worker import check_idempotency

    mock_snapshot = Mock()
    mock_snapshot.exists = False

    mock_ref = Mock()
    mock_ref.get = Mock(return_value=mock_snapshot)
    mock_ref.set = Mock()

    mock_transaction = Mock()
    mock_transaction.set = Mock()

    mock_db = Mock()
    mock_db.collection = Mock(
        return_value=Mock(document=Mock(return_value=mock_ref))
    )
    mock_db.transaction = Mock(return_value=mock_transaction)

    with patch('worker.db', mock_db):
        result = check_idempotency("msg-123")

    assert result is False, "Should return False for first time processing"
    # Crucially, check_idempotency must NOT mark the message.
    mock_ref.set.assert_not_called()
    mock_transaction.set.assert_not_called()


def test_mark_message_processed_writes_marker():
    """mark_message_processed() transactionally writes the marker."""
    from worker import mark_message_processed

    mock_snapshot = Mock()
    mock_snapshot.exists = False

    mock_ref = Mock()
    mock_ref.get = Mock(return_value=mock_snapshot)

    mock_transaction = Mock()
    mock_transaction.set = Mock()

    mock_db = Mock()
    mock_db.collection = Mock(
        return_value=Mock(document=Mock(return_value=mock_ref))
    )
    mock_db.transaction = Mock(return_value=mock_transaction)

    with patch('worker.db', mock_db):
        with patch('worker.firestore') as mock_firestore:
            # Execute the transactional function inline for the test.
            def transactional_decorator(func):
                def wrapper(transaction, ref):
                    return func(transaction, ref)
                return wrapper

            mock_firestore.transactional = transactional_decorator
            mock_firestore.SERVER_TIMESTAMP = "TIMESTAMP"

            mark_message_processed("msg-123", "job-456")

    mock_transaction.set.assert_called_once()
    # Verify the marker payload includes the job ID for bookkeeping/cleanup.
    call_args = mock_transaction.set.call_args
    payload = call_args[0][1]
    assert payload['jobId'] == "job-456"
    assert 'processedAt' in payload


def test_mark_message_processed_is_idempotent_on_existing_marker():
    """mark_message_processed() is a no-op if the marker already exists."""
    from worker import mark_message_processed

    mock_snapshot = Mock()
    mock_snapshot.exists = True  # Already marked by another worker

    mock_ref = Mock()
    mock_ref.get = Mock(return_value=mock_snapshot)

    mock_transaction = Mock()
    mock_transaction.set = Mock()

    mock_db = Mock()
    mock_db.collection = Mock(
        return_value=Mock(document=Mock(return_value=mock_ref))
    )
    mock_db.transaction = Mock(return_value=mock_transaction)

    with patch('worker.db', mock_db):
        with patch('worker.firestore') as mock_firestore:
            def transactional_decorator(func):
                def wrapper(transaction, ref):
                    return func(transaction, ref)
                return wrapper

            mock_firestore.transactional = transactional_decorator
            mock_firestore.SERVER_TIMESTAMP = "TIMESTAMP"

            mark_message_processed("msg-123", "job-456")

    # Marker already exists; we must not overwrite it.
    mock_transaction.set.assert_not_called()


def test_duplicate_messages_skip_processing():
    """callback() acks duplicate messages without running the processor."""
    from worker import callback

    message = Mock()
    message.data = json.dumps({
        "jobId": "job-123",
        "bucket": "test-bucket",
        "file": "test.csv"
    }).encode("utf-8")
    message.message_id = "msg-duplicate"
    message.ack = Mock()
    message.nack = Mock()

    with patch('worker.check_idempotency', return_value=True) as mock_check, \
         patch('worker.process_upload_local') as mock_process, \
         patch('worker.mark_message_processed') as mock_mark:
        callback(message)

    mock_check.assert_called_once_with("msg-duplicate")
    mock_process.assert_not_called()
    mock_mark.assert_not_called()
    message.ack.assert_called_once()
    message.nack.assert_not_called()


def test_first_time_message_processes_then_marks_then_acks():
    """callback() processes before marking, and marks before acking."""
    from worker import callback

    message = Mock()
    message.data = json.dumps({
        "jobId": "job-abc",
        "bucket": "test-bucket",
        "file": "test.csv"
    }).encode("utf-8")
    message.message_id = "msg-fresh"
    message.ack = Mock()
    message.nack = Mock()

    call_order = []

    def fake_process(*args, **kwargs):
        call_order.append('process')

    def fake_mark(*args, **kwargs):
        call_order.append('mark')

    def fake_ack(*args, **kwargs):
        call_order.append('ack')

    message.ack.side_effect = fake_ack

    with patch('worker.check_idempotency', return_value=False), \
         patch('worker.process_upload_local', side_effect=fake_process), \
         patch('worker.mark_message_processed', side_effect=fake_mark):
        callback(message)

    assert call_order == ['process', 'mark', 'ack'], (
        "Must process first, then mark the message as processed, then ack."
    )
    message.nack.assert_not_called()


def test_crash_mid_processing_does_not_leave_marker():
    """
    Simulate a worker crashing mid-processing. The marker must NOT be set,
    so the next Pub/Sub redelivery can retry the job instead of silently
    dropping it.
    """
    from worker import callback

    message = Mock()
    message.data = json.dumps({
        "jobId": "job-crash",
        "bucket": "test-bucket",
        "file": "test.csv"
    }).encode("utf-8")
    message.message_id = "msg-crash"
    message.ack = Mock()
    message.nack = Mock()

    class SimulatedCrash(Exception):
        pass

    with patch('worker.check_idempotency', return_value=False), \
         patch(
             'worker.process_upload_local',
             side_effect=SimulatedCrash("worker died")
         ), \
         patch('worker.mark_message_processed') as mock_mark:
        callback(message)

    # Marker must not be set - otherwise redelivery would be silently dropped.
    mock_mark.assert_not_called()
    # Outer try/except nacks on unhandled exceptions so Pub/Sub will redeliver.
    message.nack.assert_called_once()
    message.ack.assert_not_called()
