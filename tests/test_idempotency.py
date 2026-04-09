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
    # Codex P2: expiresAt must be a future timestamp (now + TTL), not the
    # current SERVER_TIMESTAMP sentinel, otherwise a Firestore TTL policy
    # would delete the marker immediately.
    import datetime as _dt
    assert isinstance(payload['expiresAt'], _dt.datetime)
    assert payload['expiresAt'] > _dt.datetime.now(_dt.timezone.utc)


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


def test_duplicate_delivery_while_job_in_progress_nacks_without_marking():
    """
    Codex P1 (r3055316xxx): if a duplicate Pub/Sub delivery arrives while
    another worker is still processing the same job (job doc is in
    PROCESSING / TRAINING / SCORING and no idempotency marker has been
    written yet because the original worker hasn't finished), callback()
    must nack WITHOUT marking the message. Otherwise:
      - If the original worker later crashes, the job stays stuck in an
        in-progress state, and
      - Because this delivery already marked the message as processed,
        any future redelivery is silently skipped as a duplicate.
    """
    from worker import callback, JobInProgressError

    message = Mock()
    message.data = json.dumps({
        "jobId": "job-inflight",
        "bucket": "test-bucket",
        "file": "test.csv"
    }).encode("utf-8")
    message.message_id = "msg-inflight-dup"
    message.ack = Mock()
    message.nack = Mock()

    with patch('worker.check_idempotency', return_value=False), \
         patch(
             'worker.process_upload_local',
             side_effect=JobInProgressError("already processing")
         ), \
         patch('worker.mark_message_processed') as mock_mark:
        callback(message)

    # Must nack for future redelivery and must NOT mark as processed.
    message.nack.assert_called_once()
    message.ack.assert_not_called()
    mock_mark.assert_not_called()


def test_job_document_not_ready_nacks_without_marking():
    """
    Codex P1 (r3053739500): if /start-job hasn't written the Firestore
    document yet and the worker picks the message up first, the worker
    raises JobDocumentNotReadyError. callback() must nack (so Pub/Sub
    redelivers) and must NOT mark the message as processed, otherwise
    the retry would be silently dropped as a duplicate.
    """
    from worker import callback, JobDocumentNotReadyError

    message = Mock()
    message.data = json.dumps({
        "jobId": "job-not-ready",
        "bucket": "test-bucket",
        "file": "test.csv"
    }).encode("utf-8")
    message.message_id = "msg-not-ready"
    message.ack = Mock()
    message.nack = Mock()

    with patch('worker.check_idempotency', return_value=False), \
         patch(
             'worker.process_upload_local',
             side_effect=JobDocumentNotReadyError("Job not ready")
         ), \
         patch('worker.mark_message_processed') as mock_mark:
        callback(message)

    # Must nack for redelivery and must NOT mark as processed.
    message.nack.assert_called_once()
    message.ack.assert_not_called()
    mock_mark.assert_not_called()


def test_is_claim_stale_true_for_old_timestamp():
    """
    _is_claim_stale should treat a claimedAt older than
    JOB_CLAIM_STALE_SECONDS as stale (eligible for takeover).
    """
    import datetime as _dt
    from worker import _is_claim_stale, JOB_CLAIM_STALE_SECONDS

    now = _dt.datetime.now(_dt.timezone.utc)
    old_claim = now - _dt.timedelta(seconds=JOB_CLAIM_STALE_SECONDS + 30)
    assert _is_claim_stale(old_claim, now=now) is True


def test_is_claim_stale_false_for_recent_timestamp():
    """A freshly-heartbeated claim must not be treated as stale."""
    import datetime as _dt
    from worker import _is_claim_stale, JOB_CLAIM_STALE_SECONDS

    now = _dt.datetime.now(_dt.timezone.utc)
    fresh_claim = now - _dt.timedelta(seconds=max(JOB_CLAIM_STALE_SECONDS // 2, 5))
    assert _is_claim_stale(fresh_claim, now=now) is False


def test_is_claim_stale_missing_timestamp_is_stale():
    """
    Legacy jobs written before `claimedAt` existed carry None; treat that
    as stale so recovery is possible without a data migration.
    """
    from worker import _is_claim_stale

    assert _is_claim_stale(None) is True


def _make_stale_claim_mocks(status_value, claimed_at, vertex_job_name=None):
    """Shared fixture for try_take_over_stale_claim tests."""
    mock_snapshot = Mock()
    mock_snapshot.exists = True

    stored = {
        'status': status_value,
        'claimedAt': claimed_at,
    }
    if vertex_job_name is not None:
        stored['vertexJobName'] = vertex_job_name

    def _snapshot_get(field, default=None):
        return stored.get(field, default)

    mock_snapshot.get = Mock(side_effect=_snapshot_get)

    mock_ref = Mock()
    mock_ref.id = f"job-{status_value}"
    mock_ref.get = Mock(return_value=mock_snapshot)

    mock_transaction = Mock()
    mock_transaction.update = Mock()

    mock_db = Mock()
    mock_db.transaction = Mock(return_value=mock_transaction)

    return mock_ref, mock_transaction, mock_db


def test_try_take_over_stale_claim_wins_when_claim_is_old():
    """
    try_take_over_stale_claim() refreshes `claimedAt` via a Firestore
    transaction and returns True when the existing claim is stale and
    the state is in-progress.
    """
    import datetime as _dt
    from worker import try_take_over_stale_claim, JobStatus, JOB_CLAIM_STALE_SECONDS

    old_claim = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(
        seconds=JOB_CLAIM_STALE_SECONDS + 60
    )

    mock_ref, mock_transaction, mock_db = _make_stale_claim_mocks(
        JobStatus.PROCESSING.value, old_claim
    )

    with patch('worker.db', mock_db), \
         patch('worker.firestore') as mock_firestore:
        def transactional_decorator(func):
            def wrapper(transaction, ref):
                return func(transaction, ref)
            return wrapper
        mock_firestore.transactional = transactional_decorator

        result = try_take_over_stale_claim(mock_ref)

    assert result is True
    mock_transaction.update.assert_called_once()
    payload = mock_transaction.update.call_args[0][1]
    assert 'claimedAt' in payload
    # Codex P1 followup: takeover must ALSO reset status to PROCESSING
    # so the downstream pipeline can restart from scratch instead of
    # hitting an invalid "training -> training" transition.
    assert payload['status'] == JobStatus.PROCESSING.value


def test_try_take_over_stale_claim_resets_training_to_processing():
    """
    Codex P1 r(stale-takeover-from-training): if the crashed worker had
    already transitioned the job past PROCESSING (e.g. to TRAINING), the
    takeover must still restart the pipeline from PROCESSING so the
    normal PROCESSING -> TRAINING -> SCORING -> COMPLETE flow works
    without hitting an invalid "training -> training" transition.
    """
    import datetime as _dt
    from worker import try_take_over_stale_claim, JobStatus, JOB_CLAIM_STALE_SECONDS

    old_claim = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(
        seconds=JOB_CLAIM_STALE_SECONDS + 60
    )

    for stuck_state in (JobStatus.TRAINING.value, JobStatus.SCORING.value):
        mock_ref, mock_transaction, mock_db = _make_stale_claim_mocks(
            stuck_state, old_claim
        )

        with patch('worker.db', mock_db), \
             patch('worker.firestore') as mock_firestore:
            def transactional_decorator(func):
                def wrapper(transaction, ref):
                    return func(transaction, ref)
                return wrapper
            mock_firestore.transactional = transactional_decorator

            result = try_take_over_stale_claim(mock_ref)

        assert result is True, f"takeover should succeed from {stuck_state}"
        payload = mock_transaction.update.call_args[0][1]
        assert payload['status'] == JobStatus.PROCESSING.value, (
            f"takeover from {stuck_state} must reset status to processing"
        )


def test_try_take_over_stale_claim_stamps_mode_from_caller():
    """
    Codex P2 r(takeover-mode-tag): when the caller passes mode="vertex",
    the takeover transaction must also persist `mode: "vertex"` on the
    recovered doc. Otherwise a duplicate delivery after a post-dispatch
    suppressor-write failure would still see an untagged doc and fall
    back to the short local stale threshold, allowing a second
    (billable) Vertex training submission.
    """
    import datetime as _dt
    from worker import (
        try_take_over_stale_claim,
        JobStatus,
        JOB_CLAIM_STALE_SECONDS_VERTEX,
    )

    very_old = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(
        seconds=JOB_CLAIM_STALE_SECONDS_VERTEX + 60
    )

    # Legacy untagged doc - no `mode` field at all.
    mock_snapshot = Mock()
    mock_snapshot.exists = True
    stored = {
        'status': JobStatus.PROCESSING.value,
        'claimedAt': very_old,
    }
    mock_snapshot.get = Mock(
        side_effect=lambda field, default=None: stored.get(field, default)
    )

    mock_ref = Mock()
    mock_ref.id = "job-legacy"
    mock_ref.get = Mock(return_value=mock_snapshot)

    mock_transaction = Mock()
    mock_transaction.update = Mock()

    mock_db = Mock()
    mock_db.transaction = Mock(return_value=mock_transaction)

    with patch('worker.db', mock_db), \
         patch('worker.firestore') as mock_firestore:
        def transactional_decorator(func):
            def wrapper(transaction, ref):
                return func(transaction, ref)
            return wrapper
        mock_firestore.transactional = transactional_decorator

        result = try_take_over_stale_claim(mock_ref, mode='vertex')

    assert result is True
    mock_transaction.update.assert_called_once()
    payload = mock_transaction.update.call_args[0][1]
    assert payload.get('mode') == 'vertex', (
        "takeover must stamp mode=vertex on a recovered legacy claim so "
        "subsequent duplicate deliveries use the Vertex stale threshold"
    )


def test_try_take_over_stale_claim_refuses_vertex_mode_short_window():
    """
    Codex P1 r(suppressor-write-failure): a job tagged `mode: "vertex"`
    must NOT be reclaimed after the 3-minute local stale window, because
    the Vertex training run itself routinely takes much longer than that.
    A claimedAt just 10 minutes old is well past the local threshold but
    comfortably inside the Vertex threshold, and takeover should refuse.
    """
    import datetime as _dt
    from worker import try_take_over_stale_claim, JobStatus, JOB_CLAIM_STALE_SECONDS

    ten_minutes_old = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=10)
    assert (
        _dt.datetime.now(_dt.timezone.utc) - ten_minutes_old
    ).total_seconds() > JOB_CLAIM_STALE_SECONDS, (
        "precondition: 10 minutes should be past the local stale threshold"
    )

    # Same fixture helper, but with mode=vertex in the stored doc.
    mock_snapshot = Mock()
    mock_snapshot.exists = True
    stored = {
        'status': JobStatus.PROCESSING.value,
        'claimedAt': ten_minutes_old,
        'mode': 'vertex',
    }
    mock_snapshot.get = Mock(
        side_effect=lambda field, default=None: stored.get(field, default)
    )

    mock_ref = Mock()
    mock_ref.id = "job-vertex-10min"
    mock_ref.get = Mock(return_value=mock_snapshot)

    mock_transaction = Mock()
    mock_transaction.update = Mock()

    mock_db = Mock()
    mock_db.transaction = Mock(return_value=mock_transaction)

    with patch('worker.db', mock_db), \
         patch('worker.firestore') as mock_firestore:
        def transactional_decorator(func):
            def wrapper(transaction, ref):
                return func(transaction, ref)
            return wrapper
        mock_firestore.transactional = transactional_decorator

        result = try_take_over_stale_claim(mock_ref)

    assert result is False, (
        "mode=vertex jobs must not be reclaimed at the local stale "
        "threshold, otherwise a post-dispatch metadata-write failure "
        "would allow a duplicate Vertex training submission"
    )
    mock_transaction.update.assert_not_called()


def test_try_take_over_stale_claim_reclaims_vertex_after_long_window():
    """
    Conversely, a `mode: "vertex"` job really is recoverable once it
    exceeds the Vertex stale threshold (4 hours). This proves the
    takeover path is still reachable for a truly crashed Vertex
    dispatcher, just with a much more conservative window.
    """
    import datetime as _dt
    from worker import (
        try_take_over_stale_claim,
        JobStatus,
        JOB_CLAIM_STALE_SECONDS_VERTEX,
    )

    very_old = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(
        seconds=JOB_CLAIM_STALE_SECONDS_VERTEX + 60
    )

    mock_snapshot = Mock()
    mock_snapshot.exists = True
    stored = {
        'status': JobStatus.PROCESSING.value,
        'claimedAt': very_old,
        'mode': 'vertex',
    }
    mock_snapshot.get = Mock(
        side_effect=lambda field, default=None: stored.get(field, default)
    )

    mock_ref = Mock()
    mock_ref.id = "job-vertex-verystale"
    mock_ref.get = Mock(return_value=mock_snapshot)

    mock_transaction = Mock()
    mock_transaction.update = Mock()

    mock_db = Mock()
    mock_db.transaction = Mock(return_value=mock_transaction)

    with patch('worker.db', mock_db), \
         patch('worker.firestore') as mock_firestore:
        def transactional_decorator(func):
            def wrapper(transaction, ref):
                return func(transaction, ref)
            return wrapper
        mock_firestore.transactional = transactional_decorator

        result = try_take_over_stale_claim(mock_ref)

    assert result is True
    mock_transaction.update.assert_called_once()


def test_try_take_over_stale_claim_refuses_when_vertex_dispatched():
    """
    Codex P1 r(vertex-stale-redispatch): if the job doc already carries
    a `vertexJobName`, Vertex owns the training run. Re-taking over
    would resubmit a duplicate (billable) Vertex job. Even if the
    stored `claimedAt` looks stale, takeover must refuse.
    """
    import datetime as _dt
    from worker import try_take_over_stale_claim, JobStatus, JOB_CLAIM_STALE_SECONDS

    old_claim = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(
        seconds=JOB_CLAIM_STALE_SECONDS + 60
    )

    mock_ref, mock_transaction, mock_db = _make_stale_claim_mocks(
        JobStatus.PROCESSING.value,
        old_claim,
        vertex_job_name="projects/x/locations/us-central1/customJobs/123",
    )

    with patch('worker.db', mock_db), \
         patch('worker.firestore') as mock_firestore:
        def transactional_decorator(func):
            def wrapper(transaction, ref):
                return func(transaction, ref)
            return wrapper
        mock_firestore.transactional = transactional_decorator

        result = try_take_over_stale_claim(mock_ref)

    assert result is False
    mock_transaction.update.assert_not_called()


def test_try_take_over_stale_claim_rejects_fresh_claim():
    """
    try_take_over_stale_claim() must NOT refresh `claimedAt` when the
    existing heartbeat is still fresh (another worker is alive).
    """
    import datetime as _dt
    from worker import try_take_over_stale_claim, JobStatus

    fresh_claim = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=5)

    mock_snapshot = Mock()
    mock_snapshot.exists = True
    def _snapshot_get(field, default=None):
        return {
            'status': JobStatus.PROCESSING.value,
            'claimedAt': fresh_claim,
        }.get(field, default)
    mock_snapshot.get = Mock(side_effect=_snapshot_get)

    mock_ref = Mock()
    mock_ref.id = "job-fresh"
    mock_ref.get = Mock(return_value=mock_snapshot)

    mock_transaction = Mock()
    mock_transaction.update = Mock()

    mock_db = Mock()
    mock_db.transaction = Mock(return_value=mock_transaction)

    with patch('worker.db', mock_db), \
         patch('worker.firestore') as mock_firestore:
        def transactional_decorator(func):
            def wrapper(transaction, ref):
                return func(transaction, ref)
            return wrapper
        mock_firestore.transactional = transactional_decorator

        result = try_take_over_stale_claim(mock_ref)

    assert result is False
    mock_transaction.update.assert_not_called()


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
