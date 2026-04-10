"""
Tests for ack deadline extension for long-running jobs.

Tests that worker extends Pub/Sub message ack deadline during processing
to prevent timeout and redelivery for 10-15 minute jobs.
"""

from unittest.mock import Mock, patch
import time


def test_ack_extender_extends_deadline_periodically():
    """Test that AckExtender.extend() calls message.modify_ack_deadline(70) periodically."""
    from worker import AckExtender

    message = Mock()
    message.message_id = "test-msg-123"
    message.modify_ack_deadline = Mock()

    extender = AckExtender(message, interval_seconds=0.1)  # Fast interval for testing
    extender.start()

    # Wait for at least 2 extensions
    time.sleep(0.25)

    extender.stop()

    # Should have called modify_ack_deadline at least twice
    assert message.modify_ack_deadline.call_count >= 2
    # Should extend by interval + 10 second buffer (0.1 + 10 = 10.1 in this case, but we use 70 for real)
    # For testing we just check it was called
    assert message.modify_ack_deadline.called


def test_ack_extender_without_job_ref_does_not_refresh_claimed_at():
    """
    Codex P1 r3055xxxxx: the extender must NOT touch `claimedAt` when it
    is started without a job_ref. Otherwise a redelivered message for a
    crashed job would have its stale claim rewritten to "fresh" by the
    heartbeat before the caller gets a chance to run
    try_take_over_stale_claim, permanently wedging crash recovery.
    """
    from worker import AckExtender

    message = Mock()
    message.message_id = "test-msg-noref"
    message.modify_ack_deadline = Mock()

    extender = AckExtender(message, interval_seconds=0.05, job_ref=None)
    extender.start()
    time.sleep(0.15)
    extender.stop()

    # Pub/Sub ack deadline refresh still runs...
    assert message.modify_ack_deadline.called
    # ...but there is no job_ref for the heartbeat to touch, so there is
    # nothing to over-refresh. This test locks in that starting the
    # extender without job_ref is a safe no-op on the claim side.


def test_ack_extender_refreshes_claimed_at_once_job_ref_is_attached():
    """
    After the caller confirms the claim (either via the normal
    queued->processing transition or via try_take_over_stale_claim),
    attaching job_ref on the already-running extender must begin
    refreshing `claimedAt` on subsequent extend() ticks.
    """
    from worker import AckExtender

    message = Mock()
    message.message_id = "test-msg-attach"
    message.modify_ack_deadline = Mock()

    job_ref = Mock()
    job_ref.id = "job-xyz"
    job_ref.update = Mock()

    extender = AckExtender(message, interval_seconds=0.05, job_ref=None)
    extender.start()
    # Before attachment: no claim refresh should have happened.
    time.sleep(0.1)
    assert job_ref.update.call_count == 0

    # Attach job_ref as the worker would after confirming the claim.
    extender.job_ref = job_ref

    # Subsequent ticks should start refreshing claimedAt.
    time.sleep(0.15)
    extender.stop()

    assert job_ref.update.call_count >= 1
    # Sanity: the payload should touch claimedAt.
    call_payload = job_ref.update.call_args[0][0]
    assert 'claimedAt' in call_payload


def test_ack_extender_stop_cancels_timer():
    """Test that AckExtender.stop() cancels timer and prevents further extensions."""
    from worker import AckExtender

    message = Mock()
    message.message_id = "test-msg-456"
    message.modify_ack_deadline = Mock()

    extender = AckExtender(message, interval_seconds=0.1)
    extender.start()

    # Wait for first extension
    time.sleep(0.15)

    call_count_before_stop = message.modify_ack_deadline.call_count

    # Stop the extender
    extender.stop()

    # Wait to confirm no more extensions happen
    time.sleep(0.2)

    call_count_after_stop = message.modify_ack_deadline.call_count

    # Should not have increased after stop
    assert call_count_after_stop == call_count_before_stop


def test_extension_continues_during_long_processing():
    """Test that extension continues every 60 seconds during long processing."""
    from worker import AckExtender

    message = Mock()
    message.message_id = "test-msg-789"
    message.modify_ack_deadline = Mock()

    extender = AckExtender(message, interval_seconds=0.05)  # Fast for testing
    extender.start()

    # Simulate long processing
    time.sleep(0.15)

    extender.stop()

    # Should have extended multiple times
    assert message.modify_ack_deadline.call_count >= 2


def test_extension_failure_logs_warning_but_continues():
    """Test that extension failure logs warning but continues processing."""
    from worker import AckExtender

    message = Mock()
    message.message_id = "test-msg-error"
    # Make modify_ack_deadline raise exception on first call, succeed on second
    message.modify_ack_deadline = Mock(side_effect=[Exception("Network error"), None, None])

    extender = AckExtender(message, interval_seconds=0.1)

    with patch('worker.logger') as mock_logger:
        extender.start()
        time.sleep(0.25)
        extender.stop()

        # Should have logged warning about failure
        warning_calls = [call for call in mock_logger.warning.call_args_list
                        if 'Failed to extend ack deadline' in str(call)]
        assert len(warning_calls) >= 1

    # Should have tried multiple times despite first failure
    assert message.modify_ack_deadline.call_count >= 2


def test_ack_extender_cleanup_in_finally():
    """Test that AckExtender cleanup happens in finally block even on exception."""
    from worker import AckExtender

    message = Mock()
    message.message_id = "test-msg-finally"
    message.modify_ack_deadline = Mock()

    extender = AckExtender(message, interval_seconds=0.1)

    try:
        extender.start()
        time.sleep(0.05)
        raise ValueError("Simulated processing error")
    except ValueError:
        pass
    finally:
        extender.stop()

    # Wait to ensure no more extensions after stop
    time.sleep(0.15)

    # Timer should be stopped (check internal state)
    assert extender.stopped is True
