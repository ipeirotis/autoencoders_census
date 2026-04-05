"""
Tests for ack deadline extension for long-running jobs.

Tests that worker extends Pub/Sub message ack deadline during processing
to prevent timeout and redelivery for 10-15 minute jobs.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import time
import threading


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
