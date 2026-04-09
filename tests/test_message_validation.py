"""
Tests for Pub/Sub message validation using Pydantic.

Tests that worker.py correctly validates required fields (jobId, bucket, file)
and rejects malformed messages.
"""

import pytest
from unittest.mock import Mock, MagicMock
import json


def test_missing_job_id():
    """Test that validate_message raises ValueError when jobId is missing."""
    from worker import validate_message

    data = {
        "bucket": "test-bucket",
        "file": "test-file.csv"
    }

    with pytest.raises(ValueError) as exc_info:
        validate_message(data)

    assert "jobId" in str(exc_info.value)


def test_missing_bucket():
    """Test that validate_message raises ValueError when bucket is missing."""
    from worker import validate_message

    data = {
        "jobId": "test-job-123",
        "file": "test-file.csv"
    }

    with pytest.raises(ValueError) as exc_info:
        validate_message(data)

    assert "bucket" in str(exc_info.value)


def test_missing_file():
    """Test that validate_message raises ValueError when file is missing."""
    from worker import validate_message

    data = {
        "jobId": "test-job-123",
        "bucket": "test-bucket"
    }

    with pytest.raises(ValueError) as exc_info:
        validate_message(data)

    assert "file" in str(exc_info.value)


def test_valid_message():
    """Test that validate_message returns PubSubMessage when all fields present."""
    from worker import validate_message, PubSubMessage

    data = {
        "jobId": "test-job-123",
        "bucket": "test-bucket",
        "file": "test-file.csv"
    }

    result = validate_message(data)

    assert isinstance(result, PubSubMessage)
    assert result.jobId == "test-job-123"
    assert result.bucket == "test-bucket"
    assert result.file == "test-file.csv"


def test_callback_acks_invalid_message_to_avoid_poison_loop():
    """
    Codex P2 (r3053739504): schema validation failures are deterministic,
    not transient. callback() must ack the message (drop it) instead of
    nacking to prevent infinite redelivery against a bad payload.
    """
    from worker import callback

    # Create mock message with missing jobId
    message = Mock()
    message.data = json.dumps({
        "bucket": "test-bucket",
        "file": "test-file.csv"
    }).encode("utf-8")
    message.message_id = "msg-123"
    message.ack = Mock()
    message.nack = Mock()

    callback(message)

    # Should ack (drop poison message), NOT nack (would redeliver forever).
    message.ack.assert_called_once()
    message.nack.assert_not_called()


def test_callback_acks_non_json_payload_instead_of_nacking():
    """
    Codex P2 (r3053739504): a message whose body is not valid JSON is a
    permanent failure for the same reason - drop it by acking.
    """
    from worker import callback

    message = Mock()
    message.data = b"this is not json {"
    message.message_id = "msg-non-json"
    message.ack = Mock()
    message.nack = Mock()

    callback(message)

    message.ack.assert_called_once()
    message.nack.assert_not_called()
