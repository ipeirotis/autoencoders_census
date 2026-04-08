"""
Tests for worker environment validation

Tests verify that the worker fails fast at startup with clear error messages
when required environment variables are missing.

Note: These tests directly call validate_environment() which reads os.getenv()
fresh each time, so monkeypatch works correctly.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Mock tensorflow and model imports before importing worker
sys.modules['tensorflow'] = MagicMock()
sys.modules['model.autoencoder'] = MagicMock()
sys.modules['dataset.loader'] = MagicMock()
sys.modules['features.transform'] = MagicMock()

# Mock load_dotenv and google.cloud.firestore.Client before importing worker.
# worker.py instantiates a Firestore client at module load time, which would
# otherwise fail under CI where no Application Default Credentials are set.
with patch('dotenv.load_dotenv'), patch('google.cloud.firestore.Client'):
    import worker


def test_validate_environment_raises_on_missing_google_cloud_project(monkeypatch):
    """Test that validate_environment() exits if GOOGLE_CLOUD_PROJECT is missing"""
    # Unset the environment variable
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("GCS_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("PUBSUB_SUBSCRIPTION_ID", "test-subscription")

    # validate_environment should exit with SystemExit
    with pytest.raises(SystemExit) as exc_info:
        worker.validate_environment()

    assert exc_info.value.code == 1


def test_validate_environment_raises_on_missing_gcs_bucket_name(monkeypatch):
    """Test that validate_environment() exits if GCS_BUCKET_NAME is missing"""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    monkeypatch.setenv("PUBSUB_SUBSCRIPTION_ID", "test-subscription")

    with pytest.raises(SystemExit) as exc_info:
        worker.validate_environment()

    assert exc_info.value.code == 1


def test_validate_environment_raises_on_missing_pubsub_subscription_id(monkeypatch):
    """Test that validate_environment() exits if PUBSUB_SUBSCRIPTION_ID is missing"""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GCS_BUCKET_NAME", "test-bucket")
    monkeypatch.delenv("PUBSUB_SUBSCRIPTION_ID", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        worker.validate_environment()

    assert exc_info.value.code == 1


def test_validate_environment_passes_when_all_vars_present(monkeypatch):
    """Test that validate_environment() returns True when all required vars are present"""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GCS_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("PUBSUB_SUBSCRIPTION_ID", "test-subscription")

    # Should not raise, should return True
    result = worker.validate_environment()
    assert result is True
