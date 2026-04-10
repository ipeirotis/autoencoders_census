"""
Pytest fixture bootstrap for the worker test suite.

Why this file exists:
    worker.py creates `db = firestore.Client(project=PROJECT_ID)` at module
    scope, which calls google.auth.default() at import time and fails in CI
    without GOOGLE_APPLICATION_CREDENTIALS:

        E   google.auth.exceptions.DefaultCredentialsError: Your default
            credentials were not found.

    Most worker tests import worker lazily inside test functions and use
    unittest.mock to stub out `worker.db`, but pytest still crashes during
    collection when even one test module (e.g. tests/test_csv_validation.py)
    does `from worker import ...` at module level. Once that collection
    error fires with `-x`, the whole run aborts before any test executes.

    pytest loads conftest.py before collecting test modules in the same
    directory, so by stubbing google.auth.default and google.cloud.firestore
    /pubsub_v1/storage client factories here we give the worker module a
    successful import path even when real credentials are absent. Individual
    tests are still free to patch `worker.db` / `worker.pubsub_v1` /
    `worker.storage` to assert on concrete behaviour.
"""

import os
import sys
from unittest.mock import MagicMock

# Provide dummy values for the env vars worker.py reads at import time so
# PROJECT_ID / BUCKET_NAME / SUBSCRIPTION_ID are non-empty strings rather
# than None (which propagates into pydantic / firestore calls deeper in the
# module and produces less-obvious failure modes).
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
os.environ.setdefault("GCS_BUCKET_NAME", "test-bucket")
os.environ.setdefault("PUBSUB_SUBSCRIPTION_ID", "test-subscription")
# Point Google auth at a non-existent file so the library does not attempt
# to hit the metadata server during import. The firestore.Client stub below
# short-circuits the call anyway; this is belt-and-braces.
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "/dev/null")


def _install_gcp_client_stubs() -> None:
    """
    Replace the real google.cloud client factories with MagicMock so
    `firestore.Client(...)`, `pubsub_v1.SubscriberClient(...)`, and
    `storage.Client(...)` return cheap mocks instead of attempting real auth.

    We patch at the package level BEFORE worker.py is imported so its
    `from google.cloud import pubsub_v1, firestore, storage` binding picks up
    the stubbed factories. Tests that want to assert on Firestore behaviour
    patch `worker.db` directly (e.g. `with patch.object(worker, "db", ...)`),
    which still works because the MagicMock we leave behind has MagicMock
    attribute access by default.

    Strategy: try importing the real GCP libraries first (works in CI where
    cryptography is healthy). If the import fails for ANY reason — including
    pyo3_runtime.PanicException from a broken cryptography C extension —
    fall back to injecting lightweight mock modules directly into
    sys.modules, which avoids the google-auth → cryptography import chain
    entirely.
    """
    try:
        from google.cloud import firestore as _firestore
        from google.cloud import pubsub_v1 as _pubsub_v1
        from google.cloud import storage as _storage

        # Real imports succeeded — patch Client factories in-place.
        _firestore.Client = MagicMock(return_value=MagicMock(name="FirestoreClient"))
        _pubsub_v1.SubscriberClient = MagicMock(
            return_value=MagicMock(name="SubscriberClient")
        )
        _pubsub_v1.PublisherClient = MagicMock(
            return_value=MagicMock(name="PublisherClient")
        )
        _storage.Client = MagicMock(return_value=MagicMock(name="StorageClient"))

        # Make firestore.transactional a pass-through so decorated functions
        # (e.g. update_job_status) keep their real implementation.
        _firestore.transactional = lambda fn: fn

    except BaseException:
        # Import failed (e.g. broken cryptography, missing packages).
        # Inject mock modules directly into sys.modules instead.
        _mock_firestore = MagicMock(name="mock_firestore_module")
        _mock_firestore.Client = MagicMock(return_value=MagicMock(name="FirestoreClient"))
        _mock_firestore.transactional = lambda fn: fn

        _mock_pubsub = MagicMock(name="mock_pubsub_v1_module")
        _mock_pubsub.SubscriberClient = MagicMock(
            return_value=MagicMock(name="SubscriberClient")
        )
        _mock_pubsub.PublisherClient = MagicMock(
            return_value=MagicMock(name="PublisherClient")
        )

        _mock_storage = MagicMock(name="mock_storage_module")
        _mock_storage.Client = MagicMock(return_value=MagicMock(name="StorageClient"))

        # Inject mock leaf modules.  Do NOT touch "google" or "google.cloud"
        # namespace packages — they are shared with google-protobuf, grpcio,
        # and TensorFlow and must keep their real __path__.
        sys.modules["google.cloud.firestore"] = _mock_firestore
        sys.modules["google.cloud.firestore_v1"] = _mock_firestore
        sys.modules["google.cloud.pubsub_v1"] = _mock_pubsub
        sys.modules["google.cloud.storage"] = _mock_storage

        # Pre-seed google.auth (and key sub-modules) with mocks so the
        # import chain never reaches the broken cryptography C extension.
        _mock_auth = MagicMock(name="mock_google_auth")
        for _auth_key in [
            "google.auth",
            "google.auth.credentials",
            "google.auth.default",
            "google.auth.transport",
            "google.auth.transport.grpc",
            "google.auth.transport.requests",
            "google.auth._default",
            "google.auth.crypt",
            "google.auth.crypt.es",
        ]:
            sys.modules.setdefault(_auth_key, _mock_auth)

        _mock_oauth = MagicMock(name="mock_google_oauth2")
        for _oauth_key in [
            "google.oauth2",
            "google.oauth2.service_account",
            "google.oauth2.credentials",
        ]:
            sys.modules.setdefault(_oauth_key, _mock_oauth)


_install_gcp_client_stubs()

# If worker has already been imported (e.g. by pytest auto-discovery of a
# sibling conftest or an earlier test module), drop it so the next
# `from worker import ...` re-runs module-level code against our stubs.
# Safe no-op when worker has not been imported yet.
sys.modules.pop("worker", None)
