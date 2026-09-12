"""
Tests for worker_http.py -- the HTTP push front-end for Cloud Run.

These spin up the real ThreadingHTTPServer on an ephemeral port and drive it
with urllib so the actual BaseHTTPRequestHandler routing + status mapping is
exercised. worker.callback is patched per-test to simulate ack / nack / raise
without touching Firestore, GCS or TensorFlow.

(conftest.py stubs the GCP client factories + required env vars, so importing
worker_http -> worker succeeds without real credentials.)
"""

import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import Mock, patch

import pytest

import worker_http


@pytest.fixture
def server():
    """Start worker_http.Handler on an ephemeral port; tear down after."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), worker_http.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _envelope(data_str="{}", message_id="msg-1"):
    message = {"data": base64.b64encode(data_str.encode()).decode()}
    if message_id is not None:
        message["messageId"] = message_id
    return {"message": message, "subscription": "projects/p/subscriptions/s"}


def _post(base_url, payload, path=None):
    path = path if path is not None else worker_http.PUSH_PATH
    body = json.dumps(payload).encode() if isinstance(payload, (dict, list)) else payload
    req = urllib.request.Request(
        base_url + path, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def _get(base_url, path):
    try:
        with urllib.request.urlopen(base_url + path, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def test_healthz_returns_200(server):
    assert _get(server, "/healthz") == 200


def test_unknown_get_path_returns_404(server):
    assert _get(server, "/nope") == 404


def test_happy_path_ack_returns_204(server):
    def fake_callback(msg):
        msg.ack()

    with patch.object(worker_http.worker, "callback", side_effect=fake_callback):
        assert _post(server, _envelope()) == 204


def test_explicit_nack_returns_503(server):
    def fake_callback(msg):
        msg.nack()

    with patch.object(worker_http.worker, "callback", side_effect=fake_callback):
        assert _post(server, _envelope()) == 503


def test_callback_exception_returns_503(server):
    def boom(msg):
        raise RuntimeError("processing blew up")

    with patch.object(worker_http.worker, "callback", side_effect=boom):
        assert _post(server, _envelope()) == 503


def test_callback_without_ack_or_nack_returns_503(server):
    # Defensive: if callback() returns without deciding, NACK so Pub/Sub retries.
    with patch.object(worker_http.worker, "callback", side_effect=lambda msg: None):
        assert _post(server, _envelope()) == 503


def test_malformed_envelope_is_dropped_204(server):
    callback = Mock()
    with patch.object(worker_http.worker, "callback", callback):
        # No "message" key -> KeyError during parse -> ACK (drop), no callback.
        assert _post(server, {"not": "a pubsub envelope"}) == 204
    callback.assert_not_called()


def test_missing_message_id_is_dropped_204(server):
    callback = Mock()
    with patch.object(worker_http.worker, "callback", callback):
        assert _post(server, _envelope(message_id=None)) == 204
    callback.assert_not_called()


def test_wrong_post_path_returns_404(server):
    callback = Mock()
    with patch.object(worker_http.worker, "callback", callback):
        assert _post(server, _envelope(), path="/some/other/path") == 404
    callback.assert_not_called()


def test_push_message_adapter_records_outcome():
    msg = worker_http.PushMessage(b"data", "id-1")
    assert msg.result is None
    msg.modify_ack_deadline(30)  # no-op, must not raise
    msg.ack()
    assert msg.result == "ack"
    msg.nack()
    assert msg.result == "nack"
