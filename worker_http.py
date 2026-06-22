"""
worker_http.py -- HTTP front-end for the worker, for Cloud Run + Pub/Sub *push*.

Why this exists
---------------
``worker.py`` runs as a long-lived process that *pulls* messages off a Pub/Sub
subscription. That model needs an always-on instance, which defeats Cloud Run's
scale-to-zero (and costs money sitting idle). This module wraps the SAME
job-processing logic (``worker.callback``) behind a tiny HTTP endpoint so
Pub/Sub can *push* a message to it instead: Cloud Run wakes the container on the
incoming request, the job runs, and the container scales back to zero when idle.

Endpoints
---------
- ``POST /pubsub/push`` -- receives the Pub/Sub push envelope and runs the job.
  Returns **204** to ACK (success, or a poison message we intentionally drop)
  and **503** to NACK (transient failure -> Pub/Sub redelivers later).
- ``GET /healthz`` -- liveness/readiness probe -> ``200 ok``.

Job duration limit
------------------
A push job runs synchronously inside the HTTP request, so it must finish within
the subscription's ack deadline (set to the 600s max) or Pub/Sub redelivers it.
Redelivery is made safe by the idempotency check + Firestore state machine in
``worker.callback()`` but wastes invocations, so in-container ``local`` mode is
meant for sub-600s jobs. For longer jobs use ``WORKER_MODE=vertex``, which makes
the request only *dispatch* a Vertex AI job and return in seconds. See the
``PushMessage.modify_ack_deadline`` comment for details.

Security
--------
The Cloud Run service is deployed with ``--no-allow-unauthenticated``, so Cloud
Run itself rejects any request that does not carry a valid OIDC token from the
Pub/Sub push service account (which is granted ``roles/run.invoker``). By the
time a request reaches this code it has already been authenticated by the
platform, so no in-app token verification is required.

Reuse
-----
``PushMessage`` mimics the small slice of the Pub/Sub ``Message`` interface that
``worker.callback`` / ``process_upload_*`` depend on (``.data``, ``.message_id``,
``.ack()``, ``.nack()``, ``.modify_ack_deadline()``) so none of the existing,
heavily-reviewed job logic has to change. The pull-mode CLI (``worker.py``)
keeps working unchanged for local development.
"""

import base64
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker_http")

# "local" = run TensorFlow inside this container; "vertex" = dispatch the job to
# a Vertex AI custom training job. worker.callback() reads this module global.
worker._processing_mode = os.getenv("WORKER_MODE", "local")

# Path Pub/Sub pushes to. Kept configurable so the deploy script and the
# subscription's push endpoint stay in sync via one env var.
PUSH_PATH = os.getenv("PUSH_PATH", "/pubsub/push")


class PushMessage:
    """Adapter exposing the slice of the Pub/Sub ``Message`` API the worker uses.

    In push mode there is no streaming-pull lease to manage: Pub/Sub decides
    redelivery from the HTTP status we return. ``ack()``/``nack()`` therefore
    just record the outcome, which the handler translates into 204/503.
    """

    def __init__(self, data: bytes, message_id: str):
        self.data = data
        self.message_id = message_id
        self.result = None  # "ack" | "nack" | None

    def ack(self) -> None:
        self.result = "ack"

    def nack(self) -> None:
        self.result = "nack"

    def modify_ack_deadline(self, seconds: int) -> None:
        # No-op on purpose. In pull mode worker.AckExtender calls this to keep a
        # long job's lease alive, but Pub/Sub *push* has no API to extend an
        # individual message's deadline -- the deadline is fixed by the
        # subscription's ackDeadlineSeconds (10s default, 600s max) and Pub/Sub
        # decides redelivery purely from the HTTP status we return. See
        # https://cloud.google.com/pubsub/docs/push.
        #
        # Consequence: a job MUST finish within the subscription ack deadline
        # (we set it to the 600s max) or Pub/Sub will redeliver it. Redelivery
        # is *safe* -- the idempotency check + Firestore state machine in
        # worker.callback() drop the duplicate (JobInProgressError -> nack, then
        # a later delivery sees the terminal state and acks) -- but it is
        # wasteful (extra Cloud Run invocations + push backoff). Local in-
        # container training is therefore intended for datasets that train
        # within ~600s (the demo-scale case; a 28k-row job runs in ~110s). For
        # larger/longer jobs set WORKER_MODE=vertex: the push request then only
        # *dispatches* a Vertex AI custom job and returns in seconds, so the
        # ack deadline is never at risk.
        return None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: str = "") -> None:
        payload = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path in ("/healthz", "/"):
            self._send(200, "ok")
        else:
            self._send(404, "not found")

    def do_POST(self) -> None:
        if self.path.rstrip("/") != PUSH_PATH.rstrip("/"):
            self._send(404, "not found")
            return

        # Parse the Pub/Sub push envelope:
        #   {"message": {"data": <base64>, "messageId": "..."}, "subscription": "..."}
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            envelope = json.loads(raw.decode("utf-8"))
            msg = envelope["message"]
            data = base64.b64decode(msg.get("data", "") or "")
            message_id = msg.get("messageId") or msg.get("message_id") or "?"
        except Exception as e:
            # A malformed envelope is not a real Pub/Sub delivery and will never
            # become valid on retry -> ACK so Pub/Sub stops resending it.
            logger.error("Bad push envelope, dropping: %s", e)
            self._send(204)
            return

        push_msg = PushMessage(data, message_id)
        try:
            # All idempotency, schema-validation, poison-drop and state-machine
            # logic lives in worker.callback() and is reused verbatim.
            worker.callback(push_msg)
        except Exception as e:
            # callback() manages its own ack/nack, but guard against anything
            # that escapes so Pub/Sub retries rather than silently dropping.
            logger.exception("Unhandled error in worker.callback: %s", e)
            self._send(503, "retry")
            return

        if push_msg.result == "nack":
            self._send(503, "retry")  # transient failure -> redeliver
        elif push_msg.result == "ack":
            self._send(204)  # success, or a poison message we chose to drop
        else:
            # callback() returned without acking/nacking (shouldn't happen).
            # Treat as transient and let Pub/Sub retry.
            logger.warning("callback() left message %s un-acked; NACKing", message_id)
            self._send(503, "retry")

    def log_message(self, fmt, *args):  # noqa: A003 - override stdlib hook
        # Suppress the default per-request stderr line; we log meaningfully above.
        return


def main() -> None:
    port = int(os.getenv("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    logger.info(
        "worker_http listening on :%s (mode=%s, push_path=%s)",
        port, worker._processing_mode, PUSH_PATH,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
