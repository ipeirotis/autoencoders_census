"""
Pub/Sub Worker - Listens for upload messages and processes survey data.

Supports two modes:
  --mode=local    (default) Process data locally without Vertex AI
  --mode=vertex   Dispatch to Vertex AI CustomContainerTrainingJob

This worker:
1. Subscribes to a Pub/Sub topic for new upload notifications
2. Receives messages containing {jobId, bucket, file} from the Express backend
3. Either processes locally or dispatches to Vertex AI
4. Writes results (outlier scores) to Firestore

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS="frontend/service-account-key.json"
    python worker.py                  # local mode (default)
    python worker.py --mode=vertex    # Vertex AI mode

Required Environment Variables:
    - GOOGLE_CLOUD_PROJECT: GCP project ID
    - GCS_BUCKET_NAME: Storage bucket for uploads
    - PUBSUB_SUBSCRIPTION_ID: Pub/Sub subscription to listen on

Optional Environment Variables (Vertex AI mode only):
    - VERTEX_STAGING_BUCKET: GCS bucket for Vertex AI staging (default: gs://autoencoders-census-staging)
    - VERTEX_SERVICE_ACCOUNT: Service account for Vertex AI jobs (default: Compute Engine default)
"""

import argparse
import os
import sys
import json
import logging
import threading
import time
import io
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import chardet
from google.cloud import pubsub_v1, firestore, storage
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from dataset.loader import DataLoader
from features.transform import Table2Vector

load_dotenv()

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
SUBSCRIPTION_ID = os.getenv("PUBSUB_SUBSCRIPTION_ID")

# Retention window for the idempotency marker in `processed_messages`.
# Used as the `expiresAt` field on each marker so that a Firestore TTL
# policy pointed at `expiresAt` sweeps records only after this many days,
# not immediately. Seven days comfortably exceeds Pub/Sub's 7-day maximum
# retained-message lifetime, so a late redelivery of an already-processed
# message will still find the marker and get ack-dropped.
IDEMPOTENCY_MARKER_TTL_DAYS = 7

# Lease / heartbeat for the per-job `processing` claim. When a worker
# transitions a job out of QUEUED it also writes `claimedAt` to the job
# document, and refreshes that timestamp periodically via the
# AckExtender heartbeat. A duplicate Pub/Sub delivery that arrives while
# the state is still in an in-progress bucket (PROCESSING / TRAINING /
# SCORING) will compare `claimedAt` against this threshold:
#   - fresh (within threshold)  -> raise JobInProgressError (nack, retry)
#   - stale (older than threshold) -> atomically take over the claim
#     and resume processing. Firestore transactions serialize the
#     takeover so two workers cannot both win.
# The threshold must be long enough to survive the worst case where the
# heartbeat thread is briefly blocked (slow GC pause, thread scheduling
# latency during CPU-bound model training) but short enough that a truly
# crashed worker is reclaimed promptly. Three times the heartbeat
# interval is a reasonable compromise.
JOB_CLAIM_HEARTBEAT_INTERVAL_SECONDS = 60
JOB_CLAIM_STALE_SECONDS = JOB_CLAIM_HEARTBEAT_INTERVAL_SECONDS * 3

# Vertex mode note: an earlier version of this file had a separate
# JOB_CLAIM_STALE_SECONDS_VERTEX = 4 hours constant used when a claim
# was tagged `mode: "vertex"`, on the theory that a long-running
# Vertex training job would otherwise be falsely considered stale.
# Codex P1 r(pre-dispatch-vertex) pointed out that this also blocked
# crash recovery for the full 4 hours when a Vertex dispatcher
# crashed BEFORE job.run() ever submitted - which is the more common
# failure mode. The vertex-dispatched guard in
# try_take_over_stale_claim now relies instead on the presence of
# `vertexJobName` on the doc: a populated vertexJobName refuses the
# takeover outright (Vertex owns the job), and an absent vertexJobName
# means pre-dispatch / unconfirmed and uses the short local window
# for fast recovery. The small residual race (crash between
# job.run() returning and the suppressor write) is mitigated by the
# retry loop around the suppressor write in process_upload_vertex.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = firestore.Client(project=PROJECT_ID)


class PubSubMessage(BaseModel):
    """Pydantic model for validating Pub/Sub message payload."""
    jobId: str = Field(..., min_length=1, description="Firestore job document ID")
    bucket: str = Field(..., min_length=1, description="GCS bucket name")
    file: str = Field(..., min_length=1, description="GCS file path")


def validate_message(data) -> PubSubMessage:
    """
    Validate message fields, raise ValueError with clear error.

    Args:
        data: Parsed JSON payload. Normally a mapping, but we tolerate
            any type and translate non-mapping inputs (list, None, str,
            int, etc.) into a ValueError so callback() can drop them as
            poison messages via its existing ack-on-ValueError path.

    Returns:
        PubSubMessage: Validated message object

    Raises:
        ValueError: If validation fails with description of missing/invalid fields
    """
    # Codex P2 r(non-object-json): PubSubMessage(**data) raises TypeError
    # when data is valid JSON but not a mapping (for example [], null,
    # a JSON string, or a number). callback() ack-drops ValueError but
    # nacks on bare TypeError, which would create an avoidable
    # redelivery loop against a deterministically bad payload. Map
    # non-mapping inputs to ValueError up front so they get the same
    # poison-message treatment as schema failures.
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid message format: expected JSON object, got "
            f"{type(data).__name__}"
        )
    try:
        return PubSubMessage(**data)
    except ValidationError as e:
        errors = '; '.join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
        raise ValueError(f"Invalid message format: {errors}")
    except TypeError as e:
        # Pydantic can also raise TypeError for unexpected kwargs
        # (e.g. {"jobId": 1, "bucket": "b", "file": "f", "extra": 2})
        # depending on config. Normalize to ValueError so the poison
        # path handles it.
        raise ValueError(f"Invalid message format: {e}")


class JobDocumentNotReadyError(Exception):
    """
    Raised when the worker tries to update a job document that has not
    yet been written by /start-job (pub/sub publish beat the Firestore
    create on a legacy/out-of-order code path). The outer callback()
    nacks the message on this exception so Pub/Sub redelivers and the
    retry can find the document.
    """


class JobInProgressError(Exception):
    """
    Raised when a duplicate Pub/Sub delivery arrives while another worker
    is actively processing the same job (the job document is in a
    non-terminal in-progress state - PROCESSING / TRAINING / SCORING -
    but no idempotency marker has been written yet because the original
    worker has not finished).

    The outer callback() nacks the message on this exception so Pub/Sub
    will redeliver later. Crucially, it does NOT write the idempotency
    marker in this case: if the original worker later crashes and
    leaves the job stuck in an in-progress state, a subsequent
    redelivery must still be able to retry. Acking (and marking) a
    duplicate that overlapped with an in-flight run would permanently
    hide that recovery path - Codex P1 r3055316xxx.
    """


def check_idempotency(message_id: str) -> bool:
    """
    Read-only check: has this Pub/Sub message already been marked processed?

    Returns True if the message was previously processed and should be
    skipped, False if it has not yet been processed.

    IMPORTANT: This function does NOT write anything. The marker is set by
    :func:`mark_message_processed` *after* job execution completes. That
    ordering prevents the "silently dropped message" failure mode: if a
    worker crashed between marking and acknowledgment, Pub/Sub would
    redeliver the message and a "mark-before-process" implementation would
    see the marker and ack the redelivered copy without performing any
    work, leaving the job stuck forever.

    Args:
        message_id: Unique Pub/Sub message ID

    Returns:
        bool: True if message was already processed, False if first time
    """
    processed_ref = db.collection('processed_messages').document(message_id)
    snapshot = processed_ref.get()
    return snapshot.exists


def mark_message_processed(message_id: str, job_id: str) -> None:
    """
    Record that a Pub/Sub message has been successfully processed.

    Must be called only AFTER :func:`process_upload_local` /
    :func:`process_upload_vertex` returns, so that a crash mid-processing
    leaves the marker unset and Pub/Sub redelivery triggers a real retry
    instead of a silent drop.

    Uses a Firestore transaction with a "set-if-not-exists" check to stay
    idempotent if two workers ever race to mark the same message.

    Args:
        message_id: Unique Pub/Sub message ID
        job_id: Job ID from message payload (for bookkeeping/cleanup)
    """
    processed_ref = db.collection('processed_messages').document(message_id)

    # Codex P2 (r3055203157): expiresAt must be a future timestamp (now + TTL),
    # not SERVER_TIMESTAMP. If a Firestore TTL policy is configured on this
    # field it would otherwise delete the marker the instant it is written,
    # leaving no deduplication cover for late redeliveries of the same
    # message ID.
    expires_at = datetime.now(timezone.utc) + timedelta(
        days=IDEMPOTENCY_MARKER_TTL_DAYS
    )

    @firestore.transactional
    def _set_if_not_exists(transaction, ref):
        snapshot = ref.get(transaction=transaction)
        if snapshot.exists:
            return  # Already marked by another worker - idempotent no-op

        transaction.set(ref, {
            'jobId': job_id,
            'processedAt': firestore.SERVER_TIMESTAMP,
            'expiresAt': expires_at,  # now + IDEMPOTENCY_MARKER_TTL_DAYS
        })

    transaction = db.transaction()
    _set_if_not_exists(transaction, processed_ref)


class AckExtender:
    """
    Periodically extends Pub/Sub message ack deadline during long-running jobs.

    Prevents message timeout and redelivery for jobs that take 10-15 minutes
    (longer than the default 10-second ack deadline). Uses threading.Timer
    to extend deadline every 60 seconds.

    Also refreshes the `claimedAt` heartbeat on the job document in the
    same cadence (Codex P1 r(stale-takeover)). A duplicate Pub/Sub
    delivery that arrives while the state is still PROCESSING / TRAINING
    / SCORING uses this timestamp to tell "another worker is alive and
    working" (don't take over) apart from "the claiming worker has
    crashed and left the job stuck" (stale -> take over). Without the
    heartbeat the worker would have to pick a single hard-coded "stale
    after N minutes" threshold that either reclaims too aggressively
    (killing a legitimately slow job) or leaves crashed jobs wedged for
    too long.

    WORK-05: Ack deadline extension pattern
    """
    def __init__(
        self,
        message,
        interval_seconds=JOB_CLAIM_HEARTBEAT_INTERVAL_SECONDS,
        job_ref=None,
    ):
        """
        Initialize AckExtender.

        Args:
            message: Pub/Sub message object with modify_ack_deadline() method
            interval_seconds: How often to extend deadline and refresh the
                job claim heartbeat (default: 60 seconds)
            job_ref: Optional Firestore DocumentReference for the job. When
                provided, every extension tick also updates `claimedAt` on
                the job doc so a concurrent duplicate delivery can tell the
                claim is still live. Best-effort: refresh failures are
                logged but do not abort the job.
        """
        self.message = message
        self.interval = interval_seconds
        self.job_ref = job_ref
        self.timer = None
        self.stopped = False

    def _refresh_claimed_at(self):
        """Best-effort heartbeat write of `claimedAt` on the job doc."""
        if self.job_ref is None:
            return
        try:
            self.job_ref.update(
                {'claimedAt': datetime.now(timezone.utc)}
            )
        except Exception as e:
            # Don't fail the job because of a heartbeat hiccup - a stale
            # takeover would only happen if the claim goes unrefreshed
            # for 3x the heartbeat interval, which requires multiple
            # consecutive refresh failures.
            logger.warning(
                f"Failed to refresh claimedAt heartbeat for "
                f"{getattr(self.job_ref, 'id', '?')}: {e}"
            )

    def extend(self):
        """Extend deadline and schedule next extension."""
        if not self.stopped:
            try:
                # Extend deadline by interval + 10 second buffer
                self.message.modify_ack_deadline(self.interval + 10)
                logger.info(f"Extended ack deadline for message {self.message.message_id}")
            except Exception as e:
                # CONTEXT.md: Log warning and continue (rely on idempotency)
                logger.warning(f"Failed to extend ack deadline: {e}")

            # Heartbeat the job claim at the same cadence so stale-takeover
            # can correctly distinguish "worker still running" from
            # "worker crashed".
            self._refresh_claimed_at()

            # Schedule next extension
            self.timer = threading.Timer(self.interval, self.extend)
            self.timer.daemon = True
            self.timer.start()

    def start(self):
        """Start periodic extension."""
        self.extend()

    def stop(self):
        """Stop extension and cancel timer."""
        self.stopped = True
        if self.timer:
            self.timer.cancel()


# Job Status State Machine (WORK-08)
from enum import Enum

class JobStatus(str, Enum):
    """Job status values with explicit state machine."""
    QUEUED = "queued"
    PROCESSING = "processing"
    TRAINING = "training"
    SCORING = "scoring"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELED = "canceled"


# TASKS.md 2.3: Structured error reporting.
# When a job fails, the worker writes a classified error payload to Firestore
# so the frontend can surface a clear, user-facing message instead of a raw
# Python traceback. Each failure has a stable `errorCode` (machine-readable,
# safe to branch on in the UI or in tests) and an `errorType` bucket that
# groups codes into the pipeline stage where the problem surfaced
# (validation of the uploaded CSV, data loading, model training, scoring,
# or an unclassified internal bug).
class ErrorType(str, Enum):
    """High-level bucket for the pipeline stage where an error occurred."""
    VALIDATION = "validation"   # CSV failed pre-processing validation
    PROCESSING = "processing"   # DataLoader / cleaning / vectorization
    TRAINING = "training"       # Model training
    SCORING = "scoring"         # Reconstruction / outlier scoring
    INTERNAL = "internal"       # Unclassified / unexpected


class ErrorCode(str, Enum):
    """Stable machine-readable error codes for the upload pipeline."""
    # Validation (pre-load)
    CSV_TOO_LARGE = "csv_too_large"
    CSV_EMPTY = "csv_empty"
    CSV_ENCODING = "csv_encoding"
    CSV_PARSE = "csv_parse"
    CSV_INCONSISTENT_COLUMNS = "csv_inconsistent_columns"
    CSV_TOO_FEW_ROWS = "csv_too_few_rows"
    CSV_TOO_FEW_COLUMNS = "csv_too_few_columns"

    # Processing (post-load, pre-train)
    LOAD_FAILURE = "load_failure"
    NO_USABLE_COLUMNS = "no_usable_columns"

    # Later pipeline stages
    TRAINING_FAILURE = "training_failure"
    SCORING_FAILURE = "scoring_failure"

    # Catch-all
    INTERNAL_ERROR = "internal_error"


class UploadValidationError(ValueError):
    """
    User-facing validation error with a stable error code.

    Subclasses ``ValueError`` so that existing ``except ValueError`` /
    ``pytest.raises(ValueError, ...)`` call sites (notably
    ``tests/test_csv_validation.py``) continue to work unchanged. Adds an
    ``error_code`` attribute (an :class:`ErrorCode` value) so the worker can
    route the error into a structured Firestore payload via
    :func:`mark_job_error`, and an ``error_type`` that defaults to
    ``VALIDATION`` but can be overridden for post-load failures that still
    want to reuse the same structured path (e.g. ``NO_USABLE_COLUMNS`` is
    raised from ``PROCESSING``).
    """
    def __init__(self, message, error_code, error_type=ErrorType.VALIDATION):
        super().__init__(message)
        self.error_code = error_code
        self.error_type = error_type


def mark_job_error(
    job_ref,
    job_id,
    message,
    error_code,
    error_type=ErrorType.PROCESSING,
):
    """
    Write a structured error state to the job's Firestore document.

    Args:
        job_ref: Firestore DocumentReference for the job.
        job_id: Job ID (for logging).
        message: Human-readable error message suitable for display.
        error_code: :class:`ErrorCode` value or plain string. Stored as a
            string so the client can branch on it.
        error_type: :class:`ErrorType` value or plain string. Defaults to
            PROCESSING since most post-validation failures land there.

    Best-effort: if the transition to ERROR is invalid (e.g. the job is
    already in a terminal state) the failure is logged but not re-raised.
    """
    code_value = (
        error_code.value if isinstance(error_code, ErrorCode) else str(error_code)
    )
    type_value = (
        error_type.value if isinstance(error_type, ErrorType) else str(error_type)
    )
    transaction = db.transaction()
    try:
        update_job_status(
            transaction,
            job_ref,
            JobStatus.ERROR,
            {
                'error': message,
                'errorCode': code_value,
                'errorType': type_value,
            },
        )
    except ValueError as transition_err:
        # Job may be in a terminal state already (e.g. CANCELED) - log and
        # move on; the real error is already captured in the caller's log.
        logger.error(
            f"Job {job_id} failed ({code_value}: {message}) but could not "
            f"update status: {transition_err}"
        )


# Define allowed transitions (WORK-08)
ALLOWED_TRANSITIONS = {
    JobStatus.QUEUED: [JobStatus.PROCESSING, JobStatus.ERROR, JobStatus.CANCELED],
    JobStatus.PROCESSING: [JobStatus.TRAINING, JobStatus.ERROR, JobStatus.CANCELED],
    JobStatus.TRAINING: [JobStatus.SCORING, JobStatus.ERROR, JobStatus.CANCELED],
    JobStatus.SCORING: [JobStatus.COMPLETE, JobStatus.ERROR, JobStatus.CANCELED],
    JobStatus.COMPLETE: [],  # Terminal state
    JobStatus.ERROR: [],     # Terminal state
    JobStatus.CANCELED: []   # Terminal state
}

# Legacy job statuses written by the pre-state-machine Express API before
# phase-01/phase-02 migrated `/start-job` and `/api/upload` to persist the
# canonical JobStatus.QUEUED value. Any job document currently sitting in
# one of these statuses was created by older client/server code and should
# be treated as "equivalent to queued" by the worker when it picks the
# message up - otherwise the state machine rejects the
# legacy -> PROCESSING transition, callback() silently acks, and the job
# is permanently dropped (Codex P2 r(legacy-status)).
LEGACY_INITIAL_STATUSES = frozenset({"uploaded", "uploading"})


def is_valid_transition(current_status, new_status):
    """
    Validate state transition is allowed.

    Args:
        current_status: Current job status (JobStatus enum, string, or None)
        new_status: New job status (JobStatus enum or string)

    Returns:
        bool: True if transition is valid, False otherwise
    """
    if current_status is None:
        # First status update must be QUEUED
        try:
            new = JobStatus(new_status)
            return new == JobStatus.QUEUED
        except ValueError:
            return False

    try:
        current = JobStatus(current_status)
        new = JobStatus(new_status)
    except ValueError:
        # Unknown status value
        return False

    return new in ALLOWED_TRANSITIONS.get(current, [])


@firestore.transactional
def update_job_status(transaction, job_ref, new_status, additional_fields=None):
    """
    Atomically update job status with validation.
    Firestore automatically retries on contention.

    Args:
        transaction: Firestore transaction object
        job_ref: DocumentReference to job document
        new_status: New status value (JobStatus enum or string)
        additional_fields: Optional dict of additional fields to update

    Raises:
        ValueError: If job not found or invalid transition
    """
    snapshot = job_ref.get(transaction=transaction)

    if not snapshot.exists:
        raise ValueError(f"Job {job_ref.id} not found")

    current_status = snapshot.get('status')

    # WORK-08: Validate state transitions
    if not is_valid_transition(current_status, new_status):
        raise ValueError(f"Invalid transition: {current_status} -> {new_status}")

    update_data = {'status': new_status}
    if additional_fields:
        update_data.update(additional_fields)

    transaction.update(job_ref, update_data)
    new_val = new_status.value if isinstance(new_status, JobStatus) else new_status
    logger.info(f"Job {job_ref.id} status: {current_status} -> {new_val}")


def _is_claim_stale(claimed_at, now=None, threshold_seconds=None):
    """
    Decide whether a job's `claimedAt` timestamp is stale (older than
    `threshold_seconds`), meaning the claiming worker's heartbeat has
    gone silent long enough that we should treat the claim as abandoned.

    Accepts timezone-aware or naive datetimes and tolerates `None`
    (missing field) by treating it as stale so legacy jobs written
    before this field existed can still be recovered.

    Args:
        claimed_at: The stored claimedAt timestamp (or None).
        now: Current time (defaults to datetime.now(utc)). Injectable
            for testing.
        threshold_seconds: The staleness threshold. Defaults to
            JOB_CLAIM_STALE_SECONDS.
    """
    if claimed_at is None:
        return True
    if now is None:
        now = datetime.now(timezone.utc)
    if threshold_seconds is None:
        threshold_seconds = JOB_CLAIM_STALE_SECONDS
    # Firestore returns timezone-aware datetimes; be defensive if we
    # see a naive value (tests, legacy data) by assuming UTC.
    if claimed_at.tzinfo is None:
        claimed_at = claimed_at.replace(tzinfo=timezone.utc)
    age = (now - claimed_at).total_seconds()
    return age > threshold_seconds


def try_take_over_stale_claim(job_ref, mode=None):
    """
    Attempt to atomically reclaim a stale in-progress job.

    Codex P1 r(stale-takeover): if a worker crashed after taking the
    queued -> processing claim but before writing a terminal state,
    subsequent redeliveries of the same Pub/Sub message will see the
    doc in an in-progress state and a naive implementation would nack
    forever (infinite redelivery loop, stuck job). This function gives
    us an escape hatch: a duplicate delivery that finds the claim
    abandoned (no heartbeat refresh for more than the mode-specific
    stale threshold) transactionally bumps `claimedAt` to now and
    returns True, letting the caller resume processing.

    Codex P1 (followup, r(stale-takeover-from-training)): the takeover
    also *resets* the status to PROCESSING, even if the current state
    is TRAINING or SCORING. Without that reset, the caller's normal
    pipeline would try a PROCESSING -> TRAINING transition and hit an
    "Invalid transition: training -> training" (or scoring -> training)
    ValueError, which the outer exception handler would then mark as
    ERROR. Crash recovery from TRAINING or SCORING would deterministically
    fail instead of restarting the pipeline. Resetting to PROCESSING
    means we redo the work (download, train, score) from scratch - that
    is the safe choice because we have no way to know how far the dead
    worker actually got before it crashed.

    Codex P1 (followup, r(vertex-stale-redispatch)): suppress takeover
    when the job doc carries a `vertexJobName`, because that indicates
    a Vertex training job has already been successfully submitted and
    Vertex itself (not this worker) owns the work. Re-taking it over
    would re-submit the same Vertex training run and double-bill.

    Codex P1 (followup, r(pre-dispatch-vertex)): the guard against
    duplicate Vertex dispatch is the `vertexJobName` field only. The
    earlier `mode: "vertex"` + 4-hour window scheme was too coarse:
    it delayed recovery of a dispatcher that crashed BEFORE
    job.run() submitted anything, for the full 4 hours. Now the only
    condition that refuses a stale takeover on a vertex-mode doc is
    `vertexJobName` being set; absent that, the short local window
    applies and a pre-dispatch crash recovers in minutes. The small
    residual race between "job.run() returned" and "suppressor write
    committed" is mitigated by the retry loop around the suppressor
    write in process_upload_vertex.

    Codex P2 (followup, r(takeover-mode-tag)): when the caller knows
    which mode it is going to run the job in (local vs vertex), it
    can pass `mode=` so the takeover transaction ALSO stamps that
    tag on the recovered claim. The tag is mainly operator-facing
    (audit / debugging) now that the stale threshold is uniform.

    Args:
        job_ref: Firestore DocumentReference for the job.
        mode: Optional "vertex" | "local" tag to persist on the
            recovered doc as part of the takeover transaction. When
            omitted, `mode` is left untouched on the doc (existing
            value preserved, legacy docs stay tagless).

    The refresh happens inside a Firestore transaction, so two workers
    that both see a stale claim cannot both take over: one wins the
    transaction, the other retries, finds the now-fresh `claimedAt`
    and returns False.

    Returns:
        bool: True if the claim was successfully taken over, False if
        the claim is still fresh, the doc is not in a recoverable
        in-progress state, or a Vertex job is already dispatched.
    """
    in_progress_states = {
        JobStatus.PROCESSING.value,
        JobStatus.TRAINING.value,
        JobStatus.SCORING.value,
    }

    @firestore.transactional
    def _takeover(transaction, ref):
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            return False
        current_status = snapshot.get('status')
        if current_status not in in_progress_states:
            return False
        # Vertex-dispatched guard (Codex P1 r(vertex-stale-redispatch)):
        # once the dispatcher has actually handed the job off to
        # Vertex (proved by vertexJobName being persisted), Vertex
        # itself owns the work. Re-taking over would re-submit the
        # same (billable) training run, so refuse the takeover
        # outright regardless of how old the claimedAt heartbeat is.
        if snapshot.get('vertexJobName'):
            return False
        # No vertexJobName means either:
        #   (a) this is a local-mode job, or
        #   (b) a Vertex-mode dispatcher claimed the doc but either
        #       crashed before job.run() submitted anything, or crashed
        #       in the brief window after job.run() returned but before
        #       the suppressor write committed.
        #
        # Codex P1 r(pre-dispatch-vertex): the previous "always use the
        # 4-hour Vertex window when mode is vertex" heuristic was too
        # coarse - it made case (b)-without-suppressor-write safer but
        # at the cost of pinning case (a) / case (b)-crashed-before-
        # dispatch for four hours on every crash. The dispatcher is a
        # fire-and-forget call that normally takes seconds, so a crash
        # there should recover promptly. Use the short local window in
        # all non-dispatched cases; the rare "suppressor write failed
        # moments after job.run() succeeded" case is mitigated by the
        # retry loop on the suppressor write itself (see
        # process_upload_vertex below).
        claimed_at = snapshot.get('claimedAt')
        if not _is_claim_stale(claimed_at):
            return False
        update_payload = {
            # Reset status back to PROCESSING so the calling pipeline
            # can restart from the beginning regardless of which
            # in-progress sub-state (processing/training/scoring) the
            # dead worker crashed in. This is a direct write that
            # bypasses the normal state machine transition check,
            # which is the whole point of this escape hatch.
            'status': JobStatus.PROCESSING.value,
            'claimedAt': datetime.now(timezone.utc),
        }
        # Codex P2 r(takeover-mode-tag): persist the mode tag on
        # recovered claims so subsequent duplicate deliveries use the
        # correct stale threshold even if the post-dispatch suppressor
        # write later fails.
        if mode is not None:
            update_payload['mode'] = mode
        transaction.update(ref, update_payload)
        return True

    transaction = db.transaction()
    took_over = _takeover(transaction, job_ref)
    if took_over:
        logger.warning(
            f"Stale claim taken over for job {getattr(job_ref, 'id', '?')} "
            f"(previous worker appears to have crashed); status reset to "
            f"PROCESSING to restart the pipeline"
        )
    return took_over


def validate_environment():
    """Validate required environment variables. Exits if any are missing."""
    # Read environment variables fresh each time (for testability)
    required_vars = {
        'GOOGLE_CLOUD_PROJECT': os.getenv("GOOGLE_CLOUD_PROJECT"),
        'GCS_BUCKET_NAME': os.getenv("GCS_BUCKET_NAME"),
        'PUBSUB_SUBSCRIPTION_ID': os.getenv("PUBSUB_SUBSCRIPTION_ID"),
    }

    missing = [name for name, value in required_vars.items() if not value]

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Set these variables before starting the worker.")
        sys.exit(1)

    logger.info("Environment validation passed.")
    return True


def validate_csv(csv_bytes, max_size_mb=100):
    """
    Validate CSV encoding, structure, and size.
    Returns (encoding, column_count, row_count) or raises
    :class:`UploadValidationError` (a ``ValueError`` subclass).

    This function implements defense-in-depth CSV validation to catch
    encoding errors, malformed CSVs, and edge cases before training starts.

    WORK-09: Encoding detection using chardet
    WORK-10: Structure validation (inconsistent row lengths)
    WORK-11: Size limit enforcement (>100MB)
    WORK-13: Edge case handling (unicode characters)
    WORK-14: Edge case handling (mostly-missing values)

    TASKS.md 2.3: Every failure path raises :class:`UploadValidationError`
    with a stable :class:`ErrorCode`, so ``process_upload_local`` can write a
    structured error payload to Firestore and the frontend can display a
    clear, user-facing message without leaking internal error strings or
    stack traces. The exception still inherits from ``ValueError`` so
    existing ``pytest.raises(ValueError, match=...)`` call sites continue
    to work.

    Args:
        csv_bytes: Raw CSV file bytes
        max_size_mb: Maximum file size in MB (default: 100)

    Returns:
        tuple: (encoding, column_count, row_count)

    Raises:
        UploadValidationError: If file too large, encoding unclear,
            structure invalid, or edge cases. Each failure path sets a
            specific :class:`ErrorCode`.
    """
    # WORK-11: Check size limit
    size_mb = len(csv_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise UploadValidationError(
            f"CSV file too large: {size_mb:.1f}MB (max {max_size_mb}MB)",
            error_code=ErrorCode.CSV_TOO_LARGE,
        )

    # WORK-09: Detect encoding. Try the chardet guess first, then fall back
    # through a small list of Windows/Western European codecs before finally
    # trying UTF-8. Doing the fallback as a candidate list (rather than just
    # jumping to UTF-8 on low confidence) keeps real cp1252 files (smart
    # quotes, em dashes, Windows Excel exports) decodable even when chardet
    # hedges with a low-confidence Mac-* guess.
    detection = chardet.detect(csv_bytes[:100000])  # Sample first 100KB
    detected_encoding = detection.get('encoding')
    detected_confidence = detection.get('confidence') or 0.0

    # Only trust chardet's guess when it is confident. On low-confidence
    # guesses (e.g. "MacLatin2" with 0.03 for a straight cp1252 file)
    # we skip the guess entirely and fall through to the well-known
    # Western European codecs - cp1252 is the real answer for almost
    # every Excel-exported CSV on Windows, so trying it explicitly is
    # more reliable than whatever obscure codec chardet picked.
    candidate_encodings = []
    if detected_encoding and detected_confidence >= 0.7:
        candidate_encodings.append(detected_encoding)
    else:
        logger.warning(
            f"Low encoding confidence: {detected_confidence:.2f} for "
            f"{detected_encoding!r}, skipping the guess and trying "
            f"standard fallback encodings"
        )
    for fallback in ("utf-8", "cp1252", "latin-1"):
        if fallback not in candidate_encodings:
            candidate_encodings.append(fallback)

    # WORK-10, WORK-13, WORK-14: Validate structure with streaming
    last_unicode_err = None
    for candidate in candidate_encodings:
        try:
            chunk_iterator = pd.read_csv(
                io.BytesIO(csv_bytes),
                encoding=candidate,
                chunksize=10000,
                engine='python'       # Better error messages, detects inconsistent columns
            )

            first_chunk = next(chunk_iterator)
            expected_columns = len(first_chunk.columns)
            total_rows = len(first_chunk)

            # Validate subsequent chunks have same structure
            for chunk in chunk_iterator:
                if len(chunk.columns) != expected_columns:
                    raise UploadValidationError(
                        f"Inconsistent column count: expected {expected_columns}, "
                        f"got {len(chunk.columns)}",
                        error_code=ErrorCode.CSV_INCONSISTENT_COLUMNS,
                    )
                total_rows += len(chunk)

            # Minimum data requirements
            if total_rows < 10:
                raise UploadValidationError(
                    f"CSV must have at least 10 rows (found {total_rows})",
                    error_code=ErrorCode.CSV_TOO_FEW_ROWS,
                )
            if expected_columns < 2:
                raise UploadValidationError(
                    f"CSV must have at least 2 columns (found {expected_columns})",
                    error_code=ErrorCode.CSV_TOO_FEW_COLUMNS,
                )

            logger.info(
                f"CSV validation passed: {total_rows} rows, {expected_columns} columns, "
                f"{candidate} encoding"
            )
            return candidate, expected_columns, total_rows

        except UnicodeDecodeError as e:
            last_unicode_err = e
            logger.info(
                f"Candidate encoding {candidate} rejected by read_csv: {e}; "
                f"trying next candidate"
            )
            continue
        except pd.errors.ParserError as e:
            raise UploadValidationError(
                f"CSV parsing error: {str(e)}",
                error_code=ErrorCode.CSV_PARSE,
            )
        except pd.errors.EmptyDataError:
            raise UploadValidationError(
                "CSV file is empty",
                error_code=ErrorCode.CSV_EMPTY,
            )
        except StopIteration:
            raise UploadValidationError(
                "CSV file is empty",
                error_code=ErrorCode.CSV_EMPTY,
            )

    # All candidate encodings failed with UnicodeDecodeError.
    raise UploadValidationError(
        f"Encoding error: no candidate encoding from {candidate_encodings} "
        f"could decode the CSV ({last_unicode_err})",
        error_code=ErrorCode.CSV_ENCODING,
    )


def is_job_canceled(job_id):
    """Quick Firestore read to check whether the job has been canceled.

    Fails closed: transient Firestore errors return False (not canceled)
    rather than propagating and crashing the caller's training loop.
    """
    try:
        snap = db.collection('jobs').document(job_id).get()
        if snap.exists:
            return snap.get('status') == JobStatus.CANCELED
        return False
    except Exception as e:
        logger.warning(f"is_job_canceled: Firestore read failed for {job_id}, "
                       f"assuming not canceled: {e}")
        return False


def _safe_float(value, default=0.0):
    """Convert value to float, returning default for non-numeric inputs."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def process_upload_local(job_id, bucket_name, file_path, message):
    """
    Process the uploaded CSV locally: download from GCS, train autoencoder,
    score outliers, and write results to Firestore.

    Args:
        job_id: Firestore job document ID
        bucket_name: GCS bucket name
        file_path: GCS file path
        message: Pub/Sub message object for ack deadline extension
    """
    # Lazy imports - TF must not be loaded at module scope
    from model.autoencoder import AutoencoderModel
    from evaluate.outliers import compute_per_column_contributions
    import tensorflow as tf

    job_ref = db.collection('jobs').document(job_id)

    # WORK-05/06: Start ack deadline extension, but *without* a job_ref
    # yet so the heartbeat only refreshes the Pub/Sub ack deadline at
    # this stage and does NOT refresh `claimedAt`. Codex P1 flagged that
    # refreshing the claim before we actually own it rewrites a
    # crashed-worker's stale claim to "fresh" on every redelivery, so
    # try_take_over_stale_claim would then always reject takeover and
    # crash recovery would be impossible. We attach the job_ref below
    # AFTER the queued -> processing transition (or stale takeover)
    # confirms that we are the rightful owner of this job.
    extender = AckExtender(
        message,
        interval_seconds=JOB_CLAIM_HEARTBEAT_INTERVAL_SECONDS,
        job_ref=None,
    )
    extender.start()

    try:
        logger.info(f"Starting local processing for job {job_id}")

        # Update status to processing using transaction (WORK-07).
        #
        # Codex P1 (r3053739500): If the worker races ahead of /start-job's
        # Firestore write and the job document does not yet exist, re-raise
        # so the outer callback() catches it and nacks the message for
        # Pub/Sub redelivery instead of silently dropping the job. With
        # phase-01's "create-then-publish" ordering in jobs.ts this race
        # should not happen in normal production traffic, but the defensive
        # branch keeps any legacy fallback or out-of-order write from
        # wedging a real user job.
        #
        # Also writes `claimedAt` alongside the status so a later
        # duplicate delivery can distinguish "another worker is actively
        # processing" from "a previous worker crashed mid-processing"
        # (see JobInProgressError + try_take_over_stale_claim below).
        try:
            transaction = db.transaction()
            update_job_status(
                transaction,
                job_ref,
                JobStatus.PROCESSING,
                additional_fields={'claimedAt': datetime.now(timezone.utc)},
            )
        except ValueError as e:
            if "not found" in str(e):
                # Bubble up so callback() nacks and we retry on redelivery
                # - do NOT let the outer try/except in this function catch it,
                # because that branch writes status=ERROR, which would stick
                # if the document later shows up.
                raise JobDocumentNotReadyError(
                    f"Job {job_id} document not yet written, retry via nack"
                ) from e

            # The transition was rejected because the job is in a state
            # other than QUEUED. Split by category:
            #
            #   - Terminal states (COMPLETE, ERROR, CANCELED): the work is
            #     done by another worker, safe to let callback() ack and
            #     mark the message as processed. Just return.
            #
            #   - In-progress states (PROCESSING, TRAINING, SCORING):
            #     There are two sub-cases, distinguished by the `claimedAt`
            #     heartbeat maintained by the AckExtender on the job doc:
            #
            #     a) Heartbeat fresh: another worker is actively running.
            #        Raise JobInProgressError so callback() nacks the
            #        duplicate delivery (without marking it processed),
            #        preserving the redelivery path if that worker later
            #        crashes (Codex P1 r3055316xxx).
            #
            #     b) Heartbeat stale: the claiming worker crashed after
            #        taking the claim but before reaching a terminal
            #        state. Without this escape hatch the state machine
            #        would forever reject in-progress -> processing, and
            #        every redelivery would nack, looping indefinitely
            #        (Codex P1 r(stale-takeover)). try_take_over_stale_claim
            #        transactionally refreshes `claimedAt` to now; if it
            #        wins the race we resume processing from the current
            #        in-progress state (the state machine happily accepts
            #        PROCESSING -> TRAINING -> SCORING -> COMPLETE from
            #        wherever we are now).
            #
            #   - Anything else (unexpected): log and return without
            #     raising. callback() will ack and the operator will
            #     see the unusual state in the logs.
            try:
                snapshot = job_ref.get()
                current_status = (
                    snapshot.get('status') if snapshot.exists else None
                )
            except Exception as read_err:
                logger.warning(
                    f"Failed to read job {job_id} state after invalid "
                    f"transition ({e}); treating as in-progress retry: "
                    f"{read_err}"
                )
                raise JobInProgressError(
                    f"Job {job_id} state unreadable after "
                    f"invalid-transition; retrying via nack"
                ) from e

            terminal_states = {
                JobStatus.COMPLETE.value,
                JobStatus.ERROR.value,
                JobStatus.CANCELED.value,
            }
            in_progress_states = {
                JobStatus.PROCESSING.value,
                JobStatus.TRAINING.value,
                JobStatus.SCORING.value,
            }
            if current_status in terminal_states:
                logger.info(
                    f"Job {job_id} already in terminal state "
                    f"{current_status}, skipping duplicate delivery"
                )
                return
            if current_status in in_progress_states:
                # Codex P2 r(vertex-dispatched-ack): if the doc carries
                # `vertexJobName`, the previous run has already handed
                # the job off to Vertex and Vertex owns it. A duplicate
                # /start-job republish (or Pub/Sub redelivery) should
                # NOT bounce around in an infinite
                # JobInProgressError -> nack -> redeliver loop; there
                # is no local work to do here, Vertex itself will
                # update the job status. Treat this as a clean skip
                # (return) so callback() acks and marks the duplicate
                # message as processed.
                if snapshot.get('vertexJobName'):
                    logger.info(
                        f"Job {job_id} already dispatched to Vertex "
                        f"(vertexJobName set), skipping duplicate local "
                        f"delivery"
                    )
                    return

                # Stale-takeover path: if the previous worker's heartbeat
                # has gone silent long enough that we consider the claim
                # abandoned, reclaim it and fall through to continue
                # processing instead of nacking. Pass mode="local" so
                # the takeover transaction keeps/stamps the tag
                # consistent with the caller.
                if try_take_over_stale_claim(job_ref, mode='local'):
                    logger.info(
                        f"Resuming job {job_id} from state "
                        f"{current_status} after stale-claim takeover"
                    )
                    # Intentionally do NOT raise here: fall through past
                    # the except block so the rest of process_upload_local
                    # runs against the existing in-progress state.
                else:
                    raise JobInProgressError(
                        f"Job {job_id} is already {current_status} with a "
                        f"fresh heartbeat; duplicate delivery nacked so a "
                        f"later redelivery can retry if the active worker "
                        f"fails"
                    ) from e
            elif current_status in LEGACY_INITIAL_STATUSES:
                # Legacy-initial-status migration path (Codex P2
                # r(legacy-status)): the doc was created by the
                # pre-state-machine Express API (or the /api/upload
                # fallback) with status="uploaded"/"uploading" instead of
                # the canonical "queued". update_job_status() just rejected
                # the transition because the state machine does not know
                # these legacy values. Without this branch, callback() would
                # ack the message and the job would be dropped forever.
                #
                # Migrate in place: force-write PROCESSING + claimedAt via
                # a fresh transaction that bypasses is_valid_transition,
                # then fall through to continue processing. Use a plain
                # transaction.update() (not update_job_status) so we are
                # not blocked by the state machine on the legacy->
                # processing step.
                logger.info(
                    f"Job {job_id} in legacy initial status "
                    f"{current_status!r}; migrating to PROCESSING and "
                    f"resuming"
                )
                migration_txn = db.transaction()

                @firestore.transactional
                def _migrate_legacy(txn):
                    snap = job_ref.get(transaction=txn)
                    if not snap.exists:
                        raise JobDocumentNotReadyError(
                            f"Job {job_id} disappeared during legacy "
                            f"migration"
                        )
                    latest = snap.get('status')
                    # If another worker (or a catch-up /start-job call)
                    # already migrated the status out from under us,
                    # just no-op: the normal in-progress branch above
                    # will handle the next redelivery.
                    if latest not in LEGACY_INITIAL_STATUSES:
                        return
                    txn.update(
                        job_ref,
                        {
                            'status': JobStatus.PROCESSING.value,
                            'claimedAt': datetime.now(timezone.utc),
                        },
                    )

                _migrate_legacy(migration_txn)
                logger.info(
                    f"Job {job_id} status: {current_status} -> "
                    f"{JobStatus.PROCESSING.value} (legacy migration)"
                )
                # Fall through past the except block to continue
                # processing against the migrated doc.
            else:
                logger.warning(
                    f"Job {job_id} in unexpected state "
                    f"{current_status!r} after failed PROCESSING transition "
                    f"({e}); acking and skipping"
                )
                return

        # Ownership is now confirmed (either via the normal queued ->
        # processing transition, or via try_take_over_stale_claim). Wire
        # the job_ref onto the heartbeat extender now so subsequent
        # extend() ticks will refresh `claimedAt`. We intentionally did
        # NOT do this earlier, because the extender's auto-refresh would
        # have rewritten a crashed worker's stale claim to "fresh"
        # before we got a chance to run stale-takeover, making crash
        # recovery impossible (Codex P1 r3055xxxxx).
        extender.job_ref = job_ref

        # 1. Download from GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        csv_bytes = blob.download_as_bytes()

        # WORK-09/10/11/13/14: Validate CSV before processing.
        # TASKS.md 2.3: UploadValidationError carries a stable error_code
        # that is routed through mark_job_error so the frontend sees a
        # structured {error, errorCode, errorType} payload instead of a
        # bare error string.
        try:
            encoding, col_count, row_count = validate_csv(csv_bytes)
            logger.info(f"CSV validation passed: {row_count} rows, {col_count} columns, {encoding} encoding")
        except UploadValidationError as e:
            mark_job_error(
                job_ref,
                job_id,
                str(e),
                error_code=e.error_code,
                error_type=e.error_type,
            )
            logger.error(f"CSV validation failed for job {job_id}: {e}")
            return

        # 2. Load Data
        # Codex P2 (r3055316yyy): pass the encoding detected by validate_csv
        # so parsing uses the same codec as validation. Otherwise a valid
        # cp1252 CSV (smart quotes, em dashes, non-ASCII categoricals) would
        # pass validation but be silently decoded as Latin-1 or UTF-8 at
        # parse time, corrupting categorical values and therefore the
        # resulting outlier scores.
        try:
            loader = DataLoader(drop_columns=[], rename_columns={}, columns_of_interest=[])
            df = loader.load_original_data(csv_bytes, encoding=encoding)
            logger.info(f"Loaded CSV data. Shape: {df.shape}")
        except Exception as load_err:
            # The CSV passed validate_csv but DataLoader still could not
            # read it - usually a pandas parse quirk that the streaming
            # validator missed. Surface a clean LOAD_FAILURE code instead
            # of leaking the raw exception message.
            mark_job_error(
                job_ref,
                job_id,
                "Failed to load CSV data. The file may be malformed.",
                error_code=ErrorCode.LOAD_FAILURE,
                error_type=ErrorType.PROCESSING,
            )
            logger.error(
                f"DataLoader.load_original_data failed for job {job_id}: "
                f"{load_err}"
            )
            return

        # 3. Calculate stats for the frontend 'Overview' tab
        stats = {
            "total_rows": len(df),
            "kept_columns": [],
            "ignored_columns": []
        }

        # 4. Data Cleaning
        process_df = df.fillna("missing").astype(str)

        # 5. Rule of 9 Filter
        cols_to_keep = []
        for col in process_df.columns:
            unique_count = process_df[col].nunique()
            if unique_count > 1 and unique_count <= 9:
                cols_to_keep.append(col)
                stats["kept_columns"].append({
                    "name": col,
                    "type": str(df[col].dtype),
                    "unique_values": unique_count,
                    "missing_values": int(df[col].isin(["NA", "nan", "missing"]).sum())
                })
            else:
                stats["ignored_columns"].append({
                    "name": col,
                    "unique_values": unique_count,
                    "missing_values": int(df[col].isin(["NA", "nan", "missing"]).sum())
                })

        process_df = process_df[cols_to_keep]
        logger.info(f"After Rule of 9: {process_df.shape} (kept {len(cols_to_keep)}, ignored {len(stats['ignored_columns'])})")

        if process_df.shape[1] == 0:
            # TASKS.md 2.3: emit a stable error code so the frontend can
            # special-case this with a specific, actionable message (users
            # hit this when every column is either constant or
            # high-cardinality). Stash `stats` so the UI can show the user
            # which columns were actually dropped.
            mark_job_error(
                job_ref,
                job_id,
                "No usable columns found. Every column was dropped because "
                "it had only one unique value or more than 9 unique values. "
                "Try a dataset with some low-cardinality categorical "
                "columns (between 2 and 9 distinct values).",
                error_code=ErrorCode.NO_USABLE_COLUMNS,
                error_type=ErrorType.PROCESSING,
            )
            try:
                job_ref.update({'stats': stats})
            except Exception as stats_err:
                logger.warning(
                    f"Failed to attach stats to error state for job "
                    f"{job_id}: {stats_err}"
                )
            logger.error(
                f"Job {job_id} produced no usable columns after Rule of 9"
            )
            return

        # 6. Vectorization (split before fitting to prevent data leakage)
        from sklearn.model_selection import train_test_split as _tts
        model_variable_types = {col: 'categorical' for col in process_df.columns}
        vectorizer = Table2Vector(model_variable_types)

        train_df, test_df = _tts(process_df, test_size=0.2)
        vectorizer.fit(train_df)
        X_train = vectorizer.transform(train_df).astype('float32')
        X_test = vectorizer.transform(test_df).astype('float32')

        # We also need the full vectorized data for scoring later
        vectorized_df = vectorizer.transform(process_df).astype('float32')

        # 7. Build and train autoencoder
        cardinalities = vectorizer.get_cardinalities(process_df.columns)
        ae_wrapper = AutoencoderModel(attribute_cardinalities=cardinalities)
        ae_wrapper.INPUT_SHAPE = X_train.shape[1:]

        input_dim = X_train.shape[1]
        model_config = {
            "learning_rate": 0.001,
            "latent_space_dim": max(2, int(input_dim * 0.1)),
            "encoder_layers": 2,
            "decoder_layers": 2,
            "encoder_units_1": int(input_dim * 0.5),
            "decoder_units_1": int(input_dim * 0.5)
        }

        keras_model = ae_wrapper.build_autoencoder(model_config)

        # Codex P1 (r3053724013): advance the state machine through TRAINING
        # before fit(). Without this, the job skipped directly from PROCESSING
        # to COMPLETE, which is not an allowed transition (SCORING -> COMPLETE
        # is the only path to the terminal state), so every successful run
        # would raise ValueError and mark the job ERROR after the model
        # actually finished training.
        transaction = db.transaction()
        update_job_status(transaction, job_ref, JobStatus.TRAINING)

        # Phase 4: cooperative cancellation during training
        class CancellationCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if is_job_canceled(job_id):
                    logger.info(f"Job {job_id} canceled during training (epoch {epoch}), stopping")
                    self.model.stop_training = True

        # Check for early cancellation before heavy training work
        if is_job_canceled(job_id):
            logger.info(f"Job {job_id} was canceled before training, aborting")
            return

        logger.info("Training autoencoder...")
        try:
            keras_model.fit(
                X_train, X_train,
                epochs=15, batch_size=32, verbose=2,
                validation_data=(X_test, X_test),
                callbacks=[CancellationCallback()]
            )
        except Exception as train_err:
            # TASKS.md 2.3: classify training failures separately from
            # validation / loading so the frontend can surface the right
            # stage in its error UI.
            mark_job_error(
                job_ref,
                job_id,
                "Model training failed. The uploaded dataset may be too "
                "small or too uniform to train on.",
                error_code=ErrorCode.TRAINING_FAILURE,
                error_type=ErrorType.TRAINING,
            )
            logger.error(f"Training failed for job {job_id}: {train_err}")
            return

        # Check cancellation after training
        if is_job_canceled(job_id):
            logger.info(f"Job {job_id} was canceled after training, aborting scoring")
            return

        # Codex P1 (r3053724013): advance TRAINING -> SCORING before scoring
        transaction = db.transaction()
        update_job_status(transaction, job_ref, JobStatus.SCORING)

        # 8. Calculate reconstruction error
        reconstruction = keras_model.predict(vectorized_df)
        if isinstance(reconstruction, list):
            reconstruction = reconstruction[0]
        mse = np.mean(np.power(vectorized_df - reconstruction, 2), axis=1)
        df['reconstruction_error'] = mse

        # 9. Get top outliers and compute per-column contributions.
        # Phase 4: reuse batch predictions, cap contributions, separate data/metadata.
        top_outliers = df.sort_values(by='reconstruction_error', ascending=False).head(100)
        top_outliers = top_outliers.replace([np.inf, -np.inf], 0).fillna("missing")

        predictions_np = reconstruction if isinstance(reconstruction, np.ndarray) else np.array(reconstruction)
        vectorized_np = vectorized_df.to_numpy()
        MAX_CONTRIBUTIONS_PER_OUTLIER = 10

        outliers_data = []
        for idx, row in top_outliers.iterrows():
            row_data = vectorized_np[idx:idx+1]
            row_pred = predictions_np[idx:idx+1]

            contributions = compute_per_column_contributions(
                row_data, row_pred, cardinalities, list(process_df.columns))

            sorted_contribs = sorted(contributions, key=lambda x: x[1], reverse=True)
            capped_contribs = sorted_contribs[:MAX_CONTRIBUTIONS_PER_OUTLIER]

            # User columns under `data`, system metadata at top level.
            row_dict = row.to_dict()
            outlier_record = {
                'data': row_dict,
                'reconstruction_error': _safe_float(row_dict.get('reconstruction_error', 0)),
                'contributions': [
                    {'column': col, 'percentage': float(pct)}
                    for col, pct in capped_contribs
                ],
            }
            outliers_data.append(outlier_record)

        # 10. Save to Firestore with transactional status update (WORK-07)
        transaction = db.transaction()
        update_job_status(transaction, job_ref, JobStatus.COMPLETE, {
            'stats': stats,
            'outliers': outliers_data,
            'processedAt': firestore.SERVER_TIMESTAMP
        })

        logger.info(f"Job {job_id} complete. Saved {len(outliers_data)} outliers to Firestore.")

    except JobDocumentNotReadyError:
        # Never swallow this: it must reach callback() so the message is
        # nacked for Pub/Sub redelivery. Do NOT mark the job ERROR - the
        # document doesn't even exist yet.
        raise
    except JobInProgressError:
        # Never swallow this either: callback() must nack (not ack+mark)
        # so Pub/Sub can redeliver if the original worker crashes.
        # Writing status=ERROR here would ALSO be wrong - another worker
        # is actively running this job and would see its next transition
        # rejected as "ERROR -> TRAINING" / similar.
        raise
    except UploadValidationError as e:
        # TASKS.md 2.3: an UploadValidationError that bubbled up past the
        # targeted handlers above is still a structured, user-facing
        # failure - route it through mark_job_error so the client sees the
        # correct error_code and errorType instead of falling through to
        # the INTERNAL_ERROR branch below.
        logger.error(
            f"Unhandled UploadValidationError for job {job_id}: {e}"
        )
        mark_job_error(
            job_ref,
            job_id,
            str(e),
            error_code=e.error_code,
            error_type=e.error_type,
        )
    except Exception as e:
        # TASKS.md 2.3: everything that falls through here is an
        # unclassified internal failure. Log the raw exception for
        # debugging, but write a generic user-facing message to Firestore
        # so the frontend does not leak Python stack strings (which can
        # include file paths and library internals). The stable
        # INTERNAL_ERROR code lets the UI surface a "contact support"
        # affordance instead of guessing what went wrong.
        logger.exception(f"Unhandled error processing job {job_id}: {e}")
        mark_job_error(
            job_ref,
            job_id,
            "An unexpected error occurred while processing this file. "
            "Please try again or contact support if the problem persists.",
            error_code=ErrorCode.INTERNAL_ERROR,
            error_type=ErrorType.INTERNAL,
        )
    finally:
        # WORK-05/06: Stop ack extension in finally block
        extender.stop()


def process_upload_vertex(job_id, bucket_name, file_path, message):
    """
    Dispatch processing to a Vertex AI CustomContainerTrainingJob.

    Args:
        job_id: Firestore job document ID
        bucket_name: GCS bucket name
        file_path: GCS file path
        message: Pub/Sub message object for ack deadline extension
    """
    from google.cloud import aiplatform

    job_ref = db.collection('jobs').document(job_id)

    # WORK-05/06: Start ack deadline extension. job_ref is intentionally
    # left unset at this point so the heartbeat only refreshes the
    # Pub/Sub ack deadline, not `claimedAt`. Codex P1 flagged that an
    # eager claim heartbeat on a redelivered message rewrites a
    # crashed worker's stale claim to "fresh" before we've even run
    # stale-takeover, which then always rejects takeover and wedges
    # crash recovery. We attach job_ref below, after the queued ->
    # processing transition (or stale-claim takeover) confirms we
    # actually own this job.
    extender = AckExtender(
        message,
        interval_seconds=JOB_CLAIM_HEARTBEAT_INTERVAL_SECONDS,
        job_ref=None,
    )
    extender.start()

    try:
        logger.info(f"Starting Vertex AI job {job_id}")

        # Codex P1 (r3055399xxx): claim the job BEFORE launching the
        # Vertex AI training job. Without this guard a duplicate Pub/Sub
        # delivery would call job.run(sync=False) again and submit a
        # second (billable) Vertex training job against the same jobId.
        # The local path already has this guard; mirror it here so both
        # modes are protected by the same state-machine claim.
        #
        # Use the same classification the local path uses for an invalid
        # transition (see process_upload_local): a document-not-found
        # error raises JobDocumentNotReadyError (nack for redelivery); a
        # non-terminal in-progress state with a FRESH `claimedAt`
        # heartbeat raises JobInProgressError (nack so we never
        # double-submit); a non-terminal state with a STALE heartbeat
        # triggers try_take_over_stale_claim so a crashed dispatcher can
        # be recovered (Codex P1 r(stale-takeover)); a terminal state
        # returns cleanly; anything unexpected logs and returns.
        try:
            transaction = db.transaction()
            update_job_status(
                transaction,
                job_ref,
                JobStatus.PROCESSING,
                # Codex P1 r(suppressor-write-failure): tag the job as
                # `mode: "vertex"` in the same transaction that claims
                # it. try_take_over_stale_claim uses this tag to pick a
                # much wider stale threshold (4 hours instead of 3
                # minutes), so even if the post-dispatch suppressor
                # write (vertexJobName / far-future claimedAt) later
                # fails, stale takeover still cannot fire during the
                # expected Vertex run window and we cannot
                # double-dispatch.
                additional_fields={
                    'claimedAt': datetime.now(timezone.utc),
                    'mode': 'vertex',
                },
            )
        except ValueError as e:
            if "not found" in str(e):
                raise JobDocumentNotReadyError(
                    f"Job {job_id} document not yet written, retry via nack"
                ) from e

            try:
                snapshot = job_ref.get()
                current_status = (
                    snapshot.get('status') if snapshot.exists else None
                )
            except Exception as read_err:
                logger.warning(
                    f"Failed to read job {job_id} state after invalid "
                    f"transition ({e}); treating as in-progress retry: "
                    f"{read_err}"
                )
                raise JobInProgressError(
                    f"Job {job_id} state unreadable after "
                    f"invalid-transition; retrying via nack"
                ) from e

            terminal_states = {
                JobStatus.COMPLETE.value,
                JobStatus.ERROR.value,
                JobStatus.CANCELED.value,
            }
            in_progress_states = {
                JobStatus.PROCESSING.value,
                JobStatus.TRAINING.value,
                JobStatus.SCORING.value,
            }
            if current_status in terminal_states:
                logger.info(
                    f"Job {job_id} already in terminal state "
                    f"{current_status}, skipping duplicate Vertex dispatch"
                )
                return
            if current_status in in_progress_states:
                # Codex P2 r(vertex-dispatched-ack): if the doc already
                # carries `vertexJobName`, a previous delivery already
                # successfully submitted the job to Vertex. Nothing
                # more for the dispatcher to do here - Vertex owns the
                # training run now. Treat the duplicate as a clean
                # skip (return) so callback() acks and marks it; the
                # alternative (raise JobInProgressError -> nack -> Pub/Sub
                # redelivers) would create an infinite redelivery loop
                # that burns worker capacity until Vertex finishes and
                # updates status on its own.
                if snapshot.get('vertexJobName'):
                    logger.info(
                        f"Job {job_id} already dispatched to Vertex "
                        f"(vertexJobName set), skipping duplicate Vertex "
                        f"dispatch"
                    )
                    return

                # Try stale-claim takeover before giving up. For Vertex
                # mode the dispatcher's active time is normally seconds
                # (just long enough to submit the job.run call), so a
                # stale `claimedAt` here almost certainly means the
                # previous dispatcher crashed before submitting anything.
                #
                # Pass mode="vertex" so the takeover transaction stamps
                # that tag on the recovered doc atomically. Otherwise a
                # legacy / untagged recovered doc would lose the mode
                # tag for any subsequent duplicate deliveries.
                if try_take_over_stale_claim(job_ref, mode='vertex'):
                    logger.info(
                        f"Resuming Vertex dispatch for job {job_id} "
                        f"from state {current_status} after stale-claim "
                        f"takeover"
                    )
                    # Fall through to submit the Vertex training job.
                else:
                    raise JobInProgressError(
                        f"Job {job_id} is already {current_status} with a "
                        f"fresh heartbeat; duplicate Vertex dispatch nacked "
                        f"to avoid double-submitting a training job"
                    ) from e
            else:
                logger.warning(
                    f"Job {job_id} in unexpected state "
                    f"{current_status!r} after failed PROCESSING transition "
                    f"({e}); acking and skipping Vertex dispatch"
                )
                return

        # Ownership is now confirmed (either via the normal queued ->
        # processing transition, or via try_take_over_stale_claim).
        # Wire the job_ref onto the heartbeat extender so subsequent
        # extend() ticks refresh `claimedAt`, matching the
        # process_upload_local ordering for the same reason (Codex P1
        # r3055xxxxx). Note: the Vertex dispatcher only needs the claim
        # heartbeat for the few seconds until job.run() returns - after
        # that, the post-dispatch suppressor path (below) stops the
        # extender and writes a far-future `claimedAt`.
        extender.job_ref = job_ref

        container_uri = f"us-central1-docker.pkg.dev/{PROJECT_ID}/autoencoder-repo/trainer:v1"
        logger.info(f"Target Image: {container_uri}")

        aiplatform.init(
            project=PROJECT_ID,
            location="us-central1",
            staging_bucket=os.getenv("VERTEX_STAGING_BUCKET", "gs://autoencoders-census-staging")
        )

        job = aiplatform.CustomContainerTrainingJob(
            display_name=f"autoencoder-{job_id}",
            container_uri=container_uri
        )

        job.run(
            args=[
                f"--job-id={job_id}",
                f"--bucket-name={bucket_name}",
                f"--file-path={file_path}"
            ],
            replica_count=1,
            service_account=os.getenv("VERTEX_SERVICE_ACCOUNT", "203111407489-compute@developer.gserviceaccount.com"),
            machine_type="n1-standard-4",
            sync=False
        )

        logger.info("Job submitted to Vertex AI.")

        # Codex P1 r(vertex-stale-redispatch): the dispatcher is about
        # to return, but the Vertex training job it just submitted will
        # keep running for many minutes. If we left the job doc as-is,
        # the `claimedAt` heartbeat would stop and after 3 minutes a
        # duplicate Pub/Sub delivery could trigger stale-claim takeover
        # and re-submit the same Vertex training run -> duplicate billing
        # and duplicate writes to the same job doc.
        #
        # Mitigations, in order:
        #
        # 1. Stop the AckExtender heartbeat BEFORE writing our suppressor
        #    values, so the heartbeat timer cannot race with us and
        #    overwrite our far-future claimedAt.
        #
        # 2. Record the Vertex job resource name on the job doc. That
        #    field gates try_take_over_stale_claim: if `vertexJobName`
        #    is set, the takeover function treats the job as externally
        #    owned and refuses to re-dispatch.
        #
        # 3. As a defense-in-depth belt to the vertexJobName suspenders,
        #    also set `claimedAt` to a timestamp far in the future
        #    (100 years) so even if some future reader ignores
        #    `vertexJobName`, _is_claim_stale still reports "fresh" and
        #    the takeover path still rejects the duplicate delivery.
        extender.stop()
        vertex_job_name = getattr(job, 'resource_name', None) or \
            getattr(job, 'name', None) or \
            f"vertex-dispatched-{job_id}"
        far_future_claim = datetime.now(timezone.utc) + timedelta(
            days=365 * 100
        )
        suppressor_payload = {
            'vertexJobName': vertex_job_name,
            'claimedAt': far_future_claim,
        }

        # Retry the suppressor write a few times to close the small race
        # window between "job.run() returned" and "takeover sees the
        # suppressor fields". If this write never lands, the stale
        # takeover path in try_take_over_stale_claim could eventually
        # reclaim the doc after JOB_CLAIM_STALE_SECONDS and submit a
        # second Vertex training run. Exponential backoff up to ~15
        # seconds total (0.5, 1, 2, 4, 8). We keep this loop tight
        # because callback() won't ack until we return, so the Pub/Sub
        # ack deadline is burning the whole time. The AckExtender has
        # already been stopped above, so we cannot push that deadline
        # further out.
        suppressor_persisted = False
        last_err = None
        delay = 0.5
        for attempt in range(5):
            try:
                job_ref.update(suppressor_payload)
                suppressor_persisted = True
                break
            except Exception as meta_err:
                last_err = meta_err
                logger.warning(
                    f"Suppressor write attempt {attempt + 1}/5 failed "
                    f"for Vertex job {job_id}: {meta_err}; retrying in "
                    f"{delay}s"
                )
                time.sleep(delay)
                delay *= 2

        if not suppressor_persisted:
            logger.error(
                f"FATAL: could not persist Vertex takeover suppressor "
                f"for job {job_id} after 5 retries (last error: {last_err}). "
                f"A duplicate Pub/Sub delivery within the next "
                f"JOB_CLAIM_STALE_SECONDS may double-dispatch the Vertex "
                f"training run; manual intervention recommended "
                f"(set vertexJobName and a far-future claimedAt on "
                f"jobs/{job_id})."
            )

        # Phase 4: reconciliation cancel. If the job was canceled while
        # we were submitting (the cancel API would have found no
        # vertexJobName and skipped the Vertex cancel), cancel the
        # pipeline ourselves now. Run this regardless of whether the
        # suppressor write succeeded — if it failed, the API has even
        # less ability to cancel, so we must do it here.
        if is_job_canceled(job_id):
            logger.info(
                f"Job {job_id} was canceled during Vertex submission, "
                "canceling the just-submitted pipeline"
            )
            try:
                job.cancel()
            except Exception as cancel_err:
                logger.error(
                    f"Failed to cancel Vertex pipeline after late cancel "
                    f"detection for job {job_id}: {cancel_err}. "
                    f"Persisting vertexCancelFailed marker for manual retry."
                )
                # Persist a marker so operators/cleanup scripts can find
                # jobs where the Vertex pipeline is still running despite
                # the app status being 'canceled'.
                try:
                    job_ref.update({'vertexCancelFailed': True})
                except Exception:
                    pass  # best-effort — already logged the real error

    except JobDocumentNotReadyError:
        # Never swallow: callback() must nack for Pub/Sub redelivery.
        raise
    except JobInProgressError:
        # Never swallow: callback() must nack without marking so a
        # crashed-worker scenario can still retry on redelivery.
        raise
    except Exception as e:
        # TASKS.md 2.3: Vertex dispatch is a thin wrapper around the
        # aiplatform SDK call, so most failures here are either
        # authentication / quota issues (internal infrastructure) or
        # malformed args (internal bugs). We classify as INTERNAL_ERROR
        # and write a generic user-facing message so the frontend does
        # not surface raw SDK stack strings.
        logger.exception(f"Failed to launch Vertex AI job {job_id}: {e}")
        mark_job_error(
            job_ref,
            job_id,
            "An unexpected error occurred while dispatching this job. "
            "Please try again or contact support if the problem persists.",
            error_code=ErrorCode.INTERNAL_ERROR,
            error_type=ErrorType.INTERNAL,
        )
    finally:
        # WORK-05/06: Stop ack extension in finally block
        extender.stop()


# Module-level processing mode, set from __main__
_processing_mode = "local"


def callback(message):
    """
    Pub/Sub callback: runs whenever a message arrives.
    """
    try:
        logger.info(f"Received message: {message.data}")

        # Codex P2 (r3053739504): deterministic schema / JSON failures are
        # poison messages, not transient errors. nack()-ing them causes an
        # infinite redelivery loop against the same bad payload, wasting
        # worker capacity (we have no DLQ configured). Ack-and-drop with a
        # log entry instead so an operator can find the payload in the logs
        # but the queue drains.
        try:
            data = json.loads(message.data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(
                f"Dropping non-JSON/invalid-encoding Pub/Sub message "
                f"{getattr(message, 'message_id', '?')}: {e}"
            )
            message.ack()
            return

        # WORK-01/02/03: Validate required fields
        try:
            validated = validate_message(data)
        except ValueError as e:
            # Codex P2 (r3053739504): schema validation failures are
            # permanent - the payload is malformed and will not become
            # valid on redelivery. Ack to drop the poison message instead
            # of nacking into an infinite redelivery loop.
            logger.error(
                f"Dropping Pub/Sub message "
                f"{getattr(message, 'message_id', '?')} with invalid schema: {e}"
            )
            message.ack()
            return

        # Use validated fields
        job_id = validated.jobId
        bucket = validated.bucket
        file_path = validated.file

        # WORK-04: Read-only idempotency check. We intentionally do NOT mark
        # the message as processed here: the marker is written AFTER
        # processing completes (mark_message_processed below). If we crashed
        # between a premature mark and ack, Pub/Sub would redeliver and we'd
        # drop the redelivered copy without ever finishing the work.
        #
        # Codex P1 (r3053917210) flags a concern that two concurrent
        # deliveries of the same message could both pass this read-only
        # check and both enter processing. That scenario is already
        # blocked by a different mechanism: the job state machine. Both
        # workers would then race to transition the job's Firestore doc
        # from QUEUED -> PROCESSING via update_job_status(). That runs
        # inside a Firestore transaction with optimistic concurrency, so
        # exactly one worker wins the transition; the loser retries,
        # sees "processing" as the current state, fails
        # is_valid_transition("processing", "processing"), and the
        # process_upload_local ValueError branch returns early without
        # running training/scoring. We avoid adding a redundant
        # "in-progress" marker here because (a) that would reintroduce
        # the original "premature mark -> silent drop on crash" failure
        # mode this function was refactored to fix, and (b) the existing
        # Firestore-transaction serialization at the job level already
        # provides the guarantee.
        if check_idempotency(message.message_id):
            logger.info(f"Message {message.message_id} already processed, skipping")
            message.ack()  # Ack duplicate message
            return

        # WORK-05/06: Process with ack extension, ack AFTER processing completes
        try:
            if _processing_mode == "vertex":
                process_upload_vertex(job_id, bucket, file_path, message)
            else:
                process_upload_local(job_id, bucket, file_path, message)
        except JobDocumentNotReadyError as not_ready:
            # Codex P1 (r3053739500): the job document hasn't been written
            # yet (race with /start-job). nack so Pub/Sub redelivers - do
            # NOT mark the message as processed, otherwise the retry would
            # be silently dropped as a duplicate.
            logger.warning(
                f"Job {job_id} not ready yet, nacking for Pub/Sub redelivery: "
                f"{not_ready}"
            )
            message.nack()
            return
        except JobInProgressError as in_progress:
            # Codex P1 (r3055316xxx): this is a duplicate Pub/Sub delivery
            # while another worker is still actively processing the same
            # job. nack so Pub/Sub redelivers later - and critically do NOT
            # call mark_message_processed, because if the original worker
            # subsequently crashes we need the next redelivery to be able
            # to retry. Acking a duplicate that overlapped an in-flight
            # run would permanently hide that recovery path.
            logger.warning(
                f"Job {job_id} is already in progress on another worker, "
                f"nacking duplicate delivery for retry: {in_progress}"
            )
            message.nack()
            return

        # WORK-04: Only now that processing is done do we record the marker,
        # so a crash mid-processing leaves the marker unset and Pub/Sub
        # redelivery triggers a real retry instead of a silent drop.
        try:
            mark_message_processed(message.message_id, job_id)
        except Exception as mark_err:
            # Failing to persist the marker isn't fatal: the job itself already
            # finished, and the worst case on redelivery is an early-return
            # due to the job state machine (terminal states reject
            # transitions). Log and continue to ack.
            logger.warning(
                f"Failed to mark message {message.message_id} as processed: {mark_err}"
            )

        # WORK-06: Ack only after processing completes successfully
        message.ack()
        logger.info(f"Message {message.message_id} processed and acknowledged.")

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        message.nack()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pub/Sub worker for survey processing")
    parser.add_argument(
        "--mode", choices=["local", "vertex"], default="local",
        help="Processing mode: 'local' runs ML locally, 'vertex' dispatches to Vertex AI (default: local)"
    )
    args = parser.parse_args()
    _processing_mode = args.mode

    # Fail fast if required env vars are missing
    validate_environment()

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

    logger.info(f"Listening for jobs on {subscription_path}...")
    logger.info(f"Processing mode: {_processing_mode}")

    future = subscriber.subscribe(subscription_path, callback=callback)
    try:
        future.result()
    except KeyboardInterrupt:
        future.cancel()
        logger.info("Worker stopped.")
