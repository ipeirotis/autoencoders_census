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
"""

import argparse
import os
import sys
import json
import logging
import threading
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = firestore.Client(project=PROJECT_ID)


class PubSubMessage(BaseModel):
    """Pydantic model for validating Pub/Sub message payload."""
    jobId: str = Field(..., min_length=1, description="Firestore job document ID")
    bucket: str = Field(..., min_length=1, description="GCS bucket name")
    file: str = Field(..., min_length=1, description="GCS file path")


def validate_message(data: dict) -> PubSubMessage:
    """
    Validate message fields, raise ValueError with clear error.

    Args:
        data: Dictionary containing message payload

    Returns:
        PubSubMessage: Validated message object

    Raises:
        ValueError: If validation fails with description of missing/invalid fields
    """
    try:
        return PubSubMessage(**data)
    except ValidationError as e:
        errors = '; '.join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
        raise ValueError(f"Invalid message format: {errors}")


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
    logger.info(f"Job {job_ref.id} status: {current_status} -> {new_status}")


def _is_claim_stale(claimed_at, now=None):
    """
    Decide whether a job's `claimedAt` timestamp is stale (older than
    JOB_CLAIM_STALE_SECONDS), meaning the claiming worker's heartbeat has
    gone silent long enough that we should treat the claim as abandoned.

    Accepts timezone-aware or naive datetimes and tolerates `None`
    (missing field) by treating it as stale so legacy jobs written
    before this field existed can still be recovered.
    """
    if claimed_at is None:
        return True
    if now is None:
        now = datetime.now(timezone.utc)
    # Firestore returns timezone-aware datetimes; be defensive if we
    # see a naive value (tests, legacy data) by assuming UTC.
    if claimed_at.tzinfo is None:
        claimed_at = claimed_at.replace(tzinfo=timezone.utc)
    age = (now - claimed_at).total_seconds()
    return age > JOB_CLAIM_STALE_SECONDS


def try_take_over_stale_claim(job_ref):
    """
    Attempt to atomically reclaim a stale in-progress job.

    Codex P1 r(stale-takeover): if a worker crashed after taking the
    queued -> processing claim but before writing a terminal state,
    subsequent redeliveries of the same Pub/Sub message will see the
    doc in an in-progress state and a naive implementation would nack
    forever (infinite redelivery loop, stuck job). This function gives
    us an escape hatch: a duplicate delivery that finds the claim
    abandoned (no heartbeat refresh for more than JOB_CLAIM_STALE_SECONDS)
    transactionally bumps `claimedAt` to now and returns True, letting
    the caller resume processing.

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
        # Vertex guard: if the dispatcher already handed the job off to
        # Vertex, it is Vertex's responsibility to run and update status.
        # Re-taking over would re-submit the same training run.
        if snapshot.get('vertexJobName'):
            return False
        claimed_at = snapshot.get('claimedAt')
        if not _is_claim_stale(claimed_at):
            return False
        transaction.update(
            ref,
            {
                # Reset status back to PROCESSING so the calling pipeline
                # can restart from the beginning regardless of which
                # in-progress sub-state (processing/training/scoring) the
                # dead worker crashed in. This is a direct write that
                # bypasses the normal state machine transition check,
                # which is the whole point of this escape hatch.
                'status': JobStatus.PROCESSING.value,
                'claimedAt': datetime.now(timezone.utc),
            },
        )
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
    Returns (encoding, column_count, row_count) or raises ValueError.

    This function implements defense-in-depth CSV validation to catch
    encoding errors, malformed CSVs, and edge cases before training starts.

    WORK-09: Encoding detection using chardet
    WORK-10: Structure validation (inconsistent row lengths)
    WORK-11: Size limit enforcement (>100MB)
    WORK-13: Edge case handling (unicode characters)
    WORK-14: Edge case handling (mostly-missing values)

    Args:
        csv_bytes: Raw CSV file bytes
        max_size_mb: Maximum file size in MB (default: 100)

    Returns:
        tuple: (encoding, column_count, row_count)

    Raises:
        ValueError: If file too large, encoding unclear, structure invalid, or edge cases
    """
    # WORK-11: Check size limit
    size_mb = len(csv_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"CSV file too large: {size_mb:.1f}MB (max {max_size_mb}MB)")

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
                    raise ValueError(
                        f"Inconsistent column count: expected {expected_columns}, "
                        f"got {len(chunk.columns)}"
                    )
                total_rows += len(chunk)

            # Minimum data requirements
            if total_rows < 10:
                raise ValueError(f"CSV must have at least 10 rows (found {total_rows})")
            if expected_columns < 2:
                raise ValueError(f"CSV must have at least 2 columns (found {expected_columns})")

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
            raise ValueError(f"CSV parsing error: {str(e)}")
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        except StopIteration:
            raise ValueError("CSV file is empty")

    # All candidate encodings failed with UnicodeDecodeError.
    raise ValueError(
        f"Encoding error: no candidate encoding from {candidate_encodings} "
        f"could decode the CSV ({last_unicode_err})"
    )


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
    # Lazy import to avoid tensorflow dependency for tests
    from model.autoencoder import AutoencoderModel

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
                # Stale-takeover path: if the previous worker's heartbeat
                # has gone silent long enough that we consider the claim
                # abandoned, reclaim it and fall through to continue
                # processing instead of nacking.
                if try_take_over_stale_claim(job_ref):
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

        # WORK-09/10/11/13/14: Validate CSV before processing
        try:
            encoding, col_count, row_count = validate_csv(csv_bytes)
            logger.info(f"CSV validation passed: {row_count} rows, {col_count} columns, {encoding} encoding")
        except ValueError as e:
            # Update job status with validation error
            transaction = db.transaction()
            update_job_status(transaction, job_ref, JobStatus.ERROR, {
                'error': str(e),
                'errorType': 'validation'
            })
            logger.error(f"CSV validation failed for job {job_id}: {e}")
            return

        # 2. Load Data
        # Codex P2 (r3055316yyy): pass the encoding detected by validate_csv
        # so parsing uses the same codec as validation. Otherwise a valid
        # cp1252 CSV (smart quotes, em dashes, non-ASCII categoricals) would
        # pass validation but be silently decoded as Latin-1 or UTF-8 at
        # parse time, corrupting categorical values and therefore the
        # resulting outlier scores.
        loader = DataLoader(drop_columns=[], rename_columns={}, columns_of_interest=[])
        df = loader.load_original_data(csv_bytes, encoding=encoding)
        logger.info(f"Loaded CSV data. Shape: {df.shape}")

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
            raise ValueError("All columns were dropped by Rule of 9 filter. No columns have 2-9 unique values.")

        # 6. Vectorization
        model_variable_types = {col: 'categorical' for col in process_df.columns}
        vectorizer = Table2Vector(model_variable_types)
        vectorized_df = vectorizer.vectorize_table(process_df).astype('float32')

        # 7. Build and train autoencoder
        cardinalities = [process_df[col].nunique() for col in process_df.columns]
        ae_wrapper = AutoencoderModel(attribute_cardinalities=cardinalities)
        X_train, X_test = ae_wrapper.split_train_test(vectorized_df, test_size=0.2)

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

        logger.info("Training autoencoder...")
        keras_model.fit(
            X_train, X_train,
            epochs=15, batch_size=32, verbose=2,
            validation_data=(X_test, X_test)
        )

        # Codex P1 (r3053724013): advance TRAINING -> SCORING before scoring
        # so that the final COMPLETE transition matches ALLOWED_TRANSITIONS.
        transaction = db.transaction()
        update_job_status(transaction, job_ref, JobStatus.SCORING)

        # 8. Calculate reconstruction error
        reconstruction = keras_model.predict(vectorized_df)
        if isinstance(reconstruction, list):
            reconstruction = reconstruction[0]
        mse = np.mean(np.power(vectorized_df - reconstruction, 2), axis=1)
        df['reconstruction_error'] = mse

        # 9. Get top outliers
        top_outliers = df.sort_values(by='reconstruction_error', ascending=False).head(100)
        top_outliers = top_outliers.replace([np.inf, -np.inf], 0).fillna("missing")
        outliers_data = top_outliers.to_dict(orient='records')

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
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")

        # Update status to error using transaction (WORK-07)
        transaction = db.transaction()
        job_ref = db.collection('jobs').document(job_id)
        try:
            update_job_status(transaction, job_ref, JobStatus.ERROR, {'error': str(e)})
        except ValueError:
            # Job may be in terminal state already, error logged anyway
            logger.error(f"Job {job_id} failed but could not update status: {e}")
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
                additional_fields={'claimedAt': datetime.now(timezone.utc)},
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
                # Try stale-claim takeover before giving up. For Vertex
                # mode the dispatcher's active time is normally seconds
                # (just long enough to submit the job.run call), so a
                # stale `claimedAt` here almost certainly means the
                # previous dispatcher crashed before submitting anything.
                if try_take_over_stale_claim(job_ref):
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
            staging_bucket="gs://autoencoders-census-staging"
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
            service_account="203111407489-compute@developer.gserviceaccount.com",
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
        try:
            vertex_job_name = getattr(job, 'resource_name', None) or \
                getattr(job, 'name', None) or \
                f"vertex-dispatched-{job_id}"
            far_future_claim = datetime.now(timezone.utc) + timedelta(
                days=365 * 100
            )
            job_ref.update({
                'vertexJobName': vertex_job_name,
                'claimedAt': far_future_claim,
            })
        except Exception as meta_err:
            # If we can't persist the suppressor fields, the worst case
            # is that a duplicate delivery will trigger a second Vertex
            # dispatch after JOB_CLAIM_STALE_SECONDS. Log so the operator
            # can see the gap and continue - the job is already running
            # on Vertex, which is the important invariant.
            logger.warning(
                f"Failed to persist Vertex takeover suppressor for "
                f"job {job_id}: {meta_err}"
            )

    except JobDocumentNotReadyError:
        # Never swallow: callback() must nack for Pub/Sub redelivery.
        raise
    except JobInProgressError:
        # Never swallow: callback() must nack without marking so a
        # crashed-worker scenario can still retry on redelivery.
        raise
    except Exception as e:
        logger.error(f"Failed to launch Vertex AI job: {e}")

        # Update status to error using transaction (WORK-07)
        transaction = db.transaction()
        job_ref = db.collection('jobs').document(job_id)
        try:
            update_job_status(transaction, job_ref, JobStatus.ERROR, {'error': str(e)})
        except ValueError:
            # Job may be in terminal state already, error logged anyway
            logger.error(f"Job {job_id} failed but could not update status: {e}")
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
