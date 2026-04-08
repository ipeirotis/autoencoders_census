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

    @firestore.transactional
    def _set_if_not_exists(transaction, ref):
        snapshot = ref.get(transaction=transaction)
        if snapshot.exists:
            return  # Already marked by another worker - idempotent no-op

        transaction.set(ref, {
            'jobId': job_id,
            'processedAt': firestore.SERVER_TIMESTAMP,
            'expiresAt': firestore.SERVER_TIMESTAMP  # For manual cleanup (7-day TTL)
        })

    transaction = db.transaction()
    _set_if_not_exists(transaction, processed_ref)


class AckExtender:
    """
    Periodically extends Pub/Sub message ack deadline during long-running jobs.

    Prevents message timeout and redelivery for jobs that take 10-15 minutes
    (longer than the default 10-second ack deadline). Uses threading.Timer
    to extend deadline every 60 seconds.

    WORK-05: Ack deadline extension pattern
    """
    def __init__(self, message, interval_seconds=60):
        """
        Initialize AckExtender.

        Args:
            message: Pub/Sub message object with modify_ack_deadline() method
            interval_seconds: How often to extend deadline (default: 60 seconds)
        """
        self.message = message
        self.interval = interval_seconds
        self.timer = None
        self.stopped = False

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

    # WORK-09: Detect encoding
    detection = chardet.detect(csv_bytes[:100000])  # Sample first 100KB
    encoding = detection['encoding']
    confidence = detection['confidence']

    if confidence < 0.7:
        logger.warning(f"Low encoding confidence: {confidence:.2f} for {encoding}, falling back to UTF-8")
        encoding = 'utf-8'

    # WORK-10, WORK-13, WORK-14: Validate structure with streaming
    try:
        chunk_iterator = pd.read_csv(
            io.BytesIO(csv_bytes),
            encoding=encoding,
            chunksize=10000,
            engine='python'       # Better error messages, detects inconsistent columns
        )

        first_chunk = next(chunk_iterator)
        expected_columns = len(first_chunk.columns)
        total_rows = len(first_chunk)

        # Validate subsequent chunks have same structure
        for chunk in chunk_iterator:
            if len(chunk.columns) != expected_columns:
                raise ValueError(f"Inconsistent column count: expected {expected_columns}, got {len(chunk.columns)}")
            total_rows += len(chunk)

        # Minimum data requirements
        if total_rows < 10:
            raise ValueError(f"CSV must have at least 10 rows (found {total_rows})")
        if expected_columns < 2:
            raise ValueError(f"CSV must have at least 2 columns (found {expected_columns})")

        logger.info(f"CSV validation passed: {total_rows} rows, {expected_columns} columns, {encoding} encoding")
        return encoding, expected_columns, total_rows

    except pd.errors.ParserError as e:
        raise ValueError(f"CSV parsing error: {str(e)}")
    except pd.errors.EmptyDataError as e:
        raise ValueError("CSV file is empty")
    except UnicodeDecodeError as e:
        raise ValueError(f"Encoding error with {encoding}: {str(e)}")
    except StopIteration:
        raise ValueError("CSV file is empty")


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

    # WORK-05/06: Start ack deadline extension
    extender = AckExtender(message, interval_seconds=60)
    extender.start()

    try:
        logger.info(f"Starting local processing for job {job_id}")

        # Update status to processing using transaction (WORK-07)
        transaction = db.transaction()
        job_ref = db.collection('jobs').document(job_id)
        try:
            update_job_status(transaction, job_ref, JobStatus.PROCESSING)
        except ValueError as e:
            logger.warning(f"Failed to update status: {e}")
            # Job may have been canceled or already completed, skip processing
            return

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
        loader = DataLoader(drop_columns=[], rename_columns={}, columns_of_interest=[])
        df = loader.load_original_data(csv_bytes)
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

        logger.info("Training autoencoder...")
        keras_model.fit(
            X_train, X_train,
            epochs=15, batch_size=32, verbose=2,
            validation_data=(X_test, X_test)
        )

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
        job_ref = db.collection('jobs').document(job_id)
        update_job_status(transaction, job_ref, JobStatus.COMPLETE, {
            'stats': stats,
            'outliers': outliers_data,
            'processedAt': firestore.SERVER_TIMESTAMP
        })

        logger.info(f"Job {job_id} complete. Saved {len(outliers_data)} outliers to Firestore.")

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

    # WORK-05/06: Start ack deadline extension
    extender = AckExtender(message, interval_seconds=60)
    extender.start()

    try:
        logger.info(f"Starting Vertex AI job {job_id}")
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
        data = json.loads(message.data.decode("utf-8"))

        # WORK-01/02/03: Validate required fields
        try:
            validated = validate_message(data)
        except ValueError as e:
            logger.error(f"Message validation failed: {e}")
            message.nack()  # Reject invalid message
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
        if check_idempotency(message.message_id):
            logger.info(f"Message {message.message_id} already processed, skipping")
            message.ack()  # Ack duplicate message
            return

        # WORK-05/06: Process with ack extension, ack AFTER processing completes
        if _processing_mode == "vertex":
            process_upload_vertex(job_id, bucket, file_path, message)
        else:
            process_upload_local(job_id, bucket, file_path, message)

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
