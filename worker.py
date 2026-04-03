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
import numpy as np
import pandas as pd
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


def check_idempotency(message_id: str, job_id: str) -> bool:
    """
    Check if message already processed using Firestore transaction.
    Returns True if already processed, False if first time.

    This prevents duplicate processing when Pub/Sub redelivers messages
    (at-least-once delivery guarantee). Uses Firestore transactions to
    handle race conditions when multiple workers check same message.

    Args:
        message_id: Unique Pub/Sub message ID
        job_id: Job ID from message payload (for logging/cleanup)

    Returns:
        bool: True if message was already processed, False if first time
    """
    processed_ref = db.collection('processed_messages').document(message_id)

    @firestore.transactional
    def mark_processed(transaction, ref):
        """Transactional read-modify-write to prevent race conditions."""
        snapshot = ref.get(transaction=transaction)
        if snapshot.exists:
            return True  # Already processed

        # Mark as processed with metadata for cleanup
        transaction.set(ref, {
            'jobId': job_id,
            'processedAt': firestore.SERVER_TIMESTAMP,
            'expiresAt': firestore.SERVER_TIMESTAMP  # For manual cleanup (7-day TTL)
        })
        return False

    transaction = db.transaction()
    return mark_processed(transaction, processed_ref)


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


def process_upload_local(job_id, bucket_name, file_path):
    """
    Process the uploaded CSV locally: download from GCS, train autoencoder,
    score outliers, and write results to Firestore.
    """
    # Lazy import to avoid tensorflow dependency for tests
    from model.autoencoder import AutoencoderModel

    try:
        logger.info(f"Starting local processing for job {job_id}")
        db.collection('jobs').document(job_id).set(
            {"status": "processing"}, merge=True
        )

        # 1. Download from GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        csv_bytes = blob.download_as_bytes()

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

        # 10. Save to Firestore
        db.collection('jobs').document(job_id).set({
            "status": "complete",
            "stats": stats,
            "outliers": outliers_data,
            "processedAt": firestore.SERVER_TIMESTAMP
        }, merge=True)

        logger.info(f"Job {job_id} complete. Saved {len(outliers_data)} outliers to Firestore.")

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        db.collection('jobs').document(job_id).set({
            "status": "error",
            "error": str(e)
        }, merge=True)


def process_upload_vertex(job_id, bucket_name, file_path):
    """
    Dispatch processing to a Vertex AI CustomContainerTrainingJob.
    """
    from google.cloud import aiplatform

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
        db.collection('jobs').document(job_id).set({
            "status": "error",
            "error": str(e)
        }, merge=True)


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

        # WORK-04: Check if already processed
        if check_idempotency(message.message_id, job_id):
            logger.info(f"Message {message.message_id} already processed, skipping")
            message.ack()  # Ack duplicate message
            return

        if _processing_mode == "vertex":
            process_upload_vertex(job_id, bucket, file_path)
        else:
            process_upload_local(job_id, bucket, file_path)

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
