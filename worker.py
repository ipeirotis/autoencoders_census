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

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
SUBSCRIPTION_ID = os.getenv("PUBSUB_SUBSCRIPTION_ID")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = firestore.Client(project=PROJECT_ID)


class PubSubMessage(BaseModel):
    jobId: str = Field(..., min_length=1)
    bucket: str = Field(..., min_length=1)
    file: str = Field(..., min_length=1)


def validate_message(data: dict) -> PubSubMessage:
    try:
        return PubSubMessage(**data)
    except ValidationError as e:
        errors = '; '.join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
        raise ValueError(f"Invalid message format: {errors}")


def check_idempotency(message_id: str, job_id: str) -> bool:
    processed_ref = db.collection('processed_messages').document(message_id)

    @firestore.transactional
    def mark_processed(transaction, ref):
        snapshot = ref.get(transaction=transaction)
        if snapshot.exists:
            return True
        transaction.set(ref, {
            'jobId': job_id,
            'processedAt': firestore.SERVER_TIMESTAMP,
            'expiresAt': firestore.SERVER_TIMESTAMP
        })
        return False

    transaction = db.transaction()
    return mark_processed(transaction, processed_ref)


class AckExtender:
    def __init__(self, message, interval_seconds=60):
        self.message = message
        self.interval = interval_seconds
        self.timer = None
        self.stopped = False

    def extend(self):
        if not self.stopped:
            try:
                self.message.modify_ack_deadline(self.interval + 10)
                logger.info(f"Extended ack deadline for message {self.message.message_id}")
            except Exception as e:
                logger.warning(f"Failed to extend ack deadline: {e}")
            self.timer = threading.Timer(self.interval, self.extend)
            self.timer.daemon = True
            self.timer.start()

    def start(self):
        self.extend()

    def stop(self):
        self.stopped = True
        if self.timer:
            self.timer.cancel()


from enum import Enum

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    TRAINING = "training"
    SCORING = "scoring"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELED = "canceled"


ALLOWED_TRANSITIONS = {
    JobStatus.QUEUED: [JobStatus.PROCESSING, JobStatus.ERROR, JobStatus.CANCELED],
    JobStatus.PROCESSING: [JobStatus.TRAINING, JobStatus.ERROR, JobStatus.CANCELED],
    JobStatus.TRAINING: [JobStatus.SCORING, JobStatus.ERROR, JobStatus.CANCELED],
    JobStatus.SCORING: [JobStatus.COMPLETE, JobStatus.ERROR, JobStatus.CANCELED],
    JobStatus.COMPLETE: [],
    JobStatus.ERROR: [],
    JobStatus.CANCELED: []
}


def is_valid_transition(current_status, new_status):
    if current_status is None:
        try:
            return JobStatus(new_status) == JobStatus.QUEUED
        except ValueError:
            return False
    try:
        current = JobStatus(current_status)
        new = JobStatus(new_status)
    except ValueError:
        return False
    return new in ALLOWED_TRANSITIONS.get(current, [])


def is_job_canceled(job_id):
    """Quick Firestore read to check whether the job has been canceled."""
    snap = db.collection('jobs').document(job_id).get()
    if snap.exists:
        return snap.get('status') == JobStatus.CANCELED
    return False


@firestore.transactional
def update_job_status(transaction, job_ref, new_status, additional_fields=None):
    snapshot = job_ref.get(transaction=transaction)
    if not snapshot.exists:
        raise ValueError(f"Job {job_ref.id} not found")
    current_status = snapshot.get('status')
    if not is_valid_transition(current_status, new_status):
        raise ValueError(f"Invalid transition: {current_status} -> {new_status}")
    update_data = {'status': new_status}
    if additional_fields:
        update_data.update(additional_fields)
    transaction.update(job_ref, update_data)
    logger.info(f"Job {job_ref.id} status: {current_status} -> {new_status}")


def validate_environment():
    required_vars = {
        'GOOGLE_CLOUD_PROJECT': os.getenv("GOOGLE_CLOUD_PROJECT"),
        'GCS_BUCKET_NAME': os.getenv("GCS_BUCKET_NAME"),
        'PUBSUB_SUBSCRIPTION_ID': os.getenv("PUBSUB_SUBSCRIPTION_ID"),
    }
    missing = [name for name, value in required_vars.items() if not value]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)
    logger.info("Environment validation passed.")
    return True


def validate_csv(csv_bytes, max_size_mb=100):
    size_mb = len(csv_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"CSV file too large: {size_mb:.1f}MB (max {max_size_mb}MB)")

    detection = chardet.detect(csv_bytes[:100000])
    encoding = detection['encoding']
    confidence = detection['confidence']
    if confidence < 0.7:
        logger.warning(f"Low encoding confidence: {confidence:.2f} for {encoding}, falling back to UTF-8")
        encoding = 'utf-8'

    try:
        chunk_iterator = pd.read_csv(io.BytesIO(csv_bytes), encoding=encoding, chunksize=10000, engine='python')
        first_chunk = next(chunk_iterator)
        expected_columns = len(first_chunk.columns)
        total_rows = len(first_chunk)
        for chunk in chunk_iterator:
            if len(chunk.columns) != expected_columns:
                raise ValueError(f"Inconsistent column count: expected {expected_columns}, got {len(chunk.columns)}")
            total_rows += len(chunk)
        if total_rows < 10:
            raise ValueError(f"CSV must have at least 10 rows (found {total_rows})")
        if expected_columns < 2:
            raise ValueError(f"CSV must have at least 2 columns (found {expected_columns})")
        logger.info(f"CSV validation passed: {total_rows} rows, {expected_columns} columns, {encoding} encoding")
        return encoding, expected_columns, total_rows
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV parsing error: {str(e)}")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except UnicodeDecodeError as e:
        raise ValueError(f"Encoding error with {encoding}: {str(e)}")
    except StopIteration:
        raise ValueError("CSV file is empty")


def process_upload_local(job_id, bucket_name, file_path, message):
    """Process the uploaded CSV locally."""
    # Lazy imports - TF must not be loaded at module scope
    from model.autoencoder import AutoencoderModel
    from evaluate.outliers import compute_per_column_contributions
    import tensorflow as tf

    extender = AckExtender(message, interval_seconds=60)
    extender.start()

    try:
        logger.info(f"Starting local processing for job {job_id}")

        transaction = db.transaction()
        job_ref = db.collection('jobs').document(job_id)
        try:
            update_job_status(transaction, job_ref, JobStatus.PROCESSING)
        except ValueError as e:
            logger.warning(f"Failed to update status: {e}")
            return

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        csv_bytes = blob.download_as_bytes()

        try:
            encoding, col_count, row_count = validate_csv(csv_bytes)
            logger.info(f"CSV validation passed: {row_count} rows, {col_count} columns, {encoding} encoding")
        except ValueError as e:
            transaction = db.transaction()
            update_job_status(transaction, job_ref, JobStatus.ERROR, {'error': str(e), 'errorType': 'validation'})
            logger.error(f"CSV validation failed for job {job_id}: {e}")
            return

        if is_job_canceled(job_id):
            logger.info(f"Job {job_id} was canceled before training, aborting")
            return

        loader = DataLoader(drop_columns=[], rename_columns={}, columns_of_interest=[])
        df = loader.load_original_data(csv_bytes)
        logger.info(f"Loaded CSV data. Shape: {df.shape}")

        stats = {"total_rows": len(df), "kept_columns": [], "ignored_columns": []}
        process_df = df.fillna("missing").astype(str)

        cols_to_keep = []
        for col in process_df.columns:
            unique_count = process_df[col].nunique()
            if unique_count > 1 and unique_count <= 9:
                cols_to_keep.append(col)
                stats["kept_columns"].append({
                    "name": col, "type": str(df[col].dtype),
                    "unique_values": unique_count,
                    "missing_values": int(df[col].isin(["NA", "nan", "missing"]).sum())
                })
            else:
                stats["ignored_columns"].append({
                    "name": col, "unique_values": unique_count,
                    "missing_values": int(df[col].isin(["NA", "nan", "missing"]).sum())
                })

        process_df = process_df[cols_to_keep]
        logger.info(f"After Rule of 9: {process_df.shape}")

        if process_df.shape[1] == 0:
            raise ValueError("All columns were dropped by Rule of 9 filter.")

        model_variable_types = {col: 'categorical' for col in process_df.columns}
        vectorizer = Table2Vector(model_variable_types)
        vectorized_df = vectorizer.vectorize_table(process_df).astype('float32')

        cardinalities = [process_df[col].nunique() for col in process_df.columns]
        ae_wrapper = AutoencoderModel(attribute_cardinalities=cardinalities)
        X_train, X_test = ae_wrapper.split_train_test(vectorized_df, test_size=0.2)

        input_dim = X_train.shape[1]
        model_config = {
            "learning_rate": 0.001,
            "latent_space_dim": max(2, int(input_dim * 0.1)),
            "encoder_layers": 2, "decoder_layers": 2,
            "encoder_units_1": int(input_dim * 0.5),
            "decoder_units_1": int(input_dim * 0.5)
        }

        keras_model = ae_wrapper.build_autoencoder(model_config)

        transaction = db.transaction()
        update_job_status(transaction, job_ref, JobStatus.TRAINING, {'stage': 'Training autoencoder model'})

        class CancellationCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if is_job_canceled(job_id):
                    logger.info(f"Job {job_id} canceled during training (epoch {epoch}), stopping")
                    self.model.stop_training = True

        logger.info("Training autoencoder...")
        keras_model.fit(X_train, X_train, epochs=15, batch_size=32, verbose=2,
                        validation_data=(X_test, X_test), callbacks=[CancellationCallback()])

        if is_job_canceled(job_id):
            logger.info(f"Job {job_id} was canceled after training, aborting scoring")
            return

        transaction = db.transaction()
        update_job_status(transaction, job_ref, JobStatus.SCORING, {'stage': 'Computing outlier scores'})

        from model.base import VAE
        predictions = keras_model.predict(vectorized_df)
        if isinstance(predictions, list):
            predictions = predictions[0]

        reconstruction_loss = VAE.reconstruction_loss(cardinalities, vectorized_df.to_numpy(), predictions)
        df['reconstruction_error'] = reconstruction_loss.numpy()

        top_outliers = df.sort_values(by='reconstruction_error', ascending=False).head(100)
        top_outliers = top_outliers.replace([np.inf, -np.inf], 0).fillna("missing")

        predictions_np = predictions if isinstance(predictions, np.ndarray) else np.array(predictions)
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

            # Structurally separate user data from system metadata.
            # User columns live under `data`; system keys (reconstruction_error,
            # contributions) live at the top level. This makes key collisions
            # with arbitrary user column names impossible.
            row_dict = row.to_dict()
            outlier_record = {
                'data': row_dict,
                'reconstruction_error': float(row_dict.get('reconstruction_error', 0)),
                'contributions': [
                    {'column': col, 'percentage': float(pct)}
                    for col, pct in capped_contribs
                ],
            }
            outliers_data.append(outlier_record)

        transaction = db.transaction()
        job_ref = db.collection('jobs').document(job_id)
        update_job_status(transaction, job_ref, JobStatus.COMPLETE, {
            'stats': stats, 'outliers': outliers_data, 'processedAt': firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Job {job_id} complete. Saved {len(outliers_data)} outliers to Firestore.")

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        transaction = db.transaction()
        job_ref = db.collection('jobs').document(job_id)
        try:
            update_job_status(transaction, job_ref, JobStatus.ERROR, {'error': str(e)})
        except ValueError:
            logger.error(f"Job {job_id} failed but could not update status: {e}")
    finally:
        extender.stop()


def process_upload_vertex(job_id, bucket_name, file_path, message):
    """Dispatch processing to a Vertex AI CustomContainerTrainingJob."""
    from google.cloud import aiplatform

    extender = AckExtender(message, interval_seconds=60)
    extender.start()

    try:
        logger.info(f"Starting Vertex AI job {job_id}")
        container_uri = f"us-central1-docker.pkg.dev/{PROJECT_ID}/autoencoder-repo/trainer:v1"

        aiplatform.init(project=PROJECT_ID, location="us-central1",
                        staging_bucket="gs://autoencoders-census-staging")

        job = aiplatform.CustomContainerTrainingJob(
            display_name=f"autoencoder-{job_id}", container_uri=container_uri)

        job.run(
            args=[f"--job-id={job_id}", f"--bucket-name={bucket_name}", f"--file-path={file_path}"],
            replica_count=1, service_account="203111407489-compute@developer.gserviceaccount.com",
            machine_type="n1-standard-4", sync=False)

        vertex_job_name = getattr(job, "resource_name", None)
        if vertex_job_name:
            db.collection('jobs').document(job_id).set({"vertexJobName": vertex_job_name}, merge=True)
            logger.info(f"Stored Vertex training pipeline resource name for job {job_id}: {vertex_job_name}")

            if is_job_canceled(job_id):
                logger.info(f"Job {job_id} was canceled during Vertex submission, canceling pipeline")
                try:
                    job.cancel()
                except Exception as cancel_err:
                    logger.warning(f"Failed to cancel Vertex pipeline after late cancel detection for job {job_id}: {cancel_err}")
        else:
            logger.warning(f"Vertex AI did not return a resource_name for job {job_id}; cancellation will be a no-op")

        logger.info("Job submitted to Vertex AI.")

    except Exception as e:
        logger.error(f"Failed to launch Vertex AI job: {e}")
        transaction = db.transaction()
        job_ref = db.collection('jobs').document(job_id)
        try:
            update_job_status(transaction, job_ref, JobStatus.ERROR, {'error': str(e)})
        except ValueError:
            logger.error(f"Job {job_id} failed but could not update status: {e}")
    finally:
        extender.stop()


_processing_mode = "local"


def callback(message):
    try:
        logger.info(f"Received message: {message.data}")
        data = json.loads(message.data.decode("utf-8"))

        try:
            validated = validate_message(data)
        except ValueError as e:
            logger.error(f"Message validation failed: {e}")
            message.nack()
            return

        job_id = validated.jobId
        bucket = validated.bucket
        file_path = validated.file

        if check_idempotency(message.message_id, job_id):
            logger.info(f"Message {message.message_id} already processed, skipping")
            message.ack()
            return

        if _processing_mode == "vertex":
            process_upload_vertex(job_id, bucket, file_path, message)
        else:
            process_upload_local(job_id, bucket, file_path, message)

        message.ack()
        logger.info(f"Message {message.message_id} processed and acknowledged.")

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        message.nack()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pub/Sub worker for survey processing")
    parser.add_argument("--mode", choices=["local", "vertex"], default="local",
                        help="Processing mode: 'local' or 'vertex' (default: local)")
    args = parser.parse_args()
    _processing_mode = args.mode

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
