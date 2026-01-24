"""
Pub/Sub Worker - Listens for upload messages and dispatches Vertex AI training jobs.

This worker runs locally and:
1. Subscribes to a Pub/Sub topic for new upload notifications
2. Receives messages containing {jobId, bucket, file} from the Express backend
3. Triggers a Vertex AI CustomContainerTrainingJob to process the data
4. The Vertex AI container downloads the CSV, runs outlier detection, and writes results to Firestore

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS="frontend/service-account-key.json"
    python worker.py

Required Environment Variables:
    - GOOGLE_CLOUD_PROJECT: GCP project ID
    - GCS_BUCKET_NAME: Storage bucket for uploads
    - PUBSUB_SUBSCRIPTION_ID: Pub/Sub subscription to listen on
"""

import os
import json
import logging
from google.cloud import pubsub_v1, firestore, aiplatform
from dotenv import load_dotenv

# worker.py
# 1. Download the file from Google Cloud Storage into memory (bytes)
# 2. Load it using loader.py

load_dotenv()
# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
SUBSCRIPTION_ID = os.getenv("PUBSUB_SUBSCRIPTION_ID")
MODEL_PATH = "cache/simple_model/autoencoder"
CONTAINER_URI = f"us-central1-docker.pkg.dev/{PROJECT_ID}/autoencoder-repo/trainer:v1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = firestore.Client(project=PROJECT_ID) # initialize Firestore DB

# Initialize connection to Vertex AI
try:
    aiplatform.init(
        project=PROJECT_ID, 
        location="us-central1",
        staging_bucket="gs://autoencoders-census-staging"
    )
except Exception as e:
    logger.warning(f"Could not init Vertex AI (Ignore this is running on a machine with gcloud auth): {e}")
    


def process_upload_job(job_id, bucket_name, file_path):
    """
    Triggers a Vertex AI Custom Job to run the training container
    """
    try:
        logger.info(f"Starting Vertex AI job {job_id}")
        logger.info(f"Target Image: {CONTAINER_URI}")

        # Initialize Vertex AI client
        aiplatform.init(
            project="autoencoders-census", 
            location="us-central1",
            staging_bucket="gs://autoencoders-census-staging"
        )

        # 1. Define the Custom Job
        job = aiplatform.CustomContainerTrainingJob(
            display_name=f"autoencoder-{job_id}",
            container_uri=CONTAINER_URI
        )

        # 2. Run the job (Asynchronous - worker doesn't wait)
        job.run(
            args=[
                f"--job-id={job_id}",
                f"--bucket-name={bucket_name}",
                f"--file-path={file_path}"
            ],
            replica_count=1,
            service_account="203111407489-compute@developer.gserviceaccount.com", # The fix for Permission Error
            machine_type="n1-standard-4",
            sync=False
        )
        
        logger.info(f"Job submitted to Vertex AI.")
        
    except Exception as e:
        logger.error(f"Failed to launch job: {e}")
        
        

    #     # 1. Download from GCS
    #     storage_client = storage.Client()
    #     bucket = storage_client.bucket(bucket_name)
    #     blob = bucket.blob(file_path)
    #     csv_bytes = blob.download_as_bytes()
        
    #     # 2. Load Data (RAW)
    #     # We use load_original_data to bypass any the 'prepare_original_dataset' binning logic
    #     # This ensures we process the raw strings exactly like main.py did during training 
    #     loader = DataLoader(drop_columns=[], rename_columns={}, columns_of_interest=[])
    #     df = loader.load_original_data(csv_bytes)

    #     logger.info(f"Successfully loaded CSV data. Shape: {df.shape}")
        
    #     # 3. Calculate States (for the 'Overview' tab)
    #     stats = {
    #         "total_rows": len(df),
    #         "kept_columns": [],
    #         "ignored_columns": []
    #     }
            
    #     logger.info(f"Stats: Kept {len(stats['kept_columns'])} columns, Ignored {len(stats['ignored_columns'])} columns.")

    #     # Outlier Detection logic
    #     logger.info("Running Outlier Detection Pipeline...")
        
    #     # A. Data Cleaning (The "missing value" Fix)
    #     process_df = df.fillna("missing").astype(str)
        
    #     # B. Rule of 9 Filter
    #     cols_to_keep = []
    #     for col in process_df.columns:
    #         unique_count = process_df[col].nunique()

    #         if unique_count > 1 and unique_count <= 9:
    #             cols_to_keep.append(col)
    #             # Add to 'Kept' stats
    #             stats["kept_columns"].append({
    #                 "name": col,
    #                 "type": str(df[col].dtype),
    #                 "unique_values": unique_count,
    #                 "missing_values": int(df[col].isin(["NA", "nan", "missing"]).sum()) # check custom missing tokens
    #             })
    #         else:
    #             # Add to 'Ignored' stats
    #             stats["ignored_columns"].append({
    #                 "name": col,
    #                 "unique_values": unique_count,
    #                 "missing_values": int(df[col].isin(["NA", "nan", "missing"]).sum()) # check custom missing tokens
    #             })

    #     process_df = process_df[cols_to_keep]
    #     logger.info(f"Model Input Shape: {process_df.shape}")
    #     logger.info(f"Stats: Kept {len(stats['kept_columns'])} columns, Ignored {len(stats['ignored_columns'])} columns.")

    #     #3 C. Vectorization
    #     model_variable_types = {col: 'categorical' for col in process_df.columns}
    #     vectorizer = Table2Vector(model_variable_types)
    #     vectorized_df = vectorizer.vectorize_table(process_df)
        
    #     # D. Float32 Conversion
    #     vectorized_df = vectorized_df.astype('float32')

    #     # E. Load Model and predict
    #     # (in a real cloud deployment, we would download the model from GCS here)
    #     logger.info(f"Loading model from {MODEL_PATH}")
        
    #     # Register the Custom Loss Function
    #     # we map the string name save in the model file to the actual Python class
    #     model = load_model(MODEL_PATH, custom_objects={
    #         "CustomCategoricalCrossentropyAE": CustomCategoricalCrossentropyAE
    #     })
        
    #     reconstruction = model.predict(vectorized_df)
    #     if isinstance(reconstruction, list):
    #         reconstruction = reconstruction[0]  # handle models with multiple outputs
        
    #     # F. Calculate Error
    #     mse = np.mean(np.power(vectorized_df - reconstruction, 2), axis=1)

    #     # Add error to the original dataframe so we can show the user their data + the score
    #     df['reconstruction_error'] = mse

    #     # G. Sort & Select Top Outliers (Limit to 100 for Frontend Performance)
    #     top_outliers = df.sort_values(by='reconstruction_error', ascending=False).head(100)

    #     # Convert to JSON-friendly format
    #     top_outliers = top_outliers.replace([np.inf, -np.inf], 0).fillna("missing")
    #     outliers_data = top_outliers.to_dict(orient='records')

    #     # 4. Save to Firestore
    #     doc_ref = db.collection('jobs').document(job_id)
    #     doc_ref.set({
    #         "status": "complete",
    #         "stats": stats,
    #         "outliers": outliers_data, # frontend will display this table
    #         "processedAt": firestore.SERVER_TIMESTAMP
    #     }, merge=True)
        
    #     logger.info(f"Saved stats to Firestore for job {job_id}")

    # except Exception as e:
    #     logger.error(f"Error processing upload job {job_id}: {e}")
    #     # write error status to Firestore so frontend knows it failed
    #     db.collection('jobs').document(job_id).set({
    #         "status": "error",
    #         "error": str(e)
    #     }, merge=True)

# Pub/Sub Listener 
subscription_id = "job-upload-topic sub"
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

def callback(message):
    """
    This function runs whenever a message arrives from Pub/Sub
    """
    try:
        logger.info(f"Received message: {message.data}")
        data = json.loads(message.data.decode("utf-8"))

        # extract info sent by the Node.js backend
        job_id = data.get("jobId")
        bucket = data.get("bucket")
        file_path = data.get("file")

        # run the upload job
        process_upload_job(job_id, bucket, file_path)

        message.ack()
        logger.info(f"Message {message.message_id} processed and acknowledged.")
    
    except Exception as e:
        logger.error(f"Error in processing message: {e}")
        message.nack()

if __name__ == "__main__":
    # This block allows you to test the worker manually 
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
    
    logger.info(f"Listening for jobs on {subscription_path}...")
    logger.info(f"Worker is ready to DISPATCH jobs to Vertex AI.")

    future = subscriber.subscribe(subscription_path, callback=callback)
    try:
        # keep the script running
        future.result()
    except KeyboardInterrupt:
        future.cancel()
    
    
    
    # parser = argparse.ArgumentParser() 
    # parser.add_argument("--job-id", default="test-job-123")
    # parser.add_argument("--bucket", required=True, help="Google Cloud Bucket Name")
    # parser.add_argument("--file", required=True, help="Path to file in bucket")
    
    # args = parser.parse_args()
    
    # process_upload_job(args.job_id, args.bucket, args.file)
