import os 
import io
import json
import logging 
import argparse
from google.cloud import storage, pubsub_v1, firestore
from dataset.loader import DataLoader
from dotenv import load_dotenv

# worker.py
# 1. Download the file from Google Cloud Storage into memory (bytes)
# 2. Load it using loader.py

load_dotenv()
# --- Configuration ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
SUBSCRIPTION_ID = os.getenv("PUBSUB_SUBSCRIPTION_ID")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = firestore.Client(project=PROJECT_ID) # initialize Firestore DB

def process_upload_job(job_id, bucket_name, file_path):
    """
    1. Downloads file
    2. Loads it via DataLoader
    3. Calculates stats
    4. Updates Firestore (Mocked for now)
    """
    try:
        # 1. Download from GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        csv_bytes = blob.download_as_bytes()
        
        # 2. Load Data using load_uploaded_csv method
        loader = DataLoader(drop_columns=[], rename_columns={}, columns_of_interest=[])
        df, types = loader.load_uploaded_csv(csv_bytes)

        logger.info(f"Successfully loaded CSV data. Shape: {df.shape}")
        
        # 3. Calculate States (for Frontend)
        stats = {
            "total_rows": len(df),
            "columns": []
        }
        
        for col in df.columns:
            col_stat = {
                "name": col,
                "type": str(df[col].dtype),
                "unique_values": df[col].nunique(),
                "missing_values": int(df[col].isnull().sum())
            }
            stats["columns"].append(col_stat)
            
        logger.info(f"Generated stats for {len(stats['columns'])} columns.")

        # 4. Save to Firestore
        doc_ref = db.collection('jobs').document(job_id)
        doc_ref.update({
            "status": "ready",
            "stats": stats,
            "processedAt": firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Saved stats to Firesotre for job {job_id}")
        
    except Exception as e:
        logger.error(f"Error processing upload job {job_id}: {e}")
        # write error status to Firestore so frontend knows it failed
        db.collection('jobs').document(job_id).update({
            "status": "error",
            "error": str(e)
        })

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
    
    # start listening in the background
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    
    try:
        # keep the script running
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
    
    
    
    # parser = argparse.ArgumentParser() 
    # parser.add_argument("--job-id", default="test-job-123")
    # parser.add_argument("--bucket", required=True, help="Google Cloud Bucket Name")
    # parser.add_argument("--file", required=True, help="Path to file in bucket")
    
    # args = parser.parse_args()
    
    # process_upload_job(args.job_id, args.bucket, args.file)
