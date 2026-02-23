"""
Vertex AI Container Entry Point - Runs inside Docker on Vertex AI.

This script is triggered by worker.py and executes the full ML pipeline:
1. Downloads CSV from GCS bucket
2. Cleans data (fill NaN, apply "Rule of 9" filter)
3. Vectorizes categorical columns (one-hot encoding)
4. Builds and trains an autoencoder (15 epochs)
5. Calculates reconstruction error for each row
6. Saves top 100 outliers to Firestore

Usage (called by Vertex AI):
    python task.py --job-id=abc123 --bucket-name=autoencoder_data --file-path=uploads/abc123/data.csv
"""

import argparse
import logging 
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage, firestore
from dataset.loader import DataLoader
from features.transform import Table2Vector
from model.autoencoder import AutoencoderModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Force the client to use YOUR specific project, not the internal Google one
db = firestore.Client(project="autoencoders-census")

def train_and_predict(job_id, bucket_name, file_path):
    try:
        logger.info(f"Starting Vertex AI Job for {job_id}")
        
        # 1. Download Data
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path) 
        csv_bytes = blob.download_as_bytes()
        
        # 2. Load and Clean
        loader = DataLoader(drop_columns=[], rename_columns={}, columns_of_interest=[])
        df = loader.load_original_data(csv_bytes)
        
        # Rule of 9 Filter
        stats = {"total_rows": len(df), "kept_columns": [], "ignored_columns": []}
        process_df = df.fillna("missing").astype(str)
        
        cols_to_keep = []
        for col in process_df.columns:
            unique_count = process_df[col].nunique()
            if 1 < unique_count <= 9:
                cols_to_keep.append(col)
                stats["kept_columns"].append({"name": col, "unique_values": unique_count})
            else:
                stats["ignored_columns"].append({"name": col, "unique_values": unique_count})
                
        process_df = process_df[cols_to_keep]
        
        if process_df.shape[1] == 0:
            raise ValueError("All columns were dropped! No columns fit the Rule of 9.")

        # 3. Vectorization & model Setup
        cardinalities = [process_df[col].nunique() for col in process_df.columns]
        model_variable_types = {col: "categorical" for col in process_df.columns}
        vectorizer = Table2Vector(model_variable_types)
        vectorized_df = vectorizer.vectorize_table(process_df).astype('float32')
        
        logger.info("Initializing AutoEncoderModel...")
        ae_wrapper = AutoencoderModel(attribute_cardinalities=cardinalities)
        X_train, X_test = ae_wrapper.split_train_test(vectorized_df, test_size=0.2)
        
        # 4. Configure & Train
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
        
        logger.info("Training Model...")
        keras_model.fit(X_train, X_train, epochs=15, batch_size=32, verbose=2, validation_data=(X_test, X_test))
        
        # 5. Predict & Score
        reconstruction = keras_model.predict(vectorized_df)
        mse = np.mean(np.power(vectorized_df - reconstruction, 2), axis=1)
        df['reconstruction_error'] = mse

        top_outliers = df.sort_values(by='reconstruction_error', ascending=False).head(100)
        top_outliers = top_outliers.replace([np.inf, -np.inf], 0).fillna("missing")
        
        # 6. Save Results
        db.collection('jobs').document(job_id).set({
            "status": "complete",
            "stats": stats,
            "outliers": top_outliers.to_dict(orient="records"),
            "processedAt": firestore.SERVER_TIMESTAMP
        }, merge=True)
        
        logger.info("Job Complete")
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        db.collection('jobs').document(job_id).set({"status": "error", "error": str(e)})
        
if __name__ == "__main__":
    # This part grabs the args passed by Vertex AI
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', type=str, required=True)
    parser.add_argument('--bucket-name', type=str, required=True)
    parser.add_argument('--file-path', type=str, required=True)
    args = parser.parse_args()
    
    train_and_predict(args.job_id, args.bucket_name, args.file_path)
