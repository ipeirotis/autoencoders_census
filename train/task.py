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
import os
import numpy as np
import pandas as pd
from google.cloud import storage, firestore
from dataset.loader import DataLoader
from evaluate.outliers import compute_reconstruction_error
from features.transform import Table2Vector
from model.autoencoder import AutoencoderModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Force the client to use YOUR specific project, not the internal Google one
db = firestore.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT", "autoencoders-census"))

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

        # 3. Vectorization & model setup (split before fitting to prevent data leakage)
        from sklearn.model_selection import train_test_split as _tts
        model_variable_types = {col: "categorical" for col in process_df.columns}
        vectorizer = Table2Vector(model_variable_types)

        train_df, test_df = _tts(process_df, test_size=0.2)
        vectorizer.fit(train_df)
        X_train = vectorizer.transform(train_df).astype('float32')
        X_test = vectorizer.transform(test_df).astype('float32')
        vectorized_df = vectorizer.transform(process_df).astype('float32')

        cardinalities = vectorizer.get_cardinalities(process_df.columns)
        logger.info("Initializing AutoEncoderModel...")
        ae_wrapper = AutoencoderModel(attribute_cardinalities=cardinalities)
        ae_wrapper.INPUT_SHAPE = X_train.shape[1:]
        
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
        #
        # TASKS.md 2.8: Use the same per-attribute categorical crossentropy
        # (normalized by log(K)) as the CLI `find_outliers` command so that
        # web UI and CLI outlier rankings match on the same data/model.
        # Previously this path used raw MSE on one-hot vectors, which is a
        # fundamentally different scoring function and produced different
        # rankings than the CLI.
        reconstruction = keras_model.predict(vectorized_df)
        if isinstance(reconstruction, list):
            reconstruction = reconstruction[0]

        reconstruction_error = compute_reconstruction_error(
            vectorized_df, reconstruction, cardinalities
        )
        df['reconstruction_error'] = reconstruction_error

        top_outliers = df.sort_values(by='reconstruction_error', ascending=False).head(100)
        
        # --- NEW: Decode Reconstruction for Visualization ---
        # 1. Align reconstruction with dataframe indices
        reconstruction_df = pd.DataFrame(
            reconstruction, 
            columns=vectorized_df.columns, 
            index=df.index
        )
        
        # 2. Extract only the outliers' reconstruction
        # Use intersection to be safe (though indices should match)
        common_indices = top_outliers.index.intersection(reconstruction_df.index)
        outlier_reconstruction = reconstruction_df.loc[common_indices]
        
        # 3. Invert transformation to get categorical values.
        # Reset to contiguous 0..N index before tabularize so internal
        # concat doesn't misalign rows, then map back to common_indices.
        try:
            contiguous = outlier_reconstruction.reset_index(drop=True)
            decoded_outliers = vectorizer.tabularize_vector(contiguous)
            decoded_outliers.index = common_indices

            # 4. Format the `top_outliers` DataFrame to show "Original -> Predicted"
            for col in decoded_outliers.columns:
                if col in top_outliers.columns:
                    original_vals = top_outliers.loc[common_indices, col].fillna("missing").astype(str)
                    predicted_vals = decoded_outliers.loc[common_indices, col].fillna("missing").astype(str)
                    
                    # zip and format
                    formatted_col = [
                        f"{orig} -> {pred}" if orig != pred else orig
                        for orig, pred in zip(original_vals, predicted_vals)
                    ]
                    
                    top_outliers.loc[common_indices, col] = formatted_col
        except Exception as e:
            logger.error(f"Failed to decocde reconstruction: {e}")
            # Continue without formatting if deserialization fails
            pass
        # --- END NEW ---

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
