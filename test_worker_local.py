import os
import sys
from unittest.mock import MagicMock, patch

# --- STEP 1: MOCK CLOUD SERVICES ---
# We mock Firestore BEFORE importing worker so it doesn't crash on init
mock_db = MagicMock()
with patch('google.cloud.firestore.Client', return_value=mock_db):
    import worker

# Redirect the worker's 'db' variable to our mock
worker.db = mock_db

def test_local_worker():
    print("🚀 Starting Local Worker Test...")
    
    # 1. Pick your local file to simulate an upload
    local_file = "data/sadc_2017only_national_full.csv"
    
    if not os.path.exists(local_file):
        print(f"❌ Error: Could not find {local_file}")
        return

    # 2. Read the real bytes (Simulating GCS download)
    print(f"📂 Loading local file: {local_file}")
    with open(local_file, "rb") as f:
        real_csv_bytes = f.read()
        
    # 3. Patch GCS to return these bytes
    # We intercept 'worker.storage.Client'
    with patch("worker.storage.Client") as MockStorage:
        # Setup the chain: client -> bucket -> blob -> download_as_bytes
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        MockStorage.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_bytes.return_value = real_csv_bytes
        
        # 4. RUN THE WORKER
        print("⚡ Running process_upload_job()...")
        try:
            worker.process_upload_job(
                job_id="test-job-local-123",
                bucket_name="fake-bucket",
                file_path="uploads/fake-file.csv"
            )
        except Exception as e:
            print(f"❌ CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            return

        # 5. VERIFY RESULTS
        # We check what the worker *tried* to write to Firestore
        doc_ref = worker.db.collection.return_value.document.return_value
        
        # Check for .set() or .update()
        if doc_ref.set.called:
            args, _ = doc_ref.set.call_args
            result_data = args[0]
            print_results(result_data)
        elif doc_ref.update.called:
            args, _ = doc_ref.update.call_args
            result_data = args[0]
            print_results(result_data)
        else:
            print("\n❌ FAILED: Worker did not attempt to save results.")

def print_results(data):
    print("\n✅ WORKER FINISHED!")
    print("-" * 30)
    print(f"Status: {data.get('status')}")
    
    if 'error' in data:
        print(f"❌ Error Reported: {data['error']}")
        return

    stats = data.get('stats', {})
    print(f"📊 Stats: {stats.get('total_rows')} rows processed")
    print(f"   - Kept Columns: {len(stats.get('kept_columns', []))}")
    print(f"   - Ignored Columns: {len(stats.get('ignored_columns', []))}")
    
    outliers = data.get('outliers', [])
    print(f"🚩 Outliers Found: {len(outliers)}")
    
    if len(outliers) > 0:
        top_outlier = outliers[0]
        score = top_outlier.get('reconstruction_error', 'N/A')
        print(f"🏆 Top Outlier Score: {score}")

if __name__ == "__main__":
    test_local_worker()