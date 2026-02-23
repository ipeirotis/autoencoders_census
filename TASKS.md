# TASKS.md -- Product Launch for Survey Outlier Detection

## 1. Stabilize the Core Pipeline

### 1.1 Unify data-loading return format
The `DataLoader` methods return inconsistent formats -- some return `(df, variable_types)`, others return `(df, metadata_dict)`. The `train` and `find_outliers` commands in `main.py` work around this with nested isinstance checks (lines 174-192, 511-526). Refactor `DataLoader.prepare_original_dataset()` and all `load_*` methods to return a consistent `(DataFrame, dict)` where the dict always has `"variable_types"` and `"ignored_columns"` keys.

### 1.2 Extract shared data-cleaning logic
The data-cleaning pipeline (fillna, astype(str), Rule of 9 filter, vectorization, float32 conversion) is duplicated between the `train` command (main.py lines 196-253), the `find_outliers` command (main.py lines 529-570), and `run_training_pipeline()` (main.py lines 57-91). Extract this into a single reusable function (e.g., `prepare_for_model(df) -> (vectorized_df, variable_types, cardinalities)`) and call it from all three places.

### 1.3 Consolidate dataset config definitions
Dataset-specific column configs (`drop_columns`, `rename_columns`, `interest_columns`) are duplicated across `main.py` (in `train`, `find_outliers`, `evaluate_on_condition`, `pca_baseline`) and `utils.py` (`define_necessary_elements`). Move all dataset configs into a single YAML or Python registry to eliminate drift between copies.

### 1.4 Fix broken test_loader.py tests
`tests/dataset/test_loader.py` references `DataLoader.DATASET_URL_2015` and `DataLoader.DATASET_URL_2017` class attributes and calls `DataLoader()` with no arguments. These no longer match the current `DataLoader.__init__` signature which requires `drop_columns`, `rename_columns`, and `columns_of_interest`. Update the tests to match the current API.

### 1.5 Remove DEBUG print statements
`main.py` contains multiple `print("DEBUG: ...")` statements (lines 138, 175-176, 195, 210, 217, 238, 480, 529, 565, etc.). Replace these with proper `logger.debug()` calls or remove them before release.

## 2. Harden the Upload Pipeline (Web UI)

### 2.1 Implement end-to-end upload processing in worker.py
The `worker.py` `process_upload_job()` function currently only dispatches to Vertex AI. The inline local-processing logic (lines 96-203) is entirely commented out. For a self-contained product, implement a local processing mode that downloads the CSV from GCS, runs the autoencoder pipeline, and writes results to Firestore -- without requiring Vertex AI.

### 2.2 Handle arbitrary user CSV uploads robustly
The current `load_uploaded_csv` path in `DataLoader` calls `prepare_original_dataset()` which bins numeric variables and applies the Rule of 9 filter. Verify this works correctly for CSVs with:
- Mixed numeric and categorical columns
- Columns with special characters, spaces, or unicode in names
- Very wide datasets (100+ columns)
- Datasets with mostly missing values
- Completely numeric datasets (no categorical columns to keep)

Add integration tests covering these edge cases.

### 2.3 Add input validation and error reporting
When a user uploads a malformed CSV (wrong encoding, not actually CSV, empty file, single-column file), the system should return a clear error message to the frontend rather than a stack trace. Add validation in the upload path and propagate structured error messages to Firestore/frontend.

### 2.4 Add API authentication
The Express API endpoints are unauthenticated (noted in README as a current limitation). Add at minimum API key authentication or session-based auth before deploying to any network accessible by others.

## 3. Improve Model Quality and Configurability

### 3.1 Make the "Rule of 9" threshold configurable
The max-unique-values threshold of 9 is hardcoded in `DataLoader.prepare_original_dataset()` (line 439) and duplicated in `main.py` `train`/`find_outliers` commands. Make this a CLI parameter and config option so users can adjust it for their specific datasets.

### 3.2 Support custom model configs for uploaded data
When a user uploads a CSV through the web UI, the system should either auto-select reasonable hyperparameters based on the data shape or allow the user to choose a config preset. Currently, the uploaded data path does not specify which config to use.

### 3.3 Add per-column outlier contribution scores
The `get_outliers_list()` function returns an aggregate reconstruction error per row. For interpretability, also compute and return per-column reconstruction error so users can see *which* survey questions a flagged respondent answered anomalously.

### 3.4 Benchmark model variants
Run systematic comparisons of AE vs. VAE (Gaussian) vs. VAE (Gumbel) vs. PCA baseline on all built-in datasets. Record metrics (accuracy, lift, ROC AUC from `evaluate` command, plus outlier detection precision from `evaluate_on_condition`) and document which approach works best under what conditions.

## 4. Frontend Polish

### 4.1 Display per-column outlier explanations
When the results table shows flagged outliers, include a breakdown of which columns contributed most to the high reconstruction error (depends on task 3.3).

### 4.2 Add progress indicator for long-running jobs
Vertex AI jobs take 10-15 minutes. The frontend polls Firestore for completion. Add a progress state (e.g., "Preprocessing", "Training", "Scoring") that the worker writes to Firestore so users see meaningful status updates instead of just "Processing...".

### 4.3 Allow downloading results as CSV
Add an export button on the results page that lets users download the outlier-scored data as a CSV file.

### 4.4 Support job cancellation
Users currently cannot cancel running jobs from the UI (noted in README). Add a cancellation endpoint that marks the job as cancelled in Firestore and, if using Vertex AI, cancels the cloud job.

## 5. Testing and CI

### 5.1 Get all existing tests passing
Run `python -m pytest tests/ -v` and fix all failures. Known issues: `test_loader.py` has stale API references, and tests may need the SADC CSV data files present in `data/`.

### 5.2 Add integration tests for the CLI pipeline
Write tests that exercise the full `train -> find_outliers` pipeline on a small synthetic dataset (no GCP dependency). Verify that the output `errors.csv` is correctly sorted by reconstruction error.

### 5.3 Add tests for the web upload path
Write tests for `DataLoader.load_uploaded_csv()` with various CSV payloads (valid, empty, malformed, unicode). Mock GCS and Firestore interactions to test the worker pipeline end-to-end.

### 5.4 Set up CI pipeline
Add a GitHub Actions workflow that runs linting and tests on each push. Pin TensorFlow and other heavy dependencies to avoid CI build failures.

## 6. Documentation and Deployment

### 6.1 Write user-facing documentation
Create a short guide explaining: what kind of data to upload, what the outlier scores mean, how to interpret per-column contributions, and what the "Rule of 9" filter does to their data.

### 6.2 Add a sample dataset and walkthrough
Include a small anonymized sample CSV in the repo (or a script to generate synthetic survey data) along with a step-by-step tutorial running the CLI pipeline and interpreting results.

### 6.3 Containerize the full stack
Create a `docker-compose.yml` that runs the React frontend, Express API, and Python worker together. This enables one-command local deployment without needing three terminals.

### 6.4 Document cloud deployment
Write deployment instructions for GCP: building and pushing the training container, setting up Pub/Sub topics/subscriptions, configuring Firestore, and deploying the frontend (Netlify config already exists in `frontend/netlify.toml`).
